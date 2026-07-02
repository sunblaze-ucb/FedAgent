# Agent-RL 引擎设计 —— rollout、异步、加速与 infra

> **本文档是什么。** FedAgent **agent-RL 子系统**的设计深剖:一个多轮 episode 如何变成训练数据
> (rollout 设计)、系统如何保持忙碌(异步模型及其边界在哪里)、它是如何变快的(作为一叠实测
> 杠杆的加速架构),以及是什么在支撑它(服务、隔离、编排、运维 infra)。
> 配套阅读:[architecture_cn.md](./architecture_cn.md)(系统 overlay、代码地图、round 循环)、
> [acceleration_cn.md](./acceleration_cn.md)(各杠杆的分析;§9 = 副本分片)、
> [acceleration_tier1_report_2026-07-01_cn.md](./acceleration_tier1_report_2026-07-01_cn.md)
> (带日期的深度验证报告)、[migration_cn.md](./migration_cn.md)(相对 verl-agent 0.3.1 fork 的保真)。
>
> **下文每一个设计选择的保真标准:** 与论文**科学等价** —— 相同算法(GRPO,G=8)、相同的
> per-turn prompting、在 3-seed 噪声内复现结论。只有在可证明不触碰科学的地方才取速度。

---

## 1. 问题形态

FedAgent 在**多轮文本环境**(WebShop 15-turn、ALFWorld ≤50-turn)上用 RL 训练 **LLM agent**
(Qwen2.5-1.5B 量级),外面套一层**联邦**包装:N 个 client × T 个 round,每个 round = 每个
client 本地 GRPO/PPO 训练 → FSDP checkpoint 的 FedAvg → 下一轮从聚合结果出发。三个性质主导了
设计:

1. **Episode 又长、又要经由 env** —— 一次 rollout 是与一个外部环境进程的对话,而不是一次
   generate() 调用。延迟同时住在两侧。
2. **原始科学是 per-turn 的("windowed")** —— 论文的 agent 每一 turn 看到的是一个*滑动窗口*
   prompt,而不是不断增长的完整历史。保真性迫使我们做一个定制的 rollout 模式。
3. **联邦把一切都乘上倍数** —— 每次 paper run 约 140 个 (client × round) 训练 job + eval。
   任何固定成本都要付 ~140 次,除非 infra 把它摊销掉。

## 2. 设计总览 —— 三个平面

```
┌─ 编排平面 (fedagent/fed/run_fed.py) ─────────────────────────────────────────────────────┐
│ round 循环 · client subprocess/persistent 生命周期 · FedAvg(torchrun) · merge · eval     │
│ 服务生命周期 (per-client + val, K 个副本) · 端口/隔离 · metrics/summary                  │
└──────────────────────────────────────────────────────────────────────────────────────────┘
          │ 启动 (subprocess 或进程内 persistent)               │ 启/停 (HTTP health)
┌─ 训练器平面 (STOCK verl 0.8, 每个 client-round) ────────────┐  ┌─ ENV 服务平面 ──────────────┐
│ Ray cluster (隔离) · RayPPOTrainer · FSDP actor/ref         │  │ 每 client 一个 FastAPI/      │
│ vLLM server-mode rollout 引擎 (+CUDA graphs)                │  │ uvicorn (×K 副本) · env pool │
│ AgentLoopManager → WindowedAgentLoopWorkers (async)         │  │ sticky session · 独立 conda  │
│   └─ 每条轨迹一个 GymTextAgentLoop 协程 ────── HTTP ────────┼──► env (per-worker textworld / │
│ FSDP→vLLM 权重同步 (ZMQ bucketed 传输)                      │  │ webshop gym) · 分片环境变量  │
└─────────────────────────────────────────────────────────────┘  └──────────────────────────────┘
```

- **训练器平面是 stock verl 0.8** —— 不是 fork。FedAgent 贡献的 agent loop、env client、
  生命周期 driver 都经由 verl 自己的扩展点注册(存在一个 2 行的 verl patch,针对权重传输
  socket 的命名空间;见 §8.3)。
- **Env 住在 HTTP 后面**(`BaseTextEnv` → 轻薄的 async client → 远程服务)。这是核心边界:它
  隔离了依赖地狱(WebShop 的 gym-0.24/pyserini/Java、ALFWorld 的 textworld 栈各自住在*自己的*
  conda 环境里),给联邦一个天然的 per-client 单元(一个服务 = 一个 client 的隐藏转移核),
  并让每一个 env 侧修复对训练器不可见。
- **编排层拥有所有生命周期** —— subprocess、服务、聚合、eval —— 也是唯一知道联邦存在的地方。

## 3. Rollout 设计

### 3.1 per-row 异步契约

verl 0.8 的 agent-loop 为**每一行数据集运行一个协程**。FedAgent 的
[`GymTextAgentLoop`](../agent_loops/gym_text_agent_loop.py) 实现了 episode 循环:

```
system_prompt → reset(seed) → [ generate(turn prompt) → env.step(action) → build next prompt ]* → done
```

- `env.reset/step` 是可 `await` 的(`BaseTextEnv`,[envs/base.py](../envs/base.py));env 实例
  **按 episode** 构造,并全程绑定到一个服务 session(sticky)。
- Seed 按 client 索引(`base_seed + round*100 + client`),因此 client 的*顺序*从不影响结果 ——
  这是 client 并行(#3)的前提条件。
- Reward/成功与否经由 `step()` 的 info 返回;循环记录轨迹,并把 token id + mask 返还给 verl。

### 3.2 Windowed vs concat —— 保真轴

| 模式 | 每 turn 的 prompt | 样本数/episode | 保真性 | 代价 |
|---|---|---|---|---|
| **windowed**(默认;即论文) | 任务 + 最近 `history_length` 个 (obs, action) 对 + 当前 obs | **每 turn 一个**(`run_episode_windowed`) | 忠实于 verl-agent 0.3.1 | 每一 turn 都是一次 vLLM **prefix-cache miss**(gen 相对 concat ~1.43×) |
| concat(stock verl) | 完整的增长历史 | 1 | stock 行为,可选启用 | prefix-cache 命中 |

值得了解的机制:
- **windowed manager**([windowed_manager.py](../agent_loops/windowed_manager.py))在
  `rollout_mode=windowed` 时替换 stock manager;它**自动映射** `agent_name=gym_text →
  gym_text_windowed`,因此同一份 env spec 驱动两种模式;`FEDAGENT_HISTORY_LENGTH`(由 run_fed
  设置:windowed=2、concat=0)覆盖 spec,使同一个文件在两种模式下都保真。
- **Batch 算术:** `train_batch_size=8 × rollout.n=8` = 64 个 episode;windowed 切片在 ALFWorld
  上把它们变成 **~3200 个 per-turn 行**(实测:`adopted 3184 per-turn rows`)—— 每行约
  440-token prompt + 约 100-token response。optimizer step 在这些*窗口*上训练,GRPO 按 episode
  分组(每个 goal G=8)。
- **推论:** per-turn 的 response 很小(~100 tokens)→ 每 turn 的 LLM 时间只有 ~0.2–0.3 s →
  天然的瓶颈候选是 *env* 一侧,而不是生成吞吐(§5、§6)。

## 4. 异步模型 —— 三层结构,以及边界在哪里

**第 1 层 —— 轨迹协程(无上限)。** 全部 64 个(paper batch 下最多 512 个)episode 作为相互
独立的 asyncio task 运行;client 侧有意**不设 semaphore**。当轨迹 A 在 await `env.step()` 时,
B..N 在生成;当 A 在生成时,B..N 在 step。env 延迟与 LLM 延迟在轨迹*之间*互相藏进对方下面。

**第 2 层 —— vLLM 动态 batching。** 所有并发的 `generate()` 调用汇入 vLLM server-mode 引擎
(每个 GPU 组一个),由其持续地 batch(CUDA graphs 预捕获,prefix caching 开启)。在 1.5B
规模下,生成吞吐实际上从来不是限制项。

**第 3 层 —— env 服务的异步 handler。** 每个服务都是带 `async def` endpoint 的 FastAPI;
阻塞式 env 工作跑在 `asyncio.to_thread` 里。并发在这一层被限制 —— 既有有意为之,也有意外:
- **有意为之:** env pool(`asyncio.Queue`、`*_pool_size`)限定活跃 session 数;`/create`
  会*阻塞*到有 env 空闲为止(有意设计 —— 见 §5)。
- **意外(2026-07-01 已修复):** ALFWorld 的 `_TW_LOCK` 把一个进程内的全部 env stepping
  串行化(tatsu PDDL parser 是进程级全局单例);WebShop 的纯 Python `env.step` 在 **GIL** 上
  竞争。一个服务进程 = 一条单行道。**这就是隐藏的边界**:在 paper batch 下,它让 ALFWorld 的
  rollout 占到训练一步的 73 %,且随 GPU 数量增加保持平坦。修复是副本分片(§6.3)——
  K 个进程 = K 把锁/GIL。

**由此得到的延迟模型**(真正卡住 `timing_s/gen` 的东西):

```
gen ≈ max( 最慢 episode 的关键路径,  Σ env-steps / (K × 每进程服务速率),  LLM )
      └ ~50 turns × (LLM 0.2-0.3s + env 86ms + HTTP)   └ 被副本除掉的锁/GIL 项
```

分片之后第一项主导(ALFWorld 上 ~52–66 s);分片之前是第二项(219 s+)。

## 5. HTTP 边界 —— 让 512 个并发 episode 保持正确的契约细节

env client([webshop_env.py](../envs/webshop/webshop_env.py) /
[alfworld_env.py](../envs/alfworld/alfworld_env.py))看上去小得出奇;每个设计点的存在,都是
因为某种故障模式逼出来的:

| 机制 | 为什么 |
|---|---|
| **每条轨迹一个 `httpx.AsyncClient`**,在 `finally` 中关闭 | 按 episode 隔离;轨迹之间不共享连接池状态 |
| **传输错误时带 backoff+jitter 的重试**(`/create`、`/reset`、`/step`) | 满 batch 时,数百个 episode 几乎同时打到一个服务;socket 在响应中途被 reset(`httpx.ReadError`)—— 重试把这场踩踏摊开 |
| **`/step` 上的幂等键 `step_id`** | `/step` 会改变 env 状态;盲目重试会重复施加。服务端对每个 id 恰好应用一次并重放缓存的响应 —— 重试安全,又不放弃 exactly-once 语义。HTTP 4xx/5xx *不*重试(真正的失步必须大声暴露)。 |
| **阻塞式 `/create`**(禁用 read-timeout,connect/write 有界) | 从 pool 借一个 env,本来就要合法地等到有 env 空闲 —— 这个等待随 batch/pool 伸缩,而不是随某个固定超时;这里曾经的硬超时杀死过整个 rollout。没有超时 → 不会重发 → 不会重复借用。 |
| **sticky session**(session_id → env,`/create` 时借出、`/close` 时归还) | episode 状态住在服务端的那个 env 实例里 |
| **副本路由**(`_pick_replica`) | 服务 URL 可以是逗号分隔的列表;每个 episode 以轮询方式绑定一个副本(PID 偏移游标,每 worker 均衡 ±1)—— 此后保持 sticky。一个实现点同时覆盖两个 env × 全部三种路由来源。 |

路由优先级(`resolve_service_url`):`FEDAGENT_SERVICE_URL_FILE`(persistent 模式的 per-client
重指向;之所以用文件,是因为一个进程的 `os.environ` 没法按 client 变化)→ 进程环境变量
(subprocess 模式)→ spec → 默认值。

## 6. Env 服务 infra

### 6.1 服务解剖(两个 env,同一形状)

FastAPI + uvicorn(单进程),lifespan 预热用并行线程构建 env pool;endpoint 为
`/health /create /reset /step /close`;per-session `asyncio.Lock`(串行化的是*同一个* session
的重试,而不是不同的 session);每请求做线程 offload。**分片环境变量桥**
(`CLIENT_ID/CLIENT_NUM/PARTITION_STRATEGY/OMEGA/...`)让服务在启动时恰好构建出它那个 client
的数据 shard —— 异质性在*这里*注入,对训练器不可见。

### 6.2 per-client 服务 + 共享的 val 服务

每轮为每个被选中的 client 起一个服务(设计 A:服务 == 该 client 的隐藏转移核),按轮惰性启动,
health 门控(`/health` 轮询,超时给得很宽 —— ALFWorld 启动时要遍历一个 8810 局游戏的集合,
约 3 分钟)。一个**未被扰动的 val 服务**(held-out split,不分片)承担每轮的全局 eval;
client 与 val 的端口带有守卫检查。

### 6.3 副本分片(`alfworld_replicas` / `webshop_replicas`,2026-07-01)

§4 第 3 层的串行化边界,在不触碰服务代码的前提下被移除:每个 client 在同一个 shard 上运行
**K 个相同的进程**(端口 `base + c*K + j`;pool 近似均分 +2 余量;val 以相同的 K 复制)。
分布相同 → 科学安全;K=1 与旧行为逐字节相同。按机制→对照→组件→端到端验证:ALFWorld 4-GPU
step **298→127.6 s**,整跑 **3509→2412 s**;WebShop(GPU-bound)−12 %。完整数据:
[带日期的报告](./acceleration_tier1_report_2026-07-01_cn.md)。

## 7. 训练器平面 —— stock verl 0.8,以及两条生命周期接缝

- **训练:** stock `RayPPOTrainer`,FSDP actor(+ref;PPO/GAE 时 +critic),GRPO advantage 在
  G=8 的 episode 组上计算;`old_log_prob`/`ref` 在 FSDP 下重算(精确优先于速度 —— 有意不取
  vLLM 的 logprobs)。
- **Rollout 引擎:** server 模式的 vLLM(每个 GPU 组一个 `vLLMHttpServer`),初始化时
  dummy-load,之后每次 rollout 经 verl 的分桶 **ZMQ IPC 传输**做 **FSDP→vLLM 权重同步**;
  引擎在两次 rollout 之间休眠,同步时唤醒。
- **接缝 1 —— 每 job 一个 cluster vs verl 的单 cluster 假设(§8.3):** verl 用 Ray job id 为
  共享的宿主机资源(权重传输的 `/tmp` socket)做命名空间,前提假设是一个共享 cluster。
  FedAgent 把每个 client/eval 跑成*各自的* Ray cluster,而相互隔离的 cluster 都铸出相同的首个
  job id → 相同的 socket 路径 → 权重发送被交叉接线(一次 44 分钟的静默死锁)。修复是每次启动
  生成唯一的 `VERL_RAY_JOB_ID`(+ 那个 2 行的 verl honor patch)。FedAvg 的 `torchrun`
  rendezvous 端口是同一类 bug(用 `--standalone` 修复)。**学到的设计规则:每一个共享宿主机
  资源都必须带一个 per-job 唯一的名字。**
- **接缝 2 —— 在 `fit()` 之外驱动 `_validate()`(worker eval):** verl 的 validation 假定
  自己处于 `fit()` 的引擎生命周期中。persistent worker eval 复现了它:先设好 `global_steps`,
  在 validate *之前*执行 `update_weights`(同步+唤醒)—— 否则 vLLM 仍然拿着 dummy 权重
  (一类 CUDA illegal-access 崩溃的根因)—— 再重新初始化 dump executor,遵守
  `val_batch_size`,结束后让引擎休眠。

## 8. 联邦编排与进程生命周期

### 8.1 三种生命周期模式(杠杆 #4)

| 模式 | 进程 | 冷启动付几次 | 适用 |
|---|---|---|---|
| subprocess(基线) | 每个 (client, round) 一个 `main_ppo_fed` | 每次 paper run ~140 次(曾占墙钟 **76–88 %**) | 最大隔离;调试 |
| `persistent: true` | 每轮一个进程,进程内按 client 重置 | 每轮一次 | — |
| `cross_round: true` | **整个 run 一个进程** | **一次** | 生产默认 |

persistent runner 通过 URL 文件(§5)为 client 重指向,在进程内重置数据/seed,并让
FedAvg/merge 保持**外部化且逐字节相同** —— 经过多轮复合后的等价性 `max|Δ|≈1e-5` 是受检查的
不变量。

### 8.2 聚合流水线

`save FSDP shards (save_contents=[model]) → torchrun --standalone matched-PG FedAvg over shards →
verl.model_merger → HF dir → next round / eval load`。精确平均、与顺序无关;PPO 通过同一路径
联邦 critic。(已知的下一个优化:直接加载 shard 以跳过 HF merge —— 见 §9"接下来"。)

### 8.3 隔离模型(是什么让同节点并发变得安全)

每个被启动的 job:`CUDA_VISIBLE_DEVICES`(互不相交的 GPU)+ `RAY_TMPDIR`(自己的 cluster)+
`VERL_RAY_JOB_ID`(自己的 socket 命名空间)+ 端口带(client `base + c*K + j`、val `val + j`,
有守卫检查)+ `--standalone` FedAvg。在这套组合之下,2–3 个并发 verl job 可以干净地共存
(GPU 验证过;#3 client 并行与 eval∥train 就是这么跑起来的)。

### 8.4 Eval 设计

Eval 是**只读的**(加载 merge 后的 model_r,不写回任何东西)→ 从构造上就零等价性风险,因此
可以自由挪动:`inline`(阻塞)/ `parallel`(互不相交的 GPU)/ `shared`(第二个引擎,0.3
util)/ `worker`(persistent 训练器的热引擎 —— 没有第二个引擎、没有冷启动)。节奏:论文的
红线 = 每轮对该轮聚合模型做一次 eval;`client_end_eval` 额外加上 per-client 的圆点。模式选择
纯粹是墙钟问题(实测排名因 env 而异 —— WebShop parallel 第一,ALFWorld worker 第一)。

## 9. 加速架构 —— 一叠实测过的杠杆

每个杠杆都攻击 `round ≈ cold-start + rollout + train-compute + eval` 中的一项,而且每一个都
暴露出下一个瓶颈(完整谱系 + 数据见
[带日期的报告](./acceleration_tier1_report_2026-07-01_cn.md) §7):

| 杠杆 | 攻击的项 | 机制 | 头条结果 |
|---|---|---|---|
| #4 persistent / cross_round | 冷启动 | 一个进程,热替换权重 | −43 % / −62 % |
| eval mode(worker/parallel) | eval | 热引擎或互不相交 GPU 上的 eval | eval ≈ 免费,移出关键路径 |
| #3 client 并行 | train-compute | 次线性 FSDP → 2×2 胜过 4-串行 | −35 %(WebShop) |
| 并发修复 | (使能项) | 共享资源的 per-job 唯一名字 | 3 个 job 共存 |
| **副本分片** | rollout(env) | K 个服务进程 = K 把锁/GIL | ALFWorld step −57 %,整跑 −31 % |

**新环境的决策规则**(可迁移的方法):在两种 GPU 数量下各跑一次 1-step `timing_s` 探针。
**gen 平坦 → env-bound → `*_replicas`。gen 随卡数伸缩 → GPU-bound → 加 GPU / #3。** 当前配方:
ALFWorld 4×H100 = `cross_round + eval_mode=worker + alfworld_replicas: 8`;1×H100 =
`alfworld_replicas: 4`;WebShop = 先加 GPU,副本可选(−12 %)。

**接下来(已识别、未构建):** 轮间管道(直接加载 shard / 进程内 FedAvg / 服务 manifest
缓存 —— 那道 −31 % vs −65 % 的差距)、#3 × 副本的组合(预测约 −18 %,需要一个并行轮
launcher)、多节点 #3。

## 10. 运维 infra(SLURM)—— 长任务的运行与看护

从惨痛教训里活下来的模式(细节见 EXPERIMENTS.md / memory):

- **耐久启动:** 把 driver 跑在一个*长生命周期 `srun --overlap` step 的前台*。`setsid nohup`
  孤儿进程活不下来 —— 启动它的 step 一退出,Slurm 的 cgroup 清理就会杀掉它(setsid 逃出的是
  session,不是 cgroup)。
- **GPFS 上的自排队 barrier 文件:** 每个 driver 向 barrier 文件追加 `[stage] rc=… wall=…`
  行 + `=== DONE ===`;下一个 driver 对它自旋等待。监控从登录节点直接读 GPFS(读不需要 srun)。
- **存活判定 = 日志 mtime 是否陈旧 + GPU 利用率** —— 永远不要 `pgrep -f <pattern>`(它会
  自匹配到兄弟 watcher;曾造成一次 53 分钟的盲区)。
- **健康噪声分诊:** teardown 时的 `DataLoader worker killed` / `Engine core died` 是无害的
  `__del__` 噪声(rc 不受影响);静默的 0 % 利用率 + 日志陈旧才是真正的死亡。
- **端口:** 每份 config 都把 client/val/副本端口带写进文档;每个实验用新的端口带,避免陈旧
  绑定。

## 11. 设计原则(蒸馏)

1. **Overlay,而非 fork。** stock verl 0.8;只用扩展点;唯一的例外是 2 行,以 patch 文件形式
   携带、由环境变量启用,未设置时逐字节相同。
2. **Env 住在 HTTP 后面。** 依赖隔离、per-client 的联邦单元,以及一个训练器永远看不见的
   加速面(副本)。
3. **先保真,再提速。** windowed 模式为论文的 per-turn prompt 付出 1.43× 的代价;速度来自
   生命周期/调度/服务 —— 这些层可证明不触碰采样。
4. **Eval 是只读的 —— 利用这一点。** 任何只读的东西都可以零风险地挪到任何地方(热引擎、
   空闲 GPU、延后执行)。
5. **每一个共享宿主机资源都要有 per-job 唯一的名字。** Ray job id、rendezvous 端口、tmp
   socket —— 这是"每 client 一个 cluster"这类设计反复出现的 bug 类别。
6. **测量 → 分解 → 修复 → 四个层级验证。** 每个杠杆都靠机制 / 对照 / 组件 / 端到端证据加上
   预先登记的预测赢得席位 —— 两次预测落空也与命中一起写进了文档。

## 另见

- [architecture_cn.md](./architecture_cn.md) —— 系统代码地图、round 循环、单个 subprocess 的解剖
- [acceleration_cn.md](./acceleration_cn.md) · [acceleration_results_cn.md](./acceleration_results_cn.md) ——
  杠杆分析 + 数字
- [acceleration_tier1_report_2026-07-01_cn.md](./acceleration_tier1_report_2026-07-01_cn.md) ——
  副本分片深度验证报告
- [acceleration_cross_env_cn.md](./acceleration_cross_env_cn.md) —— WebShop vs ALFWorld 综述
- [migration_cn.md](./migration_cn.md) —— 相对 verl-agent 0.3.1 fork 改了什么,以及为什么保真
- 英文版: [agent_rl_design.md](./agent_rl_design.md)
