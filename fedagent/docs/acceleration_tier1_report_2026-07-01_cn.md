# 深度验证报告（2026-07-01）—— env 服务副本分片与镜像相反的瓶颈

> **本文档是什么。** 2026-07-01 深度验证 campaign 的自包含报告:测了什么、**构建了什么及为什么**、
> 完整的实验数据,以及**这项新改进与此前每一个加速杠杆的关系**。配套阅读:
> [acceleration_cn.md](./acceleration_cn.md)(持续更新的分析;§9 是本次 campaign 的摘要)、
> [acceleration_cross_env_cn.md](./acceleration_cross_env_cn.md)(WebShop-vs-ALFWorld 综述)、
> [alfworld_testing_cn.md](./alfworld_testing_cn.md)(ALFWorld 策略文档;§7)、
> [acceleration_results_cn.md](./acceleration_results_cn.md)(一眼看数)。
>
> **TL;DR。** 两个基准环境的瓶颈**互为镜像**:ALFWorld 4-GPU 训练一步有 **73 %** 花在一个被锁
> 串行化的 env 服务进程里;WebShop 1-GPU 一步有 **74 %** 花在 GPU 算力上。我们构建了 **env 服务
> 副本分片**(`alfworld_replicas` / `webshop_replicas`)并在四个层级验证 —— 机制、对照、组件、
> 端到端:ALFWorld 4-GPU step **298 → 127.6 s(−57 %)**,联邦整跑 **3509 → 2412 s(−31 %)**;
> WebShop(正如其分解所预测)只有 −12 % 收益。途中还修正了两条已发布的旧结论。

**下文所有数字的公共常量**(除非另行说明):Qwen2.5-1.5B-Instruct,GRPO G=8,windowed
rollout,`train_batch_size=8 × rollout.n=8` = 64 episodes/step,seed 42,探针关闭 eval。
ALFWorld:`response_length=4096`,≤50 turns,~3200 env-steps/optimizer-step。WebShop:`webshop_15`
(15 turns),`response_length=512`,960 env-steps/step。硬件:4×H100(qgpu3021,64 核)与
1×H100(qgpu3010,8 核)。探针为 1 个 optimizer step、单次运行(±5–10 % 噪声)。

---

## 1. 起点 —— 此前的改进栈

此前每一个杠杆都攻击每轮墙钟方程
(`round ≈ cold-start + rollout + train-compute + eval`)中的某一项:

| 杠杆(时间) | 攻击的项 | 头条结果 |
|---|---|---|
| **#4 persistent / cross_round**(6 月) | 冷启动(曾占墙钟 **76–88 %**) | 每轮 −43 %,跨轮 −62 %;等价性 `max\|Δ\|≈1e-5` |
| **eval mode** inline/parallel/shared/worker(6 月) | eval 的摆放位置 | WebShop n=500:parallel 2493 < worker 2637 < inline 3090 < shared 3316 |
| **#3 client-parallel**(6 月) | 借助次线性 FSDP 的训练计算 | 2×2 GPU = 727 s vs 串行 1116 s(−35 %);暴露并修复两个并发 bug(FedAvg :29500、ZMQ `/tmp` socket `VERL_RAY_JOB_ID`) |
| **ALFWorld campaign**(2026-06-30) | 测量 | eval-mode 排名翻转(worker 3509 最快);GPU-scaling 探针:**gen 在 1/2/4 GPU 下平坦 228→225→219 s** → rollout 是 env-bound |

06-30 campaign 止于一个测量、而非修复:ALFWorld 的 rollout 时间对加卡没有反应。
**本报告就是那个修复 —— 以及证明。**

## 2. 诊断 —— gen 为什么平坦(以及为什么此前没人修)

三个事实,各自在 07-01 独立验证,合上了因果链:

1. **代码审计。** ALFWorld 服务用一把进程级全局 `threading.Lock`(`_TW_LOCK`,
   `envs/alfworld/service/server.py:180`)包住**每一次** textworld `reset`/`step`(:315、:350),
   因为 tatsu PDDL parser 是进程级全局可变单例(并发使用会破坏它的 rule 栈)。服务以**单个**
   uvicorn worker 运行;env pool 是进程内的 `asyncio.Queue` —— 所以 `alfworld_pool_size`
   绕不过这把锁,agent-loop 无上限的 async 并发全部汇入同一行代码。
2. **算术对账。** acceleration.md §2.2 早已测出这把锁 **86 ms/step** —— 但那是在 160 步的
   batch 上(13.7 s),并被归档为"已理解且有界"。在 paper 规模下,一个 windowed step 约有
   **3200** 个 env-step(实测:`adopted 3184 per-turn rows`),而 86 ms × 3200 ≈ **275 s** ——
   正好括住实测 gen 的 219–228 s。那篇文档的*常数*是对的;它的*定性裁定*没能扛住 batch 规模的放大。
3. **设计假设被反转。** 这把锁自己的注释写道:*"env 转移是毫秒级,相比 LLM 生成的**秒级**,
   pool 的真实并发收益得以保留。"* 在 windowed 模式下平均回复是 **~100 tokens/turn**(实测
   99.6):LLM 一轮只要 ~0.2–0.3 s,不是秒级。LLM 藏进了锁的下面 —— 与假设正好相反 ——
   被串行化的 env 成为地板:**占 4-GPU 一步的 73 %**。

为什么此前没修:在 6 月各次 smoke 的 batch 规模下这笔税只有几秒;直到 06-30 的 paper 规模探针
才让它成为主导,也只有 gen 平坦这一特征才定位出它*坐在哪里*。

## 3. 构建了什么 —— 副本分片,及其设计决策

**改动(已提交 `e593dd2`):** 每个 client 在*同一个*数据 shard 上运行 **K 个相同的服务进程**;
每个 episode 绑定到一个副本。

| 决策 | 选择 | 理由(被否掉的备选) |
|---|---|---|
| 在哪里打破这把锁 | **K 个独立服务进程** | (a) 给 textworld/tatsu 打 per-env parser 补丁 = 深入上游的手术,有回归风险;(b) 服务*内部*的 multiprocessing pool = 要加 IPC 层 + fork 安全工作。(c) K 个进程**零服务代码改动** —— parser 是 per-process 的,所以 K 个进程 = 白得 K 把独立的锁。 |
| Client 侧路由 | 逗号分隔的 URL 列表,只在**一个**地方处理:`envs/base.py::resolve_service_url` | 一处修改同时覆盖两个 env × 全部三种路由来源(persistent URL 文件 / 进程 env / spec);每个 env 实例在构造时选定一个副本 → 对整个 episode 天然**粘性**。 |
| 均衡策略 | **轮询**(PID 偏移游标),而非哈希 | 有界的不均衡(每个 agent-loop worker ±1)让 per-replica pool 取 `ceil(pool/K)+2` 也不会出现 `/create` 饥饿;哈希会带来 √n 量级的偏斜,需要大得多的余量。 |
| Pool 语义 | `*_pool_size` 仍是**每 client 总量**,在各副本间近似均分(+2 余量) | 保住文档化的不变量"pool ≥ gen_batch";配置保持可比。 |
| Val 服务 | 以**相同的 K** 复制 | eval 走的是同一把锁(48–140 局 × 50 turns);这份收益同样适用于每轮的 eval。 |
| 向后兼容 | `K=1` **精确**退化为旧的单服务(相同端口 `base+c`、相同日志名) | 默认路径逐字节相同;6 月的所有配置不受影响。 |
| 端口 | `base + c*K + j`;val 为 `val_port + j`;碰撞守卫改为感知端口带 | 任意 K 下各 client 的端口带保持互不相交。 |

**等价性论证(科学安全性)。** 每个副本收到该 client 完全相同的 shard 环境变量
(`CLIENT_ID`/`CLIENT_NUM`/分片旋钮)→ 相同的游戏/目标分布;episode 从同一集合 iid 采样,
所以*哪个*副本服务某个 episode 只是调度细节,与现有 pool 借用顺序同属一类。训练器平面不受
触动 —— 已由测量确认(update_actor 43.3 → 43.7 s,old_log_prob/ref 在噪声内)。

## 4. Campaign 设计

两份算力分配并行推进,由自排队 driver 驱动(每个阶段在上一个 barrier 文件写出 DONE 后启动;
沿用 06-30 的持久前台 `srun` 模式):

- **1×H100 / 8 核(剩 2.5 h walltime)** —— 先跑与代码无关的探针:WebShop 1-GPU
  分解,然后 ALFWorld K=4 / K=8,再跑 K=1+pool-64 **对照**。
- **4×H100** —— WebShop 4-GPU + 1-GPU 固定(同节点成对),然后 ALFWorld K=8 组件探针,
  然后**端到端 A/B**,最后是 WebShop 杠杆对。

每个实验都带一条**在启动前写下的可证伪数值预测**(§8)。
副本代码路径先经单元 smoke 演练(路由均衡、URL 构造、K=1 退化),再由 11 个 GPU run 覆盖
每条路径(K=1 legacy webshop+alfworld / K>1 train / K>1
train+val+cross_round+worker / K>1 webshop)。当天全部 13 个 run:rc=0。

## 5. 完整实验数据

### 5.1 ALFWorld —— 基线(2026-06-30,供参照)

| GPU 数 | gen | old_log_prob | ref | update_actor | **step** | 墙钟(1 步,含固定开销) |
|---|---|---|---|---|---|---|
| 1 | 228.3 | 52.2 | 107.9 | 140.0 | **534.5** | 1050 s |
| 2 | 225.3 | 31.9 | 33.0 | 92.2 | **386.9** | 865 s |
| 4 | 219.3 | 18.5 | 14.1 | 43.3 | **298.4** | 778 s |

gen 平坦(GPU 增至 4× 时仅 −4 %);update_actor 近似线性(3.2×);pool_size 为 8。

### 5.2 ALFWorld —— 机制扫描 + 对照(1×H100,qgpu3010,pool 总量 64)

| 配置 | K | gen | update_actor | **step** | 墙钟 |
|---|---|---|---|---|---|
| `alf_scale_g1_r1n1`(**对照**) | 1 | **217.5** | 139.5 | 511.7 | 1056 s |
| `alf_scale_g1_r4n1` | 4 | **65.8** | 138.8 | 358.1 | 714 s |
| `alf_scale_g1_r8n1` | 8 | **61.8** | 137.3 | 350.2 | 702 s |

- 对照(K=1,pool 8→64):gen 217.5 ≈ 228 基线 → **pool 大小无关紧要;锁就是全部
  答案。**(排除了"只是 pool 更大了"这种解释。)
- 同节点、同 pool、单一变量 K:**217.5 → 65.8 → 61.8**。
- K=4 ≈ K=8 → 剩余的 ~60 s 是新地板(episode 关键路径:~50 turns ×
  (LLM ~0.2–0.3 s + env 86 ms/K + HTTP));8 核节点只需要 **K=4**。

### 5.3 ALFWorld —— K=8 组件探针(qgpu3021)

| 配置 | GPU 数 | gen | old_log_prob | ref | update_actor | **step** | vs 基线 |
|---|---|---|---|---|---|---|---|
| `alf_scale_g4_r8` | 4 | **51.7** | 15.2 | 14.1 | 43.7 | **127.6** | **−57 %**(2.34×) |
| `alf_scale_g1_r8` | 1 | 62.7 | 40.7 | 108.1 | 141.0 | **358.8** | −33 % |

- 4 GPU 下 gen −76 %;**update_actor 43.3→43.7 = GPU 平面不受触动**(无副作用检查)。
- 1-GPU step 跨节点一致(358.8 vs 358.1/350.2)→ 与节点无关。
- **修复后的后果:** env 地板消失后,GPU 算力成为主导 → ALFWorld 1-vs-4-GPU
  的每步惩罚涨到 **1.79× → 2.81×**。"每 client 1 GPU"布局如今在两个 env 上都死了。

### 5.4 ALFWorld —— 端到端 A/B(06-30 eval-mode sweep 的同一配置,唯一改动是 + `alfworld_replicas: 8`)

2 client × 2 round,每轮 eval(48 局 val),`cross_round + eval_mode=worker`,train **和
val** 服务都分片:

| | 基线 `alf_em_worker` | `alf_em_worker_r8` | Δ |
|---|---|---|---|
| 总墙钟 | 3509 s | **2412 s** | **−31 %** |
| 训练 step(4 个 client-round) | 408 / 320 / 338 / 370(均值 359) | 147 / 113 / 121 / 126(均值 **127**) | **−65 %** |
| rc / val | 0 / 健康 | 0 / 健康(r2 success 0.083) | ✓ |

−31 % 与 −65 % 之间的差距定位出**下一个瓶颈**:2412 s 里 step 只占 508 s,整跑如今由
训练器冷启动、聚合/merge、服务加载与 eval 管线主导(即 Tier-2 轮间候选项)。

### 5.5 WebShop —— 首次计时分解(修正一条推断)

| 配置 | 节点 | GPU 数 | gen | old_log_prob | ref | update_actor | GPU-Σ | **step** | 墙钟 |
|---|---|---|---|---|---|---|---|---|---|
| `ws_scale_g1` | 8 核 | 1 | 50.6 | 23.8 | 35.0 | 87.9 | 146.7 (73 %) | 202.1 | 619 s |
| `ws_scale_g1b` | 64 核 | 1 | 54.6 | 26.6 | 39.1 | 100.1 | **165.8 (74 %)** | **225.2** | 543 s |
| `ws_scale_g4` | 64 核 | 4 | 44.1 | 10.2 | 8.8 | 27.9 | 46.9 (50 %) | **93.4** | 481 s |

- **WebShop 是 GPU-compute-bound** —— 与 ALFWorld 互为镜像。gen 基本平坦(54.6→44.1)但
  *很小*;GPU 算力 scaling 3.54×。
- **修正:** 已发布的"1-GPU 只慢 1.37×"是被 ~390 s 固定开销稀释的 3-step 墙钟 ——
  对账:995 ≈ 3 × 202 + 390 ✓。每步的 1-vs-4 惩罚是 **2.41×**。
- 已排除节点效应:8 核与 64 核节点之间 ±10 %。

### 5.6 WebShop —— 杠杆对(4×H100)

| 配置 | pool | K | gen | update_actor | **step** |
|---|---|---|---|---|---|
| `ws_scale_g4`(基线) | 16 | 1 | 44.1 | 27.9 | 93.4 |
| `ws_scale_g4_p64` | 64 | 1 | **50.1** ⚠ | 27.9 | 100.5 |
| `ws_scale_g4_p64r4` | 64 | 4 | **35.7** | 26.0 | **82.2(−12 %)** |

- 单独把 pool 16→64 反而**有害**(gen +14 %):同一个进程里 64 个并发 session 放大了 GIL
  竞争 —— "wave-throttle"假说被**证伪**。
- 副本给 WebShop 带来真实但有限的 −12 %(GIL 分片)。WebShop 的杠杆仍然是 GPU 数量。

## 6. 分析 —— 这些数字为什么是这个样子

**一行物理。** 一个训练 step = `gen + GPU-compute`。gen 由 env 服务串行化的部分决定;
GPU-compute 由 FSDP scaling 决定。两个环境坐在两个极端:ALFWorld = 3200 个被锁串行化的
env-step(gen ≈ 那把锁,73 %),WebShop = 960 个廉价 env step 对上一条沉重的 4096/512-token
FSDP+logprob 流水线(GPU 74 %)。因此同一个干预(副本)在一边值 **−57 %**、在另一边只值
**−12 %** —— 而*先把这个划分测清楚*(在两个 GPU 数下各跑一次 1-step `timing_s` 探针)就是可迁移
到任何新 env 的决策规则。

**为什么剩余地板是 ~60 s。** 分片之后,每副本的串行化负载(K=8 时 219/K ≈ 27 s)降到
episode 关键路径(~50 turns × ~0.3 s ≈ 15–20 s,外加尾部/HTTP)之下:此时约束 gen 的是最长的
单个 episode,而非 env 的总吞吐。更多副本帮不上忙;更短的 episode 或更少的 turns 才行
(属科学性改动 —— 超出范围)。

**为什么端到端 −31 % ≠ step −65 %。** 固定成本(一次训练器冷启动、K 次服务加载、聚合 + HF
merge、eval 引擎同步)不会随锁一起缩小。在 paper 规模(70 round × 3-epoch round)下 step
项占比更高,端到端收益会向 step 收益逼近 —— 但这个外推*尚未实测*(待办项)。

**为什么修复后惩罚反而变大(1.79× → 2.81×)。** 从 1-vs-4 比值的分子与分母里同时移除一个
与 GPU 数无关的项(env),留下的就是强 scaling 的 GPU-compute 项。反直觉但算术上必然 ——
且有策略意义:env 修复让每 client 配*更多* GPU 变得更有价值,而不是更少。

## 7. 与此前改进栈的组合方式

谱系 —— 每个杠杆都暴露出下一个瓶颈:

```
#4 persistent/cross_round    消灭冷启动(占墙钟 76–88 %)           → 暴露 rollout 与 eval
eval mode(worker/parallel)  把 eval 移出关键路径                   → 暴露训练 step
#3 client-parallel           利用次线性 FSDP(2×2 −35 %)           → 需要那两个并发修复
:29500 + VERL_RAY_JOB_ID     让任何同节点并发都安全                 → #3/eval∥train 可靠
06-30 测量                   发现 ALFWorld 一步 73 % 在 env         → 本轮:
Tier-1 副本分片              消灭 env 串行化(step −57 %)           → 下一步:GPU 算力 + 轮间管线
```

组合现状:
- **与 #4 + worker-eval 叠加:** 端到端 A/B *就是* `cross_round + worker + replicas` ——
  三者干净组合(那次 run 即生产配方)。
- **与 #3 叠加(有预测,尚未跑):** 修复后 2-GPU step ≈ 55 + 157 ≈ 210 s → 两个并行
  2-GPU client ≈ 210 s vs 串行 2×127.6 = 255 s → 再省 ~−18 %;需要 `run_fed` 里的
  parallel-round launcher(6 月 #3 的证据来自两个独立进程)。
- **与 ZMQ 修复正交:** 副本只在 *env* 平面增加进程;权重传输 socket 命名空间是
  per-verl-job 的,不受影响(worker_r8 在其下干净跑完)。
- **对既有文档的强制修正:** §2.2 "有界"(batch 规模放大)、§7.7 "WebShop
  env-latency-bound / 1.37×"(固定开销稀释)—— 两处均已就地加上带日期的修正注记。

## 8. 预测记分卡

| 预测(启动前写下) | 实测 | 裁定 |
|---|---|---|
| ALFWorld gen 219 → 40–70 s(K=8) | 51.7 | ✅ |
| ALFWorld step 298 → 120–150 s | 127.6 | ✅ |
| ALFWorld 1-GPU step → ~350 s | 350–359(两个节点) | ✅ |
| 对照:K=1+pool64 的 gen 停在 ~220 s | 217.5 | ✅ |
| 端到端 < 2500 s(vs 3509) | 2412 | ✅ |
| 小节点:更少副本即可 | K=4 ≈ K=8 | ✅ |
| WebShop gen 主导且平坦 | 基本平坦 ✓ 但只占 24–47 % —— **并不主导** | ⚠ 半错(早先反解出的"t_env ≈ 455 s/round"其实是固定开销,不是 env) |
| WebShop pool 16→64 能降 gen | gen +14 %(GIL 放大) | ❌ 证伪 |

两个失误与命中同样有信息量:它们在消耗 GPU 小时*之前*就毙掉了 WebShop 副本计划,并且被记入
文档而非悄悄丢弃。

## 9. 生产配方

| 硬件 × env | 配方 | 预期 |
|---|---|---|
| 4×H100 × ALFWorld | `cross_round: true` + `eval_mode: worker` + `alfworld_replicas: 8` + `alfworld_pool_size: 64` | step 2.34×;端到端 ≥ −31 % |
| 1×H100(8 核)× ALFWorld | 同上,`alfworld_replicas: 4` | step −33 % |
| 4×H100 × WebShop | 6 月快路径(#3 2×2 或 4-GPU+worker)+ 可选 `webshop_replicas: 4`(pool ≥ batch) | 副本再加 −12 % |
| 1×H100 × WebShop | 服务层没有魔法;每步 2.41× vs 4-GPU | 尽量避免 |
| 任何新 env | 在 2 个 GPU 数下各跑 1-step `timing_s` 探针:**gen 平坦 → `*_replicas`;gen 随卡 scaling → 加 GPU** | 决策规则 |

## 10. 局限与下一步

- **1 步、单次运行的探针**(±5–10 %);多轮稳态墙钟尚未测量 —— 端到端收益在 paper 规模下
  应当*增长*、逼近 −65 %,未验证。
- **#3 × 副本组合**未实现(需要 parallel-round launcher;预测 ~−18 %)。
- **#4 在 ALFWorld 上的单独贡献**从未再次隔离(它同时位于 A/B 两臂之内)。
- **副本启动成本**:K 次并行的游戏收集遍历(~3–5 min);manifest 缓存可以消除它。
- **下一个瓶颈**(来自 −31 % vs −65 % 的差距):轮间管线 —— 把 shard 直接加载进持久训练器
  (跳过 HF merge)、进程内 FedAvg、服务加载缓存。

## 11. 出处

- **代码:** commit `e593dd2` —— `fedagent/envs/base.py`(`_pick_replica` 路由)、
  `fedagent/fed/run_fed.py`(`alfworld_replicas`/`webshop_replicas`、train+val 复制、pool
  均分、感知端口带的碰撞守卫)。K=1 逐字节相同;单元 smoke + 11 个 GPU run 覆盖全部路径。
- **配置:** `tools/verl08_migration/accel/alfworld/alf_scale_g{4,1}_r8.yaml`、
  `alf_scale_g1_r{1,4,8}n1.yaml`、`alf_em_worker_r8.yaml`;
  `accel/webshop/ws_scale_g{1,g1b,g4}.yaml`、`ws_scale_g4_p64{,r4}.yaml`(各自的 README 有映射)。
- **Run(gitignored):** `runs/ws_scale/`、`runs/alf_scale/`、`runs/alf_em/worker_r8*` ——
  barrier 文件带有每个 run 的 rc/墙钟/timing 摘录。
- 英文版: [acceleration_tier1_report_2026-07-01.md](./acceleration_tier1_report_2026-07-01.md)
