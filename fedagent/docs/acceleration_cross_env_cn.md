# 跨环境加速对比 —— WebShop vs ALFWorld

**这篇文档一句话回答:** *哪些加速选择能从 WebShop transfer 到 ALFWorld,哪些会翻转,以及预测这一切的那条单一原则。*

这是自包含的跨环境综述。各环境的细节在
[`acceleration_cn.md`](./acceleration_cn.md)(WebShop 杠杆 + 分析)、
[`acceleration_results_cn.md`](./acceleration_results_cn.md)(WebShop 实测数)、
[`alfworld_testing_cn.md`](./alfworld_testing_cn.md)(ALFWorld 策略 + §6 结果)。两者都是
Qwen2.5-1.5B-Instruct,4×H100,GRPO(G=8),paper 设置。

> ### ⚠️ 先读这个 —— 什么能比、什么不能比
> **绝对墙钟秒数在两个环境之间不可比。** 它们的 val 规模(WebShop eval-mode sweep n=500 vs ALFWorld n=48)、
> episode 长度(15 vs 50 轮)、每步 env 重量都不同。**比的是*排名*、*相对 %* 惩罚、*机制* —— 绝不要
> "ALFWorld 3509s vs WebShop 2493s"。** 凡是*指标*要紧的数(每步 vs 整跑墙钟),都就地标注。

---

## 1. 一张表总览

| 维度 | WebShop(15-turn) | ALFWorld(50-turn) | 跨环境裁定 |
|---|---|---|---|
| **Eval-mode —— 最快** | `parallel`(2493s, n=500) | **`worker`**(3509s, n=48) | **翻转** —— worker 反超 parallel |
| **Eval-mode —— 最慢** | `shared`(3316s, n=500) | **`inline`**(4738s, n=48) | **翻转** —— inline 最慢,不是 shared |
| **Eval-mode —— 结构** | parallel < worker < inline < shared | **worker < parallel < shared < inline** | 解耦胜耦合不变;内部次序移动 |
| **每步瓶颈**(实测分解) | **GPU 算力 74%**(1-GPU 一步;gen 只占 24%) | **env(gen)73%**(4-GPU 一步) | **相反** —— 互为镜像 |
| **1-GPU 惩罚(每步)** | **2.41×**(225.2/93.4) | 修复前 1.79× → **replica 修复后 2.81×** | 两边都重;env 修好后 ALFWorld 反而更大 |
| **GPU↔rollout 耦合** | gen 基本平坦但*很小*(54.6→44.1) | **gen 在 1/2/4 GPU 下平坦(env-bound,`_TW_LOCK`)** | ALFWorld 专属地板 |
| **Replica 分片**(`*_replicas`) | step **−12%**(93.4→82.2,GIL 缓解) | step **−57%**(298→127.6);端到端 **−31%**(3509→2412 s) | **ALFWorld 的主杠杆;WebShop 上只是点缀** |
| **2-job 并发(ZMQ 修复)** | PASS(3-job) | PASS(2-job,两个 rc=0) | **transfer** |
| **持久训练器(#4)** | −43%/轮,跨轮 −62% | 已在 3509s/2412s 基线内(cross_round 开) | transfer |

**三句话:** 两个环境的瓶颈**互为镜像** —— ALFWorld 一步有 73% 花在被锁串行化的 env 服务里
(已由 `alfworld_replicas` 修复:step −57%,端到端 −31%),而 WebShop 1-GPU 一步有 74% 花在 GPU 算力上
(它的杠杆是加卡;replica 只给 −12%)。把 eval 从训练关键路径解耦在**两个**环境都赢,mode 排名在
ALFWorld 上翻转(worker 的冷启动摊销)。并发修复与环境无关,两边都成立。

---

## 2. Eval-mode 排名 —— 大翻转

同一个 4-mode sweep(inline / parallel / shared / worker),每种 = eval 跑在相对训练的不同位置。2 client × 2 round、
每轮 eval 的整跑墙钟:

| 名次 | WebShop(n=500) | ALFWorld(n=48) |
|---|---|---|
| 1(最快) | parallel 2493s | **worker 3509s** |
| 2 | worker 2637s | parallel 3620s |
| 3 | inline 3090s | shared 4560s |
| 4(最慢) | shared 3316s | **inline 4738s** |

**不变的:** 两个 **eval-解耦** 模式(`worker`、`parallel`)赢两个 **eval-耦合** 模式(`shared`、`inline`)。
eval 是否压在 4-GPU 训练关键路径上,在两个环境里都是主导因素。

**翻转的,以及为什么:**
- **`worker` 反超 `parallel`。** ALFWorld 的 eval 引擎冷启动(vLLM 初始化 + CUDA-graph capture + 加载 8810 局服务)
  *很贵*。`worker` 只付**一次**(跨轮常驻)并把 4 张卡全留给训练;`parallel` 藏了 eval 但只用 2 卡训练(+30%/步)。
  eval 越重,摊销冷启动 > 藏 eval。
- **`inline` 变最差(不是 `shared`)。** `inline` 每轮在关键路径上**重启**那个贵引擎。WebShop 上 eval 够轻,inline
  重启便宜,`shared` 的 0.3-util KV 限流才是最差;ALFWorld 上每轮重启重引擎成了主导,于是 `inline` 沉到被限流的
  `shared` 之下。

> **可比性注意。** WebShop "shared 最慢"专门是**大 val(n=500)**效应;ALFWorld 跑的是 n=48。所以 shared↔inline 的
> 次序部分是 val 规模、不是纯环境效应。与 val 规模无关的稳健论断是**机制**:ALFWorld 重量级的*每次 eval 冷启动*
> 让 `inline` 成为输家、并奖励 `worker` 的摊销。

---

## 3. GPU scaling —— 两个分解现在都测了(2026-07-01)

| 每步 | WebShop gen | WebShop GPU-Σ | WebShop step | ALFWorld gen | ALFWorld GPU-Σ | ALFWorld step |
|---|---|---|---|---|---|---|
| 1 GPU | 54.6 (24 %) | **165.8 (74 %)** | 225.2 | 228.3 (43 %) | 300.2 | 534.5 |
| 4 GPU | 44.1 (47 %) | 46.9 (50 %) | 93.4 | **219.3 (73 %)** | 75.9 | 298.4 |
| **1-vs-4 惩罚** | | | **2.41×** | | | 1.79×(修复前) |

- **瓶颈互为镜像:** WebShop = GPU-compute-bound(gen 小且基本平坦);ALFWorld = env-bound(gen 大且平坦
  —— `_TW_LOCK` 地板,后已被 replica 修复 → 修复后 step 127.6 s,惩罚涨到 **2.81×**)。
- **对早先墙钟口径数字的修正:** 旧的 "WebShop +37% / 1.37×" 是被 ~390s 固定开销稀释的 3-step 墙钟
  (995 ≈ 3×202 + 开销,恰好对账);按每步算,WebShop 的 1-GPU 惩罚是 **2.41×**。ALFWorld 的 "+21% 墙钟"
  是同一种伪影。

**新机制(仅 ALFWorld,实测):** 把每步拆成 rollout vs 训练 ——

| GPU | gen(rollout) | update_actor(训练) |
|---|---|---|
| 1 | 228.3s | 140.0s |
| 2 | 225.3s | 92.2s |
| 4 | 219.3s | 43.3s |
| scaling | **平坦(−4%)** | **~线性(3.2×)** |

`gen` 随卡数**平坦** → ALFWorld rollout 受**环境延迟约束**:被 `_TW_LOCK` 串行化的 TextWorld
服务决定生成,而非 GPU 算力。只有 `update_actor` scaling。
**实用杠杆(2026-07-01 验证):** 不是 `pool_size`(K=1 对照:pool 8→64 时 gen 停在 217.5 s ——
不管 pool 多大,锁都在串行化),而是 **service replica**(`alfworld_replicas: K` = K 个进程 = K 把锁):
gen 217.5 → 65.8(K4)→ 61.8(K8);4-GPU step 298→**127.6 s**;端到端 **3509→2412 s(−31%)**。
WebShop 的分解(实测,§3 表)恰是镜像 —— GPU-bound;`webshop_replicas: 4` 只给 −12%(GIL 缓解),
它的杠杆是 GPU 算力。

---

## 4. 并发 / ZMQ 修复 —— 与环境无关

FSDP→vLLM 权重传输死锁(每个独立 Ray 集群都取相同首个 job id `01000000` → 同一个 `/tmp` ZMQ socket → 44 分钟挂起)
及其修复(每个 verl 子进程导出 `VERL_RAY_JOB_ID` + 2 行 verl honor-override patch)完全在**与环境无关的 verl/Ray 平面**。

| | WebShop | ALFWorld |
|---|---|---|
| 测试 | 3 个并发 job(client-parallel + eval∥train) | 2 个并发训练 job,GPU {0,1}+{2,3} |
| 结果 | 修复后 PASS(rc=0) | **PASS**(两个 rc=0;A 392s,B 473s) |

ALFWorld 是*更强*的压力测试 —— 它慢的服务冷启动加宽了 socket 竞争窗口 —— 修复依然成立。这是预期结果:bug 和修复都不碰 env 服务。

---

## 5. 那条原则(为什么上面这些都成立)

ALFWorld 与 WebShop 在三个轴上不同 —— **episode 更长(50 vs 15 轮)**、**每步 env 更重(TextWorld + 进程级 `_TW_LOCK`)**、
**eval 更大更重**。每一个都改变墙钟去了哪:

```
            WebShop  ────────────────►  ALFWorld
 成本从:    GPU 算力        转移到:   eval 引擎冷启动  +  env 延迟(rollout)
```

这一个转移就预测了上面每一个结果:
- **eval 冷启动变大** → *摊销*它的模式(`worker`)赢,*重复*它的模式(`inline`)输 → **eval-mode 排名翻转**。
- **rollout 变得受 env 延迟约束** → 加卡不再帮生成(`gen` 平坦) → rollout 的杠杆变成 **env-service
  replica**(K 个进程 = K 把锁;单靠 pool 大小毫无作用),而每卡*训练*惩罚不变(它本来就与 rollout 无关)。
- **训练器平面不受触动** → 并发修复原样 transfer。

**新环境的决策规则:** 估计(a)eval 引擎冷启动成本、(b)rollout 受 env 延迟约束的程度(在两个卡数下各跑一次
1-step `timing_s` 探针:gen 平坦 = env-bound)。(a)高 → 选 `worker`/`parallel`,避开 `inline`。(b)高 → 先把
`*_replicas` 加到 per-replica 串行负载 < episode 关键路径(K≈4–8),*然后*再扩卡 —— 修复后卡数惩罚反而*变大*
(ALFWorld 1.79×→2.81×),所以两个环境都别把 client 饿到 1 GPU。

---

## 6. 已定 vs 未决

**已定(两个环境都测了):** eval-mode 排名 + 解耦-eval 原则;ZMQ 并发修复与环境无关;**两个每步分解**
(2026-07-01:WebShop GPU-bound 74% vs ALFWorld env-bound 73% —— 互为镜像);**replica 分片**(`*_replicas`)
在 ALFWorld 上端到端验证(机制 K 扫描 217→66→62 s + pool 对照 + 组件 −57% + 端到端 −31%),在 WebShop 上
实测收益有限(−12%);每步 1-GPU 惩罚(WS 2.41×,ALF 修复后 2.81× —— 1-GPU-client 布局在两边都死了)。

**未决 / 尚未单独隔离:**
- **持久训练器(#4)在 ALFWorld 的单独 A/B** —— 它*在* 3509/2412 基线*里面*(cross_round 开),但其单独贡献未再隔离。
- **#3 client-parallel × replica 组合** —— 2×2-GPU 并行 client、各带分片服务(预测比修复后的串行-4-GPU 再 ~−18%);需要 run_fed 里的 parallel-round launcher。
- **ALFWorld 全 val 数** —— 这些用 n=48;in-loop `valid_seen` 是 140,offline 集是 274(`tools/verl08_migration/eval_alfworld_by_tasktype.py`)。
- **多步稳态墙钟** —— 探针是 1 步;多轮 run 确认墙钟惩罚收敛到每步数字。

---

## 出处 & 另见
- **WebShop 数:** [`acceleration_results_cn.md`](./acceleration_results_cn.md)、
  [`acceleration_cn.md`](./acceleration_cn.md) §7.4(eval mode)/ §7.7(布局)/ §Lever #3。
- **ALFWorld 数:** [`alfworld_testing_cn.md`](./alfworld_testing_cn.md) §6(预测揭晓 + 记分卡);
  [`EXPERIMENTS.md`](../EXPERIMENTS.md) "ALFWorld acceleration economics (2026-06-30)"。
- **配置:** `tools/verl08_migration/accel/webshop/`、`…/accel/alfworld/`、`…/accel/client_parallel/`(各有 README)。
- **修复:** `tools/verl08_migration/patches/`(`VERL_RAY_JOB_ID` honor-override)。
- 英文版:[`acceleration_cross_env.md`](./acceleration_cross_env.md)。
