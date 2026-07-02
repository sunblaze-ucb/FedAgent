# ALFWorld 测试策略 —— 为什么 ALFWorld 该这么测

> **一句话论点。** ALFWorld **不需要**把整套 WebShop sweep 重跑一遍。正确性已经被覆盖,因为所有修复都在
> **env 无关**的 verl/Ray 层(§1)。真正值得花 GPU 小时的,是 ALFWorld 改变的那部分**加速经济学**——
> episode 更长(50 turn vs 15)、每步更重(TextWorld + 一把进程级全局锁)、val rollout 更大——因为有两个
> WebShop 上的结论在 ALFWorld 上可能**翻转**:*(a)* 1-GPU 惩罚是否小到让"1 卡/client + 藏 eval"变得有竞争力,
> *(b)* eval 是否大到藏不住。
>
> 配套阅读:[acceleration_cn.md](./acceleration_cn.md)(杠杆设计 + 冷启动拆解)、
> [acceleration_results_cn.md](./acceleration_results_cn.md)(本文推理所依据的 WebShop 实测数)、
> [architecture_cn.md](./architecture_cn.md)(overlay)、[running_cn.md](./running_cn.md)(如何启动)。
>
> **约定。** WebShop 基线为 4×H100、Qwen2.5-1.5B-Instruct、paper 配置。"1-GPU 惩罚"那组数(995/727/407 s)
> 在 WebShop **n=64** val 上测;eval-mode sweep 在 **n=500** 上测。每个引用的数都锚到加速文档的具体小节。

---

## 0. 结论

| 问题 | 答案 | 在哪定的 |
|---|---|---|
| 并发/传输修复需要在 ALFWorld 上重新验证吗? | **不用** —— 在 env 无关层;无论调哪个 env 服务都逐字节相同 | §1 |
| persistent / eval_mode / #3 的代码需要重新验证吗? | **不用** —— 是 verl 生命周期代码;唯一 env 相关的接缝(服务路由)两个 env 都已接好 + 验证过 | §1 |
| ALFWorld 上到底什么变了? | **墙钟分解** —— 每个杠杆的相对价值,因为 episode 长度 / 每步重量 / val 大小都不同 | §2–§3 |
| 那值得测什么? | **Tier-1**(并发 smoke + 快路径数)和 **Tier-2**(两个经济学翻转点) | §4 |
| 什么可以放心跳过? | 1-GPU 布局全套 + 4-card eval sweep 逐字重跑 | §4 |

---

## 1. 架构边界 —— 为什么 fixes 是 env 无关

FedAgent 的 env 是一个独立的 **HTTP 服务**(`BaseTextEnv` 契约,见
[architecture_cn.md](./architecture_cn.md) → *远程 env 服务*)。训练进程里 agent-loop(`GymTextAgentLoop`)
只通过 HTTP 调它。**env 在进程边界之外。** 所以整个栈分两层:

```
┌─ verl / Ray 层  (env 无关) ────────────────────────────────────────────────┐
│  FSDP actor ──ZMQ /tmp socket──► vLLM 引擎       ← 权重传输修复在这           │  (Bug #2: VERL_RAY_JOB_ID)
│  torchrun aggregate_fedavg_fsdp                  ← FedAvg 端口修复在这         │  (Bug #1: --standalone)
│  run_fed 编排 / persistent / eval_mode / #3      ← 这些 CODE 在这             │
│  metrics_logger stdout 解析 + flush              ← 在这                       │
└────────────────────────────────────────────────────────────────────────────┘
                    │  HTTP  (唯一的 env 接口)
┌─ env 服务层  (env 相关) ───────────────────────────────────────────────────┐
│  WebShop service   /   ALFWorld service          ← 只有这一层换了            │
└────────────────────────────────────────────────────────────────────────────┘
```

逐条对一下:

| 修复 / 能力 | 在哪一层 | 为什么原封不动 transfer 到 ALFWorld |
|---|---|---|
| **ZMQ 权重传输 socket**(Bug #2) | verl/Ray | socket 是 FSDP actor → vLLM 引擎,在 trainer *内部*。撞车(每个隔离的 Ray cluster 都分到同一个首 job id `01000000` → 同一个 `/tmp` socket 路径 → 死锁)和 `VERL_RAY_JOB_ID` 修复,无论 HTTP 另一端是哪个 env 服务都逐字节相同。 |
| **FedAvg rendezvous 端口**(Bug #1) | verl/Ray | 聚合器(`torchrun --standalone`)操作的是 **FSDP checkpoint 分片** —— 根本不碰 env。 |
| **persistent / cross_round / eval_mode / #3** | verl/Ray | 这是 `run_fed` 编排 + verl 生命周期。*唯一* env 相关的接缝是**服务路由**(`WEBSHOP_SERVICE_URL` vs `ALFWORLD_SERVICE_URL`)—— 而这两条路由都已实现 **且** GPU 验证过(per-client routing,acceleration.md §7.3)。 |
| **stream / stdout flush** | verl/Ray | `metrics_logger.py` 解析 client 的 `training.log` stdout;完全不感知 env。 |

**结论:正确性已经覆盖 ALFWorld。** 重跑整套 WebShop sweep,大部分是在重新确认 env 边界本就保证不变的东西。
真正的问题不是*能不能跑通*,而是*多快* —— 而这恰恰是 ALFWorld 真正不一样的地方。

---

## 2. 墙钟分解 —— 加速逻辑依赖的是什么

每轮墙钟 ≈ **冷启动 + rollout + 训练计算**,再加一个 eval 项。每个杠杆恰好攻击其中一项。所以杠杆的价值取决于
**那一项有多大** —— 而各项的大小由 env 决定。这就是 ALFWorld 会不一样的根源。

| 杠杆 | 攻击的项 | WebShop 1.5B 实测 | 来源 |
|---|---|---|---|
| **#4 persistent** | **冷启动**(只付一次,而不是 ~140×) | ramp ≈ 2.5 min 热 / 5–14 min 冷;占 paper run 墙钟 **~76–88%**;**−43%** per-round、**−62%** cross-round | accel §2.1;results §2 |
| **#3 / GPU-scaling** | **训练计算 + vLLM 生成**(FSDP/TP 并行) | t1(4)=**558** → t1(2)=**725** → t1(1)=**995**;4 卡只比 2 卡快 **1.30×**(次线性) | results §4;accel §7.7 |
| **eval-hiding**(#1 / worker) | **eval** 项 | eval **407 s**(n=64)藏在 **995 s** 训练下;n=500 sweep:parallel 2493 / worker 2637 / inline 3090 / shared 3316 | accel §7.7;results §3 |

每个杠杆的相对价值,就是它攻击的那一项的相对大小。改变各项大小(env 就是干这个的),排名就会动。

---

## 3. ALFWorld 怎么改变每一项(可证伪的预测)

ALFWorld 相对 WebShop:**episode 更长**(50 turn vs 15)、**每步更重**(TextWorld 模拟 vs WebShop 检索),
以及——决定性地——一把**进程级全局 `_TW_LOCK`**,把*每一次* textworld `reset`/`step` 都串行化(tatsu PDDL
parser 是进程级可变单例;约 86 ms/step 串行,env 层没有并行——WebShop 没有这把锁;accel §2.2)。逐项推:

### 3.1 冷启动占比 ↓ → 杠杆 #4 的*相对*收益变小
rollout 项变大(50 turn × 重 env),冷启动 ~不变 → 冷启动**占比**缩小。所以"消灭冷启动"的 #4 在 ALFWorld 上买到的
相对收益,比 WebShop 的 **−43% / −62%** **更小**。#4 仍值得开着(免费、从不伤害),只是没那么主导。

### 3.2 1-GPU 惩罚 ↓ → "1 卡/client" 布局结论可能松动甚至翻转 ⭐
这是最有意思的一条。GPU 数量影响**训练计算 + vLLM 生成吞吐**;它**不碰** env 延迟(HTTP 打 CPU env 服务,
GPU 无关——而且 ALFWorld 的 env 步进被 `_TW_LOCK` *本就串行化*,与 GPU 数无关)。ALFWorld 的 rollout 远更
**env-latency-bound**,所以墙钟里 GPU 无关的等待占比更大 → 砍 GPU 4→2→1 只切到**更小**的一块。

- WebShop 1-GPU 惩罚 = **995 / 727 = +37%**(+268 s/轮)—— 大到足以让文档判"2 client × 1 GPU + 2 卡藏 eval"
  **不是**快路径(accel §7.7;report §9.1)。
- **ALFWorld 预测:惩罚更小(也许 +15–25%)** → "2 个 client 各 1 卡 + 2 卡藏 eval"这个布局在 ALFWorld 上可能
  反而变得**有竞争力**。这正是 **Tier-2 的 t1(1) vs t1(2)** 要测的翻转点。

### 3.3 eval 成本 ↑ → 藏 eval 更重要,但可能藏不住
ALFWorld 的 val 是个大 rollout,而且 `_TW_LOCK` 让它在 env 层无法并行。三个后果:

- **藏 eval 省得更多**(eval 更贵)→ eval-hiding 的价值**上升**。
- **但"藏住"的前提是 eval < 一个训练轮。** 一个 50-turn × `_TW_LOCK` 串行 × 重步的 val 可能**比一个训练轮还长**
  → eval **溢出**训练窗口 → eval-hiding 杠杆在 ALFWorld 上可能**失效**。*这必须实测。*
- **`shared` 几乎肯定输。** "shared 在大 val 上最慢"(WebShop n=500 的发现,results §3)会叠加:shared 的
  0.3-util 减 KV eval 引擎会节流一个本就 env 串行的 ALFWorld val → 几乎确定确认 `shared` 不适合 ALFWorld。
- **`worker` 对 `inline` 的领先变小。** worker 的优势是省掉 eval 冷启动;当 eval *rollout 本身*巨大时,省掉的
  冷启动只是更小的比例 → eval-mode 排名可能重排。

> **相对粗估的量级订正。** ALFWorld 的 in-loop val 是 **140 games**(`valid_seen`,50-turn——
> `config/envs/alfworld_val.yaml`),**不是** 274。完整的 **274 trial** 集 = `valid_seen`(140) +
> `valid_unseen`(134),由 `tools/verl08_migration/eval_alfworld_by_tasktype.py` 在**最终模型**上**离线**评测,
> 不在 loop 里。所以按原始 game 数,ALFWorld in-loop(140)其实比 WebShop(n=500)**还小**。eval 仍然暴涨的原因
> **不是** game 数 —— 是 **episode 长度(50 vs 15)× 每步更重 × `_TW_LOCK` 串行化**。这套机制也正是 §3.2 成立的原因:
> env 步进与 GPU 数无关地被串行化。

---

## 4. 测试设计 —— 一个预测,一个测

每个 tier 直接对应上面一个预测。用能解决翻转点的最小集合去跑。

### Tier 1 —— 并发 smoke + ALFWorld 快路径数(最高价值,一把跑,三个答案)
**布局:** 2 个 ALFWorld client × 2 卡并发 + `persistent` + `eval_mode=worker`。

为什么最高价值:ALFWorld 服务冷启动**慢**(要 load game collection),这恰好是 **`/tmp` socket race 窗口最宽**的
地方 → 对 ZMQ 权重传输修复(Bug #2)的**最强**压力测试。一把跑拿到:✅ ALFWorld 快路径数,✅ ALFWorld 上 4 卡
两 job 共存,✅ 并发修复最难的 case —— 确认它们在真实 ALFWorld 负载下确实 env 无关。

### Tier 2 —— ALFWorld 专属的经济学(两个翻转点)
1. **t1(1) vs t1(2)** —— 1-GPU 惩罚是否真的更小/翻转(§3.2)?测"1 卡/client + 藏 eval"是否变得有竞争力。
2. **140-game val 上的 eval-mode 小 sweep** —— eval 藏不藏得住、哪个 mode 赢(§3.3)?预期 `shared` 输;
   看 `worker` 是否仍领先 `inline`,以及是否有任何 mode 能把 eval 压在一个训练轮以内。

这才是唯一真正 **ALFWorld 专属**的 cadence 结论 —— 无法从 WebShop 推出来。

### 跳过(结论 transfer;不值 GPU 小时)
- **1-GPU 布局全套**逐字重跑。
- **4-card eval sweep** 逐字重跑。

两者都从 WebShop transfer;只有 Tier-2 里的*翻转点*是新的。

---

## 5. 机制 / 坑(测之前要接好的)

| 项 | 细节 |
|---|---|
| **独立 conda env** | ALFWorld 服务跑在自己的 conda env(`verl-agent-alfworld`),是与 py3.12 训练器(`fedagent-verl08`)**不同的解释器**。得先把服务起起来(`envs/alfworld/service/run_service.sh`)。 |
| **冷启动更慢** | 服务启动时要 load game collection → `/health` 等待要给**够**。并发起多个服务会加重**主机 CPU 争用** —— 错峰起或把 timeout 放宽。 |
| **val split** | in-loop val = `alfworld_val_split: eval_in_distribution` = **`valid_seen`(140)** → round→success 曲线。完整 **274**(`valid_seen`+`valid_unseen`)+ 按 task-type 拆分,来自在**最终**模型上跑 `tools/verl08_migration/eval_alfworld_by_tasktype.py`。`alfworld_task_types`(`""`=全部 6 类;否则 `1=Pick..6=Pick2`)选拆分子集。 |
| **pool 大小** | `alfworld_pool_size ≥ gen_batch` —— TextWorld env 池要够一个 rollout batch(`fed/run_fed.py` DEFAULTS / [`fed/README.md`](../fed/README.md))。 |
| **`_TW_LOCK`** | 一把进程级 `threading.Lock` 包住每次 textworld `reset`/`step`(tatsu PDDL parser 是进程级可变单例),把 ALFWorld env 步进串行化(~86 ms/step;~13.7 s/windowed client-step,对比 legacy 的 ~0.9 s 并行 Ray actors)。已知、有界 —— 也是 ALFWorld env-latency-bound(§3.2)和 eval 难藏(§3.3)的**根因**。WebShop 没有这把锁。 |
| **context 预算** | 50-turn ALFWorld episode 配宽窗口(paper 设 `rollout.max_model_len=16384`、`response_length=8192`);agent loop 硬防溢出。确认 verbose 房间里不会在 `done` 前截断(`config/envs/alfworld.yaml` GPU-VERIFY 注)。 |

---

## 6. 结果 —— 预测逐条揭晓(2026-06-30,1.5B,4×H100,qgpu3021)

三个 tier 全部跑完。**设置:** in-loop val 缩到 **n=48**(`alf_em` 配置用
[`alfworld_val_48.yaml`](../../tools/verl08_migration/accel/alfworld/alfworld_val_48.yaml),140 个
`valid_seen` 里的 48),好让 4-mode sweep 可控;训练 = sweep 的最小 `epochs=1, total_training_steps=1`/轮。
除注明外均为墙钟。配置:
[`tools/verl08_migration/accel/alfworld/`](../../tools/verl08_migration/accel/alfworld/)。

### 6.1 Tier-1 —— 并发:**PASS** ✅
两个独立 ALFWorld 训练 job(GPU {0,1}+{2,3}),各自独立 Ray 集群 + ALFWorld 服务,两个同时在共享 `/tmp` socket
上做 FSDP→vLLM 权重同步 —— 正是修复前死锁那条路径。两个都 `rc=0`,16 分钟跑完(A 392s,B 473s;B 的 +22% 是两个
8810 局服务之间的主机 CPU/RAM 争用,不是 GPU/正确性问题)。**`VERL_RAY_JOB_ID` 修复在 ALFWorld 更重的双服务负载下
成立** → §1 的 env-无关论断在真实负载下被确认。(`alf_conc_{A,B}.yaml`。)

### 6.2 Tier-2 scaling —— 揭晓 §3.2(1-GPU 惩罚预测)
1/2/4 GPU 的 `timing_s/step`(`alf_scale_g{1,2,4}.yaml`,eval off,1 步):

| GPU | step | gen(rollout) | update_actor |
|---|---|---|---|
| 1 | 534.5s | 228.3s | 140.0s |
| 2 | 386.9s | 225.3s | 92.2s |
| 4 | 298.4s | 219.3s | 43.3s |
| scaling | — | **几乎不变(−4%)** | **~线性(3.2×)** |

- **机制确认 ✅:** `gen` 随卡数几乎不变 → ALFWorld rollout 受**环境延迟约束**(被 `_TW_LOCK` 串行化、
  `pool_size` 节流的 env 服务决定,而非 GPU 算力)。§3.2 的前提被直接测到。
- **量级预测(+15–25%)在每步层面太乐观 ✗:** **每步** 1-GPU 惩罚是 **+38%**(534.5/386.9)≈ WebShop 的 +37%
  —— 并**没有**变小。平坦的 env-bound gen 把它*压低*了(纯算力会是 ~+90%),但 `update_actor` 仍 scaling,所以每步
  与 WebShop 持平。
- **墙钟 1-GPU 惩罚是 +21%**(1050/865s)—— 但那是**单步探针伪影**:~490s 固定开销(服务加载 + Ray/vLLM 初始化
  + 拆除)不 scaling,在单步上稀释了。真实多步 run 里固定开销被摊薄,墙钟惩罚会爬回每步的 +38%。**所以 §3.2
  "1-GPU 变得有竞争力"的松动在稳态下不成立** —— 布局结论与 WebShop 一致。(ALFWorld rollout 的杠杆是 `pool_size`,不是加卡。)

### 6.3 Tier-2 eval-mode 小 sweep —— 揭晓 §3.3
2 client × 2 round,每轮 eval,48 局 val(`alf_em_{inline,parallel,shared,worker}.yaml`):

| mode | 墙钟 | 对比最快 |
|---|---|---|
| **worker** | **3509s** | — |
| parallel | 3620s | +3% |
| shared | 4560s | +30% |
| inline | 4738s | +35% |

- **排名 `worker < parallel ≪ shared < inline`。** eval-**解耦** {worker, parallel} 比 eval-**耦合**
  {shared, inline} 快 **~25–30%** → 把 eval 从训练关键路径上解耦才是 ALFWorld 重 eval 下的关键;*怎么*解耦
  (常驻 worker vs 并发 GPU)是平局。
- **§3.3 "shared 输/最慢" → 错 ✗。** shared(4560)**赢了** inline(4738)。ALFWorld 上,inline 每轮重启*重量级*
  eval 引擎的代价 > shared 0.3-util 限流,所以**最慢的是 inline**,不是 shared。(WebShop "大 val 下 shared 最慢"没 transfer。)
- **§3.3 "worker 对 inline 的领先变小" → 没变小 ✗。** worker 仍比 inline 快 **26%** —— eval 越重,它摊销的冷启动
  是*更大*的奖励。worker 还险胜 parallel(跨轮 4-GPU 训练 + 摊销冷启动 > parallel 的"藏起来但只 2-GPU 训练"的 eval)。

### 6.4 预测记分卡

| § | 预测 | 裁定 |
|---|---|---|
| §3.2 机制 | rollout env-bound → gen 对 GPU 不敏感 | ✅ 确认(gen 平坦 228→219) |
| §3.2 量级 | 1-GPU 惩罚缩到 +15–25% | ✗ 每步 +38% ≈ WebShop;只有单步*墙钟*是 +21%(固定开销稀释) |
| §3.3 shared | shared 输/最慢 | ✗ shared(4560)赢 inline(4738);**inline** 最慢 |
| §3.3 worker | worker 对 inline 领先变小 | ✗ worker 仍领先 26% |
| §4 Tier-1 | ZMQ 修复在 ALFWorld 负载下 env-无关 | ✅ PASS,两个 rc=0 |

**净结论:** 本文推理依赖的*机制*(env-bound rollout;解耦 eval 才关键)被**确认**;两个*量级/排序*预测**翻转**了
—— 每步 1-GPU 惩罚没缩小,且 eval-mode 输家是 **inline**(不是 shared)。ALFWorld 单节点快路径 = **`worker` 或
`parallel`**(比 inline 快 ~25–30%)。run 日志:gitignored `runs/alf_em`、`runs/alf_scale`、`runs/alf_conc`。

---

## 7. Tier-1 修复 —— env-service replica 分片干掉 env-bound 地板(2026-07-01)

§6.2 那个 env-bound 的 `gen`(219–228 s,随卡数平坦)就是 `_TW_LOCK` 串行化:tatsu PDDL parser
是进程全局的,一个服务进程 = 一把锁 = env stepping 单线排队
(86 ms × ~3200 步/optimizer-step ≈ 整个 gen)。**修复:** `alfworld_replicas: K` —— 每个 client
对*同一个* game shard 跑 K 个相同的服务进程,session 在 client 侧 round-robin 分摊
(`resolve_service_url` 接受逗号分隔的 URL 列表;run_fed 对训练**和 val** 服务都做 replica)。episode
分布不变 → 科学上安全。验证链(1.5B,batch 8×8,GPU 实测):

| 层级 | 结果 |
|---|---|
| 机制(同节点 K 扫描,pool 64) | gen **217.5 (K1) → 65.8 (K4) → 61.8 (K8)** |
| 对照(K=1,pool 8→64) | gen 217.5 ≈ 228 → **pool 无关;这把锁就是全部原因** |
| 4×H100 组件(K=8) | gen **219→51.7(−76%)**,step **298→127.6(−57%)**;update_actor 未动 |
| 1×H100 组件(两个节点) | step **534→350–359(−33%)**;**8 核节点上 K=4 就够** |
| **端到端**(§6.3 worker 配置 + K=8) | **3509 → 2412 s(−31%)**;训练步 −65%;val 健康 |

残余 gen ≈ 52–66 s = episode 关键路径(新地板)。**修复后的后果:** GPU 算力现在占主导 → 每步
1-GPU 惩罚涨到 **1.79× → 2.81×**,所以 1-GPU-per-client 的想法彻底死了。**ALFWorld 生产配方:
`cross_round + eval_mode=worker + alfworld_replicas: 8`(4×H100)/ `alfworld_replicas: 4`
(1×H100,−33%)。** 细节 + WebShop 对照(GPU-bound,replica 只 −12%):
[acceleration_cn.md](./acceleration_cn.md) §9。

---

## 一句话

ALFWorld 不该"都重测一遍":正确性靠 env 无关的 fix(§1),已经覆盖。真正值得花 GPU 小时的,是因为 ALFWorld
rollout *更长 / 更重 / 更大*而改变的**加速经济学** —— 尤其那两个可能**翻转**的点:*1-GPU 惩罚是否小到让布局结论
松动(§3.2)*,以及 *eval 是否大到藏不住(§3.3)*。Tier-1 一把确认修复在真实 ALFWorld 负载下 env 无关 **并**拿到
快路径数;Tier-2 解决两个翻转点。**两个都跑了(§6):** 修复成立(并发 PASS)、快路径 = `worker`(3509s) ——
且两个预测都*翻转*了:每步 1-GPU 惩罚没缩小(+38% ≈ WebShop),eval-mode 输家是 **inline**(不是 shared)。

## 另见
- [acceleration_cross_env_cn.md](./acceleration_cross_env_cn.md) —— **WebShop vs ALFWorld 并排对比**（§6 结果蒸馏成一张主表 + transfer/翻转原则）
- [acceleration_cn.md](./acceleration_cn.md) —— 杠杆设计、冷启动拆解、等价性审计
- [acceleration_results_cn.md](./acceleration_results_cn.md) —— 本文推理依据的 WebShop 实测数
- [acceleration_report_cn.md](./acceleration_report_cn.md) —— 完整的加速 walkthrough
- [architecture_cn.md](./architecture_cn.md) —— overlay、两层架构、远程 env 服务
- [running_cn.md](./running_cn.md) —— 启动 `run_fed.py`(eval mode、GPU、ALFWorld 服务)
