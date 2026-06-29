# FedAgent verl-0.8 —— 性能分析与加速方案

> **状态：** 设计 + 分析文档。迁移在*功能上*已经收尾（windowed GRPO+PPO 与
> concat 联邦循环均在 GPU 上验证通过）；本文档关注的是**速度**，而非正确性。
> 英文版见 [acceleration.md](acceleration.md)。
>
> **TL;DR：**
> 1. **为什么慢：** 经实测，wall-clock 中有 **76%（0.5B、warm cache、本次会话）→ 88%（1.5B smoke）**
>    花在了**每个 (client×round) 的子进程冷启动**上（Ray + FSDP + vLLM + kernel 编译；
>    warm 2.5 min / cold 5–14 min），在一次论文级 run 中要重复约 140 次。真正的训练步只占
>    约 12–24%。见 §2.6。
> 2. **verl 的 async 在运行，但只在 rollout 层**（一个 batch 内的 episode 并发派发给
>    vLLM）；**pipeline 层（client / round / eval）完全串行** —— 这是最大的尚未利用的
>    并行度。
> 3. **四个杠杆，按 ROI 排序：** `#2 env prewarm`（最安全、零数值影响）→ `#4 持久化 trainer/vLLM`
>    （唯一能消灭那 88% 的单节点杠杆，带数值等价风险）→ 多节点 `#3 parallel
>    clients` + `#1 eval∥train`。
> 4. **铁律：** 在*同一组* GPU 上硬并行 eval+train 是 VRAM/kernel 争用，而非
>    加速；真正的收益要么*消除*工作（#4），要么*资源隔离*（#1/#3，多节点）。
>
> **✅ 本次会话结果（#4 已 GPU 验证 —— 见 §3 Lever #4 / §7）：** 持久化 trainer 已构建
>（overlay 方式，无需 fork verl）并集成进 `run_fed`。逐轮（`persistent: true`）：一个 2 轮
> 联邦 GRPO 循环在 **515 s 完成，对比子进程 909 s = −43%**。跨轮（`cross_round: true`，§7.2）：
> **整个 run 只付一次冷启动，342 s = −62%**。两者最终聚合后的模型都与子进程路径
> **数值等价**（full-loop max|Δ|=1.13e-5，bf16 噪声）。PPO critic reload（§7.1）以及 **per-client
> service 路由 —— 已在真实 WebShop env 上验证**（§7.3，32/32 个 episode 分流到各自的服务）——
> 也已落地；cross-round + per-round eval 会 OOM 并自动回退到 per-round（§7.4）。它是唯一不破坏
> 可复现性的单节点杠杆。windowed 默认值的 blocker 也已修复（§7.5）。

---

## 0. 范围与参考

- **➜ 配套文档。** *本*文档是**分析与方案**。另见：
  [acceleration_report.md](acceleration_report.md) —— **完整的端到端走查**（每个杠杆与
  特性的深入讲解、调查 + 修正、所有结果）；以及
  [acceleration_results.md](acceleration_results.md) —— **结果一览**（状态表 + 数字）。
- **Repo（overlay）：** `/gpfs/projects/b1222/userdata/canyu/kangyu/fedagent`
- **verl 0.8 源码（editable）：** `/gpfs/projects/b1222/userdata/canyu/kangyu/others/verl/verl`
- **基准：** 与 FedAgent 论文的科学等价（在 3-seed 噪声范围内复现）。
  这里的每一项加速都先看*它是否扰动数值*，再看速度。
- **证据：** 下文所有计时均来自集群上真实 run 的日志（行内引用），而非估算。

---

## 1. 当前状态 —— 已实现的内容

### 1.1 迁移架构：thin overlay，不 fork
FedAgent 通过一个薄薄的 `fedagent/` overlay 运行在**原生 verl 0.8** 之上（不 fork verl）。verl 以
editable 方式安装，overlay 添加了联邦编排器、env 服务、忠实的
windowed rollout、FedProx 以及论文配置生成器 —— verl 在磁盘上没有任何 patch
（唯一的运行时 patch 是从 overlay 应用的一个*作用域受限、tag 门控*的 monkeypatch；见 §1.3）。

### 1.2 联邦循环：`fedagent/fed/run_fed.py` —— 每个 (client, round) 一个子进程
编排器与 verl 无关。对每一轮 `r`：

```
model_r = base                 (r == 1)
        = round_{r-1}/aggregated/hf   (r > 1)   # merged FedAvg'd FSDP shards
for each selected client c (SEQUENTIAL):                 # run_fed.py:854
    python -m fedagent.main_ppo_fed  model.path=model_r  default_local_dir=round_r/client_c ...
        -> FSDP actor (+critic for PPO) checkpoint shards
FedAvg:  torchrun aggregate_fedavg_fsdp.py  --client-actor-dirs c0,c1 ...   # run_fed.py:865
merge :  python -m verl.model_merger  ->  round_r/aggregated/hf            # run_fed.py:866
eval  :  (every round) a SEPARATE val-only subprocess (inline) | hot engine (worker)   # run_fed.py
```

关键性质（均已在代码中确认）：
- **每个 (client, round) 一个全新的 OS 子进程** —— 干净隔离，保证 GPU 释放。
  一轮内的 client 是**串行的**（`for c in selected`，`run_fed.py:854`）。
- **轮次是硬屏障：** 第 `r+1` 轮在 `FedAvg(r)`+`merge(r)` 产出 `model_r` 之前无法启动。
- **Eval 是独立子进程**（`eval_global`，`run_fed.py:501-544`，`trainer.val_only=true`）→ 它
  要付出*第二次*完整的冷启动，并且它在该轮**之后**运行，不阻塞任何东西但被它阻塞。
- **Env 服务**（WebShop/ALFWorld）是**每个 client 的远程 FastAPI 服务**，按轮**惰性**
  启动（仅该轮被选中的 client），在聚合前拆除（`run_fed.py:849-863`）。
  未被扰动的 **val 服务**只启动一次并保持运行。
- Baselines（federated / centralized / local）、FedProx（经由 `sitecustomize.py`）以及
  未被扰动的 eval 曲线（`val_before_train`、`test_freq`）都已接好线。

### 1.3 两种 rollout 模式（忠实-vs-原生 这条轴）
| 模式 | 它是什么 | 采样 | history | 默认 | 如何实现 |
|---|---|---|---|---|---|
| **windowed** | 论文的 per-turn rollout | 每 **turn** 1 个 | `history_length=2` 模板 | ✅ 是 | `WindowedAgentLoopManager` |
| **concat** | 原生 verl GymTextAgentLoop | 每 **episode** 1 个 | 完整 history | 可选启用 | 原生 `AgentLoopManager` |

`run_fed.inject_rollout_mode`（`run_fed.py:490-498`）把 windowed manager 注入到 train 与 eval
**两条**命令中，除非存在显式的 `manager_class` 覆盖。

**windowed 修复**（`fedagent/agent_loops/windowed_manager.py`）：windowed 把一个 episode 展开成
N 个 per-turn 行，这违反了 verl 0.8 的"每个输入 prompt 1 个 sample"约定（train 中静默截断，
eval 中 AssertionError 崩溃）。overlay 在*不 fork verl* 的前提下修复了这一点：通过一个 tag 门控的
作用域 monkeypatch 改写 `DataProto.slice`/`union` + 一个 worker 侧的 eval-collapse + `adjust_batch` 风格的
除数 padding（对应 legacy 的 `del batch; batch = gen_batch_output` + `adjust_batch`）。它是
幂等且隔离的（已验证对 concat run 0 泄漏）。

### 1.4 在 GPU 上验证通过的内容
- **windowed GRPO** 联邦循环已闭环（per-turn 行折叠、FedAvg、per-episode eval）。
- **windowed PPO** 联邦循环已闭环（actor **和** critic 每轮都做 FedAvg）。
- **concat**（经由 `rollout_mode=concat`）已闭环；monkeypatch 隔离成立。

### 1.5 已经存在（并正在使用）的 async —— verl 0.8 agent-loop rollout
verl 0.8 用一个 **async agent loop** 取代了 legacy verl-agent 的 batched-sync rollout：

```
one training step:
  train_batch_size × rollout.n  episodes launched as async coroutines
  each episode (async def):  await env.reset(); await generate(); await env.step(); await generate(); ...
  vLLM dynamically batches the concurrent generate() requests across all episodes
```

这套 async **正在运行** —— WebShop PPO 论文配置会启动 `64×8 = 512` 条轨迹，它们全部
同时 `reset()`（这就是我们加固过的 env-service "connection storm" 的来源）。具体来说：
- `GymTextAgentLoop.run()` / `WindowedGymTextAgentLoop.run_episode_windowed()` 都是 `async def`；
  `server_manager.generate()`、`env.reset()`、`env.step()` 全都被 `await` → 一个 batch 内的 episode
  并发推进。
- WebShop/ALFWorld env client 使用 `httpx.AsyncClient`；服务是 FastAPI/uvicorn 的 async handler，
  配一个 `asyncio.Queue` env 池 → I/O 层非阻塞。
- vLLM 把并发的 generation 请求批处理 —— 这是 async rollout 的核心收益。

**所以：rollout 层用了 async。联邦 pipeline 层没有。** §2 讲的就是这个差距。

---

## 2. 深入分析 —— 为什么还没有加速

要点：**正在运行的工作大多不是训练。** 在一次 windowed PPO 联邦 smoke 上，
总 wall-clock 是 **31m33s**，但所有训练步的 `timing_s/step` 之和只有
**~223.5s** → **wall-clock 的约 88% 是固定开销**，几乎没有一点是真正的 RL 步。

按影响程度排序，有三个不同的原因。

### 2.1 ⭐ 主导因素：每个 (client×round) 子进程的**冷启动**（~88%）
每个 `python -m fedagent.main_ppo_fed` 子进程在训练第 1 步之前都会重建**整个**
栈。下面是从一个真实的 **1.5B / max_seq_len=8192** run 日志测得
（`_scratch/c4_final/homog/round_1/client_0/training.log`）：

```
20:17:49.776  Ray local instance started           (worker.py:2003)
   │   ~2 min: worker-pool construction, FSDP actor/ref/critic init + weight load,
   │           CUDA-graph capture-size sweep, kernel prep
20:19:58.178  flashinfer autotuning  (39 ms — CACHED here; see note)   (autotuner.py:256)
20:20:02.017  vLLM "Initializing a V1 LLM engine"  seed=0, prefix_caching=True, dummy load
20:20:20      LLMServerManager ready: 4 server addresses                (ray_trainer log)
   →  first training step begins
20:20:20+     step:1   timing_s/step=37.94  (gen=30.9, update_actor=1.78, update_weights=2.14)
```

对这个配置而言，**冷启动 ≈ 2.5 min**，*且 kernel cache 是 warm 的*。**关键细节：** 上面的
flashinfer autotune 之所以是 **39 ms，是因为 JIT/inductor cache 已经 warm**。在**冷节点 /
首次 run** 时，flashinfer JIT + `torch.compile`/inductor 编译**不会**被缓存，同一阶段会
膨胀到我们观察到的 **5–14 min**。无论哪种情况，它都是**一次性 setup**：

| 冷启动阶段 | warm 成本 | 持久化进程能跳过吗？ | 证据 |
|---|---|---|---|
| Python import + Hydra | ~1–2 s | ✅ 跳过 | — |
| Ray init | ~2–5 s | ✅ 跳过（集群保持运行） | log L5 |
| Worker pool + FSDP actor/ref/critic init + weight load | ~60–90 s | ✅ 跳过；改为 hot-swap weights | log L920–921 |
| CUDA-graph capture + flashinfer/torch.compile kernels | warm ~30 s / **cold 5–14 min** | ✅ **完全跳过**（kernel 保持已编译） | log L863–886 |
| vLLM engine init + KV-cache alloc | ~18 s | ✅ 跳过；改为 `update_weights`（~0.5–2 s） | log L882–978 |
| Dataloader + agent-loop manager | ~3 s | ✅ 变便宜（仅重建 dataset） | — |

这套 setup 在**每个 (client × round)** *外加* **每次 eval** 都要付一次。一次论文级 run 是
~`clients_per_round × rounds (+ evals)` ≈ **140 个子进程**。每个 2.5–14 min，光是冷启动
就是**好几个小时**，把实际训练的 ~38 s/step 衬托得微不足道。**这就是那 88%。** 一个保持
存活并 hot-swap weights 的进程只需付**一次**。

### 2.2 步内的减速（确实在运行的那 ~12%）
两项忠实性/env 成本让即便在运行的那 12% 也比 legacy 更慢：

1. **Windowed 打断了 vLLM prefix cache。** Windowing 每个 turn 都移动 context window → 每个 turn
   都是 prefix-cache *miss*；concat 复用不断增长的 prefix → hit。实测 **windowed gen ~43 s vs
   concat gen ~30 s**（同一节点，1.43×）。这是*忠实性的代价*，是有意接受的 ——
   不是 bug。（步级：gen 约占 38 s 步的 81%，所以这直接影响 wall-clock。）
2. **`_TW_LOCK` 把 ALFWorld 的 env stepping 串行化了。** ALFWorld 服务在每次 textworld
   `reset`/`step` 周围持有一个**进程全局的 `threading.Lock`**，因为 tatsu PDDL parser 是一个
   进程全局的可变单例。实测 **86 ms/step → 160 个串行步 = 13.7 s** 的纯
   锁串行 env 时间（每个 windowed client-step），对比 legacy 的**并行 per-env Ray actor
   （~0.9 s）**。**WebShop 没有这种锁**（所以这是 ALFWorld 独有的）。async rollout 的 env 侧
   并发是真实的，但被这把锁限流成了单线排队。

> §2.2 的净效应：即使编排并行度无限大，windowed 仍比 concat *每步*慢约 1.47×，
> 而且 ALFWorld 还额外加了一笔串行化的 env 税。这些都已被理解且有界。

### 2.3 尚未利用的并行度 —— 整个 pipeline 层都是串行的
`run_fed.py` 中没有任何东西重叠：

| 今天的串行 | 独立吗？ | 能重叠吗？ | 阻塞点 |
|---|---|---|---|
| **一轮内的 client**（`for c in selected`，L854） | ✅ 是 —— 每个读 `model_{r-1}`，写自己的 ckpt；FedAvg 与顺序无关 | ✅ 能 | GPU 争用（单节点）→ 需要更多 GPU/节点 |
| **轮次** | ❌ 否 —— `train(r+1)` 需要 `FedAvg(r)` | ❌ 不能（硬屏障） | 数据依赖 |
| **eval(r)**（merge 之后，L896） | ✅ 是 —— 只读 `model_r` | ✅ **能，与 `train(r+1)` 重叠** | 两者都要 GPU |
| **下一轮的 env warmup** | ✅ 是 —— 服务是 CPU，client 是确定性的 | ✅ 能（与 FedAvg/merge/eval 重叠） | 无 —— 纯调度 |

### 2.4 依赖图（*能*重叠的部分）
```
                        ┌────────────► eval(r)            (reads model_r; pure measurement)
merge(r) ──► model_r ───┤
                        └────────────► train_client_c(r+1) ∀c  (reads model_r)
within round r:  train_client_0(r), train_client_1(r), ...  are siblings reading model_{r-1}
                 FedAvg(r) is the JOIN (needs ALL clients of round r)
```
因此合法的重叠恰好是：**(a) 一轮内的 client**、**(b) eval(r) ∥ train(r+1)**、
**(c) env-warmup(r+1) ∥ post-train(r)**。跨轮训练不能重叠。

### 2.5 铁律（关键所在）
在**单一共享 GPU 组**上，硬并行 eval+train（或两个 client）**不是加速 —— 而是
VRAM + kernel 争用。** 真正的收益要么需要 (i) **消除**工作（#4：不再反复付冷启动），要么需要
(ii) **资源隔离**（#1/#3：给每个并发作业各自的 GPU/节点）。任何忽视这一点的"async pipeline"
只是在分时复用同一组 kernel。

### 2.6 实证确认（本次会话，GPU —— qgpu3022/qgpu3013，4×H100）
验证上述分析的全新 run。

**Baseline（TinyGuess，0.5B，4-GPU，concat，warm caches，3 steps/client，federated 2 clients × 2 rounds，rc=0）：**

| 每个 (client,round) 子进程 | 均值 |
|---|---|
| total wall | **125 s** |
| step-compute（3 steps） | 31 s（≈10 s/step） |
| **固定开销（冷启动 + teardown）** | **94 s = ~76%** |

- 即便在*最廉价*的情形下（tiny model、warm kernel + fs cache、3 steps），**每个子进程约 76% 不是训练。** 放大到 1.5B + 8192-ctx + ~140 个子进程，这就是在 windowed PPO smoke 上测得的那 **88%**。光是 import（process→Ray）warm 是 **27 s，而冷 GPFS 读时是 86 s**。
- → **#4（持久化进程）才是那个杠杆**：它把这约 94 s 只付*一次*，而不是约 140 次。

**Windowed 默认值崩溃 —— 已确认为 release blocker。** `rollout_mode=windowed`（新默认值）+ 原生 `agent.yaml`（只注册了 `gym_text`）→ `AttributeError: 'GymTextAgentLoop' object has no attribute 'run_episode_windowed'`（[windowed_manager.py](../agent_loops/windowed_manager.py)#L152）。windowed 此前只在*显式*的 `_scratch/windowed_poc` agent 配置下才是绿的；任何基于原生 `agent.yaml` 的配置都会崩溃。→ **暂停的 windowed 配置迁移**（注册 `gym_text_windowed`）是 windowed-as-default 的前置条件。

**#2 的收益依配置而定（往往 ≈0）。** WebShop *homogeneous* 服务（`partition_strategy=""`）在**几秒内**就健康 —— 它们不构建 catalog，所以 prewarm 没有什么可重叠的。#2 只对**昂贵 warmup**的 arm（catalog_split 大 catalog、ALFWorld 游戏集合）才划算，而即便如此，每子进程的冷启动（#4）也远盖过 env warmup。**结论：#2 仍是一个次要的、可选启用的杠杆** —— 正确且就绪（CPU 已验证），但不是 wall-clock 所在之处。

**子进程路径的脆弱性（更多 #4 的动机）。** 在一个节点上连续 4 次冷启动期间，我们看到 vLLM `/dev/shm` 的 `KeyError`，以及 teardown 期间的 `DataLoader worker killed (SIGKILL)`（仅靠 client 间 5 s 的等待才幸免；legacy 会轮询 `nvidia-smi` 直到 GPU 空闲）。4-GPU rollout 还触发了 `data size must be divisible by force_group_size × micro_batch` —— 与 windowed 修复处理的同一类可整除性问题，这里发生在一份手搓配置上。持久化进程彻底绕开了这种反复的 alloc/teardown 抖动。

---

## 3. 加速杠杆

### Lever #2 —— 预热下一轮的 env 服务 *（最廉价、最安全、单节点收益）*
把第 `r+1` 轮的 env 池预热（WebShop/ALFWorld 需要数分钟）与第 `r` 轮的 FedAvg/merge/eval 重叠。
- **为何安全：** 纯调度，**零数值影响**。CPU-only 服务与 GPU 聚合重叠 ——
  无资源争用。
- **可行性：** `select_clients(r+1, …)` 是确定性的（seed 为 `base_seed + round - 1`，
  `run_fed.py:557-571`）→ 下一轮的 client 在第 `r` 轮结束前就已知。端口按 client 索引
 （`base_port + client_id`）→ 无跨轮冲突，**除非**某个 client 在连续两轮都被选中
 （处理方式：对重叠部分跳过 prewarm，或复用存活的服务）。
- **Patch 形态**（`run_fed.py`）：添加 `prewarm_next_round_services(cfg, env_base, r)`（调用现有的
  `start_*_services`，传入 `select_clients(r+1)`）；在 client 循环**之后**、`fedavg` **之前**调用它；
  把 handle 暂存到 `prewarmed_next_services[r+1]`，在下一次迭代顶部直接采用它们而非重新启动；
  在最终轮 / 失败时拆除。保持
  "≤ clients_per_round 存活"不变式（重叠期间最多短暂约 2×）。
- **收益：** 节省当前阻塞每轮启动的 env-warmup 分钟数。**但 §2.6 实测 WebShop homog warmup 约为数秒 → 那里收益 ≈ 0；只对昂贵 warmup 的 arm 才有实质意义**（catalog_split 大 catalog、ALFWorld）。有界且依配置而定。

### Lever #4 —— 跨 client 持久化 trainer / vLLM *（最大的单节点杠杆；带数值等价风险）*
让**一个** `RayPPOTrainer` 在所有 (client, round) 调用间保持存活；在 client 之间，通过
`update_weights`（verl 本就每步运行的同一调用，~0.5–2 s）把 actor（和 critic）hot-swap 到该轮的
聚合模型，把 dataloader 重指向下一个 client 的 env/seed，**重建 optimizer/scheduler**，
再跑 E 个 epoch。**这是唯一攻击那 88% 的杠杆** —— 它把冷启动只付*一次*，而非约 140 次。

> **✅ 原型已构建 + GPU 已验证（本次会话）—— 见 §7。** 仅 overlay（不 fork verl）：
> `fedagent/fed/persistent_{patch,task_runner,main}.py` + 一个 `sitecustomize` 门控。每个 client 的
> reset 是 `reload_client_model` → 重指 `engine.model_config.local_path` + `engine.initialize()`
>（已验证一次调用即可重建 module+optimizer+scheduler）+ `del engine._fedprox_w_t`。在一个
> 2-client/1-round TinyGuess smoke 上做 A/B：**207 s vs 327 s（−37%）**，且各 client 的 checkpoint 与
> 子进程路径**等价（max\|Δ\|≈1e-6，bf16 噪声）**。该 reset 复现了 fresh-Adam / fresh-LR /
> 丢弃的 FedProx-anchor；vLLM-RNG 发散风险在这里的 checkpoint 层面没有出现。
> 现已在**完整循环**层面确认（2 轮最终聚合模型 max\|Δ\|=1.13e-5，§7.1）以及 **PPO**（critic
> reload 已验证，§7.1）；剩余：在更大 step 数上确认。

- **可行性：** **可以，但有注意事项。** verl 本就每步重新进入"load-weights → push-to-rollout →
  train"；worker group / vLLM / FSDP 本就被设计为长生命周期。架构上没有任何东西
  禁止用一个 trainer 驱动 N 个 client。注意事项**全都关于数值
  等价**，而非机制。
- **最干净的接缝：** 把 `RayPPOTrainer.fit()` 的主体抽成一个可重入的 `_fit_one_client()`，
  由外层循环驱动。自然的切点是 **`ray_trainer.py:1383-1410`** —— 它*之上*的一切
 （`init_workers()`）是一次性的；从 `self.global_steps = 0`（L1383）往下已经是 per-`fit()`，
  会变成 per-client。overlay 驱动的接缝是 **`run_fed.py:854-861`**（`for c in selected`
  循环从 `subprocess.Popen` 变成 `trainer.train_client(c, model_r, seed_c)`）。
- **尖锐细节：** FedAvg 聚合器**只**写 `model_world_size_*` 分片并**剥离
  optimizer/extra 分片**（`aggregate_fedavg_fsdp.py:75-76`）。所以你**不能**原样复用
  `load_checkpoint`（它会断言 optim 分片存在）。改用 `load_contents=["model"]` 加载
 （或把 merged-HF weights 在内存中推入 `engine.module`），然后 `update_weights()` 到 vLLM。

#### 每个 client 的 reset 清单（等价审计规范）
子进程设计的标志性性质：**每个 client 从全新的 Adam 矩、step 0 的全新
cosine schedule、以及 FedProx anchor = 刚加载的聚合模型 开始。** 持久化
进程必须手动复现其中每一项：

| # | 状态 | 持久化路径必须… | 引用 |
|---|---|---|---|
| 1 | actor weights | 加载聚合的 FSDP 分片（model-only）→ `update_weights()` | aggregator:75; ray_trainer:1387 |
| 2 | **Adam m/v** | **重建 optimizer**（不要保留上一个 client 的）—— *最大的陷阱* | transformer_impl:451 |
| 3 | **LR scheduler** | 在 step 0 **重建 scheduler**；若 `len(dataloader)` 不同则重算 `total_training_steps` | ray_trainer:438 |
| 4 | `global_steps` | 每个 client 重置为 0（已在 L1383） | ray_trainer:1383 |
| 5 | torch/numpy/py RNG | 每个 client 重新 seed（否则 stream *会续上*） | transformer_impl:135 |
| 6 | **env seed** | 重建 `AgenticDataset` 以让新的 `FEDAGENT_BASE_SEED` 生效（只在 `__init__` 时读取） | agentic_dataset:55 |
| 7 | dataloader iterator + sampler RNG | 每个 client 重建 `train_dataloader`/sampler | ray_trainer:374 |
| 8 | vLLM GPU weights | 由 #1 的 `update_weights` 覆盖 | vllm_rollout/utils:288 |
| 9 | vLLM KV/prefix cache | flush（自动，**当且仅当** swap 经由 `ServerAdapter.update_weights`） | vllm_rollout:194 |
| 10 | **vLLM sampler RNG** | **无干净的 reset API** —— 把 per-request `seed` 注入 `SamplingParams` | vllm_async_server:505 |
| 11 | **FedProx anchor `w_t`** | swap 后 `del engine._fedprox_w_t`（否则会锚到上一个 client 的模型） | fedprox:35 |
| 12 | ref-policy weights | 把 ref 重指到已 swap 的聚合 weights | engine_workers:452 |
| 13 | critic (PPO) | 对 `critic_wg` 做与 #1–#3 相同的处理（GRPO：跳过） | ray_trainer:1008 |
| 14 | GPU mem 碎片 | 每次 swap 做 `aggressive_empty_cache(force_sync=True)`（运维性的，非数值性） | engine_workers:738 |

#### Top-3 等价风险（已排序）
1. **Optimizer / LR / FedProx-anchor 残留**（#2,#3,#11）—— 静默地改变第一个之后每个 client 的
   优化轨迹；在日志里看不见。聚合器通过剥离 optim 分片*强制*执行 fresh-Adam 不变式；
   持久化必须复制它。**强制重建。**
2. **vLLM sampler RNG 连续性**（#10）—— engine 只在构造时 seed 一次；没有 API 能重新 seed 它，
   所以采样 stream *会跨 client 续上*。它决定每个 client 看到哪些轨迹 →
   即便 weights/env-seed 完全相同，rollout 也会偏离子进程路径。只能部分
   修复（per-request seed）。可能迫使"等价"被定义为**统计意义上的**，而非 bit-exact。
3. **过期的 dataset/env seed + KV cache**（#6,#7,#9）—— 复用 dataset 对象会把每个 client 钉在
   client-0 的目标上（一个*正确性* bug）；绕过 `update_weights` 会留下过期的 prefix block。
   修起来直接，但漏了就是静默的。

> **先例：** legacy verl-agent 0.3.1 与当前 overlay 都是 subprocess-per-(client,round)
> —— 两者都从未让 trainer 保持存活（legacy 甚至在 client 之间轮询 `nvidia-smi` 直到 GPU 空闲）。
> 所以 #4 是一项**真正全新的能力**，没有可移植的参考：它必须在*进程内*解决
> legacy 靠进程死亡解决的问题。→ 采纳前必须对子进程路径做 A/B 验证。

### Lever #3 —— 一轮内并行 client *（小模型单节点就赢 —— 已 GPU 验证；大模型才需多节点）*
`for c in selected` → 并发，每个 client 在自己的 GPU 子集/节点上。**与串行数值完全一致**
（FedAvg 与顺序无关；每个 client 的 env seed 是 `base_seed + round*100 + client`
（`run_fed.py:635`），*按 client 索引，不依赖顺序*）。
把 4 GPU 拆成 2+2 会改变 FSDP `world_size` → 分片布局（聚合器动态读 `world_size_of`，能处理；相同 global batch ⇒ 相同数值）。

**单节点已 GPU 验证（2 client × 2 GPU，1.5B，paper 设置）—— 而且快 ~35%,不是打平。**
两个独立的 verl/Ray/vLLM job 在 4 卡节点上**干净共存**：引擎各占不相交的卡对（4 张各 6519 MiB），无 Ray 端口 /
GPU / `/dev/shm` 冲突 —— 隔离只靠各自的 `CUDA_VISIBLE_DEVICES` + `RAY_TMPDIR`。时序：

| arm | 墙钟 |
|---|---|
| `t1` —— 1 client, 4 GPU | 558s |
| `t1` —— 1 client, 2 GPU | 725s |
| **#3 —— 2 client × 2 GPU 并发** | **727s** |
| 顺序 —— 2 client × 4 GPU | 2×558 = **1116s** |

赢在**小模型下 FSDP 多卡扩展是 sub-linear 的**：1.5B 上 4 卡只比 2 卡快 `725/558 = 1.30×`（FSDP
all-gather/reduce-scatter 开销 + env-latency-bound 的 WebShop rollout + 固定冷启动,在单卡算力小的时候占比都更大）。
所以 4→2+2 拆开、两个 client 并发,反而打赢"各用满 4 卡顺序跑"。**这推翻了之前"单节点 #3 = 争用"的说法** —— 小模型上是真收益。
*注意*:大模型 4 卡扩展接近 2× 时,单节点 #3 打平/更慢 → 那才是真正需要 **≥2 节点(一 client 一节点)** 的场景。

**并发暴露并修掉了一个真实 bug(FedAvg rendezvous 端口)。** 聚合步
（`torchrun --nproc_per_node=ws aggregate_fedavg_fsdp.py`）用了 torchrun **默认的 c10d rendezvous `localhost:29500`**,
两个 client 同时聚合就**抢 29500** → 一个聚合中途死 `rc=1`（`CUDA_VISIBLE_DEVICES`/`RAY_TMPDIR` 隔离不了 TCP 端口;
这也意味着**任意两个 `run_fed` 同节点聚合都不安全**）。修法:`torchrun --standalone`（自动选空闲端口）+ 清掉聚合进程
继承的 `MASTER_*`/`RANK`/`WORLD_SIZE`（`run_fed.py fedavg()`）。只动**聚合的通信端口** —— FedAvg 数学、rollout、eval
都不变;PPO critic 的 FedAvg 也走这条。已 GPU 验证:之前会崩的并发 A+B,现在**两个都** `rc=0`、无 `EADDRINUSE`。
*排查注*:表面症状 `DataLoader worker killed (SIGKILL)` 是**红鲱鱼**(两个 run 都打的良性 `__del__` teardown 噪声;
dmesg 无 OOM-killer,节点 RAM 966 G / `/dev/shm` 504 G / cgroup 无限都很宽),真正的失败是下一步的 29500 冲突。

### Lever #1 —— eval(r) ∥ train(r+1) *（需要额外 GPU；有界）*
两者都读 `model_r`（§2.4）→ 独立。在备用分配上跑 eval，同时第 `r+1` 轮训练。
eval **每 round** 都发生（per-round 红线，§7.4），且每次 `inline` eval 本身就是一个完整的冷启动子进程
—— 所以让它变便宜/可重叠*每一轮*都有意义，而非偶尔。**与 #4 叠加：** 有了持久化 trainer，eval 变成
`update_weights(model_r) + val pass`（数秒，而非冷启动）—— 这正是 **`eval_mode=worker`**（§7.4），无需
额外 GPU；`parallel` 把它重叠到空闲卡上。纯测量 ⇒ **零数值风险。**

### 按硬件的 ROI 排序
| 硬件 | 先做 | 大杠杆 | 跳过（争用） |
|---|---|---|---|
| **单 4-GPU 节点**（默认） | **#2**（免费、零风险） | **#4**（唯一触及那 88% 的）；**小模型用 #3**（1.5B 上 2×2 = −35%,已 GPU 验证 §Lever #3） | #1（需空闲 GPU）；#3 在*大*模型上（4 卡扩展 ≈2× ⇒ 打平） |
| **≥2 节点 / 8 GPU** | #2 + **#3**（巨大、bit-equivalent） | #1 在备用分配上免费；#4 仍是终极的 GPU-hour 收益 | — |

---

## 4. 科学等价小结（项目基准）
| 杠杆 | 数值影响 | 裁定 |
|---|---|---|
| #2 env prewarm | 无（纯调度） | ✅ 安全 |
| #1 eval ∥ train | 无（测量） | ✅ 安全 |
| #3 parallel clients | 无（FedAvg 与顺序无关、seed 按 client 索引） | ✅ 安全；单节点 2×2 已 GPU 验证（1.5B −35%）+ FedAvg rendezvous 端口 bug 已修（`--standalone`，§Lever #3） |
| #4 persistent trainer | **未测出**：smoke max\|Δ\|≈1e-6，full-loop max\|Δ\|=1.13e-5；若漏掉 reset 则风险高 | ✅ 已验证（§7）含完整循环 + PPO critic reload；需在更大 step 数上确认 |

---

## 5. 实验计划 + GPU 验证
1. **Baseline 计时（夯实那 88%）：** 给 `run_fed.py` 加上 per-phase wall-clock 埋点
   （`service_start`、`run_client` 总耗时 + 它的进程内冷启动、`fedavg`、`merge`、`eval`）。
   跑一个小的联邦 smoke；输出一张 phase 分解表。*（这是 §2.1 的实证锚点。）*
2. **#2 A/B：** 在 WebShop/ALFWorld 上用同一配置，开/关 prewarm；比较 round-start 延迟。
3. **#4 spike + 等价 A/B：** 原型化 `_fit_one_client()`，在一个进程内驱动 2 个 client；
   把 per-step 指标（loss、reward、advantages）与子进程路径对比，**可能处 bit-for-bit、
   否则统计意义上**（vLLM RNG 注意项，风险 #2）。
4. **#3（若有多节点）：** 在 2 个分配上并发跑 2 个 client；确认
   聚合模型在 fp 噪声范围内与串行 run 一致。

## 6. 推荐路线图（分阶段）
- **Phase 0 —— 测量。** 埋点 + 给一个 smoke 做 baseline 计时（把那 88% 从"汇总数字"变成
  一张全新的 per-phase 表）。*低风险，高信息量。*
- **Phase 1 —— #2 prewarm。** 实现 + A/B。零数值风险，立竿见影的单节点收益。
- **Phase 2 —— #4 原型。✅ 已完成 + GPU 验证（本次会话，§7）。** 仅 overlay：per-round −43%
  （§7.1）、**cross-round −62% 且整个 run 只有一次冷启动（§7.2）**、PPO critic reload（§7.1）
  —— 全程 full-loop max\|Δ\|=1.13e-5 EQUIVALENT（TinyGuess）；**per-client service 路由已在
  真实 WebShop env 上验证**（§7.3，32/32 个 episode 分流到各自的服务）。cross-round + per-round
  eval 的 OOM 已 GPU 确认，并由**回退到 per-round 持久化自动处理**（§7.4）。
  **剩余：** 一次论文级（1.5B）持久化 run + 一次更大 step 的等价 A/B。
- **Phase 3 —— 多节点 #3 / #1。** 当 ≥2 个分配可用时，并行 client 并把
  eval 浮到备用节点上。

---

## 7. 本次会话结果 —— GPU 验证的 #4 原型

**已构建（仅 overlay，不 fork verl）：**
- [persistent_patch.py](../fed/persistent_patch.py) —— `reload_client_model`（`ONE_TO_ALL`），通过
  一个 deferred import hook（对照 FedProx）挂到 worker 类上，使它落到每个 Ray FSDP worker 上；
  由 [sitecustomize.py](../../sitecustomize.py) 中的 `FEDAGENT_PERSISTENT=1` 门控。
- [persistent_task_runner.py](../fed/persistent_task_runner.py) —— `PersistentFedTaskRunner(TaskRunner)`：
  `init_workers()` 一次，然后 per-client `fit()`，其间用 `_reset_for_client`。
- [persistent_main.py](../fed/persistent_main.py) —— Hydra 入口（`run_ppo(config, task_runner_class=…)`）。
- [compare_fsdp_checkpoints.py](../../tools/verl08_migration/compare_fsdp_checkpoints.py) —— tensor diff。

**每个 client 的 reset（机制已验证）：** `reload_client_model` 重指
`engine.model_config.local_path` → `engine.initialize()`（transformer_impl.py:183→543：`_build_module`
读 `local_path` → 新 weights；`_build_optimizer`/`_build_lr_scheduler` → fresh Adam + fresh
schedule）→ `del engine._fedprox_w_t`（重新锚定 FedProx）。Dataloader 按 seed 重建；driver RNG 重新 seed。

**A/B（TinyGuess，0.5B，4-GPU，concat，GRPO，2 clients × 1 round，2 steps，匹配 seed 142/143）：**

| 指标 | persistent | subprocess |
|---|---|---|
| wall-clock | **207 s**（1 次冷启动） | 327 s（2 次冷启动 + FedAvg + merge） |
| **节省** | — | **120 s = 37%** |
| client-0 ckpt max\|Δ\| | 4.0e-6 → **等价** | （参考） |
| client-1 ckpt max\|Δ\| | 7.6e-6 → **等价** | （参考） |

在最小的情形下，持久化 trainer **数值忠实（1e-6 = bf16 噪声）且快 37%**。这里的
节省 = 每轮 `(clients_per_round − 1)` 次冷启动；配合 `cross_round`（§7.2），进程跨越整个 run，
捕获**全部约 140 次冷启动**（总共只付一次）。

### 7.1 已集成进 `run_fed`（`persistent: true`）—— 完整联邦循环
现在一整轮的 client 经由 `run_round_persistent`（[run_fed.py](../fed/run_fed.py)）在一个进程内训练：
它写一份 plan JSON，启动 `persistent_main` 一次，然后扫描各 client 的 checkpoint → 下游的
**同一套** FedAvg/merge 照常运行（字节级一致）。一个 **2 轮 TinyGuess GRPO** A/B：

| | persistent（`persistent: true`） | subprocess |
|---|---|---|
| 2 轮联邦循环 | **闭环 rc=0，515 s** | 闭环 rc=0，**909 s** |
| **节省** | — | **394 s = −43%**（避免约 2 次冷启动） |
| 最终聚合模型（round_2） | **max\|Δ\|=1.13e-5，mean 1.7e-7 → 等价**（atol 1e-4） | （参考） |

整个循环跑完后，round-2 最终聚合 actor 是**逐 tensor 等价**的（最差 tensor 为
`layers.15.self_attn.o_proj.weight`，1.13e-5 —— bf16 往返噪声），所以这份加速是**免费**的：
持久化路径的 hot-swap reset 精确复现了子进程路径的 fresh-Adam / fresh-LR / fresh-dataloader，
不仅是单轮，而是经由 FedAvg 跨轮复合后依然等价。

每轮节省 = `(clients_per_round − 1)` 次冷启动。**PPO critic reload —— 已 GPU 验证。**
一个 2-client、`adv_estimator=gae` 的持久化 smoke 闭环 rc=0，且 value model 每轮都做 FedAvg：
client 1 的 per-client reset 重建了 critic engine（`TrainingWorker` 上的 `reload_critic_model`）
以产出它的 `.../global_step_2/critic`，随后与 client 0 的 critic 做 FedAvg → 聚合后的 `actor`
**和** `critic` 都写出，循环闭环。（这首先需要在 [fedagent_ppo.yaml](../config/fedagent_ppo.yaml)
里加一个 `critic:` 块 —— verl 的 gae 路径要求显式设置 value model 的 micro-batch；GRPO 下 critic
被禁用，故该块无效化。）WebShop/ALFWorld 在持久化模式下由 per-client service 路由（§7.3）解除阻塞。

### 7.2 跨轮持久化（`cross_round: true`）—— 已 GPU 验证
per-round 路径仍然*每轮*付一次冷启动。`cross_round: true` 让**一个进程在整个 run 期间保持
存活**：一轮的 client 训练 + 保存之后，worker 写一个 `done_<r>` 信号并**空转**（持有其 GPU），
此时编排器运行**同一套外部 FedAvg/merge**（字节级一致 → 等价性得以保持）；编排器随后发布
`plan_round_<r+1>.json` + touch `go_<r+1>`，worker 重置到 merged 模型并训练下一轮 —— 全在同一进程内。
聚合器初始化一个*独立*的 NCCL world 且每 rank 用 ~1 GB，因此它与暂停的 worker 共存于同一组 H100。

一个 **2 轮 TinyGuess GRPO** A/B（与 §7.1 相同 seed，可直接对比）：

| | cross-round | per-round persistent | subprocess |
|---|---|---|---|
| cold-starts（whole run） | **1** | 2 | 4 |
| wall（2 轮循环） | **342 s** | 515 s | 909 s |
| **vs subprocess** | **−62%** | −43% | （参考） |
| 最终聚合模型（round_2） | **max\|Δ\|=1.13e-5 → EQUIVALENT** | 1.13e-5 → EQUIVALENT | （参考） |

整个 run 恰好**一次** `Started a local Ray instance`；worker 在 `stop` 时以 rc=0 退出。最终
模型是逐 tensor 等价的（最差 tensor 为 `layers.11.mlp.gate_proj.weight`，1.13e-5）—— reset 等价
论证*跨越 FedAvg 边界*依然成立，所以在一个进程内跨越所有轮次在保真度上零成本。实现：
[run_fed.py](../fed/run_fed.py) `BgProc` + `_wait_signal` + `stop_persistent_cross_round`；
[persistent_task_runner.py](../fed/persistent_task_runner.py) `_wait_next_round`（worker 侧的外层循环）。

### 7.3 Per-client service 路由（WebShop/ALFWorld）—— 已在真实 WebShop 上 GPU 验证
持久化模式下所有 client 共享一个进程，所以进程级 env 路由（`WEBSHOP_SERVICE_URL`）无法给每个
client 各自的服务。修复：一个**文件通道**。driver 在每次 `fit()` 之前用当前 client 的 URL
（`base_port + client_id`）改写 `$FEDAGENT_SERVICE_URL_FILE`（`_route_service`）；共享的 agent-loop
worker —— 它们在一个*独立*进程中按 episode 构建 env —— 读取该文件（`resolve_service_url`，优先级：
file > env-var > config > default）。与拓扑无关（文件在共享 FS 上），从而绕开了要伸进 async
agent-loop worker 进程的麻烦。已单元测试（file 胜过 env-var；per-client 切换无需重启即生效；
missing-file 回退；tinyguess no-op），**并在真实 `verl-agent-webshop` 服务上端到端验证**：一次
2-client 持久化 smoke（端口 8100/8101）rc=0 关闭，`route client 0 → :8100`、`route client 1 → :8101`，
且 —— 决定性地 —— **每个 client 自己的服务恰好服务了它的 32 个 episode**（env seed 11 vs 12 各异）。
路由坏掉会把全部 64 个发到一个服务（或默认 `:8080`）；32/32 各自分流就是判别性证据。

### 7.4 Eval/训练 GPU 共享（`eval_mode`：inline / parallel / shared / worker）
**Eval cadence（曲线是什么）。** 论文那条 "server-aggregated" 红线 = **每个 round 对该 round 聚合后的
全局模型在 shared 无扰动 val set 上评一次**，**每 round 一个点**（`val_before_train` 额外加 round-0 的
base 点）。它**不**由 `test_freq` 门控 —— 那个旋钮是 verl 的 *job 内部* step cadence（每 round
`epochs_per_round` 步，只触发 `is_last_step`，即 per-client 的 "client-end" 点）。对该 round 聚合模型评
一次 = 论文 per-client `val_before_train`(step-0) **average 的期望**：一个 round 里所有 client 都从*同一个*
聚合模型出发，所以单次 shared eval 给同一条曲线，只花一小部分 rollout 成本。门控：每 round `run_eval`
（[run_fed.py](../fed/run_fed.py)）；`worker` 在 `i==0` 评该 round 的*起始*模型，其余 mode 在 merge 后评。

**为什么 eval 是 read-only（零等价风险）。** eval 加载 merged `model_r`、在 val set 上打分、不向训练写回
任何东西（不碰 RNG/data/weights）。所以 `eval(model_r)` 不管 async / parallel / deferred / 在热引擎上跑，
都给出**逐 bit 相同的训练轨迹** —— 不像 lever #4（要复现 reset 状态）。唯一约束是显存。

**资源物理。** vLLM **预留** `gpu_memory_utilization × 总显存` 作 KV cache，**与模型大小无关**。同一张 GPU
上两个 vLLM 引擎冲突：0.6 + 0.6 > 1.0 → `cross_round` OOM（`Free 28.9 < desired 47.5 GiB`）。所以
"eval ∥ train" 要么 **GPU 不相交**、要么 **缩小的第二引擎**、要么 **复用那一个热引擎** —— 故 4 种 mode（`eval_mode`）：

| mode | 机制 | 何时用 | 代价 |
|---|---|---|---|
| **inline**（默认）| merge 后阻塞式 eval，用 `n_gpus_per_node` 张卡 | 训练**占满**整节点 → eval 在它的窗口独占整节点 | eval cold-start、在关键路径上；`cross_round`+inline 会 OOM → 自动回退到 per-round |
| **parallel** | eval 放在**不相交的 GPU 子集**（`CUDA_VISIBLE_DEVICES`），与下一轮训练并发；非阻塞启动 + 延迟收集 | 训练用 **< 整节点** 的卡（如 4 选 2）→ 空闲卡 async | 需要 `n_gpus_per_node + eval_gpus ≤ 节点`；离开关键路径 |
| **shared** | **第二个** eval vLLM 与 worker 共享 GPU，降到 `eval_gpu_mem_util=0.3` | 单节点、`cross_round`、无空闲卡 | eval 串行 + 每轮一次 cold-start（KV 池小）；保住 `cross_round` |
| **worker** | cross-round worker 用**自己的热 vLLM**（verl `_validate()`）评该 round 起始模型 —— 无第二引擎 | 单节点、`cross_round`、无空闲卡（论文的 4-GPU 场景）| **无 OOM、无 eval cold-start**；串行但便宜；需要下述 FSDP→vLLM 权重同步 |

**inline 自动回退。** `cross_round` + `eval_mode=inline` 会 OOM（worker 占着卡）→ `run()` 回退到 per-round
持久化。`parallel`/`shared`/`worker` 通过隔离/缩小/复用引擎保住 `cross_round` 速度。

**GPU 已验证对比**（2-client × 2-round WebShop，每 round eval，0.5B；cross-round 基线 = 342s 无 eval，
per-round 持久化基线 = 909s；训练用节点 4 张卡里的 2 张）：

| eval_mode | 进程基线 | wall-clock | val_curve r0/r1/r2 | eval OOM? | 需要空闲卡? |
|---|---|---|---|---|---|
| **inline** | per-round 持久化 | 1018s | −0.6 / −0.6 / −0.6 | n/a（串行）| 否 |
| **parallel** | cross-round | **690s** | −0.6 / −0.6 / −0.6 *(相同)* | 无（不相交 GPU）| **是（≥2）** |
| **shared** | cross-round | 874s | −0.6 / −0.6 / −0.6 *(相同)* | 无（0.3 util）| 否 |
| **worker** | cross-round | **703s** | −0.6 / −0.6 / −0.6 *(相同)* | 无（一个引擎）| 否 |

4 条 `val_curve` 逐 bit 相同 —— eval 是 read-only，`eval_mode` 只改变 eval *在哪/何时*跑，不改结果。4 个都
`rc=0`（`shared`/`worker` 在 shutdown 时有无害的 `__del__` DataLoader teardown 噪声，run 仍干净关闭）。

**饱和 4-GPU 论文场景 → `worker`。** `n_gpus_per_node=4` 时没有空闲卡，`parallel` 作为 *4-train* 不适用（但
可以 **2 train + 2 eval** 跑,见下）、`shared` 每轮付一次 eval cold-start。`worker` 复用**热** rollout 引擎
（无第二 vLLM → 无 OOM、无 cold-start），每轮 eval 便宜、保住 `cross_round` 速度。端到端已验证（2-round
smoke 上 **703s** vs shared **874s**；下面 1.5B/n=500 进一步修正：`shared` 其实是*最慢*的）。

**1.5B 论文设置、4 卡对比（已 GPU 验证）。** 上面 0.5B 表卡在 −0.6 地板；在**论文设置**下重跑（1.5B、G=8、
`webshop_15` 15-turn、response 512、**n=500 val**、100-client uniform partition 2/round、seed 42、2 round），
每个模式都用**满 4 卡节点** —— 三个 4 训练卡，`parallel` 用 **2 训练 + 2 eval**。（`n_gpus_per_node` 不是算法
参数 —— FSDP 切 2 卡 vs 4 卡是同样的数学 —— 所以四个都与论文算法一致。）

| eval_mode | GPU 布局 | 墙钟 | rc |
|---|---|---|---|
| **parallel** | 2 训练 + 2 eval | **2493s** | 0 |
| **worker** | 4 训练（热引擎 eval） | 2637s | 0 |
| **inline** | 4 训练（merge 后阻塞 eval） | 3090s | 0 |
| **shared** | 4 训练 + 第二 eval 引擎 @ 0.3 util | **3316s** | 0 |

四个都干净跑通 —— 连 `shared` 的共存第二引擎、`parallel` 的拆分都**没 OOM**。排序 `parallel < worker <
inline < shared` —— 两个被 0.5B 地板掩盖的发现：
- **val 集大时 `shared` 翻转成最慢。** 0.5B/n=8 时 `shared`(874s) 快过 `inline`(1018s)；1.5B/**n=500** 时
  `shared`(3316s) *最慢*，超过 `inline`(3090s)。原因：`shared` 缩小的 KV（0.3-util）eval 引擎压低了 batch
  并发，500 episode 的 eval 被卡住 —— 这个惩罚**随 val 集大小放大**，n=8 时看不见。所以 val 大时别选 `shared`。
- **`parallel` 靠把贵 eval 藏起来取胜。** 它在不相交 2 卡上的满 util eval 引擎和下一轮训练重叠（n=500 eval
  在关键路径外）；`worker` 紧随其后（满 util 热引擎，串行但便宜）；`inline` 每轮付 cold-start **且**阻塞在 eval 上。

各模式的 val 数值差异来自 **eval 采样**（temp=0.4，n=500 但只有 3–25 个 success）和 eval 路径，不是训练：跨模式
**权重等价性**已直接验证（worker vs inline 1.5B 聚合，max|Δ| 3.8e-6 / 7.6e-6），所以 `eval_mode` 仍不改变轨迹。

**`worker` 需要的修复**（verl 生命周期，[persistent_task_runner.py](../fed/persistent_task_runner.py)）。
verl 的 `_validate()` 是为在 `fit()` 的引擎生命周期*内部*运行设计的；从持久化循环驱动它就要复现这套生命周期：
1. **`global_steps`** —— `_validate()` 读它做 step 标签；只有 `fit()` 设它 → 缺失时 seed `=0`。
2. **FSDP→vLLM 权重同步**（*CUDA 崩溃的真正根因*）—— verl 用 **dummy** 权重初始化 rollout vLLM，且在
   `init_workers` 后让它**睡着**（`ray_trainer.py:972`）；真权重由 `checkpoint_manager.update_weights` 每次
   rollout 前同步。worker-eval 在该 round 的 `fit()` *之前*跑，没同步就拿 dummy 权重 →
   **CUDA illegal-memory-access / EngineDeadError**。修复：`_validate` 前 `update_weights`（同步+唤醒）、
   之后 `sleep_replicas` —— 复刻 `fit()` 的 `ray_trainer.py:1387`。（`enforce_eager` 只是*移动*了症状，不是修复。）
3. **dump executor** —— 每次 `fit()` 在结尾关闭 verl 的 dump `ThreadPoolExecutor`；下一轮 worker-eval 会向
   死掉的 executor 提交（`cannot schedule new futures`）。修复：`_validate` 前若已关闭则重建，正如 `fit()`
   在 `ray_trainer.py:1369` 所做。
4. **`val_batch_size`** —— 遵从 `config.data.val_batch_size`（stock verl）而非 `len(val)`，免得把整个
   WebShop/ALFWorld val set 一个 batch 打出去（env-service storm）。

**Per-client "client-end" 圆点（`client_end_eval`，默认关）。** 红线评的是该 round 的*聚合*模型；论文还把每个
client **训练后**的模型画成一个圆点（每 round 每 client 一个）。开 `client_end_eval: true` 会在**无扰动 val
set** 上每 round 多评 `clients_per_round` 次，并在 `federated_summary.json` 里和 `val_curve` 并列写出
`client_curve`（每个 `(round, client)` 一条）。按 `eval_mode` 分两条路径：
- **orchestrator**（inline/parallel/shared）：`eval_client` 把 client 训练后的 actor 合并到
  `round_<r>/client_<c>/hf`，再走正常的 `eval_global` 路径打到**无扰动 val service** —— 绕开 within-job 路由
  问题（env 分不清 train rollout 和 val rollout，无法在 job 中途换 service URL，所以 client *自己*的 job
  没法在干净 val set 上自评）。必须在 `cleanup_round_checkpoints` **之前**跑（它读 client shard；合并出的
  `hf` 会留下）。
- **worker**（热引擎）：每个 client 的 `fit()` 之后，`_worker_validate(r, client_id=c)` 在**热**rollout 引擎上
  评刚训完的模型 —— 不合并、不起第二个 service。

两条路径都已 GPU 验证（2×2 WebShop smoke）：`client_curve` = 4 个圆点（r1c0、r1c1、r2c0、r2c1，全 `−0.6`），
与 3 点红线 `val_curve`（r0 base、r1/r2 聚合）一致。默认关 —— 红线是主曲线，圆点是可选诊断。

### 7.5 Windowed 默认值 release blocker —— 已修复
新的 `rollout_mode=windowed` 默认值在原生 `agent.yaml` 上崩溃
（`AttributeError: GymTextAgentLoop has no run_episode_windowed`，§2.6）。在**不**编辑
7 份 env spec、也不重新生成 176 份论文配置的前提下修复：
- [agent.yaml](../config/agent.yaml) 注册了 `gym_text_windowed` → `WindowedGymTextAgentLoop`。
- [windowed_manager.py](../agent_loops/windowed_manager.py) 的 `_run_agent_loop` **自动映射**
  `agent_name=gym_text → gym_text_windowed`，因此一份共享的 env spec 同时驱动两种模式。
- `FEDAGENT_HISTORY_LENGTH`（由 `run_fed` 按 rollout_mode 设置：**windowed=2 / concat=0**，优先级
  高于 spec；由 `alfworld_env`/`webshop_env` 读取）让同一份 spec 在任一模式下都忠实 ——
  windowed 拿到论文的 2-history 模板，concat 拿到 `history_length=0`（由 GymTextAgentLoop 拥有
  history）。**✅ GPU 已验证：** 之前会崩溃的 windowed 默认配置现在端到端跑通整个
  联邦循环（4 steps，FedAvg + merge，`FEDERATED LOOP CLOSED` rc=0，无 `AttributeError`）。

### 7.6 剩余（已排序）
1. **更大 step / 真实 env 的等价 A/B** —— 等价目前在 TinyGuess 上以 2 步验证
   （max\|Δ\|=1.13e-5）；在更多 step、以及一次真实 env 的持久化 A/B 上确认。
2. **论文配置：** 已重新生成到 windowed `response_length=512` 预算（176 文件，§8.1）；一次
   真实论文级（1.5B）持久化 run 是最终的集成检查。
3. **vLLM sampler-RNG / /dev/shm teardown** 在长时程下（良性 teardown 噪声 —— 退出时的
   `DataLoader worker killed` / `resource_tracker KeyError`，rc 仍为 0；持续观察，若真咬到再加
   `aggressive_empty_cache` / `SamplingParams.seed`）。

---

## 8. 实现参考（本次会话的改动）

下面的一切都是**仅 overlay（不 fork verl）**，且目前**本地/未提交**。

### 8.1 新增 / 改动的文件

**新增（lever #4 —— 持久化 trainer）：**

| 文件 | 用途 | 关键符号 |
|---|---|---|
| [persistent_patch.py](../fed/persistent_patch.py) | 通过 deferred import hook 把 per-client reset 方法挂到 verl worker 类上 | `reload_client_model`（ActorRolloutRefWorker）、`reload_critic_model`（TrainingWorker）、`install_deferred_persistent_patch()` |
| [persistent_task_runner.py](../fed/persistent_task_runner.py) | `PersistentFedTaskRunner(TaskRunner)`：init_workers 一次，fit-per-client；**跨轮外层循环**；**per-client service 路由**；**`eval_mode=worker`** 热引擎 eval + **client-end 圆点**（§7.4）| `run()`、`_reset_for_client(spec)`、`_wait_next_round(xdir,r)`、`_route_service(spec)`、`_worker_validate(r, client_id=None)`（update_weights+global_steps+dump-executor+sleep_replicas；`client_id` → client-end 圆点）、`_should_worker_eval(r)`（每-round 门控）|
| [persistent_main.py](../fed/persistent_main.py) | Hydra 入口 → `run_ppo(config, task_runner_class=ray.remote(...)(PersistentFedTaskRunner))` | `main()` |
| [compare_fsdp_checkpoints.py](../../tools/verl08_migration/compare_fsdp_checkpoints.py) | 逐 tensor 的 FSDP-checkpoint 等价 diff | `compare_dir(a,b,atol)` |

**改动：**

| 文件 | 改动 |
|---|---|
| [sitecustomize.py](../../sitecustomize.py) | 门控 `FEDAGENT_PERSISTENT=1` → `install_deferred_persistent_patch()`（每个 Ray worker 拿到 reset 方法） |
| [run_fed.py](../fed/run_fed.py) | **#4 per-round：** `persistent` 标志 + `run_round_persistent()` + run-loop 分支。**#4 cross-round：** `cross_round` 标志 + `BgProc`（行缓冲日志）+ `_wait_signal` + `stop_persistent_cross_round` + signal-file 握手。**routing：** `client_service_url()` + plan `service_url` + `FEDAGENT_SERVICE_URL_FILE`。**eval modes（§7.4）：** `eval_mode` inline/parallel/shared/worker —— `_build_eval`/`eval_global`/`launch_eval_async`/`collect_eval`；per-round eval **每 round**（`if do_eval:`，非 `r%test_freq`）；`cross_round`+inline → 自动回退 per-round。**client-end 圆点（§7.4）：** `client_end_eval` 标志 + `eval_client()`（合并 client actor → `client_<c>/hf` + 在 val service 上 eval，清理之前）+ `merge_to_hf(out_hf=)`/`_build_eval(client_id=)`；在 summary 中写出 `client_curve`。**metrics：** flush BgProc + 解析启动日志（cross-round），让 `metrics.json` 不为 `[]`。**#2：** `prewarm_next_round_services()`。**windowed：** `history_length_env()`。**#3（并发聚合，§Lever #3）：** `fedavg()` 用 `torchrun --standalone` + 清 `MASTER_*`/`RANK`/`WORLD_SIZE`,让同节点两个 client/实验聚合时不撞默认 rendezvous 端口 29500 |
| [fedagent_ppo.yaml](../config/fedagent_ppo.yaml) | 新增 `critic:` 块（PPO/gae value-model micro-batch；GRPO 下无效化） |
| `fedagent/config/paper/*.yaml`（176） | **重新生成**到 windowed `response_length=512` 预算（原为 `6144`/`8192`）；经由 `tools/verl08_migration/gen_paper_configs.py --out fedagent/config/paper` |
| [base.py](../envs/base.py) | `resolve_service_url(env_var, cfg, default)` —— 文件通道路由 helper（file > env-var > config > default） |
| [agent.yaml](../config/agent.yaml) | 注册 `gym_text_windowed → WindowedGymTextAgentLoop` |
| [windowed_manager.py](../agent_loops/windowed_manager.py) | `_run_agent_loop` 自动映射 `agent_name → {name}_windowed` |
| [alfworld_env.py](../envs/alfworld/alfworld_env.py)、[webshop_env.py](../envs/webshop/webshop_env.py) | `history_length` 读 `FEDAGENT_HISTORY_LENGTH`；service URL 经由 `resolve_service_url`（per-client 文件路由） |

### 8.2 持久化 trainer —— 端到端走查

**(1) 多进程 patch 的问题与修复。** `reload_client_model` 必须存在于 worker 类上
*在每个 Ray FSDP-worker 进程内*，而不仅仅是 driver。在解释器启动时急切 import
`verl.workers.engine_workers` 会在 Ray 设置 per-rank `CUDA_VISIBLE_DEVICES` 之前就拉入 FSDP engine
→ "Duplicate GPU detected"。所以 `install_deferred_persistent_patch()` 装载一个一次性的 `MetaPathFinder`，
它包裹 `engine_workers` 的 `exec_module`，并在 verl import 它的那一刻（设备分配之后）附加
这些方法 —— 与 FedProx 的延迟模式完全相同。`sitecustomize.py` 在每个带
`FEDAGENT_PERSISTENT=1` + verl 在 PYTHONPATH 上的进程中调用它。*已验证：* `[persistent] … attached` 打印在
driver、TaskRunner actor 以及全部 4 个 FSDP worker 中。

**(2) plan。** `run_round_persistent` 写入 `round_<r>/persistent_plan.json` =
`[{client, model_path, critic_path, seed, out_dir, exp}, …]`，对应该轮被选中的 client（它们共享
该轮的 `model_path`/`critic_path`；`seed = base_seed + round*100 + client`，**与** `run_client` **完全相同**）。
它启动一个 `persistent_main`，带 `FEDAGENT_PERSISTENT=1` +
`FEDAGENT_PERSISTENT_PLAN=<path>`，然后扫描 `round_<r>/client_<c>/checkpoints/global_step_*/actor` 并
返回 `{client: (actor_dir, critic_dir)}`，于是**现有的** `fedavg()`/`merge_to_hf()` 原样运行。

**(3) 每个 client 的 reset**（`_reset_for_client` —— 复现一个全新子进程免费获得的东西）：

| reset | 如何 | verl 锚点 |
|---|---|---|
| actor weights+optimizer+scheduler | `reload_client_model` → 重指 `engine.model_config.local_path` + `engine.initialize()`（一次调用重建 module(新 weights)+optimizer(零 Adam)+scheduler） | transformer_impl.py:183→543 |
| ref policy | 同样的 `_reset_engine(self.ref.engine)`（forward_only → 仅 weights） | engine_workers.py:537 |
| critic (PPO) | `reload_critic_model`（TrainingWorker.engine 重建） | engine_workers.py:165 |
| FedProx anchor | reset 后 `del eng._fedprox_w_t`（能在 `initialize()` 后存活） | fedprox.py:37 |
| dataset/env seed | `os.environ[FEDAGENT_BASE_SEED]=seed` 然后 `_create_dataloader(None…)`（driver 侧） | ray_trainer.py:374 |
| global_steps | 原生 `fit()` 设 `=0` 然后 `+=1`（免费） | ray_trainer.py:1383 |
| vLLM weights + KV/prefix | reload 后原生 `fit()` 的 `update_weights()`（完整 `load_weights` + cache flush） | ray_trainer.py:1387 |
| driver RNG | `random/np/torch.manual_seed(seed)` + `torch.cuda.empty_cache()` | — |

然后 `trainer.fit()` 跑 E 个 epoch 并保存该 client 的 checkpoint。

### 8.3 Windowed release-blocker —— 机制

三个协同的部件让**同一份** env spec（`agent_name: gym_text`，无 `history_length`）驱动
两种 rollout 模式，因此**无需编辑 env spec、也无需重新生成 176 份配置**：
1. **Registry** —— `agent.yaml` 现在有 `gym_text_windowed → WindowedGymTextAgentLoop`（带
   `run_episode_windowed` 的子类）。
2. **Auto-map** —— `windowed_manager._run_agent_loop` 在 `_agent_loop_registry` 中解析
   `f"{agent_name}_windowed"`（回退到 `agent_name`）。concat 路径（原生 manager）继续用 `gym_text`。
3. **History** —— `FEDAGENT_HISTORY_LENGTH`（run_fed：windowed=`windowed_history_length` 默认 2，concat=0；
   优先级高于 spec）由 `alfworld_env`/`webshop_env` 读取。windowed → 论文的 2-history
   per-turn 模板；concat → `0`，于是 `GymTextAgentLoop` 拥有不断增长的 chat。

### 8.4 复现

```bash
# lever #4 A/B (full 2-round federated GRPO, TinyGuess, 4 GPU): persistent vs subprocess
python -m fedagent.fed.run_fed --config _scratch/accel/persist_full.yaml   # persistent: true
python -m fedagent.fed.run_fed --config _scratch/accel/subproc_full.yaml   # persistent: false
python tools/verl08_migration/compare_fsdp_checkpoints.py \
  --a .../subproc_full_out/round_2/aggregated/checkpoints/global_step_0/actor \
  --b .../persist_full_out/round_2/aggregated/checkpoints/global_step_0/actor --atol 1e-4

# windowed-default no-crash check (rollout_mode defaults to windowed):
python -m fedagent.fed.run_fed --config _scratch/accel/tinyguess_windowed_check.yaml

# enable lever #4: `persistent: true` (per-round) or `cross_round: true` (one process, whole run).
```

---

> **状态备注（本次会话）：** windowed 默认值 release blocker（§7.5）与 lever #4 —— per-round
> 持久化（§7.1）、**跨轮持久化（§7.2）**、PPO critic reload（§7.1）、**per-client
> service 路由（§7.3）** —— 均**已完成 + GPU 验证**（cross-round 在 TinyGuess 上：整个 run 一次
> 冷启动，−62%，max\|Δ\|=1.13e-5 EQUIVALENT；路由在**真实 `verl-agent-webshop`** env 上：32/32 个
> episode 分流到各自的 per-client 服务），仅 overlay，**未提交**。
>
> **五个 review 缺口 —— 本次会话已解决：**（1）**真实 env 路由** ✅ 已 GPU 验证（§7.3）；
>（2）**cross-round + `val_env_spec`** ✅ —— GPU 确认持有 GPU 的 worker 会让 eval 的 vLLM OOM，
> 现在**自动回退到 per-round 持久化**（§7.4），保住 eval 曲线；（3）**论文配置**
> ✅ 已重新生成到 windowed `response_length=512`（176 文件，原为 `6144`/`8192`，§8.1）；（4）
> **metrics 丢失** ✅ 已修复 —— 此前是 `BgProc` 缓冲了日志（行缓冲 + 解析前 flush + cross-round
> 启动日志路径；parser 本身没问题）；（5）**teardown 噪声** —— 退出时的 `DataLoader worker killed` /
> `resource_tracker KeyError` 是**良性**的（rc=0），记录在 §7.6。windowed 的
> *history-length* 迁移已被 auto-map + `FEDAGENT_HISTORY_LENGTH` 取代；（3）中的
> *response-length* 重新生成是另一个、现已完成的部分。**剩余：** 一次论文级（1.5B）
> 持久化 run + 一次更大 step 的等价 A/B（§7.6）。
