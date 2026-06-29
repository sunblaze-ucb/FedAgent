# FedAgent verl-0.8 加速 —— 完整工程报告

> **对加速 + 验证这条工作线的权威、端到端记述**：问题、设计空间、每一个杠杆与特性
> *深入展开*（机制 → 等价性论证 → 实测结果）、各项调查（包括走过的弯路与修正），
> 以及如何把这一切用起来。
>
> **三份加速文档，按用途区分：**
> - [acceleration.md](acceleration.md) —— 最初的**分析与计划**（冷启动剖析、铁律、杠杆设计、等价性风险审计）。
> - [acceleration_results.md](acceleration_results.md) —— **快速结果**参考（状态表 + 数字一览）。
> - **本文档** —— **完整走查**：一切内容，按顺序，附带推理与实证发现。
>
> **约定。** GPU 为 4×H100（qgpu30xx）。基准 = 在 3-seed 噪声范围内复现论文。**EQUIVALENT** =
> FSDP-checkpoint `max|Δ| ≤ 1e-4`（bf16 噪声地板），对比 stock-verl 子进程基线。模型 =
> Qwen2.5-1.5B-Instruct（smoke：0.5B / TinyGuess）。在 **stock verl 0.8 之上的薄 `fedagent/` overlay，不 fork。**

---

## 1. 问题与基准

FedAgent 是联邦 agent-RL：每一轮，少数几个 **client** 各自训练一个本地策略（GRPO/PPO，针对远程 env
服务的多轮 rollout），然后服务器把它们的权重 **FedAvg** 成一个新的全局模型。原始代码是 verl-agent 0.3.1
的 fork；本次迁移把它重新实现为 **stock verl 0.8 之上的薄 overlay**（不 fork）。不可妥协的**基准是科学
等价**：每一项加速都必须*先*判定它是否扰动数值（checkpoint `max|Δ|`），*再*看速度。一次更快但把训练
权重改动超过 bf16 地板的 run 是一次失败，而非加速。

论文规模：**100 clients，每轮 2 个，70 rounds，每轮 3 epochs，GRPO G=8**（也有 PPO 变体），4-GPU FSDP，
Qwen2.5-1.5B-Instruct，WebShop + ALFWorld，外加异质性 arm —— **176 个配置**。

## 2. 为什么一开始没有加速 —— 冷启动论点

GPU 实测，**wall-clock 中有 ~76%（0.5B、warm cache）到 ~88%（1.5B smoke）是每个 (client×round) 的子进程
冷启动**：Ray init + FSDP shard load + vLLM engine init + kernel 编译，在一次论文级 run 中要付 **~140×**
（70 rounds × 2 clients）。verl 的 async agent-loop rollout *确实*在 client 内部被使用，但**联邦 pipeline**
（clients → rounds → eval）是完全串行的子进程。所以真正的杠杆不是去挤那 ~12% 的计算 ——
而是**不再反复付冷启动**。

**铁律（关键所在）。** 在单一饱和节点上，"同时跑两件事"并不帮忙 —— 它们只是争抢同一组
GPU/VRAM/PCIe。真正的收益要么需要**(i) 消除工作**（不再反复付冷启动），要么需要**(ii) 资源隔离**
（给每个并发作业各自的 GPU/节点）。这套框架驱动了整个设计：#4 消除；#1/#3 隔离。（铁律的一个重要
例外稍后浮现 —— 见 §9。）

## 3. 设计空间 —— 四个杠杆

| 杠杆 | 思路 | 适用情形 | 数值风险 |
|---|---|---|---|
| **#4 persistent trainer** | 一个进程跨 client/round；进程内 reset | 单节点 —— **那个**杠杆 | 唯一带风险的（必须复现 reset 状态）|
| **#1 eval ∥ train** | 在空闲 GPU 上把 eval(model_r) 与 train(round r+1) 重叠 | 需要 ≥1 张空闲 GPU | 无（eval 只读）|
| **#3 client-parallel** | 一轮的 client 并发训练 | 多节点理想；小模型单节点就赢（§9）| 无（FedAvg 与顺序无关）|
| **#2 env prewarm** | 重叠下一轮的 env-service warmup | 次要 / 可选启用 | 无（纯调度）|

按硬件的 ROI：**单 4-GPU 节点** → 做 #2（免费）+ #4（那 88%）+ 小模型用 #3；**≥2 节点** → 再加
#3（巨大、bit-equivalent）+ #1 在备用分配上免费。

---

## 4. 杠杆 #4 —— persistent trainer（那个大杠杆）

**机制。** 不再每个 (client, round) 起一个全新子进程，而是让一个 Ray/FSDP/vLLM 进程保持存活，并在
client 之间**进程内 reset**。这个 reset 把引擎重指到下一个 client 的模型 + 清掉 FedProx anchor，在一次
`engine.initialize()` 中重建 module（新 weights）+ optimizer（fresh Adam）+ LR scheduler。两种作用域：
- **`persistent: true`**（per-round）：每轮一个进程，在该轮的 client 间复用（每轮一次冷启动）。
- **`cross_round: true`**（whole run）：**整个 run 一个进程** —— 冷启动只付**一次**。`<out>/_xround/` 中的
  signal-file 握手让 worker 空转（持有 GPU），此时编排器运行*同一套*外部 FedAvg/merge，然后在 merged
  模型上恢复。聚合器初始化一个独立的每 rank ~1 GB 的 NCCL world，因此它与暂停的 worker 共存。

FedAvg/merge 保持**外部且字节级一致**于子进程路径 —— 这正是维持等价性的所在。

**结果（TinyGuess GRPO，2-round，匹配 seed）：**

| arm | wall | Δ | 最终聚合 `max|Δ|` |
|---|---|---|---|
| subprocess（baseline）| 909s | — | — |
| `persistent`（per-round）| 515s | **−43%** | `1.13e-5` → EQUIVALENT |
| `cross_round`（whole run）| 342s | **−62%** | `1.13e-5` → EQUIVALENT |

等价性**跨轮复合**地穿过 FedAvg 边界依然成立（最差 tensor 为
`layers.15.self_attn.o_proj.weight`，mean 1.7e-7）。**PPO critic reload 已 GPU 验证**：gae 路径每个 client
重建 critic engine 并对 actor **和** critic 做 FedAvg（这需要给 `fedagent_ppo.yaml` 加一个 `critic:` 块 ——
一个同时影响子进程与持久化 PPO *两者*的既有缺口；GRPO 下无效化）。

**Per-client reset = 等价关键面。** 头号风险是 optimizer/LR/FedProx-anchor 残留（必须完全 reset，而非从
上一个 client 继承）以及 vLLM sampler RNG 状态。per-client reset 清单 + 排序后的风险见 `acceleration.md`
§Lever #4。

**Per-client service 路由（真实 WebShop/ALFWorld）。** 一个持久化 worker 无法为每个 client 起一个全新
env，所以路由用一个**文件通道** `FEDAGENT_SERVICE_URL_FILE`：driver 在每次 `fit()` 之前用当前 client 的 URL
（`base_port + c`）改写它；共享的 agent-loop worker 经由 `resolve_service_url`（优先级 file > env-var >
config > default）读取它。已 GPU 验证：一次 2-client smoke 恰好让每个不同的 service 各服务它的 32 个 episode
（seed 11 vs 12），证明路由没有泄漏到默认值。

---

## 5. Eval/train 的 GPU 共享 —— 四种 `eval_mode`

Eval 是**只读**的：它加载 merged `model_r`、给它打分、不向训练写回任何东西（无 RNG/data/weights）。所以
`eval(model_r)` 不管 inline / async / 在热引擎上跑，都给出**逐 bit 相同的训练轨迹** —— `eval_mode` 只改变
eval *在哪/何时*跑。唯一约束是 GPU 显存：vLLM 预留 `gpu_memory_utilization × VRAM`，与模型大小无关，所以
同一张 GPU 上两个引擎冲突（0.6+0.6>1.0）。故四种 mode：

| mode | 机制 | 何时用 |
|---|---|---|
| **inline**（默认）| merge 后阻塞式 eval，用所有节点 GPU | 训练占满节点 |
| **parallel**（= #1）| eval 放在**不相交的** GPU 子集，与下一轮训练并发；async 启动 + 延迟收集 | 训练用 < 整节点的 GPU |
| **shared** | 第二个 eval vLLM 在缩小的 `eval_gpu_mem_util=0.3` 上、在 worker 的 GPU 上 | 单节点、无空闲 GPU |
| **worker** | cross-round worker 用它**自己的热 vLLM**（verl `_validate()`）做 eval —— 无第二引擎 | 单节点、饱和（论文场景）|

**0.5B，2-client × 2-round WebShop，每 round eval**（val 卡在 −0.6 地板 → 跨模式逐 bit 相同，确认只读）：

| eval_mode | base | wall | val r0/r1/r2 |
|---|---|---|---|
| inline | per-round persistent | 1018s | −0.6 / −0.6 / −0.6 |
| **parallel** | cross-round | **690s** | identical |
| shared | cross-round | 874s | identical |
| **worker** | cross-round | **703s** | identical |

**1.5B，PAPER 设置（G=8，webshop_15 15-turn，response 512，n=500 val），4-card，2 rounds** —— 全部 `rc=0`，**无 OOM**：

| eval_mode | GPU 布局 | wall |
|---|---|---|
| **parallel** | 2 train + 2 eval | **2493s** |
| **worker** | 4 train（热 eval）| 2637s |
| inline | 4 train（阻塞）| 3090s |
| shared | 4 train + 第二引擎 @0.3 | **3316s** |

**0.5B 地板掩盖的发现：在大 val 集上 `shared` 翻转成*最慢*。** 在 0.5B/n=8 时 shared（874s）快过
inline（1018s）；在 1.5B/**n=500** 时 shared（3316s）最慢，因为它缩小 KV（0.3-util）的 eval 引擎压低了 batch
并发 —— n=500 的 eval 被卡住，这个惩罚*随 val 集大小放大*。所以 val 集大时 `shared` 是错误选择；
`parallel` 取胜（满 util 的 eval 重叠在关键路径外），`worker` 紧随其后。

**`worker` 需要四个 verl 生命周期修复**（它在 `fit()` 的生命周期*之外*驱动 `_validate()`）：
1. **`global_steps`** —— `_validate()` 读它；只有 `fit()` 设它 → 缺失时 seed `=0`。
2. **FSDP→vLLM 权重同步 = CUDA 崩溃的真正根因** —— verl 用 **dummy** 权重初始化 rollout vLLM，在
   `init_workers` 后睡着；真权重由 `checkpoint_manager.update_weights` 每次 rollout 同步。一次在 `fit()`
   *之前*的 worker-eval 会拿到 dummy 权重 → CUDA illegal-memory-access / EngineDeadError。修复：
   `_validate` 前 `update_weights`（同步+唤醒）、之后 `sleep_replicas`（镜像 `fit()`）。（`enforce_eager`
   只是*移动*了症状 —— 不是修复。）
3. **dump executor** —— `fit()` 在结尾关闭 verl 的 dump ThreadPoolExecutor；下一轮的 worker-eval 会向
   死掉的 executor 提交 → 若已关闭则重新 init。
4. **`val_batch_size`** —— 遵从 `config.data.val_batch_size`，而非 `len(val)`，免得把整个 WebShop/ALFWorld
   val 集一个 batch 打出去（env-service storm）。

---

## 6. Eval cadence 语义（一处值得明说的修正）

论文那条 "server-aggregated" 红线是**对该 round 聚合模型评一次，每 round 一次** —— *不*由 `test_freq` 门控。
`test_freq` 是 verl 的 **job 内部** step cadence（每 round `epochs_per_round` 步，只触发 `is_last_step`）。一次
对该 round 聚合模型的 shared eval 等于论文 per-client `val_before_train`(step-0) **average 的期望**（该 round
所有 client 都从*同一个*聚合模型出发）→ 同一条曲线，只花一小部分 rollout 成本。代码：全局 eval 门控是
`if do_eval:`（每 round），而非 `r % test_freq`；`do_eval = bool(val_env_spec)`（所以 `val_env_spec: ""` 把 eval
完全关掉 —— 在 §9 用来隔离训练时间）。

---

## 7. Client-end eval —— per-client "圆点"标记

红线评的是该 round 的*聚合*模型；论文还把**每个 client 训练后的模型**画成一个圆点（每 round 每 client
一个）。`client_end_eval: true`（默认关）在**无扰动 val 集**上每 round 多加 `clients_per_round` 次 eval，并在
`val_curve` 旁边写出一条 `client_curve`。

**within-job 路由问题（为什么这不平凡）。** 在一个 client 的训练 job 期间，env-service URL 被路由到*那个
client 的*（被扰动的）service，而 agent-loop worker 分不清 train rollout 和 val rollout，无法在 job 中途换
URL。所以一个 client 的*自己*的 job 没法在干净集上自评。两条路径在 per-client 路由**之外**解决它：
- **orchestrator**（inline/parallel/shared）：`eval_client` 把 client 训练后的 actor 合并到
  `round_<r>/client_<c>/hf`，再走正常的 `eval_global` 路径打到**无扰动 val service**（必须在
  `cleanup_round_checkpoints` *之前*跑 —— 它读 client shard；合并出的 `hf` 会留下）。
- **worker**：每个 `fit()` 之后，`_worker_validate(r, client_id=c)` 在**热**引擎上给刚训完的模型打分。

两者都已 GPU 验证：`client_curve` = 4 个圆点（r1c0/r1c1/r2c0/r2c1），与 3 点红线一致。

---

## 8. 等价性验证 —— 方法论、结果，以及 eval-noise 的微妙之处

**方法论。** 匹配 arm 的 A/B，只在加速机制上不同（eval 关、cleanup 关以保住 shard），逐 tensor 对比
（`tools/verl08_migration/compare_fsdp_checkpoints.py`，atol 1e-4）。

| A/B | actor `max|Δ|` | 裁定 | 备注 |
|---|---|---|---|
| WebShop **GRPO**（subprocess vs cross-round）| **9.8e-6** | EQUIVALENT | |
| **PPO**（subprocess vs cross-round）| **1.16e-5** | EQUIVALENT | backbone ~1e-4；critic **value-head** `score.weight` 5.92e-2 |
| 1.5B cross-**mode**（worker vs inline 聚合）| 3.8e-6 / 7.6e-6 | EQUIVALENT | |

**PPO value-head 的 5.92e-2** *不是*发散：critic 的 value head（shape (1,896)）是一个随机初始化，不会跨 arm
复现；backbone 匹配到 ~1e-4，而 **actor** 匹配到 1.16e-5，因为 advantage normalization 把 value-head 的偏移从
policy gradient 里洗掉了。对训练好的策略无害。

**eval-noise 的微妙之处（1.5B）。** 在 0.5B 时 val_curve 跨模式*逐 bit 相同* —— 但那是**失败地板**（模型
总失败 → 不管怎么采样都是 −0.6）。在 1.5B 地板消失，val 用 `temperature=0.4` 采样，所以 val *数值*跨模式
略有差异（例如 base eval 在 8 个 episode 上 1.21 vs 1.24；或 500 里 3 vs 17 个 success）—— **eval 采样噪声，
不是训练发散。** 证据：**权重**等价（worker vs inline 1.5B 聚合 `3.8e-6 / 7.6e-6`），即便 val 数值不同。所以
等价性主张落在 **checkpoint 层面**；eval 数值是只读的，带采样噪声。

---

## 9. #3 client-parallel 调查（侦探故事）

**测试。** 在一个 4-GPU 节点上**并发**跑两个 client，各占不相交的一对（A=0,1 / B=2,3），1.5B，论文设置，
eval 关 —— 然后问：(a) 两个 verl/Ray/vLLM job 能共存吗？(b) 2×2-parallel 比 2×4-sequential 快吗？

**(a) 共存：能。** 两个引擎加载在不相交的卡对上（4 张卡各 6519 MiB），无 Ray-port / GPU /
`/dev/shm` 冲突。隔离只需 per-job 的 `CUDA_VISIBLE_DEVICES` + `RAY_TMPDIR`（独立的 temp 目录 → 每个 Ray
自己挑空闲端口）。这推翻了"两个 verl job 不能共享一个节点"的此前担忧。

**(b) 崩溃与取证。** 一个 run（A）失败 `rc=1`；表面症状是
`DataLoader worker killed by signal: Killed`。默认嫌疑是 OOM —— 但**三个内存来源全都空闲**：节点 RAM
966 G / 1 TB，`/dev/shm` 504 G（48 M 已用），cgroup `memory.max` = unlimited。**dmesg 没有 OOM-killer。** 所以
这个 SIGKILL 不是内核。完整日志讲出了真相：
- A 的训练*成功了*：`step 1/2/3`、`Training Progress 100%`、`[Rank 0/1] Saved model`、`[fed] client 0
  round 1 OK` 带 reward。`DataLoader killed` 是 `Exception ignored in: ...__del__` —— **GC 期间良性的 teardown
  噪声**，A 和 B *两者*都打。
- A 在**下一**步失败：`FedAvg actor round 1 FAILED`。FedAvg 是 `torchrun --nproc_per_node=ws
  aggregate_fedavg_fsdp.py`，它用 torchrun 的**默认 c10d rendezvous `localhost:29500`**。A 和 B 在数秒内
  训练完成，并发启动它们的 FedAvg `torchrun` → **两者都抢 29500 → 冲突 → 一个死掉。**
  （`CUDA_VISIBLE_DEVICES`/`RAY_TMPDIR` 隔离不了 TCP 端口。）

所以 `DataLoader SIGKILL` 是**红鲱鱼**；真正的 bug 是 FedAvg rendezvous 端口冲突 —— 它也使*任意两个*
`run_fed` 在一个节点上聚合都不安全。

**修复**（`run_fed.py fedavg()`）：`torchrun --standalone`（自动选空闲端口）+ 清掉聚合进程继承的
`MASTER_*`/`RANK`/`WORLD_SIZE`。只动**聚合器的通信端口** —— FedAvg 数学、rollout、eval 不变；PPO-critic
FedAvg 走同一条路径。**已验证**：之前会失败的那个并发 A+B 现在两个都 `rc=0`，无 `EADDRINUSE`。

**速度裁定（以及第二处修正）。**

| arm | wall |
|---|---|
| `t1` —— 1 client, 4 GPU | 558s |
| `t1` —— 1 client, 2 GPU | 725s |
| **#3 —— 2 client × 2 GPU, concurrent** | **727s** |
| sequential —— 2 client × 4 GPU | 2×558 = **1116s** |

**#3 快约 35% —— 不是我此前预测的"打平"。** 我先前的推理（"计算守恒 → 拆分把每个 client 减半 →
同样 wall"）假设了 4-GPU *线性*扩展。实测 4 GPU 对 1.5B 只比 2 GPU 快 `725/558 = 1.30×` —— **sub-linear**：
FSDP all-gather/reduce-scatter 开销 + env-latency-bound 的 WebShop rollout（15 turns，固定的 per-turn service
延迟）+ 固定冷启动，在单卡计算量小的时候占比都更大。所以把 4→2+2 拆开、两个 client 并发，反而打赢
sequential-at-full-4-GPU。这是**铁律的例外**：对小模型，单节点 #3 *是*真收益。**注意：** 对于 4-GPU 扩展
≈2× 的大模型，单节点 #3 打平/更慢 → 那才是真正需要 **≥2 节点（一 client 一节点）** 的场景；编排器的
外部 FedAvg 已经支持它，它需要一个并行的多节点 launcher（尚未构建）。

---

## 10. 论文配置验证（接线 + 可行性）

**真实**的 `uniform/Qwen2.5-1.5B/main/grpo/webshop` 配置（100 clients，2/round，G=8，webshop_15 15-turn，
response 512 / prompt 4096，n=500 val，val temp 0.4），在 `worker` mode 下封顶 2 rounds：**rc=0，循环闭合。**
这是在*实际*论文配置（不是 smoke）上的第一次 run，它压测并通过了每一个新面：100-client partition、
per-client 路由、4 GPU 上的 **G=8 内存**、完整的 3-epoch round、n=500 eval。

**计时分解**（经由 artifact mtime）：冷启动 + base eval 707s；一个 G=8 训练 round（2 clients ×
3 epochs）**496s**；一次 n=500 eval **630s**。注意 eval *比训练 round 还贵* —— 所以 eval cadence 是大规模下
主要的时间杠杆。

**70-round 可行性：** ≈ `70×475 + 71×630` ≈ **22h**（每 round eval）或 `70×475 + 15×630` ≈ **12h**
（`test_freq=5`）—— **一个节点放得下**（一个 4×H100 分配 + ~1.5 天 walltime）。即便只跑 2 rounds，val 也朝
**正确**方向移动（success base `0.022 → 0.034`，n=500）—— 对完整 run 是个好兆头。

完整的 70-round 曲线（× 3 seeds × WebShop/ALFWorld × GRPO/PPO × 异质性 arm）是一场**多节点、
多日的 campaign**，尚未运行。

---

## 11. 杠杆 #1 与 #2（简述）

- **#1 eval ∥ train** 正是 `eval_mode=parallel`（§5）—— eval(r) 在不相交的 GPU 上与 train(r+1) 重叠。
  已 GPU 验证；零数值风险（只读）。与 #4 叠加：一个 persistent trainer 把 eval 变成
  `update_weights + val pass`（数秒），这就是 `eval_mode=worker`。
- **#2 env prewarm**（`prewarm_next_round_services`，默认关）：把 service 启动从 health-wait 中拆出，在
  round 顶部采用下一轮的 service。CPU 已验证，但**对同质 WebShop 收益 ≈0**（service 数秒就 warm ——
  没什么可重叠的）。只对昂贵 warmup 的 arm（catalog_split 大 catalog、ALFWorld 游戏集合）才有实质意义，
  即便如此，每子进程的冷启动（#4）也远盖过它。仍是次要/可选启用。

---

## 12. 如何运行 —— 配置参考

| 标志 | 效果 |
|---|---|
| `persistent: true` | #4 per-round persistent base（每轮一次冷启动）|
| `cross_round: true` | #4 cross-round（整个 run 一个进程）|
| `eval_mode:` `inline`/`parallel`/`shared`/`worker` | eval/train 的 GPU 共享（§5）|
| `eval_gpus: N` | `parallel`：给 eval 的 GPU 数（train 拿 `n_gpus_per_node`；和 ≤ 节点）|
| `eval_gpu_mem_util: 0.3` | `shared`：第二个 eval 引擎的 KV 池 |
| `client_end_eval: true` | per-client 圆点 → `client_curve`（§7）|
| `val_env_spec: ""` | eval 关（隔离训练；§6）|
| `test_freq: N` | verl 的 job 内部 step cadence（**不是**全局红线门控）|

**配方。**
- **单 4-GPU 节点（论文默认）：** `cross_round: true` + `eval_mode: worker` —— 饱和节点的答案（无第二引擎、
  无 OOM、无 eval 冷启动）。
- **有空闲 GPU：** `eval_mode: parallel` —— 最快（eval 重叠在关键路径外）。
- **小模型，每轮 ≥2 client，单节点：** #3 client-parallel（每个 client 起一个 `run_fed` 钉到不相交的
  GPU 子集；`--standalone` 的 FedAvg 修复让并发聚合安全）—— 约 35% 收益。

---

## 13. 待办与路线图

- **完整 70-round 复现** —— 接线已验证；完整曲线未跑（≈12–22h/config；3-seed × env × algo ×
  异质性矩阵是一场多节点、多日的 campaign）。
- **多节点 #3** —— 未实现。单节点 2×2 已验证（对*小*模型取胜）；大模型需要一个一 client 一节点的
  并行 launcher（外部 FedAvg 已经支持它）。
- **#2 env prewarm** —— 已实现 + CPU 验证，但其非零收益（昂贵 warmup 的 arm）尚未 GPU 演示。
- **长时程下的 vLLM sampler RNG** —— 在论文的 70 rounds 上观察 reset-等价性 + `/dev/shm` teardown
  （smoke 在 2 rounds 通过）。

---

## 14. 附录 —— 完整数字 + file/symbol 映射

**所有 GPU 验证的数字，集中一处：**

| 测量项 | 值 | 配置 |
|---|---|---|
| 冷启动占 wall 的比例 | 76%（0.5B）→ 88%（1.5B）| GPU 实测 |
| #4 per-round | 515s（−43% vs 909s）| TinyGuess GRPO 2-round |
| #4 cross-round | 342s（−62%）| TinyGuess GRPO 2-round |
| #4 / equiv-A/B GRPO | `max|Δ|` 1.13e-5 / 9.8e-6 | TinyGuess / WebShop |
| equiv-A/B PPO | actor 1.16e-5（value-head 5.92e-2 无害）| WebShop PPO |
| eval modes 0.5B | inline 1018 / parallel 690 / shared 874 / worker 703 | 2c×2r WebShop, eval/round |
| eval modes 1.5B paper | parallel 2493 / worker 2637 / inline 3090 / shared 3316 | 4-card, G=8, n=500, 2r |
| cross-mode 权重等价 1.5B | 3.8e-6 / 7.6e-6 | worker vs inline 聚合 |
| #3 scaling | t1(4)=558, t1(2)=725 (1.30×) | 1 client, eval 关, paper |
| #3 parallel vs sequential | 727s vs 1116s (−35%) | 2 client, 1.5B |
| paper 单位成本 | 475s/train-round, 630s/n=500-eval | 1.5B worker, G=8 |
| paper 70-round 估算 | 12h (test_freq=5) / 22h (every-round) | 一个节点 |

**关键文件**（overlay）：`fed/run_fed.py`（编排器：persistent/cross_round、路由、eval modes、
client-end eval、`fedavg() --standalone` 修复）、`fed/persistent_{patch,task_runner,main}.py`（持久化
worker + `_worker_validate`）、`envs/base.py`（`resolve_service_url`）、`tools/verl08_migration/{aggregate_fedavg_fsdp,
compare_fsdp_checkpoints}.py`。逐 symbol 细节：`acceleration.md` §8。
