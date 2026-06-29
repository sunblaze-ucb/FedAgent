# FedAgent verl-0.8 —— 加速与验证结果

> **一眼看清"验证了什么 + 数字"。** 两份配套文档：要看**完整、详细的
> 走查**（机制、调查过程、修正）请读 [acceleration_report.md](acceleration_report.md)；
> 要看原始的**分析与方案**（冷启动剖析、铁律、杠杆设计、等价性风险
> 审计）请读 [acceleration.md](acceleration.md)；要看*如何复现论文*，见 [reproducing.md](reproducing.md)。
>
> **约定。** 所有 GPU run 都在 4×H100 节点（qgpu30xx）上。**Bar** = 在 3-seed
> 噪声范围内复现论文。**EQUIVALENT** = FSDP-checkpoint `max|Δ| ≤ 1e-4`（bf16 噪声
> 地板）对比原生 verl 子进程基线。Model = Qwen2.5-1.5B-Instruct（smoke 用 0.5B / TinyGuess）。
> 以薄薄的 `fedagent/` overlay 构建在**原生 verl 0.8 之上 —— 不 fork**。

---

## 0. 裁定

加速 overlay 在 checkpoint 层面与子进程基线**数值等价**，同时把单节点
wall-clock 砍掉 **−43% 到 −62%**。主导成本 —— 每个 (client×round) 的**冷启动**
（Ray + FSDP + vLLM init，占 wall-clock 的 **~76–88%**）—— 现在**每个 run 只付一次**，
而非论文规模下的约 140 次。全部 4 种 eval-mode、per-client 的 "client-end" 圆点、
以及单节点 client-parallel 均已 GPU 验证；**真实论文配置端到端跑通**。

## 1. 总状态表

| 领域 | 内容 | 验证？ | 标志性结果 | 详情 |
|---|---|---|---|---|
| **#4 持久化 trainer** | 一个进程跨越 clients/rounds | ✅ GPU | per-round **−43%**、cross-round **−62%**；`max|Δ|=1.13e-5` | §2 |
| **eval modes** | inline / parallel / shared / worker | ✅ GPU（0.5B + 1.5B）| 全部跑通；`val` 一致（eval 只读）| §3 |
| **#1 eval ∥ train**（= `parallel`）| 在不相交 GPU 上重叠 eval | ✅ GPU | 最快的模式（1.5B **2493s**）| §3 |
| **#3 client-parallel** | 一个节点上 client 并发 | ✅ GPU | 1.5B 2×2 = **−35%**（+ 端口 bug 已修）| §4 |
| **#2 env prewarm** | 重叠 env-service warmup | ⚠️ 仅 CPU | 对同质 WebShop 收益 ≈0 → 次要/可选启用 | §7 |
| **等价性 A/B** | accel 权重 == 子进程 | ✅ GPU | GRPO actor **9.8e-6**、PPO actor **1.16e-5** | §5 |
| **client-end eval（圆点）** | per-client 训练后标记 | ✅ GPU（两条路径）| `client_curve`、4 个圆点 | §6 |
| **论文配置** | 真实 `main/grpo/webshop` 1.5B | ✅ 接线（2-round）| 端到端跑通；完整 70-round ≈ **12–22h** | §6 |

---

## 2. #4 —— 持久化 trainer（大杠杆）

**一个** Ray/FSDP/vLLM 进程横跨 clients（per-round）或**整个 run**（cross-round），其间做
进程内的 per-client reset，而非每次都开一个全新子进程。FedAvg/merge 仍在外部、字节级一致。

| arm（TinyGuess GRPO，2-round，匹配 seed）| wall | Δ vs 子进程 | 最终聚合 `max|Δ|` |
|---|---|---|---|
| 子进程（基线）| 909s | — | — |
| **`persistent: true`**（per-round）| 515s | **−43%** | `1.13e-5` → EQUIVALENT |
| **`cross_round: true`**（整个 run）| 342s | **−62%** | `1.13e-5` → EQUIVALENT |

- **PPO critic reload —— 已 GPU 验证**：`adv_estimator=gae` 的持久化 run 每个 client 重建
  critic engine，并对 actor **和** critic 都做 FedAvg（需要 `fedagent_ppo.yaml` 里有一个 `critic:` 块）。
- 等价性**经由 FedAvg 跨轮复合**后依然成立，而不仅仅是 per-client。
- 机制 + per-client reset 清单 + top-3 等价性风险：`acceleration.md` §Lever #4。

## 3. Eval modes（`eval_mode`：inline / parallel / shared / worker）

Eval 是**只读**的 → 模式改变只影响 eval *在哪/何时*跑，绝不改变训练后的权重。Cadence = 那条
per-round 红线，**每个 round** 都评；`client_end_eval` 额外加 per-client 圆点（§6）。详情：`acceleration.md` §7.4。

**0.5B，2-client × 2-round WebShop，每 round eval**（val 触及 −0.6 地板，故跨模式字节级一致）：

| eval_mode | 进程基线 | wall | val r0/r1/r2 | 需要空闲 GPU？ |
|---|---|---|---|---|
| inline（默认）| per-round 持久化 | 1018s | −0.6 / −0.6 / −0.6 | 否 |
| **parallel**（= #1）| cross-round | **690s** | 相同 | **是（≥2）** |
| shared | cross-round | 874s | 相同 | 否 |
| **worker** | cross-round | **703s** | 相同 | 否（复用热引擎）|

**1.5B，PAPER 设置（G=8，webshop_15 15-turn，response 512，n=500 val），4 卡，2 rounds** —— 全 `rc=0`、**无 OOM**：

| eval_mode | GPU 布局 | wall |
|---|---|---|
| **parallel** | 2 train + 2 eval | **2493s** |
| **worker** | 4 train（热 eval）| 2637s |
| inline | 4 train（阻塞）| 3090s |
| shared | 4 train + 第二引擎 @0.3 | **3316s** |

- **`shared` 在大 val 集上翻转成*最慢***：它缩小 KV（0.3-util）的 eval 引擎把 n=500 eval 卡住 ——
  这个惩罚随 val 集大小放大（在 0.5B/n=8 地板下看不见，那里 shared 还快过 inline）。
- `val` 数值跨模式差异来自 **eval 采样**（temp=0.4，500 里只有 3–25 个 success），**而非**训练：
  跨模式**权重等价性**已直接确认（worker vs inline 1.5B 聚合 `max|Δ| 3.8e-6 / 7.6e-6`）。
- **占满 4-GPU 的论文场景 → `worker`**（无空闲 GPU → `parallel` 作为 4-train 不适用；`worker` 复用
  热 vLLM → 无第二引擎、无 OOM、无冷启动）。

## 4. #3 —— client-parallel（单节点，已 GPU 验证）

两个 client 在不相交的 GPU 对（A=0,1 / B=2,3）上**并发**训练，1.5B，paper 设置，eval 关。

| arm | wall |
|---|---|
| `t1` —— 1 client, 4 GPU | 558s |
| `t1` —— 1 client, 2 GPU | 725s |
| **#3 —— 2 client × 2 GPU，并发** | **727s** |
| 顺序 —— 2 client × 4 GPU | 2×558 = **1116s** |

- **快 ~35%，不是打平。** 对 1.5B 而言 4-GPU FSDP 只比 2-GPU 快 `725/558 = 1.30×`（sub-linear：
  FSDP comm + env-latency-bound 的 rollout + 固定冷启动在小规模下占比更重）→ 把 4→2+2 拆开就赢。
  **注意：** 大模型（4-GPU≈2×）→ 打平 → 那才是真正需要多节点（一 client 一节点）的场景。
- **共存**：两个 verl/Ray/vLLM job 干净共享一个节点（不相交的卡对，6519 MiB ×4）；隔离 =
  每个 job 各自的 `CUDA_VISIBLE_DEVICES` + `RAY_TMPDIR`。
- **发现并修复一个 bug**：FedAvg `torchrun` 用了默认的 c10d rendezvous `localhost:29500` → 并发
  聚合相撞 → 一个死 `rc=1`。修法：`torchrun --standalone` + 清掉 `MASTER_*`/`RANK`/`WORLD_SIZE`
  （`run_fed.py fedavg()`）；只动聚合的通信端口，数学不变。已验证：并发 A+B 两个都 `rc=0`。
  （`DataLoader SIGKILL` 症状是**红鲱鱼** —— 良性的 `__del__` teardown 噪声；无 OOM。）完整
  取证：`acceleration.md` §Lever #3。

## 5. 等价性 A/B（项目基准）

匹配的 arm **仅**在加速机制上不同（eval 关、cleanup 关），逐 tensor 对比
（`tools/verl08_migration/compare_fsdp_checkpoints.py`，atol 1e-4）：

| A/B | actor `max|Δ|` | 裁定 | 备注 |
|---|---|---|---|
| WebShop **GRPO**（子进程 vs cross-round）| **9.8e-6** | EQUIVALENT | |
| **PPO**（子进程 vs cross-round）| **1.16e-5** | EQUIVALENT | backbone ~1e-4；critic **value-head** 5.92e-2 = 未复现的随机初始化，**无害**（被 advantage norm 冲掉）|
| 1.5B 跨**模式**（worker vs inline 聚合）| 3.8e-6 / 7.6e-6 | EQUIVALENT | eval mode 从不改变训练 |

## 6. Client-end eval（圆点）+ 论文配置接线

**Client-end 圆点（`client_end_eval: true`，默认关）** —— 论文的 per-client 训练后标记，
在无扰动 val set 上评，与 `val_curve` 并列发出为 `client_curve`。两条路径都已 GPU 验证（各 4
个圆点）：**orchestrator**（合并 client actor → `client_<c>/hf` → 在 val service 上评，cleanup 之前）
和 **worker**（热引擎 `_worker_validate(client_id)`）。详情：`acceleration.md` §7.4。

**论文配置接线** —— 真实 `uniform/Qwen2.5-1.5B/main/grpo/webshop`（G=8、webshop_15、response 512、
100-client partition 2/round、val temp 0.4），2 rounds 跑 `worker` 模式：**rc=0、loop closed**。已演练且
通过：100-client partition、per-client routing、G=8 显存、完整 3-epoch rounds、n=500 eval。Val 朝
**正确**方向移动（success base `0.022 → 0.034`，n=500）。单位成本 ≈ **475s/training-round、630s/n=500-eval**
→ 一次完整 **70-round** 标志性 run ≈ **12h**（`test_freq=5`）/ **22h**（每 round）—— **单节点可容纳**。

## 7. 如何运行

| flag | 效果 |
|---|---|
| `persistent: true` | #4 per-round 持久化基线 |
| `cross_round: true` | #4 cross-round（整个 run 一个进程）|
| `eval_mode:` `inline`/`parallel`/`shared`/`worker` | eval/train GPU 共享（§3）|
| `eval_gpus: N` / `eval_gpu_mem_util: 0.3` | `parallel` GPU 拆分 / `shared` 第二引擎 KV |
| `client_end_eval: true` | per-client 圆点 → `client_curve`（§6）|
| `val_env_spec: ""` | eval 关（隔离训练）|

- **单 4-GPU 节点：** `cross_round: true` + `eval_mode: worker` 是占满节点时的默认。
- **有空闲 GPU：** `eval_mode: parallel`（最快 —— 把 eval 重叠到关键路径之外）。
- **小模型、≥2 clients/round、单节点：** `#3` client-parallel（2×2）是约 35% 的收益（每个 client
  跑一个 `run_fed`、钉到不相交的 GPU 子集；`--standalone` 的 FedAvg 修复让并发聚合安全）。

## 8. 待办项

- **#2 env prewarm** —— 已实现（`prewarm_next_round_services`，默认关），CPU 已验证，但对同质
  WebShop 收益 ≈0（服务在数秒内 warm）。只对昂贵-warmup 的 arm（catalog_split 大 catalog、
  ALFWorld 游戏集合）才有实质意义。
- **多节点 #3** —— 未实现。单节点 2×2 已验证（对*小*模型是收益）；大模型需要
  一 client 一节点的并行（orchestrator 的外部 FedAvg 支持它；需要一个并行 launcher）。
- **完整 70-round 复现** —— 接线已验证，完整曲线尚未跑（≈12–22h/config；3-seed band +
  ALFWorld + PPO + 异质性 arm = 一场多节点、多日的 campaign）。
