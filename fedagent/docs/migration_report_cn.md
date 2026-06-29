# FedAgent verl-0.8 迁移 —— 完整工程报告

> **权威的、端到端的记述**：把 FedAgent 从内置的 **verl-agent 0.3.1 fork** 迁移到
> 作为薄 overlay 的 **stock verl 0.8**：路线决策、环境构建、那些棘手的
> 兼容性问题及各自的解决方式、两种 rollout 模式、保真度记录，以及
> 验证历程。
>
> **迁移文档，按用途划分：**
> - [architecture.md](architecture.md) —— **overlay 是如何搭建的**（扩展点、round 循环、数据流）。
> - [migration.md](migration.md) —— **保真度记录**（改了什么、那些科学关键的对齐、验证状态）。
> - **本文档** —— **完整走查**：策略 + 那些棘手技术问题的*深入剖析* + 历程。
> - [acceleration_report.md](acceleration_report.md) —— 独立的**加速与验证**工作线（建立*在*本次迁移之上）。
>
> **约定。** 分支 `migrate/verl-0.8.0`。基准 = **科学等价**（在 seed 噪声范围内复现论文的
> 结论，*而非*逐 bit 相同的曲线）。GPU = 4×H100，经由 `srun --overlap`。verl 0.8
> 源码（editable）：`others/verl`。overlay 模式的参考：VAGEN-Lite（`others/VAGEN`）。

---

## 1. 使命与基准

FedAgent 是**面向 LLM agent 的联邦 RL**：每一轮，少数几个 client 各自训练一个本地 policy（GRPO/PPO，
针对 WebShop/ALFWorld 的多轮 rollout），然后 server 把它们的权重做 **FedAvg**。原始代码
**fork 了 verl-agent 0.3.1** 并把联邦逻辑织进了 trainer。本次迁移把它重新实现为一个**作用在
stock verl 0.8 上的薄 overlay —— 不 fork** —— 这样框架就能跟随上游，无需维护 fork。

**基准是科学等价**：在 3-seed 噪声范围内复现论文的*结论*（input-dynamics 不对称性、
异质性扫描的排序、baseline 之间的关系）。这允许使用 verl 0.8 的原生 rollout 而不必逐 bit 复现
fork 的 rollout，但它禁止对**科学红线**做任何改动：task-vs-env 异质性可独立扫描；
确定性的 per-client-id 分配
（`RandomState(base_seed+client_id)`，`base_seed=42`）；validation 永远在**未被扰动**的 env 上；uniform
FedAvg；FedProx 锚定到 round 起始权重；预算匹配的 `N=100 / M=2 / T=70 / E=3` 协议。

## 2. 路线决策 —— fork → overlay

fork（0.3.1）patch 了 `verl/trainer/.../ray_trainer_fed.py`，**只有一个**原因：为了向 trainer 注入
`traj_collector.multi_turn_loop`（多轮 rollout）+ 一个 GiGPO estimator。**verl 0.8 有一个原生的
`AgentLoopManager`**（async 多轮 rollout 作为一等的接缝）—— 所以那个原因**没有了**。选定的路线
（"**Route B**"，VAGEN 风格）：一个独立的 `fedagent/` 包，它**把 verl 0.8 当作库 import**
并插入它的公开扩展点，驱动**stock `RayPPOTrainer`** 而不做修改。

| 扩展点 | FedAgent 插入的内容 |
|---|---|
| `data.custom_cls` | `AgenticDataset` —— 每个 env 实例输出一行 env-spec（而非静态文本） |
| agent-loop 注册表（`agent.yaml`） | `GymTextAgentLoop` —— 在 verl 的 async 接缝上做多轮 rollout |
| Hydra `searchpath` | `fedagent_ppo.yaml` 叠加在 verl 的 stock `ppo_trainer` 之上 |
| 解释器启动（`sitecustomize.py`） | FedProx 近端项，由 `FEDPROX_MU` 门控 |
| 进程边界（HTTP） | WebShop / ALFWorld 远程 env 服务 |

有两份关键代码**逐字**保留（与 verl 无关 → 零迁移风险）：异质性构造
（`partition_strategy.py`，约 3.7k LOC —— 科学皇冠上的明珠）以及 WebShop/ALFWorld **引擎**（MDP）。
0.3.1 的 `core/` 控制平面（约 2.8k LOC 的 server + script-builder）**没有**沿用 —— 它假设了一个
`config['verl']` schema + `model_world_size_1` 单 rank checkpoint；精简的 `fed/run_fed.py` 取代了它。

## 3. 环境（`fedagent-verl08`）—— 依赖传奇

一个新的 conda env（**py3.12**）：verl 0.8 + **vllm 0.11.0** + sglang 0.5.2 + flashinfer 0.3.1，仅 FSDP
（`USE_MEGATRON=0`）。这次构建暴露了五个值得记录的坑（全部已修；脚本在 `tools/verl08_migration/`）：

1. **flash-attn 是必需的。** verl 0.8 的 `ray_trainer` 会**无条件地**调用
   `unpad_input → flash_attn.bert_padding` —— `sdpa` 并*不能*绕过它（一个早期的错误信念）。预构建的 wheel 需要
   GLIBC_2.32（节点是 2.28）→ **从源码构建**：FA 2.7.4.post1，`FLASH_ATTN_CUDA_ARCHS=90`，conda gcc-11 +
   cuda-12.1 nvcc，`--no-deps`（`build_fa.sh`）。
2. **永远不要在没有 `--no-deps` 的情况下 `pip install --force-reinstall`** —— 它会级联成裸 `torch`，从 PyPI 拉
   torch 2.12+cu130，在 12.8 驱动上破坏 CUDA。正确的 torch = **2.8.0+cu128**（vllm 0.11 的 pin）。
3. **CUDA-13 时代的 `nvidia-*-cu13` pip 包会覆盖 cu12 的 `.so`**（共享的 `nvidia/<lib>/lib/` 命名空间，
   最后安装的胜出）→ torch 在 12.8 驱动上加载 NCCL 2.29.7 → 在 FSDP
   param-broadcast 时 `ncclUnhandledCudaError`。修复：卸载 cu13 孤儿包 + `--force-reinstall` 那组 torch 三件套（已归档于
   `_scratch/archived_diagnostics/_fix_nvidia_stack.sh`）。
4. **sglang 拉来了 numpy 2.4**（破坏 vllm 的 numba，需要 ≤2.2）→ pin `numpy==2.2.6`。
5. verl 的 `copy_to_local` 拒绝带尾随 `/` 的 model path。

## 4. 架构简述

两个平面（完整细节：[architecture.md](architecture.md)）。**控制平面** —— `fed/run_fed.py`，
联邦 round 循环；它**从不 import verl**（一个 client 只是一个子进程）。**框架内 hook** ——
`envs/`、`agent_loops/`、`data/`、`fedprox.py`，经由扩展点运行*在* verl client 进程*内*。

round 循环（subprocess-per-(client,round) 是*原始*路径；持久化/跨轮加速见
[acceleration_report.md](acceleration_report.md) §4）：

```
ROUND r:  for each selected client c (sequential):
            python -m fedagent.main_ppo_fed   model.path=model_r   FEDAGENT_BASE_SEED=base_seed+r*100+c
            → round_r/client_c/.../actor   (FSDP shards, ws = n_gpus)
          FedAvg:  torchrun aggregate_fedavg_fsdp.py  → round_r/aggregated/.../actor
          merge:   verl.model_merger  → round_r/aggregated/hf   → model_{r+1}
```

`model_1 = base`；`model_r = round_{r-1}/aggregated/hf`。PPO（`gae`）以同样的方式 federate **critic**。

## 5. 已解决的棘手问题

### 5.1 Checkpoint 兼容性 —— FSDP-shard 传奇

verl 0.8 把 FSDP checkpoint 保存为 per-rank 的 `model_world_size_{WS}_rank_{R}.pt` + 一个**新的** `fsdp_config.json`
（记录 FSDP 版本 + world_size）。本次迁移最早最难的问题就是*如何对这些分片做 FedAvg*。

- **一次虚惊，然后真相。** 一个合成 spike 暗示 FSDP1 保存的是 **ShardedTensor**，它**无法**被
  单进程 `torch.load`（"world size at save 2, at load 1"）。在**真实的** verl-0.8 训练 checkpoint 上，
  这些参数是 **DTensor**，单进程加载**完全没问题** —— ShardedTensor 错误是该 spike 自己合成保存路径
  的产物，而非 verl 的。
- **FedAvg = 在匹配的 process group 下回写。** 经验证/安全的方法：把聚合作为
  `torchrun --nproc_per_node=ws aggregate_fedavg_fsdp.py` 来跑 —— 每个 rank 从每个 client 加载**它自己的**
  rank 分片，**原地**平均本地值（`_get_local` 处理 DTensor/ShardedTensor/plain），然后
  `torch.save` 回去 —— 与一次 verl save **在字节结构上一致**，所以下一轮原样加载它。
  *不要*把 verl 分片加载进一个新 wrap 的 model 再重新保存：verl 的 transformer auto-wrap 切分参数的方式
  与整模型 wrap 不同 → 在 SHARDED_STATE_DICT 下类型不匹配。**已验证 FedAvg-exact**
  （`max|resumed − mean| = 0.0`）。
- **重新进入 = Option C（model_merger → HF → `model.path`）。** 第 r+1 轮 == 第 1 轮，只是把 `model.path` 换成
  merged HF 目录（fresh optimizer —— 恰好是原始的 "load-aggregated, fresh-optimizer"）。`model_merger`
  从 `<local_dir>/huggingface` 读取 HF config 并写出一个完整的 HF 目录，无需 patch。**注意：**
  `model_merger` 会 cast 到 **bf16**，所以每个 round 边界都会截断 fp32 聚合后的权重 —— 在
  等价基准之内，但若 Phase-8 出现 drift 则切到 Option B（`resume_from_path`，仅 model 加载）。

### 5.2 async agent-loop 接缝

verl 0.8 的 generation 是**仅 async** 的，经由 `experimental/agent_loop/`，**按行**派发（每个 `AgentLoop`
只看到**一**行 dataset）。fork 的 batched-synchronous `multi_turn_loop` 没有等价物 → env 必须变成
**per-instance async**（`reset/step/system_prompt/close`）。overlay 的 `GymTextAgentLoop`（`@register("gym_text")`，
一个 `AgentLoopBase` 子类）在原生接缝上为每行驱动一个 `BaseTextEnv`
（`reset → generate → parse action → env.step → …`），返回一个 concat 的 `AgentLoopOutput`，其 `response_mask`
在 agent token 上为 1、在 observation token 上为 0（这样 PPO/GRPO 只在 action 上训练）。**Phase 0(b) 证明了
接缝**：一个 custom AgentLoop 在 **stock** `main_ppo` 上跑了一个完整的 GRPO 循环并发出了规范的 0.8 FSDP layout ——
与 FedAvg 步骤（5.1）所消费的内容相互一致。

### 5.3 远程 env 服务（依赖隔离）

WebShop（Java/pyserini/gym 0.24）、ALFWorld（TextWorld/Fast-Downward）和 trainer（verl 0.8）有**相互
冲突的依赖**。所以每个 env 在**它自己的 conda env 里**作为**它自己的 HTTP 服务**运行，**每个 client 一个**；
trainer（`fedagent-verl08`）经由 HTTP 与它们通信。这些服务 `sys.path`-注入**内置引擎** ——
*原始代码所运行的同一份代码* —— 所以 MDP 不变。`run_fed` 只启动该轮被选中 client 的服务
（同时存活的 ≤ `clients_per_round`），等待 `/health`，在聚合前按轮拆除；一个共享的
**未被扰动的 val** 服务在整个 run 期间保持运行。

两项健壮性发现：(i) **`/step` 风暴** —— 负载下的重试乱序重放了 step → 把服务做成了
**per `step_id` 幂等**（per-session 的 `asyncio.Lock` + 单槽重放缓存 + 乱序时返回 409）。(ii)
**ALFWorld 吞吐瓶颈**是一个进程全局的 `_TW_LOCK`：tatsu PDDL parser 是一个可变单例，所以
所有 textworld 操作都串行化（实测 86 ms/step → 每个 client 约 13.7 s 的串行 env-stepping；fork 把它
跨 Ray-actor 进程并行化了，每个 env 一个）。修复（仅工程，仅 ALFWorld）是 N 个 worker 进程，每个一个
textworld env —— 尚未应用。

### 5.4 异质性注入

partition 构造被**逐字**复制（numpy-only 的函数体；文件本身 import matplotlib，所以只把
函数提取出来），由 `RandomState(base_seed + client_id)` 作为 key → 一个 client 的 shard 与 0.3.1 baseline
逐 bit 相同。两个层级：**env 级**（catalog）是**服务侧**的 —— `run_fed` 把
`PARTITION_STRATEGY/CLIENT_ID/ENV_DIV/…` 作为环境变量传入，服务从真实的 shuffle 后的 `server.goals`
构建*那个 client 的* catalog；**task 级**（goal 分布）是 `AgenticDataset._partition_specs` 接缝。**关键
修复：** `WEBSHOP_SERVICE_URL`（env）必须对 spec 的 `service_url` 具有**权威性** —— 否则两个 client 都
打到 `:8080`，异质性就**坍塌**了。

### 5.5 无 fork 的 FedProx

FedProx 是对 `FSDPEngine.optimizer_step` 的一次单方法 patch（第一步快照 `w_t`，此后加 `mu·(w − w_t)`），
经由 repo-root 的 **`sitecustomize.py`** 应用（在每个进程的解释器启动时自动 import
—— client + Ray worker —— 由 `FEDPROX_MU` 门控）。它有意**不**做成 Ray `runtime_env` worker hook：那会
**覆盖 verl 的 per-worker `CUDA_VISIBLE_DEVICES`** 分配 → "Duplicate GPU detected"。`mu=0` → 普通 FedAvg。

## 6. 两种 rollout 模式 —— 忠实-vs-原生 这条轴

verl 0.8 的原生 rollout 是 **concat**（每个 episode 一个 sample，完整逐字 history）；论文训练的是
**windowed**（per-turn samples，`history_length=2` legacy 模板）。两者都保留，由一个 flag
`rollout_mode: windowed (default) | concat` 切换，已在 GPU 上验证。

**为什么 windowed 在 stock verl 上很难。** per-turn 展开使一个 prompt 产出*许多*训练行，但 stock
verl 0.8 **硬性强制每个输入 prompt 1 个训练 sample**（`fit()` 把 gen 输出切到
`num_sampled_prompts`；`_validate()` 把 test batch 1:1 union）。一个会扩行的 manager 在 train 中会被
**静默截断**（损坏：episode-uid ↔ turn-row 错位），在 eval 中会**崩溃**（`AssertionError: 4 vs 76`）。修复（在
`windowed_manager.py` 中，**不 fork verl**）：
- 对 `DataProto.slice` 做作用域受限的 monkeypatch（tag 过的展开 batch → 不截断）+ `DataProto.union`（tag 过的
  长度不同的 `other` → 采纳它，pad 到 mini-batch 除数，丢弃 tag），由 `len != len` 守护，使得
  长度匹配的 union 和**所有** eval 都不受影响；
- **mini-batch 可整除性** —— `make_iterator` 硬断言 `batch % mini_batch == 0`；`use_dynamic_bsz` *并不*
  起作用（它只影响 micro 切分）。`_compute_size_divisor = lcm(ppo_mini·n, world_size, [micro·ws], [critic
  terms if gae])` + 把动态的 per-turn batch pad 到它（镜像 fork 的 `adjust_batch`）；
- **两个级联修复**，因为 `union` 返回 `other`（丢弃了 stock 本会从 `self` merge 的内容）：重新 merge
  `meta_info`（携带 `temperature`，否则 `KeyError`）并强制加上 per-turn 的 `non_tensor`（携带 `uid`，否则
  在 advantage 处 `KeyError`）；eval 把每个 episode **坍塌**为 1 行（最后一个 turn + 广播 return）以保持 1:1。

**忠实性修正**（两个早期信念是错的）：(1) 论文的 GRPO **没有**做 per-trajectory
dedup —— fork 的 `seen_pairs` dedup 由一个默认禁用它的 flag 门控，所以 mean/std 是在**所有**
per-turn samples 上算的 = **stock verl 0.8 grpo**。所以不需要 custom estimator；`grpo_traj` 仅 opt-in。(2)
**invalid-action 惩罚是 per-turn 的**（在自己 action 无效的那些 turn 上 −0.1），叠加在 base
episode return 之上 —— 而不是统一的 per-episode 减法。

**A/B（回答"会有加速吗"）：windowed 约 ~1.47× 更*慢*，不是加速。** 同样的 16 episodes/step：
windowed 58.5 s/step vs concat 39.8 s；差距来自 **vLLM prefix-cache 被打断**（windowing 每个 turn 都移动 window
→ cache miss；gen 43.0 vs 30.1 s）+ per-turn 展开（160 vs 16 个训练行 → `update_actor` 5.8×）。
**选择 windowed 是为了忠实性 + 长 episode 可行性**（ALFWorld 50 turns：concat context 爆炸并
在 response cap 处截断；windowed 保持有界）—— **而非**速度。

## 7. 保真度记录（精简版）

完整记录：[migration.md](migration.md)。迁移期间验证过的科学关键对齐：

- **算法** —— GRPO **G=8**（`rollout.n=8`）；stock verl 在内部把 `ppo_mini_batch_size` 乘以 `rollout.n`，
  所以原始的 "1 update/rollout-batch" 是 `ppo_mini_batch_size=8`（GRPO）/ 64（PPO），**而非** 64×8。
- **Trajectories/step = `train_batch_size × rollout.n`**，对 PPO *和* GRPO 都成立（在 fork 源码中无条件）→
  原始的 GRPO 跑 8×8=64，PPO 跑 64×8=512；**`rollout.n` 对 PPO 必须保持 8**（降到 1 是不忠实的）。
- **稀疏 reward** `{0,10}` + per-turn 的 `0.1×n_invalid` 惩罚（移到了 agent-loop；每 episode 总量相同）。
- **Round-threaded 数据 seed** `base_seed + round*100 + client`（普通取模会坍塌 round 项 → 每个
  client 每轮看到相同的 goal）。
- **完整 E epochs/round**（`total_training_steps: 0` → smoke 的 step-cap 不会泄漏到论文 run）；val 在共享的
  **未被扰动**服务上，temperature 0.4，held-out split。
- **Config-generator 修复**（`gen_paper_configs.py`）：WebShop `search_return_n` 200（env-het）/ **50**（其他地方，
  匹配原始 baseline）；ALFWorld `max_turns=50` + 用于 50-turn transcript 的 16384-token context。

## 8. 验证状态

| 路径 | 状态 |
|---|---|
| TinyGuess（in-process） | ✅ GPU 端到端 |
| **WebShop GRPO federated** | ✅ GPU 完整 2-round 循环（train → FedAvg → merge → round 2 → eval） |
| **WebShop PPO（gae critic federation）** | ✅ GPU（windowed smoke：actor **和** critic 的 FedAvg + merge） |
| **windowed GRPO + PPO** train+eval+loop | ✅ GPU-green（per-turn 行已训练、eval per-episode、无 4-vs-76 崩溃） |
| **concat** rollout | ✅ GPU-green（1 sample/episode；隔离已确认 —— 未加载 windowed monkeypatch） |
| ALFWorld（service + `max_turns=50`） | code-audited；GPU-VERIFY pending（50 turns 处 OOM/截断） |
| 加速 overlay（#4 / eval modes / #3） | ✅ GPU + 等价 —— 见 [acceleration_report.md](acceleration_report.md) |

本次迁移还解锁了**加速与验证**工作线（持久化 trainer、eval 模式、
client-parallel、等价 A/B）—— 建立在这个 overlay 之上，单独记录。

## 9. 配置矩阵

论文配置（`fedagent/config/paper/`）与原始树 1:1 镜像 —— `uniform/<model>/<setting>/<algo>/`、
`env_heterogeneity/`、`task_heterogeneity/`、`decentralized/`，**176 个配置**（模型尺寸
1.5B/3B/7B + Llama-3.2-3B；GRPO + PPO；WebShop + ALFWorld；各异质性分支）。一个有意的偏离：
centralized/local baseline 用 `T=70 × E=3`（= 210 epochs）而非 1 round × 210，因为 runner 从**rounds**
（round-threaded seed）汲取 goal 多样性 —— 总 epoch 相同、覆盖正确。完整矩阵 → run 命令：
[reproducing.md](reproducing.md)。

## 10. 坑与运维笔记

- **Compute-node 的 `/tmp` 从 login node 不可见** —— 对某个 run 的 checkpoint 做带外 `ls`/`cat` 必须
  经由 `srun --overlap --jobid=<JID>`。把脚本放在 GPFS（`_scratch/`）上，而非 login 本地的 scratchpad。
- **不要 `pkill -f <pattern>`**，当 `<pattern>` 出现在 wrapper 自己的命令行里时 —— 它会自匹配并
  杀掉 srun step（瞬间退出，零输出）。`run_fed` 无论如何都会管理服务生命周期。
- 解释器拆解时（atexit，在 `fit()` 已保存之后）那个良性的 `RuntimeError: DataLoader worker ... killed by signal: Killed`
  是**噪声**（exit 0）—— 见 [acceleration_report.md](acceleration_report.md) 的 §9，那里它曾被误以为是
  崩溃原因（其实不是）。
- 0.8 的 config 重命名：`checkpoint.contents` → `checkpoint.save_contents` / `.load_contents`。

## 11. 未决事项

- **ALFWorld 50-turn GPU-verify** —— 确认在 16384-token / 50-turn 预算下无 OOM / prompt 截断。
- **ALFWorld 服务并行** —— 用 N 个 worker 进程（每个一个 textworld env）替换进程全局的 `_TW_LOCK`
  → ALFWorld rollout 约快 ~22%（仅工程，无科学影响）。
- **完整论文复现** —— wiring 已验证（见 acceleration_report §10）；3-seed × model × env × algo ×
  heterogeneity 矩阵是一项多节点、多日的 campaign。
- **byte-exact windowed-obs vs legacy** —— 唯一剩下的忠实性审计。

---

## 附录 —— phase 时间线

| phase | 里程碑 | 状态 |
|---|---|---|
| 0(a) | checkpoint round-trip + FedAvg（matched-PG, write-back） | ✅ exact |
| 0(b) | stock `main_ppo` 上的 custom async AgentLoop（接缝证明） | ✅ |
| 1 | `fedagent/` 包（entry、config、base env、agent-loop、dataset） | ✅ GPU（TinyGuess） |
| 2 | WebShop 作为远程服务 | ✅ GPU smoke |
| 3 | ALFWorld + 两种 rollout 模式（concat / windowed） | ✅ GPU（windowed + concat green） |
| 4 | 异质性（env-level 服务侧 + task-level dataset 接缝） | ✅（env-level GPU；首次联邦科学 run） |
| 5 | FedProx（`sitecustomize`）+ `json_logs/metrics.json` logger | ✅ |
| 6 | FedAvg 聚合核心（`aggregate_fedavg_fsdp.py`） | ✅ 在真实 verl checkpoint 上验证 |
| 6/7 | 联邦循环闭环（`run_fed.py`，model_merger 重新进入） | ✅ GPU |
| 8 | 科学等价验证 | ✅（GRPO/PPO A/B —— acceleration_report §8） |
| (accel) | 持久化 trainer、eval 模式、client-parallel | ✅ —— [acceleration_report.md](acceleration_report.md) |
