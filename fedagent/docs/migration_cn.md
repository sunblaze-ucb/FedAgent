# 迁移与保真记录

本次发布把 FedAgent 以一个薄薄的 overlay（`fedagent/`）重新实现在 **stock verl 0.8** 之上，
取代了原先的 **verl-agent 0.3.1 fork**。衡量标准是**科学等价** ——
在 seed 噪声范围内复现论文的结论，而非逐 bit 一致的输出。本
文档记录了改了什么、保持了什么不变、应用了哪些保真修复，以及
验证状态。运行中的实验日志见 [`../EXPERIMENTS.md`](../EXPERIMENTS.md)。

## 改了什么（以及为什么）

| 方面 | 原版（verl-agent 0.3.1 fork） | 本 overlay（stock verl 0.8） | 为什么 |
|---|---|---|---|
| verl | fork；联邦逻辑织进 trainer | stock，作为库导入；**不 fork**（一处 2 行的例外 —— 见下注） | 跟随 upstream，无需维护 fork |
| 控制平面 | `core/custom_fed_server.py` + 一个被正则改写的 base bash 脚本 | [`fed/run_fed.py`](../fed/README.md) —— 每个 (client,round) 一个子进程 | 干净、与 verl 无关 |
| Env 执行 | 进程内 verl-agent env manager | **远程 HTTP env 服务**，每个 client 一个 | conda 依赖隔离 |
| Hooks | patch 进 vendored tree | verl 扩展点（`custom_cls`、agent-loop registry、Hydra `searchpath`） | stock trainer 不动 |
| Config schema | 嵌套 `verl:/federated:/data_preprocess:` | flat keys → `run_fed.py` | 匹配精简的 overlay |
| Checkpoints | `model_world_size_1` 单 rank | FSDP shards → `aggregate_fedavg_fsdp.py` → `verl.model_merger` | verl 0.8 原生 FSDP |
| FedProx | trainer 内 | `sitecustomize.py`，门控于 `FEDPROX_MU` | 避免覆盖 verl 的 per-worker GPU 分配 |
| 算法 / 异质性 / 协议 | GRPO G=8 / PPO；两级 het；N=100/M=2/E=3/T=70 | **完全相同** | 科学等价 |

> **唯一一处 verl 例外。** "不 fork" 仍是原则，只有一处有意的 2 行 patch（FSDP→vLLM weight-transfer
> socket，捕获在 `tools/verl08_migration/patches/` 下，从而无需 fork 即可复现；见
> [acceleration.md](./acceleration.md) §7.7）。它加固同节点并发的 verl job，仅 client-parallel /
> eval-parallel run 才需要 —— 依然没有需要维护的 fork。

## 环境保真：引擎是复用的，不是重写的

WebShop 与 ALFWorld 远程服务**通过 `sys.path` 注入并 import 原始引擎**，
从 vendored 的 `fedagent/envs/<name>/engine/`（经由 `importlib`）—— 与**原始 FedAgent 跑的是同一份代码**。
因此 MDP 不变：

- **WebShop** —— `WebAgentTextEnv` / `SimServer` / `engine.py` / `goal.py` 以及
  `webshop_projection` action parser 都是逐字加载的。评分奖励 `get_reward`、
  `{0,10}` 稀疏训练奖励（won 当且仅当 `done and score==1.0`）、action validity、
  catalog 文件、**seed-42** goal shuffle，以及 `val=goals[0:500]/train=goals[500:]`
  切分都完全相同。异质性数学（catalog-split、preference、coverage、
  hardness、bm25/lookalike/rank）是 `partition_strategy.py` 的**逐字拷贝**。
- **ALFWorld** —— `AlfredTWEnv` / TextWorld、`alfworld_projection` parser、game
  loader、`10 × won` 奖励、6 种 task type，以及 `uniform/preference/coverage/
  hardness/env_disjoint` partition 集合都原封不动地复用。

不同的是**包装/驱动**（HTTP 服务 + verl 0.8 原生的多轮
agent-loop，取代 fork 的进程内 rollout）—— 对 policy 而言是等价信息，
而非对环境的改动。

## 科学关键对齐

这些在迁移审计中被核验，并在出现偏差时修复（每项记录见
`../EXPERIMENTS.md`；其中的代码 B1–B-G2）：

- **算法** —— GRPO，group size **G = 8**（`adv_estimator=grpo`，
  `actor_rollout_ref.rollout.n=8`）。Stock verl 0.8 在内部把 `ppo_mini_batch_size` 乘以
  `rollout.n`；而 fork 乘的是它自己的 `actor_rollout_ref.rollout.n`，且该值被**断言 `== 1`**
  （其 main_ppo.py:168 —— 分组来自 `env.rollout.n`，从不缩放 minibatch）。因此原版始终以
  **64 行全局 minibatch** 步进：GRPO 对 64 行批做 1 次更新/步，PPO 对 512 行批做 **8 次
  更新/步**。在 verl 0.8 上复现同样的 64 行 minibatch，**两种算法都用**
  `ppo_mini_batch_size=8` prompts —— *不是* PPO 用 64（那会把 512 行批融合成单次 8 倍大的
  更新）。（PPO 配置最初误发为 64；已修正为 8，见 `../EXPERIMENTS.md`。）
- **每步 Trajectories = `train_batch_size × rollout.n`，PPO 与 GRPO 都如此。** 已在
  verl-agent 源码中确认（`agent_system/multi_turn_rollout/rollout_loop.py:448` 目标为
  `train_batch_size * rollout.n`；`:504` 执行 `gen_batch.repeat(rollout.n)`），两者都是
  **无条件的** —— *不*门控于 `adv_estimator`，且 PPO 用的是同一个 `multi_turn_loop`。
  所以原版跑的是 **GRPO 8×8 = 64** 和 **PPO 64×8 = 512** trajectories/step；新
  configs 把两者都精确复现。⚠️ **PPO 的 `rollout.n` 必须保持 8** —— 把它降到 1
  会得到 64/step，*不忠实*于论文。（已复查的虚惊：新 PPO **没有**
  相对 legacy 多做 8× rollout —— legacy 本来就做了 512/step。）
- **稀疏奖励 + 非法动作惩罚** —— `{0,10}` 配一个 `0.1 × n_invalid` 惩罚
  （惩罚从 trainer actor 移到了 agent-loop；每个 episode 的总和相同）。
- **Task-heterogeneity 在运行时对真实 shuffle 过的 `server.goals` 做分区**（不是离线
  重建）—— 所以每个 client 的 shard 与原版一致。
- **轮次穿线的 data seed** —— `FEDAGENT_BASE_SEED = base_seed + round*100 + client`，且
  服务用 `random.Random(seed)` 抽取 goals（一个朴素的取模会让 round 项坍缩，
  使每个 client 每轮都看到相同的 goals）。
- **完整的 E epochs/round** —— `total_training_steps: 0` → `null`（smoke 的 step-cap 绝不能
  泄漏进论文 run）；`save_freq` 保存该轮的最后一步；`resume_mode=disable`（联邦在轮次层面
  拥有“resume”）。
- **验证** —— 一个共享的未扰动 val 服务，`test_freq=5`、`val_before_train`、
  val temperature 0.4，跑在论文的 held-out splits 上。

## 烘焙进 config 生成器的保真修复

`tools/verl08_migration/gen_paper_configs.py`（它产出 176-config 的论文树）
应用了 WebShop/ALFWorld 实现审计浮现出的三个修复：

1. **WebShop `search_return_n`（BM25 top-K）。** 原版只在 env-het arm 上抬高它
   （这些 arm 会扰动 catalog/search，需要 target 可达），在 uniform / task-het / decentralized / baseline
   run 上保留**引擎默认值 50**。迁移版曾在所有地方硬编码 200，这会让
   non-het baseline 变得更容易。现在：env_heterogeneity/ arm 用 **200**，其它地方用 **50** ——
   与原版 baseline 一致。
2. **ALFWorld `max_turns = 50`**（原为 12）。原版跑的是 50-turn episode；更小的 cap
   只会降低 ALFWorld 成功率。设在 `config/envs/alfworld.yaml` + `alfworld_val.yaml`。
3. **ALFWorld context window**，按 **windowed**（per-turn、`history_length=2`）默认 rollout
   来调整 —— 正是它改变了 context 的尺寸。每一 turn 是一个训练样本，其 prompt 是有界的
   windowed 模板（task + 最近 2 个 (obs,action) + 当前 obs），而非不断增长的 transcript，
   所以旧的增长式 transcript 预算（`max_model_len=16384`、`response_length=8192`）已不复存在。
   ALFWorld `client_overrides` 现在用 `rollout.max_model_len=2560`、`response_length=512`
   （prompt `2048`，对应简短的房间文本）；WebShop 用 `rollout.max_model_len=4608`、
   `response_length=512`（prompt `4096`，对应较长的商品页面）。`rollout.n` 保持 G=8。

> 修复 #2/#3 是 **GPU-VERIFY**：确认在目标硬件上 50 turn 不会 OOM / prompt 截断；
> 若 episode 在 `done` 之前就被截断，则进一步抬高 `max_model_len`。

## Config tree

论文 configs（`fedagent/config/paper/`）在结构与命名上 1:1 镜像原始 `config/` 树
—— `uniform/<Model>/<setting>/<algo>/`、`env_heterogeneity/`、
`task_heterogeneity/{grpo,ppo}/{env}/`、`decentralized/` —— 共 176 个 config（见
[reproducing.md](./reproducing.md)）。唯一有意的偏离：**centralized/local
baseline 用 T=70 × E=3（=210 epochs）**，而非原版的 1 round × 210 epochs，
因为 verl-0.8 runner 从 **rounds** 取 goal 多样性（轮次穿线的 seed），所以单
轮会重复相同的 goals。总 epoch 相同；goal 覆盖正确。

## 残留差异

**良性的管道差异（无 MDP 影响）：** 多轮 history 是 verl 0.8 原生的 concat-chat，
而非 fork 重新渲染的模板（等价信息）；非法动作惩罚施加在
agent-loop 里，而非 trainer；goal sampling 用了一个不同（仍可复现）的 RNG，所以
每个 seed 的 trajectory 与 0.3.1 不是逐 bit 一致。

**Baseline 动力学（改名后的 rd-70_ep-3 centralized/local 配置）：** 每一轮都是从 merge 后的
HF 权重启动的全新子进程（`save_contents=[model]`、`resume_mode=disable`），所以 T=70×E=3 的
baseline 继承了联邦 arm 的每轮语义 —— Adam 动量每 3 个 epoch 重新初始化、`use_kl_loss` 的
参考策略重新锚定到每轮的起始模型（原版 1×210 的 baseline 用同一个 optimizer 和固定的
base-model KL 锚），且一轮内的 E 个 epoch 重放该轮的 goal 抽取（原版每个 epoch 重新抽取：
210 次 vs 这里 70 次）。与联邦 arm 在算力与动力学上对齐，但 legacy 论文的 baseline 数字
在本栈上无法逐点重推 —— 需要时请用 paper-reproduce 分支。

**Seed 复现轴：** 3-seed arm 变化的是 `base_seed`（42/21/84 —— client 选择 + 每轮 goal 抽取）；
原版变化的是 `SHUFFLE_SEED`（在固定选择下重洗训练池）。两者都是 val split 固定下合法的
seed 噪声轴，但不是同一个随机源（在 `gen_paper_configs.py` 中有记录）。

**GPU 待验证：** PPO（`gae`）critic 联邦与 decentralized ablation 已
config-parse + 代码审计，但尚未端到端 smoke-run；更大的 backbone（Qwen2.5-3B/7B、
Llama-3.2-3B）与完整 70 轮预算尚未在本栈上运行。GRPO 联邦路径在两个环境上都**已**
以真实论文配置端到端 GPU 验证（WebShop；ALFWorld 2026-07-02/03，含 50-turn 预算 ——
无 OOM/截断，见 [acceleration_final_2026-07-03.md](./acceleration_final_2026-07-03.md)）。

## 验证状态

（截至 2026-07-09；运行记录见 [`../EXPERIMENTS.md`](../EXPERIMENTS.md)）

| 路径 | 状态 |
|---|---|
| TinyGuess（进程内） | 已端到端 GPU 验证 |
| **WebShop GRPO 联邦** | **GPU 验证：完整多轮循环**（train → FedAvg → merge → 下一轮 → eval），含真实论文配置 |
| **ALFWorld GRPO 联邦**（service + max_turns=50） | **GPU 验证：真实论文配置 2 轮运行**（2026-07-02/03）；50-turn 预算无 OOM/截断 |
| WebShop PPO（gae critic 联邦） | config-parse + 代码审计；未做 GPU-smoke-run —— 请用 2026-07-09 修正后的 `ppo_mini_batch_size` 配置 |
| ALFWorld PPO | config-parse + 代码审计；未做 GPU-smoke-run |
| Decentralized ablation | config-parse + 代码审计；未做 GPU-smoke-run |
| 更大 backbone（3B/7B/Llama）/ 70 轮全预算 | 尚未在本栈上运行 |

## 另见

- [migration_report.md](./migration_report.md) —— **完整的迁移走查**：路线决策、
  环境搭建的曲折，以及那些硬骨头（checkpoint/agent-loop/env-service/windowed）*深入*版。
  *本*文档是凝练的保真记录；那一份是完整的工程账。
- [architecture.md](./architecture.md) —— overlay 是怎么搭起来的
- [reproducing.md](./reproducing.md) —— 论文 config 矩阵
- [`../EXPERIMENTS.md`](../EXPERIMENTS.md) —— 运行中的实验日志 + 每项修复的细节
