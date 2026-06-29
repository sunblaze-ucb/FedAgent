# 复现论文

这是 FedAgent **verl-0.8 overlay** 的逐实验复现指南 —— 这层薄薄的封装在
**未修改的 verl 0.8** 上重跑论文的配置矩阵。矩阵的每个 cell 都是
[`../config/paper/`](../config/paper/) 下的一个 YAML，且每个 cell 都用一条命令运行：

```bash
python -m fedagent.fed.run_fed --config fedagent/config/paper/<...>.yaml
```

请与 [`../config/README.md`](../config/README.md)（四种配置类型与 `paper/`
命名约定）、[`../fed/README.md`](../fed/README.md)（runner 内部机制 —— 轮次循环、FedAvg、
baselines、eval）、[`./heterogeneity.md`](./heterogeneity.md)（het arm 所实例化的两级异质性
套件）以及 [`./running.md`](./running.md)（硬件旋钮与 CLI 覆盖）一起阅读。这是
**科学等价**复现，而非逐 bit 一致 —— 见[保真度说明](#scientific-equivalence-not-bit-identical)。

---

## 前置条件

- **Conda 环境 `fedagent-verl08`**（py3.12，stock verl 0.8）。先激活它；
  `run_fed` 会把 `PYTHONPATH` 设到 repo 根，使 `fedagent` 与根目录的
  `sitecustomize.py`（FedProx）在它 spawn 的每个子进程里都可导入。
- **一个 4-GPU 节点。** `paper/` 配置钉死 `n_gpus_per_node: 4`（FSDP world
  size 4）；`--n-gpus` 可覆盖它。
- **env service 环境。** WebShop 与 ALFWorld arm 每个 client 都对接一个远程 HTTP
  服务；`run_fed` **自己启动这些服务**，但它们的 conda 环境 / 数据必须已安装并在 PATH 上。
  `tinyguess` 在进程内运行。
- **模型。** 每个配置把 `model_path` 设为一个 HF id
 （如 `Qwen/Qwen2.5-1.5B-Instruct`），会自动下载。在离线集群上用
  `--model-path <local snapshot>` 指向一个预取的目录。

> ### ℹ️ Hardness arm 使用一个随仓库发布的 reference-labels 文件
>
> **task-heterogeneity Hardness** 配置（`p-hardness_success_std-*`）是**唯一**需要外部输入的
> cell：每个都引用一个 `trajectories_file` —— 来自 reference policy 的
> `task_id`→success-label 映射。**这些随 `data/hardness/` 发布**（原始的**trained-checkpoint**
> 标签 —— Qwen2.5-1.5B 微调，完整训练池：WebShop 6,402 goals / 27.8 % easy，
> ALFWorld 3,553 games / 59.4 % easy），所以 Hardness cell 开箱即用。
>
> 这八个 Hardness 配置恰好引用两个路径 ——
> `data/hardness/qwen2.5-1.5b_webshop_trajectories.json` 与
> `data/hardness/qwen2.5-1.5b_alfworld_trajectories.json`（het backbone 在两个 env 上
> 都是 Qwen2.5-1.5B）。若要用不同 backbone 重新生成，请用一个 **trained** checkpoint 作为
> reference（**不要**用 base instruct 模型 —— zero-shot 在 WebShop 上严格成功率只有约 1.4 %，
> 这会使 easy/hard split 崩塌）：
>
> ```bash
> python -m tools.verl08_migration.gen_hardness_trajectories \
>   --config fedagent/config/examples/webshop/scaled/hardness.yaml \
>   --model  <trained Qwen2.5-1.5B checkpoint> --num-goals 6410 \
>   --output fedagent/data/hardness/qwen2.5-1.5b_webshop_trajectories.json
> ```
>
> （ALFWorld 标签来自原始的 verl-agent 推理 pipeline。）schema 是
> `{"trajectories": [{"task_info": {"task_id": ...}, "traj_info":
> {"success": ...}}, ...]}`。见
> [`../data/hardness/README.md`](../data/hardness/README.md) 与
> [`./heterogeneity.md`](./heterogeneity.md#hardness--beta-skewed-easyhard-mix-over-success-labels)。

- **ALFWorld arm** 以 `max_turns: 50`（原始的 `max_steps=50`）驱动 episode，配一个
  **加宽的 context window**（`rollout.max_model_len=16384`，`response_length=8192`）。这在
  `config/envs/alfworld.yaml` 中标注为 **GPU-VERIFY**：在你的 GPU 上确认 50 turn 时
  无 OOM / prompt 截断，若 verbose 房间在 `done` 前就被截断，则调高 `max_model_len`。

---

## 一条命令的运行范式

下面每条命令都是完整调用；唯一变化的是配置路径。CLI flag（`--rounds --clients
--n-gpus --base-seed --fedprox-mu --local-client-id --model-path`）覆盖 YAML。

```bash
conda activate fedagent-verl08

# Uniform main table, GRPO, WebShop (Qwen2.5-1.5B):
python -m fedagent.fed.run_fed --config \
  fedagent/config/paper/uniform/Qwen2.5-1.5B-Instruct/main/grpo/fed_webshop_grpo_total-100_cl-per-rd-2_rd-70_ep-per-cl-3_min-goals-per-cl-100_p-uniform.yaml

# Environment-level heterogeneity: catalog split (div 0.7):
python -m fedagent.fed.run_fed --config \
  fedagent/config/paper/env_heterogeneity/catalog_split/fed_webshop_grpo_total-100_cl-per-rd-2_rd-70_ep-per-cl-3_min-goals-per-cl-100_p-catalog_split_div-0.7_keep-0.7.yaml

# Task-level heterogeneity: preference skew (omega 0.99):
python -m fedagent.fed.run_fed --config \
  fedagent/config/paper/task_heterogeneity/grpo/webshop/fed_webshop_grpo_total-100_cl-per-rd-2_rd-70_ep-per-cl-3_min-goals-per-cl-100_p-preference_omega-0.99.yaml

# PPO arm (federates the critic too — adv_estimator: gae):
python -m fedagent.fed.run_fed --config \
  fedagent/config/paper/uniform/Qwen2.5-1.5B-Instruct/main/ppo/fed_webshop_ppo_total-100_cl-per-rd-2_rd-70_ep-per-cl-3_min-goals-per-cl-100_p-uniform.yaml

# ALFWorld (uniform main, GRPO):
python -m fedagent.fed.run_fed --config \
  fedagent/config/paper/uniform/Qwen2.5-1.5B-Instruct/main/grpo/fed_alfworld_grpo_total-100_cl-per-rd-2_rd-70_ep-per-cl-3_min-goals-per-cl-100_p-uniform.yaml

# Baselines (same family, different mode):
python -m fedagent.fed.run_fed --config \
  fedagent/config/paper/uniform/Qwen2.5-1.5B-Instruct/centralized/grpo/fed_webshop_grpo_total-1_cl-per-rd-1_rd-70_ep-per-cl-3_min-goals-per-cl-100_p-uniform.yaml
python -m fedagent.fed.run_fed --config \
  fedagent/config/paper/uniform/Qwen2.5-1.5B-Instruct/local_client1/grpo/fed_webshop_grpo_total-100_cl-per-rd-1_rd-70_ep-per-cl-3_min-goals-per-cl-100_p-uniform.yaml
```

联邦协议已烘焙进 `paper/` 配置并与论文一致：
**N = 100** 个 client（`total_clients`）、每轮采样 **M = 2** 个（`clients_per_round`）、
**T = 70** 轮（`total_rounds`）、**E = 3** 个本地 epoch（`epochs_per_round`）。每一轮都从上一轮
合并后的 FedAvg 模型出发训练所选 client，再重新聚合，并（每 `test_freq` 轮）在共享的未扰动
val set 上为全局模型打分。

---

## 实验矩阵

`config/paper/` 下共 176 个配置，镜像原始论文结构。**main table 是跨 WebShop +
ALFWorld 的 4-backbone uniform sweep**；heterogeneity 与 decentralized 家族只在单一 backbone
（Qwen2.5-1.5B-Instruct）上运行。

| 家族 | 配置目录 | Backs（论文产物） | Backbones | 数量 |
|---|---|---|---|---|
| **[Uniform (main)](#1-uniform-the-main-table)** | `uniform/<Model>/{main,main_seed1,main_seed2}/{grpo,ppo}/` | Main table **FedAgent** 行 + 训练动态曲线 | 4 | (in 112) |
| **[Uniform (baselines)](#1-uniform-the-main-table)** | `uniform/<Model>/{centralized,local_client1-3}/{grpo,ppo}/` | Main table **Centralized** + **Local Agent** 行 | 4 | (in 112) |
| **[Env heterogeneity](#2-environment-level-heterogeneity-the-worst-case-non-robust-study)** | `env_heterogeneity/<strategy>[_ppo]/` | env-variant 图（worst-case non-robust） | Qwen2.5-1.5B | 16 |
| **[Task heterogeneity](#3-task-level-heterogeneity-the-robust-study)** | `task_heterogeneity/{grpo,ppo}/{webshop,alfworld}/` | task-het 图，6 个 sub-type × benchmark 面板（robust） | Qwen2.5-1.5B | 24 |
| **[Decentralized](#4-decentralized-protocol-ablations)** | `decentralized/{ep_per_round_change,samples_change,selected_cl_change}/{grpo,ppo}/` | protocol-sensitivity 消融图 | Qwen2.5-1.5B | 24 |

`uniform/` 家族（112）是四个 backbone —— `Qwen2.5-1.5B-Instruct`、
`Qwen2.5-3B-Instruct`、`Qwen2.5-7B-Instruct`、`Llama-3.2-3B-Instruct` —— 各有
7 种 run（`main`、`main_seed1`、`main_seed2`、`centralized`、`local_client1-3`）
× `{grpo, ppo}` × `{webshop, alfworld}`。het / decentralized 家族只有 Qwen2.5-1.5B。
`config/paper/` 镜像原始 tree 的结构与命名；内容是 verl-0.8 的 `run_fed` 配置，可用
`tools/verl08_migration/gen_paper_configs.py` 重新生成（见
[`../config/README.md`](../config/README.md)）。

每个 `paper/` 文件名都是自描述的；唯一在一次 heterogeneity sweep 内变化的是末尾的
`p-*` token：

```
fed_webshop_grpo_total-100_cl-per-rd-2_rd-70_ep-per-cl-3_min-goals-per-cl-100_p-uniform.yaml
        │       │      │           │          │         │              │           └ partition / perturbation strategy
        │       │      │           │          │         │              └ |X_i| (min goals per client)
        │       │      │           │          │         └ E (epochs per client per round)
        │       │      │           │          └ T (communication rounds)
        │       │      │           └ M (clients sampled per round)
        │       │      └ N (total clients)
        │       └ RL algorithm (grpo | ppo)
        └ benchmark (webshop | alfworld)
```

---

## 1. Uniform: the main table

**Backs：** headline table —— **FedAgent**、**Centralized**、**Local Agent** 行，覆盖全部四个
backbone、两个 benchmark、两种算法 —— 外加训练动态的 validation-success 曲线
（Qwen2.5-1.5B 上 FedAgent vs Centralized）。

### 布局

```
config/paper/uniform/
  <Model>/                         # Qwen2.5-1.5B / 3B / 7B-Instruct, Llama-3.2-3B-Instruct
    main/          {grpo,ppo}/     # FedAgent, seed 42   ─┐
    main_seed1/    {grpo,ppo}/     # FedAgent, seed 21    ├ 3-seed FedAgent rows
    main_seed2/    {grpo,ppo}/     # FedAgent, seed 84   ─┘
    centralized/   {grpo,ppo}/     # Centralized baseline
    local_client1/ {grpo,ppo}/     # Local Agent baseline, client 21
    local_client2/ {grpo,ppo}/     # Local Agent baseline, client 42
    local_client3/ {grpo,ppo}/     # Local Agent baseline, client 84
```

每个叶子恰好持有两个配置，每个 benchmark 一个（`fed_webshop_*.yaml`、
`fed_alfworld_*.yaml`）。

### 行 → 配置映射

| Table 行 | 配置子目录 | 联邦形状（文件名） | 由谁选定模式 |
|---|---|---|---|
| **FedAgent** | `main/`, `main_seed1/`, `main_seed2/` | `total-100_cl-per-rd-2_rd-70_ep-per-cl-3` | `total_clients: 100`（FedAvg） |
| **Centralized** | `centralized/` | `total-1_cl-per-rd-1_rd-70_ep-per-cl-3` | `total_clients: 1` |
| **Local Agent** | `local_client{1,2,3}/` | `total-100_cl-per-rd-1_rd-70_ep-per-cl-3` | `local_client_id ≥ 0` |

三行都把**总优化预算固定在 T·E = 70·3 = 210 个本地 epoch**，使比较是 compute-matched 的：
Centralized 在 pooled 数据上训练（单个 client 的 FedAvg 即恒等），Local Agent 每轮钉住同一个
client 的 shard 且无联邦，FedAgent 把同样的 210 个 epoch 跨 `M = 2` 个 client、在 `T = 70` 轮上
分配，轮间做 FedAvg。三个 Local index **21 / 42 / 84** 是这些配置在确定性 partition 下钉住的
goal shard。

> **为什么 baseline 用 T = 70（而不是 1 轮 × 210 epoch）？** 原始把 baseline 跑成单个 210-epoch
> 轮。在本 overlay 中 goal variety 是**按轮**抽取的（round-threaded 数据 seed
> `FEDAGENT_BASE_SEED = base_seed + round*100 + client_id` 每轮为每个 client 重新抽取 goal），
> 所以 1 轮 baseline 会重复同一次 goal 抽取。保持 **70 轮**在同样的 210-epoch 预算下复现相同的
> goal 覆盖；单个 client/shard 的每轮 FedAvg 是 no-op。见
> [`../fed/README.md`](../fed/README.md#baseline-modes)。

### 运行

```bash
conda activate fedagent-verl08

# FedAgent (seed 42), WebShop-GRPO, default backbone:
python -m fedagent.fed.run_fed --config \
  fedagent/config/paper/uniform/Qwen2.5-1.5B-Instruct/main/grpo/fed_webshop_grpo_total-100_cl-per-rd-2_rd-70_ep-per-cl-3_min-goals-per-cl-100_p-uniform.yaml

# Centralized baseline, WebShop-GRPO:
python -m fedagent.fed.run_fed --config \
  fedagent/config/paper/uniform/Qwen2.5-1.5B-Instruct/centralized/grpo/fed_webshop_grpo_total-1_cl-per-rd-1_rd-70_ep-per-cl-3_min-goals-per-cl-100_p-uniform.yaml

# Local Agent baseline (client 21), WebShop-GRPO:
python -m fedagent.fed.run_fed --config \
  fedagent/config/paper/uniform/Qwen2.5-1.5B-Instruct/local_client1/grpo/fed_webshop_grpo_total-100_cl-per-rd-1_rd-70_ep-per-cl-3_min-goals-per-cl-100_p-uniform.yaml

# A larger backbone, FedAgent ALFWorld-GRPO with Qwen2.5-7B:
python -m fedagent.fed.run_fed --config \
  fedagent/config/paper/uniform/Qwen2.5-7B-Instruct/main/grpo/fed_alfworld_grpo_total-100_cl-per-rd-2_rd-70_ep-per-cl-3_min-goals-per-cl-100_p-uniform.yaml

# The PPO appendix counterpart (federates the critic too):
python -m fedagent.fed.run_fed --config \
  fedagent/config/paper/uniform/Qwen2.5-1.5B-Instruct/main/ppo/fed_webshop_ppo_total-100_cl-per-rd-2_rd-70_ep-per-cl-3_min-goals-per-cl-100_p-uniform.yaml
```

### 说明

- **Backbones：** 切换 `<Model>` 目录即可复现 table 的另一块；模型钉在每个配置内的
  `model_path`（离线：`--model-path <snapshot>`）。
- **GRPO vs PPO：** `grpo/` 配置支撑正文（GRPO，group size **G = 8**）；同级的
  `ppo/` 配置支撑 PPO 附录，并设 `adv_estimator: gae`，使 **critic 与 actor 一同被联邦**。
- **训练动态曲线：** 由 Qwen2.5-1.5B 的 `main/grpo` 与 `centralized/grpo` 的 `val_curve`
  构建（见 [Outputs](#outputs)）。

---

## 2. Environment-level heterogeneity: the worst-case-non-robust study

**Backs：** WebShop env-variant 图（GRPO 与 PPO 并列）。环境级异质性通过
**transition kernel / catalog** 进入 —— policy 只能通过 successor state 感知它，*而非*来自
prompt —— 所以联邦目标对它是 **worst-case non-robust**（论文的负面结果）。task partition 在每个
env-level run 中都保持 **uniform**，因此任何发散都仅归因于 transition 扰动。**仅 WebShop**
（ALFWorld 没有 catalog/search 可扰动），仅 Qwen2.5-1.5B。WebShop 的 search/transition pipeline
可拆为四个阶段，五种策略横跨它们进行扰动。

| 策略（dir） | Pipeline 阶段 | 旋钮 | Sweep 点（GRPO） | PPO sibling |
|---|---|---|---|---|
| `catalog_split/` | content | `env_div`, `keep_ratio` | `div ∈ {0.0, 0.3, 0.7, 1.0}`, `keep 0.7` | `div 1.0` only |
| `field_subset_index/` | encoding | `variant_n` | `N ∈ {4, 8}` | `N 4` only |
| `bm25_reweighting/` | matching | `variant_n` | `N ∈ {4, 8}` | `N 4` only |
| `lookalike_injection/` | content + matching | `variant_n` | `N ∈ {2, 4}` | `N 4` only |
| `rank_wrapper/` | rendering | `variant_n` | `N 4` | `N 4` |

也就是 11 个 GRPO + 5 个 PPO = **16** 个配置。注意这种不对称：GRPO 目录 sweep 多个点，
但每个 `*_ppo` 目录只持有用于 GRPO-vs-PPO 对比的**单个最发散点** —— 不要期望完整的 PPO
sweep。目录/文件名 token（如 `bm25_reweighting`、`field_subset_index`）镜像原始论文名；
`run_fed` 实际消费的值是短策略 id（`bm25_reweight`、`bm25_field_subset`、`lookalike`、
`rank_wrapper`、`catalog_split`）。

这些 arm 设 `search_return_n: 200`（论文的 BM25 top-K），因为扰动 catalog/search 否则会把
target 推出可达范围；uniform、task-het、decentralized 与 baseline 的 WebShop run 使用引擎默认
`50`，那才是匹配原始非 het 数字的设置。

### 运行

```bash
conda activate fedagent-verl08

# Catalog Split, full divergence, GRPO:
python -m fedagent.fed.run_fed --config \
  fedagent/config/paper/env_heterogeneity/catalog_split/fed_webshop_grpo_total-100_cl-per-rd-2_rd-70_ep-per-cl-3_min-goals-per-cl-100_p-catalog_split_div-1.0_keep-0.7.yaml

# Lookalike Injection, GRPO vs PPO (the worst-case contrast):
python -m fedagent.fed.run_fed --config \
  fedagent/config/paper/env_heterogeneity/lookalike_injection/fed_webshop_grpo_total-100_cl-per-rd-2_rd-70_ep-per-cl-3_min-goals-per-cl-100_p-lookalike_injection_N-4.yaml
python -m fedagent.fed.run_fed --config \
  fedagent/config/paper/env_heterogeneity/lookalike_injection_ppo/fed_webshop_ppo_total-100_cl-per-rd-2_rd-70_ep-per-cl-3_min-goals-per-cl-100_p-lookalike_injection_N-4.yaml
```

### 说明

- **Validation 始终在未扰动的 WebShop 环境上**（`val_env_spec` 强制关闭扰动 kwargs），
  所以该指标隔离的是聚合后的泛化，而非 per-client 过拟合。
- 要复现一个图点，请同时运行 GRPO 配置**和**它的 `*_ppo` sibling，各 3 seed。
- 关于 per-stage 构造与 per-client variant-assignment seeding，见
  [`./heterogeneity.md`](./heterogeneity.md#arm---knob---paper-config-map)。

---

## 3. Task-level heterogeneity: the robust study

**Backs：** task-het 图 —— **6 个面板**，每个对应一个 (sub-type × benchmark)：
**Preference**、**Coverage**、**Hardness**，各在 WebShop 与 ALFWorld 上。任务级异质性
**通过 prompt** 进入 policy（task descriptor 是可观测的），所以联邦目标对它是 **robust**
（论文的正面结果）。基础联邦形状与 uniform main run 完全相同；**只有 partition 策略不同。**
Qwen2.5-1.5B，两个 benchmark，两种算法。

### 布局

```
config/paper/task_heterogeneity/
  grpo/ {webshop,alfworld}/        # 6 configs each
  ppo/  {webshop,alfworld}/        # 6 configs each      = 24
```

每个叶子持有各 sub-type 的两个端点：

| Sub-type | 策略 | 文件名 token | 端点（near-uniform → extreme） |
|---|---|---|---|
| **Preference** | `preference` | `p-preference_omega-*` | `omega 0.01` → `omega 0.99` |
| **Coverage** | `coverage` | `p-coverage_std-*` | `std 256` → `std 1` |
| **Hardness** | `hardness` | `p-hardness_success_std-*` | `success_std 256` → `success_std 1` |

> **命名说明。** Preference 旋钮用 `omega`（环境变量 `OMEGA`）；代码仍接受一个 legacy 别名
> `tau`/`TAU` 表示同一个 Dirichlet spread，它与论文符号 $\tau$（可观测的 task descriptor）
> **无关**。各处一律优先用 `omega`。见 [`./heterogeneity.md`](./heterogeneity.md)。

> **Hardness 需要 labels 文件** —— 先生成
> `data/hardness/qwen2.5-1.5b_<env>_trajectories.json`；见
> [前置条件标注](#ℹ️-hardness-arm-使用一个随仓库发布的-reference-labels-文件)。
> 另外两个 sub-type 无需外部输入。

### 运行

```bash
conda activate fedagent-verl08

# Preference, extreme heterogeneity, WebShop-GRPO:
python -m fedagent.fed.run_fed --config \
  fedagent/config/paper/task_heterogeneity/grpo/webshop/fed_webshop_grpo_total-100_cl-per-rd-2_rd-70_ep-per-cl-3_min-goals-per-cl-100_p-preference_omega-0.99.yaml

# Coverage, near-uniform, ALFWorld-GRPO:
python -m fedagent.fed.run_fed --config \
  fedagent/config/paper/task_heterogeneity/grpo/alfworld/fed_alfworld_grpo_total-100_cl-per-rd-2_rd-70_ep-per-cl-3_min-goals-per-cl-100_p-coverage_std-256.yaml

# Hardness, extreme, WebShop-GRPO (REQUIRES the labels file above):
python -m fedagent.fed.run_fed --config \
  fedagent/config/paper/task_heterogeneity/grpo/webshop/fed_webshop_grpo_total-100_cl-per-rd-2_rd-70_ep-per-cl-3_min-goals-per-cl-100_p-hardness_success_std-1.yaml
```

要复现一个面板，请为相关 benchmark 运行相关 sub-type 的**两个**端点，各 3 seed；PPO 附录变体
使用 `ppo/` sibling。partition 由 `partition_dataset(strategy, ...)` 实现，经配置里的
`partition_strategy` 选定。

---

## 4. Decentralized: protocol ablations

**Backs：** protocol-sensitivity 消融图。这些 sweep 通过在同质（`p-uniform`）baseline 上**一次只
变一个**联邦旋钮、并保持优化预算可比，来证明默认 `(M = 2, E = 3, |X_i| = 100)` 的合理性。
Qwen2.5-1.5B，两个 benchmark，两种算法。

### 布局与各 sweep 所变的内容

```
config/paper/decentralized/
  selected_cl_change/  {grpo,ppo}/   # vary M (clients sampled per round)
  ep_per_round_change/ {grpo,ppo}/   # vary E (local epochs), T scaled to hold ~210 epochs
  samples_change/      {grpo,ppo}/   # vary |X_i| (tasks per client)
```

| Sweep | 旋钮 | 存在的点（baseline 点位于 `uniform/`） |
|---|---|---|
| `selected_cl_change/` | `M`（`cl-per-rd`） | `cl-per-rd-1`, `cl-per-rd-4`（`M=2` 点即 uniform main run） |
| `ep_per_round_change/` | `E × T` | `rd-210_ep-1`, `rd-42_ep-5`（T 与 E 成反比缩放以保持约 210 epoch；`E=3/T=70` 是 baseline） |
| `samples_change/` | `\|X_i\|`（`min-goals`） | `min-goals-500`, `min-goals-1000`（`100` 点即 uniform main run） |

每个叶子持有其各点的 WebShop 与 ALFWorld 变体：3 sweep × 2 点 × 2 env × 2 algo = **24** 个配置。

### 运行

```bash
conda activate fedagent-verl08

# M = 4 clients/round, WebShop-GRPO:
python -m fedagent.fed.run_fed --config \
  fedagent/config/paper/decentralized/selected_cl_change/grpo/fed_webshop_grpo_total-100_cl-per-rd-4_rd-70_ep-per-cl-3_min-goals-per-cl-100_p-uniform.yaml

# E = 5 local epochs (T = 42 rounds), ALFWorld-GRPO:
python -m fedagent.fed.run_fed --config \
  fedagent/config/paper/decentralized/ep_per_round_change/grpo/fed_alfworld_grpo_total-100_cl-per-rd-2_rd-42_ep-per-cl-5_min-goals-per-cl-100_p-uniform.yaml

# |X_i| = 1000 tasks/client, WebShop-GRPO:
python -m fedagent.fed.run_fed --config \
  fedagent/config/paper/decentralized/samples_change/grpo/fed_webshop_grpo_total-100_cl-per-rd-2_rd-70_ep-per-cl-3_min-goals-per-cl-1000_p-uniform.yaml
```

### 说明

- 每个 sweep 的 **baseline 点**（`M=2`、`E=3`、`|X_i|=100`）*不*在此重复 —— 它就是
  [§1](#1-uniform-the-main-table) 里对应的 `uniform/Qwen2.5-1.5B-Instruct/main` run。
- `ep_per_round_change/` 把 `total_rounds` 与 `epochs_per_round` 成反比缩放，使本地-epoch 总预算
  保持在 210 附近，从而隔离 round/epoch 的权衡，而非总算力。
- PPO 对应物位于 `ppo/` sibling。

---

## 三 seed 复现

main table 报告三个 seed。它们已经是**独立的配置** —— `main`、`main_seed1`、`main_seed2` ——
仅在 `base_seed` 上不同：

| Run kind | `base_seed` |
|---|---|
| `main` | 42 |
| `main_seed1` | 21 |
| `main_seed2` | 84 |

`base_seed` 同时驱动 per-round 的 client 选择与 round-threaded 的数据 seed
（`FEDAGENT_BASE_SEED = base_seed + round*100 + client_id`），所以三次 run 探索不同的 client
调度与 goal 抽取。你也可以通过在 CLI 上覆盖任意 base 配置来复现某个 seed：

```bash
python -m fedagent.fed.run_fed --config <main config> --base-seed 21   # == main_seed1
```

het / decentralized 家族只发布单个 seed（`base_seed: 42`）；用 `--base-seed 21` / `--base-seed 84`
重跑即可得到它们的 3-seed error bar。

---

## Baselines（centralized & local）vs federated

runner 从配置推导模式 —— 没有单独的 flag（见
[`../fed/README.md`](../fed/README.md#baseline-modes)）：

| 模式 | 由谁选定 | 行为 |
|---|---|---|
| **federated** | `total_clients: 100`（默认） | 每轮在 2 个采样 client 间做 FedAvg。 |
| **centralized** | `total_clients: 1` | 一个模型在 pooled 数据上；单个 client 的 FedAvg 即恒等，所以该 run 是持续的中心化训练。 |
| **local** | `local_client_id >= 0`（`clients_per_round: 1`） | 论文的 *Local Agent Training*：每轮钉住同一个 client 的数据 shard，无联邦。 |

三个 local 配置钉住 100-way partition 的不同 client：

| 配置目录 | `local_client_id` |
|---|---|
| `local_client1/` | 21 |
| `local_client2/` | 42 |
| `local_client3/` | 84 |

`--local-client-id` 可为任意 base 配置覆盖它。

**Epoch 预算。** 两个 baseline 都运行 **T = 70 × E = 3 = 210 epoch**，匹配联邦 arm 的总量。
原始论文把 baseline 跑成 1 轮 × 210 epoch；在本 overlay 中单个 client/shard 的每轮 FedAvg 是
no-op，但**goal variety 是按轮抽取的**（round-threaded 数据 seed 每轮为每个 client 重新抽取
goal），所以 runner 保持 **70 轮**以复现该 variety —— 同样的总 epoch、同样的 goal 覆盖。

---

## 算力预算

所有报告实验合计约 **1,800 H100 GPU-hours**。Per-config（单 seed）估计，在默认的 4 × H100 节点上：

| Benchmark × algorithm | Wall-clock (4 × H100) | GPU-hours / config |
|---|---|---|
| WebShop GRPO  | ~24 h | ~93 |
| WebShop PPO   | ~29 h | ~117 |
| ALFWorld GRPO | ~29 h | ~117 |
| ALFWorld PPO  | ~35 h | ~140 |

GPU-hours = wall-clock × 4 GPU。每个报告的 mean ± std cell 要乘 **3 seed**，并乘以给定图或
table 块中的 sweep 点 / backbone 数量。要在开发期缩减成本，可降低 group size
（`gen_paper_configs.py --group-size 2` 重新生成一个 cheap-smoke 矩阵），或用 `--n-gpus` 在更少
GPU 上运行；见 [`./running.md`](./running.md)。

---

## Outputs

每次 run 把所有内容写在配置的 `output_dir` 下：

- **`federated_summary.json`** —— per-round provenance（所选 client、每轮起始的模型、聚合后的
  actor + HF 路径、PPO 的 critic chain），外加 `mode`、`partition_strategy`、最终模型，以及
  **`val_curve`**。
- **Per-round 日志** —— `round_<r>/client_<c>/training.log`、
  `round_<r>/aggregated/{aggregate,merge}_*.log`，以及 per-service 日志
  （`webshop_service_client<c>.log` / `alfworld_service_client<c>.log`）。
- **`round_<r>/client_<c>/json_logs/metrics.json`** —— 每个 client 的 `training.log` 被重新解析为
  FedAgent plot schema（`[{"step", "metrics"}, ...]`）。
- **未扰动的 val success 曲线** —— `eval_global` **每轮**在共享的未扰动 val 服务上为聚合后的全局
  模型打分（论文那条 per-round "server-aggregated" 红线，每轮一个点），并由
  `val_before_train: true` 把 base 模型加为 round-0 点、`val_temperature: 0.4`。对该轮聚合模型在
  共享 val set 上评一次即为该轮的 per-round 点：因为该轮每个 client 都从*同一个*聚合模型出发，
  这单次 eval 等于论文 per-client `val_before_train`(step-0) average 的期望 —— 同一条曲线，只花
  一小部分 rollout 成本。`test_freq: 5` **不是**这个全局 eval —— 它是 verl 的 *job 内部* step
  cadence（在 `epochs_per_round` 步/轮时只触发 `is_last_step`，即 per-client 的 "client-end"
  标记）。曲线落在 `federated_summary.json`（`val_curve`），而 round-`r` 的 eval dump 位于
  `round_<r>/eval/`。

`tools/verl08_migration/summarize_fed_run.py` 对一个 run 目录做后处理。

> **磁盘。** 已消费的 FSDP shard 在每次 merge 后删除（`cleanup_checkpoints`，默认开），
> 保留每个 `training.log` 与 merged HF；峰值磁盘大致保持在一轮的量级。

---

## Scientific-equivalence, not bit-identical

本 overlay 复现论文的**科学** —— 相同的联邦协议（N/M/T/E）、相同的算法（GRPO G = 8，
PPO/GAE 配联邦 critic）、相同的异质性构造、以及相同的未扰动-val 测量 —— 在 **stock verl 0.8**
上、无 trainer fork。它**不是**与原始 verl-agent 0.3.1 栈逐 bit 一致（不同的 rollout 引擎、
FSDP checkpoint 布局与 RNG 线程）。完整的保真度记录 —— 什么被保留、什么改变、为何 —— 见
[`./migration.md`](./migration.md)。
