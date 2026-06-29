# Architecture

FedAgent 是**面向 LLM agent 的联邦强化学习**。本文档解释 `fedagent/` 这个包如何将其实现为
**原生 verl 0.8 之上的一层 thin overlay** —— 什么东西在哪里运行，以及一个联邦 round 实际上是
怎样执行的。

## Design principle: overlay, not fork

原始的 FedAgent fork 了 verl-agent 0.3.1，并把联邦逻辑*织入*了 trainer。本版本把
**原生 verl 0.8 作为库 import**，并通过 verl 的公开扩展点添加全部功能 —— **不存在被 patch 过的
verl 树**：

| Extension point | What FedAgent plugs in |
|---|---|
| `data.custom_cls` | [`data/agentic_dataset.py`](../data/README.md) —— 发出 env-spec 行，而非静态文本 |
| agent-loop registry (`agent.yaml`) | [`agent_loops/`](../agent_loops/README.md) —— `GymTextAgentLoop`、multi-turn rollout |
| Hydra `searchpath` | [`config/fedagent_ppo.yaml`](../config/README.md) —— 叠加在 verl 原生的 `ppo_trainer` 之上 |
| interpreter startup (`sitecustomize.py`) | FedProx proximal term，由 `FEDPROX_MU` 门控 |
| process boundary (HTTP) | [`envs/webshop/service/`](../envs/webshop/service/README.md)、[`envs/alfworld/service/`](../envs/alfworld/service/README.md) —— 远程 env |

好处：verl 0.8 的 trainer、FSDP engine、async agent-loop rollout 以及 model merger 都被
**原样使用**，因此该框架能跟随上游而无需维护 fork。

## Two planes

**Control plane** —— [`fed/run_fed.py`](../fed/README.md)。联邦 round 循环。它与 verl 无关：
它从不 import verl；一个 client 只是一个子进程
（`python -m fedagent.main_ppo_fed`）。它编排子进程、FedAvg 与 merge。

**In-framework hooks** —— `envs/`、`agent_loops/`、`data/`、`fedprox.py`。这些运行在 verl client
进程*内部*，经由上面的扩展点触达。

## Code map: `fedagent/` file → role

实时 overlay 中的每个 first-party 文件，按 subpackage 分组。每个 subpackage 还有
各自的 README（已链接）含代码级细节；本表是那张一屏索引。这里**没有**遗留的
`core/` / `eval/` / `scripts/` 控制平面 —— 整个联邦循环就是
[`fed/run_fed.py`](../fed/README.md) 加上下面这些进程内 hook。

### `fed/` — control plane ([README](../fed/README.md))

| File | Role |
|---|---|
| `fed/run_fed.py` | 联邦 round 循环。与 verl 无关的 driver：每个 (client, round) 启动一个 client **子进程**，启停 per-client + val env 服务，FedAvg 各 FSDP 分片，merge 成 HF，推进 round，运行 eval。Functions: `run`、`select_clients`、`run_client`、`fedavg`、`merge_to_hf`、`cleanup_round_checkpoints`、`eval_global`、`start_*_services`。 |
| `fed/metrics_logger.py` | 把每个 client 的 verl `training.log` stdout 解析成 FedAgent plot schema（`[{"step", "metrics"}]`）下的 `json_logs/metrics.json`。在不 fork verl 的 `Tracking` 的前提下恢复了可测量性。 |
| `fed/persistent_main.py`、`fed/persistent_patch.py`、`fed/persistent_task_runner.py` | 可选的**持久 trainer** 路径（lever #4，[acceleration.md](./acceleration.md)），在 `persistent=true`/`cross_round=true` 时使用。`persistent_main.py` 是 `main_ppo_fed` 的对应物，驱动一个 `PersistentFedTaskRunner`（`init_workers()` 一次，然后按一个 JSON plan 逐 client 调 `fit()`），以避免 per-client 子进程冷启动；`persistent_task_runner.py` 就是那个 runner；`persistent_patch.py`（经由 `sitecustomize.py`，由 `FEDAGENT_PERSISTENT=1` 门控）装上 `reload_client_model` 这个 worker 方法，把存活中的 FSDP engine 重新指向下一个聚合模型。 |

### `envs/` — env contract + clients ([README](../envs/README.md))

| File | Role |
|---|---|
| `envs/base.py` | `BaseTextEnv` —— per-instance 的 async env 契约（`system_prompt` / `reset` / `step`），每个 dataset 行对应一个 env 对象。与 VAGEN 的 `GymBaseEnv` 对齐。 |
| `envs/registry.py` | `ENV_REGISTRY`，把行的 `env_name` → env class；`make_env(...)`。注册 `TinyGuess`、`WebShop`、`ALFWorld`。 |
| `envs/tiny_guess.py` | `TinyGuessEnv` —— 无依赖的进程内 guess-the-number env。是接线 smoke test，不属于研究套件。 |
| `envs/webshop/webshop_env.py` | `WebShopEnv` —— 到 WebShop 服务的 thin async **HTTP client**。把 action 文本送进去，把 verl-agent `WEBSHOP_TEMPLATE` observation 格式化出来。 |
| `envs/webshop/service/server.py` | WebShop 远程服务（FastAPI）。预热一个 `WebAgentTextEnv` 池；提供 `/create`·`/reset`·`/step`·`/close`；用原始的 `webshop_projection` 在服务端解析 action；从 environment 读取异质性 `env_kwargs`。运行在 `verl-agent-webshop` conda env 中。([README](../envs/webshop/service/README.md)) |
| `envs/webshop/service/run_service.sh` | WebShop 服务的启动脚本（port、conda env、把 vendored `engine/` 放上 path）。 |
| `envs/alfworld/alfworld_env.py` | `AlfworldEnv` —— 到 ALFWorld 服务的 thin async **HTTP client**。镜像 `WebShopEnv`；使用 verl-agent 的 `ALFWORLD_TEMPLATE_NO_HIS`。 |
| `envs/alfworld/service/server.py` | ALFWorld 远程服务（FastAPI）。一次性构建 `AlfredTWEnv`，预热一个 `batch_size=1` 的 textworld env 池；按 seed 经 `env.seed(seed)` 选 game；用 `alfworld_projection` 解析 action。运行在 `verl-agent-alfworld` conda env 中。([README](../envs/alfworld/service/README.md)) |
| `envs/alfworld/service/run_service.sh` | ALFWorld 服务的启动脚本（port、conda env、`$ALFWORLD_DATA` / `$ALF_CONFIG`）。 |

### `agent_loops/` — rollout ([README](../agent_loops/README.md))

| File | Role |
|---|---|
| `agent_loops/gym_text_agent_loop.py` | `GymTextAgentLoop`（`@register("gym_text")`）—— verl `AgentLoopBase` 子类，在 verl 原生的 async seam（`reset → generate → decode → env.step → …`）上为每行驱动一个 `BaseTextEnv`。返回一个 concat 的 `AgentLoopOutput`，其 `response_mask` 在 agent token 上为 1、在 observation token 上为 0，因此 PPO/GRPO 只在 action 上训练。它是 verl-agent 的 `TrajectoryCollector.multi_turn_loop` 的 verl-0.8 替代物。 |

### `data/` — dataset hook ([README](../data/README.md))

| File | Role |
|---|---|
| `data/agentic_dataset.py` | `AgenticDataset` —— verl 的 `data.custom_cls`，从一个 env-spec YAML（`name`/`n_envs`/`max_turns`/`agent_name`/`config`）**每个 env instance** 发出一行，各自带一个不同的 seed。非 tensor 列作为 kwargs 流向 `AgentLoop.run()`。`_partition_specs` 是*预留的* per-client 异质性接缝（设计上应读 `PARTITION_STRATEGY`/`CLIENT_ID`/… → `hetero/`），但目前是一个 identity no-op —— 异质性是在服务端注入的，不在这里。 |

### `hetero/` — heterogeneity constructions ([README](../hetero/README.md))

| File | Role |
|---|---|
| `hetero/webshop_task.py` | **Task-level** Preference (omega)：每个 client 一个 category-skewed（Dirichlet）的 goal 分布，全 catalog。`preference_for_client(...) → goal_idxs`。 |
| `hetero/webshop_coverage.py` | **Task-level** Coverage (xi)：Beta-sized 的 per-client goal 数量，带受控的 cross-client overlap，全 catalog。`coverage_for_client(...)`。 |
| `hetero/webshop_hardness.py` | **Task-level** Hardness (xi')：从一个预先算好的 per-task success 文件得出的 easy-vs-hard skew，全 catalog。`hardness_for_client(...)`（需要一个 `trajectories_file`）。 |
| `hetero/webshop_catalog_split.py` | **Env-level** Variant 1 —— Catalog Split：每个 client 拿到一个不相交的 product catalog + goal 切片（hidden-kernel divergence P_i）。`load_webshop_data`、`catalog_split_for_client(...)`。 |
| `hetero/webshop_env_variants.py` | **Env-level** Variants 2–5：Field-Subset Index、BM25 Reweighting、Lookalike Injection、Rank Wrapper。为每个发出服务端的 `env_kwargs` 覆盖。 |
| `hetero/_beta_sizing.py` | 共享的 Beta-distribution sizing 原语（`default_r`、`generate_client_sizes`、`assign_with_overlap`），供 Coverage/Hardness 使用。 |

> 每种构造把其核心 partition 主体从 verl-agent 的 `partition_strategy.py` **逐字**复制过来
> （`base_seed=42`），使一个 client 的分片与 0.3.1 baseline 逐 bit 一致；只有围绕它的那层 thin 公开
> API 是新的。见 [heterogeneity.md](./heterogeneity.md)。

### `config/` — Hydra configs ([README](../config/README.md))

| Path | Role |
|---|---|
| `config/fedagent_ppo.yaml` | 叠加在 verl **原生** `ppo_trainer` 之上的训练配置（经 `hydra.searchpath` → `$VERL_CFG`）。设定 `adv_estimator`、`data.custom_cls`、batch sizes；机器路径来自 CLI。 |
| `config/agent.yaml` | 供 verl 的 `AgentLoopManager` 消费的 agent-loop registry：把 `agent_name: gym_text` → `GymTextAgentLoop._target_`。 |
| `config/envs/*.yaml` | 由 `AgenticDataset` 读取的 env-spec 文件（`tiny_guess`、`webshop_15`、`webshop_15_ppo`、`webshop_15_val`、`alfworld`、`alfworld_val`、…）—— per-run 的 env 池 + turn 预算。 |
| `config/examples/**` | 供 `run_fed.py` 用的**运行配置**（`examples/tinyguess_2cl_2rd.yaml`、`examples/webshop/`、`examples/alfworld/`）：smoke、scaled WebShop arms、ALFWorld、FedProx。顶层**没有** `fed_*.yaml` —— 只有 `agent.yaml` + `fedagent_ppo.yaml`。 |
| `config/paper/` | 论文矩阵（`fed_*.yaml` 运行配置就在这里）：`uniform/<model>/`、`task_heterogeneity/{grpo,ppo}/`、`env_heterogeneity/<variant>{,_ppo}/`、`decentralized/`。见 [reproducing.md](./reproducing.md)。 |

### Top-level overlay modules

| File | Role |
|---|---|
| `main_ppo_fed.py` | client 入口：`python -m fedagent.main_ppo_fed`。加载 `config/fedagent_ppo.yaml` 并运行 verl 的**原生** `run_ppo`；import agent-loop 模块以触发其 `@register`。它是 verl-agent 那个被 fork 的 `verl/trainer/main_ppo_fed.py` 的 verl-0.8 替代物。 |
| `fedprox.py` | FedProx proximal term，作为对 `FSDPEngine.optimizer_step` 的单方法 monkeypatch（在第一步 snapshot 全局权重 `w_t`，此后加 `mu*(w - w_t)`）。经 `FEDPROX_MU` 启用；不 fork verl。 |
| `EXPERIMENTS.md` | 正在进行的实验日志。 |
| `README.md` | 包概览（[up one level](../README.md)）。 |

### Runtime dependencies (outside `fedagent/`)

| Path | Role |
|---|---|
| `envs/{webshop,alfworld}/engine/` | **vendored 的 WebShop/ALFWorld 引擎**（+ 原始的 `partition_strategy.py`、`*_projection` action parser）。由 env 服务 `sys.path`-注入，使 environment MDP 与原始 FedAgent 用的是*同一份代码* —— 现在**不再带任何 verl-agent 依赖**。trainer 本身是**原生 verl 0.8**。 |
| `sitecustomize.py`（repo root） | 在每个位于 `PYTHONPATH` 的进程（client + Ray workers）解释器启动时由 CPython 自动 import。由 `FEDPROX_MU` 门控，它应用 `fedprox.py` 的 patch —— 刻意**不**做成 Ray `runtime_env` hook（那会破坏 per-worker 的 `CUDA_VISIBLE_DEVICES`）。 |
| `tools/verl08_migration/aggregate_fedavg_fsdp.py` | FedAvg 核心。在 `torchrun --nproc_per_node=world_size` 下运行：每个 rank 跨 client 就地平均自己那份 FSDP 分片并重新保存，在字节结构上与一个 verl checkpoint 一致，从而下一 round 原样加载它。由 `run_fed.py` 的 `fedavg` shell 出去调用。 |

## The federated round loop

`run_fed.py` 运行 `T` 个 round。每个 round 把被选中的 client 作为**独立的子进程**训练，
FedAvg 它们的 FSDP checkpoint，merge 成一个 HuggingFace 模型，下一 round 从那个 merged 模型开始：

```
base model ─┐
            ▼
   ROUND r:                          (select_clients: seeded per round)
   ┌─────────────────────────────────────────────────────────────────┐
   │  for each selected client c (SEQUENTIAL):                         │
   │     python -m fedagent.main_ppo_fed                               │
   │         actor_rollout_ref.model.path = model_r                    │
   │         trainer.default_local_dir   = round_r/client_c/ckpt       │
   │         env FEDAGENT_BASE_SEED = base_seed + r*100 + c            │
   │         env WEBSHOP_SERVICE_URL = client c's service             │
   │     → round_r/client_c/.../actor   (FSDP shards, ws = n_gpus)     │
   └─────────────────────────────────────────────────────────────────┘
            │  client actor dirs
            ▼
   FedAvg:  torchrun --nproc_per_node=ws aggregate_fedavg_fsdp.py
            --client-actor-dirs c0,c1  --output-actor-dir round_r/aggregated/.../actor
            ▼
   merge:   python -m verl.model_merger merge --backend fsdp
            → round_r/aggregated/hf            (complete HF model)
            │
            └──> model_{r+1} = round_r/aggregated/hf   ← the loop closes here
```

`model_1 = base model`；当 `r > 1` 时 `model_r = round_{r-1}/aggregated/hf`。PPO
（`adv_estimator=gae`）以同样的方式联邦 **critic**，与 actor 并行进行。

`run_fed.py` 中相关的 functions：`run`（driver）、`select_clients`、`run_client`、
`fedavg`、`merge_to_hf`、`cleanup_round_checkpoints`、`eval_global`。

## Anatomy of one client subprocess

```
python -m fedagent.main_ppo_fed                       (verl stock run_ppo + FedAgent config)
  └─ verl PPO/GRPO trainer
       ├─ AgenticDataset (data.custom_cls)            → N env-spec rows, seeded by FEDAGENT_BASE_SEED
       ├─ GymTextAgentLoop (agent-loop registry)      → multi-turn rollout per row
       │     reset → generate → parse action → env.step → repeat (until done / max_turns)
       │     └─ BaseTextEnv: WebShopEnv / AlfworldEnv  → HTTP → remote env service
       ├─ advantage (GRPO group of G — base 4, paper arms 8 — or GAE w/ critic)
       └─ actor update → FSDP checkpoint shards
```

env client（`envs/webshop/webshop_env.py`、`envs/alfworld/alfworld_env.py`）是一个**thin
HTTP client**；笨重的 WebShop/ALFWorld 引擎运行在远程服务里。见
[envs/](../envs/README.md)。

## End-to-end data flow

从 driver 一路向下到 weights 的一条 trace，映射到上面那些文件：

```
fed/run_fed.py  (control plane, no verl import)
  │  per round r, per selected client c:
  ▼
python -m fedagent.main_ppo_fed           (subprocess; loads config/fedagent_ppo.yaml)
  │  runs verl STOCK run_ppo
  ▼
data/agentic_dataset.py  AgenticDataset    (data.custom_cls)
  │  env-spec YAML (config/envs/*) + hetero/ slice (PARTITION_STRATEGY, CLIENT_ID, …)
  │  → N rows, one per env instance, each a distinct seed
  ▼
agent_loops/gym_text_agent_loop.py  GymTextAgentLoop   (agent.yaml registry, one per row)
  │  reset → server.generate → decode action → env.step → append obs   (loop ≤ max_turns)
  ▼
envs/registry.py → BaseTextEnv  (WebShopEnv / AlfworldEnv)   ── HTTP ──►  remote service
                                                                          (envs/*/service/server.py
                                                                           in its own conda env;
                                                                           server-side *_projection
                                                                           parse + engine step)
  ◄── obs, reward, done, info{success}  ──────────────────────────────────
  │  concat AgentLoopOutput (response_mask: agent tokens = 1)
  ▼
verl PPO/GRPO  → advantage → actor (and PPO critic) update  → FSDP checkpoint shards
  │  round_r/client_c/.../actor
  ▼  (back in run_fed.py, after all clients in the round)
tools/verl08_migration/aggregate_fedavg_fsdp.py   (torchrun, ws ranks)   FedAvg shards in place
  ▼
verl.model_merger merge --backend fsdp   → round_r/aggregated/hf        (complete HF model)
  │
  └──► model_{r+1} = round_r/aggregated/hf      (next round starts here; eval_global scores it)
```

`metrics_logger.py` 在每个 client 之后运行，发出 `json_logs/metrics.json`；`fedprox.py`
（经由 `sitecustomize.py`，由 `FEDPROX_MU` 门控）在本地更新期间把 actor 锚定到 `model_r`。

## Remote env services (and why)

WebShop、ALFWorld 与 trainer 有**相互冲突的依赖**（WebShop 的
Java/pyserini/gym 0.24；ALFWorld 的 TextWorld/Fast-Downward/torchvision；verl 0.8）。所以每个
environment 作为它**自己的、运行在自己 conda env 里的 HTTP 服务**，每个 client 一个服务：

```
trainer (fedagent-verl08)  ──HTTP──>  client 0 service (verl-agent-webshop, :8080)
                           ──HTTP──>  client 1 service (verl-agent-webshop, :8081)
                                      ...
                           ──HTTP──>  shared unperturbed VAL service (:8090)
```

（一次只有该 round 被选中的 client 在线，所以**最多 `clients_per_round` 个 per-client 服务
同时存活** —— 不是整支舰队。）

`run_fed.py` 每个 round **惰性**启动 per-client 服务（仅该 round 被选中的 client），经由
`start_webshop_services` / `start_alfworld_services`，等待每个 `/health`，并在聚合**之前**
**按 round 拆除它们**；只有那个 shared 未被扰动的 VAL 服务在整个 run 期间保持运行，并在结束时停止。
这些服务从 `fedagent/envs/<name>/engine/` `sys.path`-注入 vendored 的引擎 —— 与**原始 FedAgent 用的
同一份代码**，所以 environment MDP 不变（见 [migration.md](./migration.md)）。这个隔离也正是为何
服务包位于 `fedagent/` 的顶层，而非 `envs/` 之下。

## Heterogeneity injection

`run_fed.py` 把 `partition_strategy` + 它的旋钮作为环境变量传给每个 client 的服务
（`PARTITION_STRATEGY`、`OMEGA`、`SIZE_STD`、`SUCCESS_STD`、`ENV_DIV`、`KEEP_RATIO`、
`VARIANT_N`、`CLIENT_ID`、`CLIENT_NUM`、…）。服务调用 [`hetero/`](../hetero/README.md)
从真实 shuffle 过的 `server.goals` 构建*那个 client 的*数据分片。两个层次：
environment（catalog）与 task（goal 分布）。见 [heterogeneity.md](./heterogeneity.md)。

## FedProx

当 `fedprox_mu > 0` 时，`run_fed.py` 在 client env 里设 `FEDPROX_MU`。repo-root 的
`sitecustomize.py` 在**每个**进程（client + 它的 Ray workers）解释器启动时运行，并由
那个变量门控，patch FSDP optimizer step 以加上 proximal term。
它刻意**不**做成 Ray `runtime_env` hook（那会破坏 verl 的 per-worker
`CUDA_VISIBLE_DEVICES`）。`mu = 0` → 纯 FedAvg。

## Evaluation

一个 single **未被扰动**的 validation 服务（完整 env、held-out val 切片、无异质性）
**每个 round** 都给**聚合后的全局模型**打分（`if do_eval: run_eval(current_model, r)`）—— 外加
round 0 的 base 模型（`val_before_train`）—— 采样温度为 `val_temperature`。（`test_freq` *不是*
这个门：它是 verl 的 within-job step 节奏，用于 per-client 的"circle"标记，而非这条全局红线。）
`eval_global` 运行一次 verl `val_only` pass，并把 round→success/reward 曲线解析进
`federated_summary.json`。一次失败的 eval 绝不会中止 run（它是测量，不是循环）。

## Outputs

每次 run，在 `output_dir/` 之下：`round_*/client_*/training.log` + `json_logs/metrics.json`
（FedAgent plot 格式）、`round_*/aggregated/hf`（该 round 的全局模型）、各 per-service
日志，以及 `federated_summary.json`（round 历史 + 未被扰动的 val 曲线）。被消费掉的
FSDP 分片在每次 merge 后被删除（`cleanup_checkpoints`），以把磁盘占用限制在约一个 round。

## See also

- [running.md](./running.md) —— 如何运行它（modes、GPUs、baselines、FedProx、eval）
- [configuration.md](./configuration.md) —— 每一个 config key
- [reproducing.md](./reproducing.md) —— 论文配置矩阵
- [migration.md](./migration.md) —— 相对 verl-agent 0.3.1 改了什么，以及保真记录
- [migration_report.md](./migration_report.md) —— 完整的迁移工程报告（路线决策、依赖 saga、checkpoint/agent-loop/env-service/windowed 深入剖析）
