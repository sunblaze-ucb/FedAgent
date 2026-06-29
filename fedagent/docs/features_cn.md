# 关键特性详解

本文把 **FedAgent verl-0.8 overlay** 的每一项核心能力展开为开启它的具体
**config key**，以及在 `fedagent/` 中实现它的**源文件**。

FedAgent 是叠加在**原生 verl 0.8** 之上的一层薄 overlay（从 verl-agent-0.3.1
fork 迁移而来）。每个实验都是一份扁平 YAML，由单一入口驱动：

```bash
python -m fedagent.fed.run_fed --config fedagent/config/<experiment>.yaml
```

这些 config key 是 [`run_fed.py`](../fed/run_fed.py) 中那个扁平的 `DEFAULTS`
dict；少数 CLI flag 会覆盖 YAML
（`--model-path --output-dir --rounds --clients --n-gpus --base-seed --port-base
--fedprox-mu --local-client-id`）。任何 verl 特有的设置都会作为 **Hydra override**
通过 `client_overrides:` 列表传给每个 client（每一项都是一个
`key=value` 字符串，应用于 `python -m fedagent.main_ppo_fed`）。完整的字段
参考见 [configuration.md](./configuration.md)。

## 目录
1. [Algorithms — federated GRPO & PPO](#1-algorithms)
2. [Models — any HuggingFace backbone](#2-models)
3. [Environments — WebShop & ALFWorld](#3-environments)
4. [Two-level heterogeneity](#4-two-level-heterogeneity)
5. [Aggregation — FedAvg / FedProx](#5-aggregation)
6. [Baselines — federated / centralized / local](#6-baselines)
7. [Federation protocol](#7-federation-protocol)
8. [FSDP & scaling](#8-fsdp--scaling)
9. [Evaluation](#9-evaluation)
10. [Logging — W&B-free](#10-logging)
11. [Extensibility](#11-extensibility)

---

## 1. Algorithms

联邦 **GRPO** 与 **PPO**，作为 verl trainer 的联邦对应版本。每个被选中的 client
在自己的子进程里跑一次本地 verl 更新
（`python -m fedagent.main_ppo_fed`）；driver 对得到的 FSDP
checkpoint 做 FedAvg，并从 merge 后的模型进入下一轮。GRPO 用 group
rollout 且没有 critic；PPO 增加一个 value model，与 actor 一起被联邦化。

**Configure**

| 能力 | Key | Where | Source |
|---|---|---|---|
| 算法选择 | `adv_estimator: grpo`（默认）或 `gae` | `run_fed.py` DEFAULTS | [`fed/run_fed.py`](../fed/run_fed.py) |
| GRPO group size **G** | `actor_rollout_ref.rollout.n=8` | `client_overrides`（paper arm = 8；base 默认 4） | [`config/fedagent_ppo.yaml`](../config/fedagent_ppo.yaml) |
| GRPO actor loss | `actor_rollout_ref.actor.use_kl_loss=true`、`kl_loss_coef=0.01`、`kl_loss_type=low_var_kl`、`entropy_coeff=0.001` | base config（每个 arm 都继承） | [`config/fedagent_ppo.yaml`](../config/fedagent_ppo.yaml) |
| PPO — 联邦化 critic | `adv_estimator: gae`（+ `critic.*` overrides） | DEFAULTS + `client_overrides` | [`fed/run_fed.py`](../fed/run_fed.py) |
| Per-client trainer 入口 | `python -m fedagent.main_ppo_fed`（跑 verl 原生的 `run_ppo`） | — | [`main_ppo_fed.py`](../main_ppo_fed.py) |
| Multi-turn rollout | `actor_rollout_ref.rollout.agent.default_agent_loop: gym_text` | base config + [`config/agent.yaml`](../config/agent.yaml) | [`agent_loops/gym_text_agent_loop.py`](../agent_loops/gym_text_agent_loop.py) |

对于 **PPO**（`adv_estimator: gae`），`run_fed.py` 启用 value model，逐轮设置
`critic.model.path`（第 1 轮的 critic = base 模型的 backbone；其后为聚合后的
critic），并且每轮对 actor 和 critic **两者**都做 FedAvg
（`fedavg(..., kind="actor")` 与 `kind="critic"`）。client 必须保存 value model
（在 `client_overrides` 中设 `critic.checkpoint.save_contents=[model]`），它才能被
聚合。轮次机制见 [`fed/README.md`](../fed/README.md)。

新增一个 RL 算法 → [extending.md](./extending.md)。

## 2. Models

任意 **HuggingFace** causal-LM backbone。paper 扫描了 **Qwen2.5-1.5B / 3B /
7B-Instruct** 与 **Llama-3.2-3B-Instruct**。

**Configure**

| 能力 | Key | Where |
|---|---|---|
| Base model（第 1 轮） | `model_path`（`""` → 自动发现本地的 Qwen2.5-0.5B snapshot） | DEFAULTS / `--model-path` |
| Attention 实现 | `+actor_rollout_ref.model.override_config.attn_implementation=sdpa`（由 driver 添加） | [`fed/run_fed.py`](../fed/run_fed.py) |
| PPO value model 初始化 | `critic.model.path`（由 driver 逐轮设置，非固定） | [`fed/run_fed.py`](../fed/run_fed.py) |

每一轮都从**上一轮 merge 后的 HF 模型**
（`round_{r-1}/aggregated/hf`）开始训练，该模型由 `verl.model_merger` 产出。模型获取、
缓存位置以及离线集群 → [installation.md](./installation.md)。

## 3. Environments

真实 agent benchmark **WebShop**（电商搜索-购买）与 **ALFWorld**
（在 TextWorld 上的具身家务任务），外加一个进程内的 **TinyGuess** wiring probe。
WebShop 与 ALFWorld 作为**远程 HTTP 服务**运行（它们的依赖相互冲突，所以
各自住在自己的 conda env 里）；driver 为**每个 client 启动一个服务**，从而每个
client 都拿到自己的环境 / 隐藏的 transition kernel。

**Configure**

| 能力 | Key | Where | Source |
|---|---|---|---|
| 环境选择 | `env_kind: tinyguess \| webshop \| alfworld` | DEFAULTS | [`fed/run_fed.py`](../fed/run_fed.py) |
| Env spec（turns / pool） | `env_spec: config/envs/<name>.yaml` | DEFAULTS | [`config/envs/`](../config/envs) |
| 数据集 adapter（verl `custom_cls`） | `custom_cls_path` → `data.custom_cls.path` | DEFAULTS | [`data/agentic_dataset.py`](../data/agentic_dataset.py) |
| WebShop 服务启动器 | `webshop_run_service`、`webshop_base_port`（client `c` → `+c`）、`webshop_pool_size`、`search_return_n` | DEFAULTS | [`envs/webshop/service/`](../envs/webshop/service/server.py) |
| ALFWorld 服务启动器 | `alfworld_run_service`、`alfworld_base_port`（client `c` → `+c`）、`alfworld_pool_size`、`alfworld_train_eval`、`alfworld_task_types` | DEFAULTS | [`envs/alfworld/service/`](../envs/alfworld/service/server.py) |
| 服务健康等待 | `service_health_timeout`（秒） | DEFAULTS | [`fed/run_fed.py`](../fed/run_fed.py) |

env-spec 文件（如 [`config/envs/webshop_15.yaml`](../config/envs/webshop_15.yaml)、
[`config/envs/alfworld.yaml`](../config/envs/alfworld.yaml)）设置 `n_envs`、`max_turns`
以及 `gym_text` agent 名；per-client 的服务 URL 由 driver 作为
`WEBSHOP_SERVICE_URL` / `ALFWORLD_SERVICE_URL` 注入。新增一个环境 →
[extending.md](./extending.md)。

## 4. Two-level heterogeneity

核心研究特性：沿两条结构上各异的轴线的一套 **client-partition 策略**，
通过 `partition_strategy` 加上各策略自己的旋钮来选择。driver 经由 env var
（`PARTITION_STRATEGY`、`OMEGA`、`SIZE_STD`、…）把它们转发给每个 client 的 env 服务；
服务再分派到 [`fedagent/hetero/`](../hetero/) 下对应的模块。

**Task-level** —— client 之间的*任务分布*不同（通过 prompt 可观测）：

| `partition_strategy` | Knob key(s) | Source |
|---|---|---|
| `preference` | `omega` | [`hetero/webshop_task.py`](../hetero/webshop_task.py) |
| `coverage` | `size_std` | [`hetero/webshop_coverage.py`](../hetero/webshop_coverage.py) |
| `hardness` | `success_std`、`trajectories_file`（必需） | [`hetero/webshop_hardness.py`](../hetero/webshop_hardness.py) |
| `task_disjoint` | （不相交的 goal 切片，完整 catalog） | [`hetero/webshop_catalog_split.py`](../hetero/webshop_catalog_split.py) |

**Environment-level** —— client 之间的 *transition kernel* 不同（对 policy 隐藏）：

| `partition_strategy` | Knob key(s) | Source |
|---|---|---|
| `catalog_split` | `env_div`、`keep_ratio` | [`hetero/webshop_catalog_split.py`](../hetero/webshop_catalog_split.py) |
| `bm25_field_subset` | `variant_n` | [`hetero/webshop_env_variants.py`](../hetero/webshop_env_variants.py) |
| `bm25_reweight` | `variant_n` | [`hetero/webshop_env_variants.py`](../hetero/webshop_env_variants.py) |
| `lookalike` | `variant_n` | [`hetero/webshop_env_variants.py`](../hetero/webshop_env_variants.py) |
| `rank_wrapper` | `variant_n` | [`hetero/webshop_env_variants.py`](../hetero/webshop_env_variants.py) |

`partition_strategy: ""`（ALFWorld 用 `uniform`）是同质 / i.i.d.
baseline。`min_goals_per_client` 设定每个 client 的任务数；`base_seed` 让
client → data 的分配是确定性的。完整构造与 paper 映射见
[heterogeneity.md](./heterogeneity.md)；新增一个 partition →
[extending.md](./extending.md)。

## 5. Aggregation

每轮的服务端合并是 **FedAvg**（对各 client FSDP shard 的加权参数均值）。verl 0.8
保存的是 per-rank 的 `ShardedTensor` shard，无法单进程加载，所以聚合器在一个
匹配 world-size 的 process group 下运行
（`torchrun --nproc_per_node = save-time world_size`）：每个 rank 就地把自己那份 rank
shard 跨 client 平均。**FedProx** 通过在每个 optimizer step 给 actor 梯度加上一个
`μ·(w − w_t)` 项，让每个 client 靠近本轮的全局模型 ——
它改变的是 **client** 更新；服务端仍用 FedAvg 聚合。

**Configure**

| 能力 | Key | Where | Source |
|---|---|---|---|
| FedAvg（默认） | （无 key —— 每轮总是运行） | — | [`tools/verl08_migration/aggregate_fedavg_fsdp.py`](../../tools/verl08_migration/aggregate_fedavg_fsdp.py) |
| FedAvg 权重 | `weights`（`""` → 均匀；否则为逗号分隔、和为 1） | DEFAULTS（仅 YAML —— 无 CLI flag） | [`tools/verl08_migration/aggregate_fedavg_fsdp.py`](../../tools/verl08_migration/aggregate_fedavg_fsdp.py) |
| Merge shard → HF | （自动 —— `verl.model_merger merge --backend fsdp`） | — | [`fed/run_fed.py`](../fed/run_fed.py)（`merge_to_hf`） |
| FedProx | `fedprox_mu`（>0 启用；`0` ≡ FedAvg） | DEFAULTS → `--fedprox-mu` | [`fedagent/fedprox.py`](../fedprox.py) |

FedProx 的注入**不**走 Ray `runtime_env` hook（那会覆盖掉 verl 的
per-worker `CUDA_VISIBLE_DEVICES`）：driver 在每个 client 的环境里设置 `FEDPROX_MU`，
而 repo 根目录的 [`sitecustomize.py`](../../sitecustomize.py) —— 在 `PYTHONPATH` 上的每个进程于解释器启动时
被自动 import —— 调用
`fedagent.fedprox.install_deferred_patch()`（当 `verl` 存在时 fail-closed）。它装好一个
`sys.meta_path` hook，在 verl 首次 import 其 FSDP-engine 模块的那一刻才 monkeypatch
`FSDPEngine.optimizer_step` —— 也就是**在** Ray worker 拿到它的 per-rank
`CUDA_VISIBLE_DEVICES` **之后**。（如果在解释器启动时就 eager 地 import `FSDPEngine`，反而会
在设备分配之前把 torch/verl 拉进来，在多 GPU 下破坏 per-rank GPU 隔离，
"Duplicate GPU detected"。）Eval pass 会清掉 `FEDPROX_MU`，使 proximal 项在 validation
期间绝不触发。新增一条聚合规则 → [extending.md](./extending.md)。

## 6. Baselines

三种 regime 共享同一套 driver 与 config schema；模式由 client key 推断。

| Mode | 如何选择 | 行为 | Source |
|---|---|---|---|
| **Federated** | 默认（`total_clients` > 1，`local_client_id: -1`） | 每轮在 `clients_per_round` 个采样到的 client 上做 FedAvg | [`fed/run_fed.py`](../fed/run_fed.py) |
| **Centralized** | `total_clients: 1`（+ `partition_strategy: ""`） | 在汇总数据上的单一模型；单个 client 的 FedAvg 是恒等，所以循环就是 `total_rounds × epochs_per_round` 的持续训练 | [`fed/run_fed.py`](../fed/run_fed.py) |
| **Local** | `local_client_id: k >= 0`（配合 `total_clients: N`） | paper 的 "Local Agent Training"：每轮固定 N-way partition 中的一个 client，单独训练它，无联邦 | [`fed/run_fed.py`](../fed/run_fed.py) |

`select_clients()` 做确定性的 per-round 采样（seed = `base_seed + round −
1`）；local 模式固定那唯一一个 client 并只启动它的 env 服务
（`participating_client_ids()`）。带例子的讲解 → [running.md](./running.md)。

## 7. Federation protocol

完整协议都可在扁平 config 中配置（所有 key 都在 `run_fed.py`
DEFAULTS 里）：

| Symbol | Key | 含义 |
|---|---|---|
| `N` | `total_clients` | client 池大小 |
| `M` | `clients_per_round` | 每轮采样并训练的 client 数 |
| `T` | `total_rounds` | 联邦轮数 |
| `E` | `epochs_per_round` | 每个被选中 client 的本地 epoch 数（`trainer.total_epochs`） |
| `\|Xᵢ\|` | `min_goals_per_client` | 每个 client 的任务数 |
| seed | `base_seed` | 确定性的 client→data + client-selection seed |
| — | `total_training_steps` | per-client-round 的 step 上限（`> 0` 用于 smoke；`<= 0` → 完整的 `E` 个 epoch） |
| — | `save_freq` | 一个 client round 内的 checkpoint 节奏 |
| — | `wait_between_clients` | 让 Ray/GPU 在 client 之间释放的秒数 |
| — | `cleanup_checkpoints` | 每次 merge 后删除已消费的 FSDP shard（保留 HF + 日志） |

paper 的 config 文件名编码了协议
（如 `…total-100_cl-per-rd-2_rd-70_ep-per-cl-3_min-goals-per-cl-100…`）；解码器
与完整的 176-config 矩阵见 [reproducing.md](./reproducing.md) 和
[configuration.md](./configuration.md)。driver 串入一个 per-(round, client) 的 data
seed（`FEDAGENT_BASE_SEED = base_seed + round·100 + client_id`），使每个 client 每轮
都从它固定的 shard 重新抽取 goal。

## 8. FSDP & scaling

更大的 backbone（3B / 7B）通过 **FSDP** 训练，可选 CPU offload；run 可从
单 GPU 扩展到整节点。

**Configure**

| 能力 | Key | Where |
|---|---|---|
| 每节点 GPU 数 | `n_gpus_per_node`（= FedAvg `nproc_per_node`） | DEFAULTS → `--n-gpus` |
| Actor offload | `actor_rollout_ref.actor.fsdp_config.param_offload` / `.optimizer_offload` | `client_overrides` |
| Ref-policy offload | `actor_rollout_ref.ref.fsdp_config.param_offload` | base config / `client_overrides` |
| Critic offload（PPO） | `critic.fsdp.param_offload` / `.optimizer_offload` | `client_overrides` |
| vLLM tensor-parallel | `actor_rollout_ref.rollout.tensor_model_parallel_size` | base config / `client_overrides` |
| vLLM 显存 | `actor_rollout_ref.rollout.gpu_memory_utilization` | `client_overrides` |

save-time 的 world size（从 `fsdp_config.json` 读取）会被自动检测，并用作
聚合器的 `nproc_per_node`，使 FedAvg 匹配训练时的 shard 布局。硬件 / 扩展
矩阵 → [running.md](./running.md)。

## 9. Evaluation

聚合后的**全局**模型每 `test_freq` 轮在一个共享、**未扰动**的 validation 服务
（完整 env，held-out split）上打分，这样每个 arm 都在同一组固定集合上被衡量。Eval 是一次
verl 的 val-only pass（generate + score，无训练、无 critic），且绝不中止主循环 ——
一次失败的 eval 只记一条 warning 并继续。

**Configure**

| 能力 | Key | Where | Source |
|---|---|---|---|
| 启用 eval | `val_env_spec`（`""` → 无 eval） | DEFAULTS | [`fed/run_fed.py`](../fed/run_fed.py)（`eval_global`） |
| Eval 节奏 | `test_freq`（每 K 轮 + 最终轮） | DEFAULTS | [`fed/run_fed.py`](../fed/run_fed.py) |
| Round-0 baseline | `val_before_train`（在第 1 轮之前也评一次 base 模型） | DEFAULTS | [`fed/run_fed.py`](../fed/run_fed.py) |
| 采样 temp | `val_temperature`（paper = 0.4） | DEFAULTS | [`fed/run_fed.py`](../fed/run_fed.py) |
| 共享 val 服务端口 | `webshop_val_port`、`alfworld_val_port`、`alfworld_val_split` | DEFAULTS | [`fed/run_fed.py`](../fed/run_fed.py)（`start_val_service`） |

`eval_global()` 转储 verl 的 per-sample validation JSONL，`summarize_val_dump()`
把它归约为 `{n, success_rate, reward_mean}`；round → success 曲线被写入
`federated_summary.json`（`val_curve`）。val env-spec 文件是
[`config/envs/webshop_15_val.yaml`](../config/envs/webshop_15_val.yaml) 和
[`config/envs/alfworld_val.yaml`](../config/envs/alfworld_val.yaml)。

## 10. Logging

Weights & Biases 已被**移除** —— 不需要 tracking 账号或 key。verl 只向
console 记录（base config 里的 `trainer.logger: [console]`），driver 再把每个 client 的
`training.log` 后处理成 FedAgent 的 metrics schema。

**Configure / artifacts**

| 能力 | Key / path | Source |
|---|---|---|
| Console logger | `trainer.logger: [console]` | [`config/fedagent_ppo.yaml`](../config/fedagent_ppo.yaml) |
| Per-client JSON metrics | `round_<r>/client_<c>/json_logs/metrics.json` | [`fed/metrics_logger.py`](../fed/metrics_logger.py) |
| Run summary | `<output_dir>/federated_summary.json`（per-round 溯源 + `val_curve`） | [`fed/run_fed.py`](../fed/run_fed.py) |
| Per-client / 服务 / eval 日志 | `output_dir` 下的 `training.log`、`*_service_client*.log`、`eval.log` | [`fed/run_fed.py`](../fed/run_fed.py) |

`write_metrics_json()` 把 verl 的 per-step console 转储解析成
`[{"step": int, "metrics": {...}}, …]` —— 与 FedAgent 绘图工具消费的
schema 相同 —— 且无需修改 verl。

## 11. Extensibility

FedAgent 的设计目标不仅是可复现，更是可扩展。

| 添加… | Where | Guide |
|---|---|---|
| 一个新的**环境 / 数据集** | [`fedagent/envs/`](../envs/) + [`config/envs/`](../config/envs) | [extending.md](./extending.md) |
| 一种新的**异质性**（client partition） | [`fedagent/hetero/`](../hetero/) | [heterogeneity.md](./heterogeneity.md) |
| 一个新的 **RL 算法**（GRPO/PPO 之外） | `client_overrides` / verl trainer | [extending.md](./extending.md) |
| 一种新的**聚合**（FedAvg/FedProx 之外） | [`tools/verl08_migration/aggregate_fedavg_fsdp.py`](../../tools/verl08_migration/aggregate_fedavg_fsdp.py) / [`fedagent/fedprox.py`](../fedprox.py) | [extending.md](./extending.md) |

另见：[configuration.md](./configuration.md)（完整 key 参考） ·
[heterogeneity.md](./heterogeneity.md)（two-level 套件） ·
[`fed/README.md`](../fed/README.md)（round-loop 内部机制）。
