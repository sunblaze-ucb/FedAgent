# Key features, detailed

This expands each headline capability of the **FedAgent verl-0.8 overlay** into the
concrete **config key** that turns it on and the **source file** in `fedagent/` that
implements it.

FedAgent is a thin overlay on **stock verl 0.8** (migrated from a verl-agent-0.3.1
fork). Every experiment is one flat YAML driven by a single entry point:

```bash
python -m fedagent.fed.run_fed --config fedagent/config/<experiment>.yaml
```

The config keys are the flat `DEFAULTS` dict in
[`run_fed.py`](../fed/run_fed.py); a few CLI flags override the YAML
(`--model-path --output-dir --rounds --clients --n-gpus --base-seed --port-base
--fedprox-mu --local-client-id`). Anything verl-specific is passed through to each
client as a **Hydra override** in the `client_overrides:` list (each entry is a
`key=value` string applied to `python -m fedagent.main_ppo_fed`). The complete field
reference is in [configuration.md](./configuration.md).

## Contents
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

Federated **GRPO** and **PPO**, as federated counterparts of the verl trainers. Each
selected client runs a local verl update in its own subprocess
(`python -m fedagent.main_ppo_fed`); the driver FedAvgs the resulting FSDP
checkpoints and re-enters the next round from the merged model. GRPO uses group
rollouts and no critic; PPO adds a value model that is federated alongside the actor.

**Configure**

| Capability | Key | Where | Source |
|---|---|---|---|
| Algorithm select | `adv_estimator: grpo` (default) or `gae` | `run_fed.py` DEFAULTS | [`fed/run_fed.py`](../fed/run_fed.py) |
| GRPO group size **G** | `actor_rollout_ref.rollout.n=8` | `client_overrides` (paper arms = 8; base default 4) | [`config/fedagent_ppo.yaml`](../config/fedagent_ppo.yaml) |
| GRPO actor loss | `actor_rollout_ref.actor.use_kl_loss=true`, `kl_loss_coef=0.01`, `kl_loss_type=low_var_kl`, `entropy_coeff=0.001` | base config (inherited by every arm) | [`config/fedagent_ppo.yaml`](../config/fedagent_ppo.yaml) |
| PPO — federate the critic | `adv_estimator: gae` (+ `critic.*` overrides) | DEFAULTS + `client_overrides` | [`fed/run_fed.py`](../fed/run_fed.py) |
| Per-client trainer entry | `python -m fedagent.main_ppo_fed` (runs verl's stock `run_ppo`) | — | [`main_ppo_fed.py`](../main_ppo_fed.py) |
| Multi-turn rollout | `actor_rollout_ref.rollout.agent.default_agent_loop: gym_text` | base config + [`config/agent.yaml`](../config/agent.yaml) | [`agent_loops/gym_text_agent_loop.py`](../agent_loops/gym_text_agent_loop.py) |

For **PPO** (`adv_estimator: gae`), `run_fed.py` enables the value model, sets
`critic.model.path` per round (round-1 critic = the base model's backbone; thereafter
the aggregated critic), and FedAvgs **both** actor and critic each round
(`fedavg(..., kind="actor")` and `kind="critic"`). Clients must save the value model
(`critic.checkpoint.save_contents=[model]` in `client_overrides`) so it can be
aggregated. See [`fed/README.md`](../fed/README.md) for the round mechanics.

Adding a new RL algorithm → [extending.md](./extending.md).

## 2. Models

Any **HuggingFace** causal-LM backbone. The paper sweeps **Qwen2.5-1.5B / 3B /
7B-Instruct** and **Llama-3.2-3B-Instruct**.

**Configure**

| Capability | Key | Where |
|---|---|---|
| Base model (round 1) | `model_path` (`""` → auto-discover a local Qwen2.5-0.5B snapshot) | DEFAULTS / `--model-path` |
| Attention impl | `+actor_rollout_ref.model.override_config.attn_implementation=sdpa` (added by the driver) | [`fed/run_fed.py`](../fed/run_fed.py) |
| PPO value model init | `critic.model.path` (set per round by the driver, not pinned) | [`fed/run_fed.py`](../fed/run_fed.py) |

Each round trains from the **previous round's merged HF model**
(`round_{r-1}/aggregated/hf`), produced by `verl.model_merger`. Model acquisition,
cache location, and offline clusters → [installation.md](./installation.md).

## 3. Environments

Real agent benchmarks **WebShop** (e-commerce search-and-buy) and **ALFWorld**
(embodied household tasks on TextWorld), plus an in-process **TinyGuess** wiring probe.
WebShop and ALFWorld run as **remote HTTP services** (their dependencies conflict, so
each lives in its own conda env); the driver launches **one service per client** so
each client gets its own environment / hidden transition kernel.

**Configure**

| Capability | Key | Where | Source |
|---|---|---|---|
| Environment select | `env_kind: tinyguess \| webshop \| alfworld` | DEFAULTS | [`fed/run_fed.py`](../fed/run_fed.py) |
| Env spec (turns / pool) | `env_spec: config/envs/<name>.yaml` | DEFAULTS | [`config/envs/`](../config/envs) |
| Dataset adapter (verl `custom_cls`) | `custom_cls_path` → `data.custom_cls.path` | DEFAULTS | [`data/agentic_dataset.py`](../data/agentic_dataset.py) |
| WebShop service launcher | `webshop_run_service`, `webshop_base_port` (client `c` → `+c`), `webshop_pool_size`, `search_return_n` | DEFAULTS | [`envs/webshop/service/`](../envs/webshop/service/server.py) |
| ALFWorld service launcher | `alfworld_run_service`, `alfworld_base_port` (client `c` → `+c`), `alfworld_pool_size`, `alfworld_train_eval`, `alfworld_task_types` | DEFAULTS | [`envs/alfworld/service/`](../envs/alfworld/service/server.py) |
| Service health wait | `service_health_timeout` (seconds) | DEFAULTS | [`fed/run_fed.py`](../fed/run_fed.py) |

The env-spec files (e.g. [`config/envs/webshop_15.yaml`](../config/envs/webshop_15.yaml),
[`config/envs/alfworld.yaml`](../config/envs/alfworld.yaml)) set `n_envs`, `max_turns`,
and the `gym_text` agent name; the per-client service URL is injected by the driver as
`WEBSHOP_SERVICE_URL` / `ALFWORLD_SERVICE_URL`. Adding a new environment →
[extending.md](./extending.md).

## 4. Two-level heterogeneity

The core research feature: a suite of **client-partition strategies** along two
structurally distinct axes, selected with `partition_strategy` plus per-strategy
knobs. The driver forwards them to each client's env service via env vars
(`PARTITION_STRATEGY`, `OMEGA`, `SIZE_STD`, …); the service dispatches to the matching
module under [`fedagent/hetero/`](../hetero/).

**Task-level** — clients differ in their *task distribution* (observable through the prompt):

| `partition_strategy` | Knob key(s) | Source |
|---|---|---|
| `preference` | `omega` | [`hetero/webshop_task.py`](../hetero/webshop_task.py) |
| `coverage` | `size_std` | [`hetero/webshop_coverage.py`](../hetero/webshop_coverage.py) |
| `hardness` | `success_std`, `trajectories_file` (required) | [`hetero/webshop_hardness.py`](../hetero/webshop_hardness.py) |
| `task_disjoint` | (disjoint goal slice, full catalog) | [`hetero/webshop_catalog_split.py`](../hetero/webshop_catalog_split.py) |

**Environment-level** — clients differ in the *transition kernel* (hidden from the policy):

| `partition_strategy` | Knob key(s) | Source |
|---|---|---|
| `catalog_split` | `env_div`, `keep_ratio` | [`hetero/webshop_catalog_split.py`](../hetero/webshop_catalog_split.py) |
| `bm25_field_subset` | `variant_n` | [`hetero/webshop_env_variants.py`](../hetero/webshop_env_variants.py) |
| `bm25_reweight` | `variant_n` | [`hetero/webshop_env_variants.py`](../hetero/webshop_env_variants.py) |
| `lookalike` | `variant_n` | [`hetero/webshop_env_variants.py`](../hetero/webshop_env_variants.py) |
| `rank_wrapper` | `variant_n` | [`hetero/webshop_env_variants.py`](../hetero/webshop_env_variants.py) |

`partition_strategy: ""` (or `uniform` for ALFWorld) is the homogeneous / i.i.d.
baseline. `min_goals_per_client` sets the per-client task count; `base_seed` makes the
client → data assignment deterministic. The full construction and paper mapping are in
[heterogeneity.md](./heterogeneity.md); adding a new partition →
[extending.md](./extending.md).

## 5. Aggregation

Server-side combination each round is **FedAvg** (a weighted parameter mean of the
clients' FSDP shards). verl 0.8 saves per-rank `ShardedTensor` shards that cannot be
loaded single-process, so the aggregator runs under a matched-world-size process group
(`torchrun --nproc_per_node = save-time world_size`): each rank averages its own rank
shard across clients in place. **FedProx** keeps each client near the round's global
model by adding a `μ·(w − w_t)` term to the actor gradient on every optimizer step —
it changes the **client** update; the server still aggregates by FedAvg.

**Configure**

| Capability | Key | Where | Source |
|---|---|---|---|
| FedAvg (default) | (no key — always runs each round) | — | [`tools/verl08_migration/aggregate_fedavg_fsdp.py`](../../tools/verl08_migration/aggregate_fedavg_fsdp.py) |
| FedAvg weights | `weights` (`""` → uniform; else comma-separated, sums to 1) | DEFAULTS (YAML only — no CLI flag) | [`tools/verl08_migration/aggregate_fedavg_fsdp.py`](../../tools/verl08_migration/aggregate_fedavg_fsdp.py) |
| Merge shards → HF | (auto — `verl.model_merger merge --backend fsdp`) | — | [`fed/run_fed.py`](../fed/run_fed.py) (`merge_to_hf`) |
| FedProx | `fedprox_mu` (>0 enables; `0` ≡ FedAvg) | DEFAULTS → `--fedprox-mu` | [`fedagent/fedprox.py`](../fedprox.py) |

FedProx is injected **without** a Ray `runtime_env` hook (which would clobber verl's
per-worker `CUDA_VISIBLE_DEVICES`): the driver sets `FEDPROX_MU` in each client's
environment, and the repo-root [`sitecustomize.py`](../../sitecustomize.py) — auto-imported
at interpreter startup in every process on `PYTHONPATH` — calls
`fedagent.fedprox.install_deferred_patch()` (fail-closed when `verl` is present). That arms a
`sys.meta_path` hook which monkeypatches `FSDPEngine.optimizer_step` the moment verl first
imports its FSDP-engine module — i.e. **after** the Ray worker has its per-rank
`CUDA_VISIBLE_DEVICES` set. (Importing `FSDPEngine` eagerly at interpreter startup instead
pulls in torch/verl before device assignment and breaks per-rank GPU isolation at multi-GPU,
"Duplicate GPU detected".) Eval passes scrub `FEDPROX_MU` so the proximal term never fires
during validation. Adding a new aggregation rule → [extending.md](./extending.md).

## 6. Baselines

Three regimes share the same driver and config schema; the mode is inferred from the
client keys.

| Mode | How to select | Behaviour | Source |
|---|---|---|---|
| **Federated** | default (`total_clients` > 1, `local_client_id: -1`) | FedAvg across `clients_per_round` sampled clients each round | [`fed/run_fed.py`](../fed/run_fed.py) |
| **Centralized** | `total_clients: 1` (+ `partition_strategy: ""`) | one model on the pooled data; FedAvg of one client is the identity, so the loop is `total_rounds × epochs_per_round` of continued training | [`fed/run_fed.py`](../fed/run_fed.py) |
| **Local** | `local_client_id: k >= 0` (with `total_clients: N`) | the paper's "Local Agent Training": pin one client of the N-way partition every round, train it alone, no federation | [`fed/run_fed.py`](../fed/run_fed.py) |

`select_clients()` does deterministic per-round sampling (seed = `base_seed + round −
1`); local mode pins the one client and launches only its env service
(`participating_client_ids()`). Worked examples → [running.md](./running.md).

## 7. Federation protocol

The full protocol is configurable in the flat config (all keys in `run_fed.py`
DEFAULTS):

| Symbol | Key | Meaning |
|---|---|---|
| `N` | `total_clients` | size of the client pool |
| `M` | `clients_per_round` | clients sampled & trained each round |
| `T` | `total_rounds` | number of federated rounds |
| `E` | `epochs_per_round` | local epochs per selected client (`trainer.total_epochs`) |
| `\|Xᵢ\|` | `min_goals_per_client` | tasks per client |
| seed | `base_seed` | deterministic client→data + client-selection seed |
| — | `total_training_steps` | per-client-round step cap (`> 0` for smokes; `<= 0` → full `E` epochs) |
| — | `save_freq` | checkpoint cadence within a client round |
| — | `wait_between_clients` | seconds to let Ray/GPU release between clients |
| — | `cleanup_checkpoints` | delete consumed FSDP shards after each merge (keeps HF + logs) |

Paper config filenames encode the protocol
(e.g. `…total-100_cl-per-rd-2_rd-70_ep-per-cl-3_min-goals-per-cl-100…`); the decoder
and the full 176-config matrix are in [reproducing.md](./reproducing.md) and
[configuration.md](./configuration.md). The driver threads a per-(round, client) data
seed (`FEDAGENT_BASE_SEED = base_seed + round·100 + client_id`) so each client re-draws
goals from its fixed shard every round.

## 8. FSDP & scaling

Larger backbones (3B / 7B) train via **FSDP** with optional CPU offload; runs scale
from one GPU to a full node.

**Configure**

| Capability | Key | Where |
|---|---|---|
| GPUs per node | `n_gpus_per_node` (= FedAvg `nproc_per_node`) | DEFAULTS → `--n-gpus` |
| Actor offload | `actor_rollout_ref.actor.fsdp_config.param_offload` / `.optimizer_offload` | `client_overrides` |
| Ref-policy offload | `actor_rollout_ref.ref.fsdp_config.param_offload` | base config / `client_overrides` |
| Critic offload (PPO) | `critic.fsdp.param_offload` / `.optimizer_offload` | `client_overrides` |
| vLLM tensor-parallel | `actor_rollout_ref.rollout.tensor_model_parallel_size` | base config / `client_overrides` |
| vLLM memory | `actor_rollout_ref.rollout.gpu_memory_utilization` | `client_overrides` |

The save-time world size (read from `fsdp_config.json`) is auto-detected and used as
the aggregator's `nproc_per_node`, so FedAvg matches the training shard layout. The
hardware / scaling matrix → [running.md](./running.md).

## 9. Evaluation

The aggregated **global** model is scored each `test_freq` rounds on a shared,
**unperturbed** validation service (full env, held-out split), so every arm is measured
on the same fixed set. Eval is a verl val-only pass (generate + score, no training, no
critic) and never aborts the loop — a failed eval logs a warning and continues.

**Configure**

| Capability | Key | Where | Source |
|---|---|---|---|
| Enable eval | `val_env_spec` (`""` → no eval) | DEFAULTS | [`fed/run_fed.py`](../fed/run_fed.py) (`eval_global`) |
| Eval cadence | `test_freq` (every K rounds + final round) | DEFAULTS | [`fed/run_fed.py`](../fed/run_fed.py) |
| Round-0 baseline | `val_before_train` (also eval the base model before round 1) | DEFAULTS | [`fed/run_fed.py`](../fed/run_fed.py) |
| Sampling temp | `val_temperature` (paper = 0.4) | DEFAULTS | [`fed/run_fed.py`](../fed/run_fed.py) |
| Shared val service ports | `webshop_val_port`, `alfworld_val_port`, `alfworld_val_split` | DEFAULTS | [`fed/run_fed.py`](../fed/run_fed.py) (`start_val_service`) |

`eval_global()` dumps verl's per-sample validation JSONL and `summarize_val_dump()`
reduces it to `{n, success_rate, reward_mean}`; the round → success curve is written
into `federated_summary.json` (`val_curve`). The val env-spec files are
[`config/envs/webshop_15_val.yaml`](../config/envs/webshop_15_val.yaml) and
[`config/envs/alfworld_val.yaml`](../config/envs/alfworld_val.yaml).

## 10. Logging

Weights & Biases is **removed** — no tracking account or key is needed. verl logs to
console only (`trainer.logger: [console]` in the base config), and the driver
post-processes each client's `training.log` into the FedAgent metrics schema.

**Configure / artifacts**

| Capability | Key / path | Source |
|---|---|---|
| Console logger | `trainer.logger: [console]` | [`config/fedagent_ppo.yaml`](../config/fedagent_ppo.yaml) |
| Per-client JSON metrics | `round_<r>/client_<c>/json_logs/metrics.json` | [`fed/metrics_logger.py`](../fed/metrics_logger.py) |
| Run summary | `<output_dir>/federated_summary.json` (per-round provenance + `val_curve`) | [`fed/run_fed.py`](../fed/run_fed.py) |
| Per-client / service / eval logs | `training.log`, `*_service_client*.log`, `eval.log` under `output_dir` | [`fed/run_fed.py`](../fed/run_fed.py) |

`write_metrics_json()` parses verl's per-step console dump into
`[{"step": int, "metrics": {...}}, …]` — the same schema the FedAgent plotting tools
consume — with no verl modification.

## 11. Extensibility

FedAgent is built to be extended, not only reproduced.

| Add… | Where | Guide |
|---|---|---|
| a new **environment / dataset** | [`fedagent/envs/`](../envs/) + [`config/envs/`](../config/envs) | [extending.md](./extending.md) |
| a new **heterogeneity** (client partition) | [`fedagent/hetero/`](../hetero/) | [heterogeneity.md](./heterogeneity.md) |
| a new **RL algorithm** (beyond GRPO/PPO) | `client_overrides` / verl trainer | [extending.md](./extending.md) |
| a new **aggregation** (beyond FedAvg/FedProx) | [`tools/verl08_migration/aggregate_fedavg_fsdp.py`](../../tools/verl08_migration/aggregate_fedavg_fsdp.py) / [`fedagent/fedprox.py`](../fedprox.py) | [extending.md](./extending.md) |

See also: [configuration.md](./configuration.md) (full key reference) ·
[heterogeneity.md](./heterogeneity.md) (the two-level suite) ·
[`fed/README.md`](../fed/README.md) (round-loop internals).
