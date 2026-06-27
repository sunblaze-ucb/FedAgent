# `fed/` — the federated training spine for FedAgent on stock verl 0.8

This package closes the FedAgent federated RL loop **on top of unmodified verl 0.8**.
There is **no trainer fork**: each client is a plain training subprocess
(`python -m fedagent.main_ppo_fed`) that runs verl's stock PPO/GRPO trainer with its
native async agent-loop rollout. `fed/` orchestrates the rounds — select clients, run
each one, FedAvg their FSDP checkpoints, merge to HuggingFace, and re-enter the next
round from the aggregated model.

```
fed/
├── run_fed.py         # the federated driver (round loop, FedAvg, merge, eval, baselines)
├── metrics_logger.py  # parse each client's training.log -> json_logs/metrics.json
├── __init__.py
└── README.md          # this file
```

See the top-level [`../README.md`](../README.md) for the project overview and
[`../docs/`](../docs/) for installation, configuration, and reproduction guides.
Per-client environment services live in [`../envs/webshop/service/`](../envs/webshop/service/) and
`../envs/alfworld/service/`.

---

## What `run_fed.py` does

`run(cfg)` is the driver. For each round `r = 1 … total_rounds`:

1. **Choose the starting model.** Round 1 trains from the base model
   (`model_path`, or an auto-discovered local Qwen2.5-0.5B-Instruct via
   `discover_model()`). Round `r > 1` trains from round `r-1`'s **merged FedAvg model**
   (`round_{r-1}/aggregated/hf`).
2. **Select clients.** `select_clients(r, total_clients, clients_per_round, base_seed)`
   deterministically samples `clients_per_round` of `total_clients`
   (RNG seed `base_seed + r - 1`, so the schedule is reproducible on resume). When
   `clients_per_round >= total_clients`, all clients participate.
3. **Train each selected client sequentially.** `run_client(...)` shells out to
   `python -m fedagent.main_ppo_fed` with Hydra overrides for the data spec, model path,
   agent-loop config, per-round checkpoint dir, `total_epochs=epochs_per_round`, GPU
   count, and `resume_mode=disable` (federation owns "resume" at the round level). It
   returns the newest `global_step_K/actor` FSDP-shard dir (`latest_actor_dir`), plus the
   sibling `critic` dir for PPO (`critic_dir_for`). `wait_between_clients` seconds pass
   between clients to let Ray/GPU fully release.
4. **FedAvg the actor shards.** `fedavg(...)` runs
   `tools/verl08_migration/aggregate_fedavg_fsdp.py` under
   `torchrun --nproc_per_node=<world_size>` (world size auto-detected from the shards by
   `world_size_of`), averaging the clients' shards into
   `round_r/aggregated/checkpoints/global_step_0/actor`. `weights` (if set) gives a
   weighted FedAvg; otherwise it is uniform.
5. **Merge to HuggingFace.** `merge_to_hf(...)` runs `python -m verl.model_merger merge
   --backend fsdp` to turn the aggregated shards into a complete HF model dir
   (`round_r/aggregated/hf`). This becomes the next round's `model.path`. **The loop
   closes here.**
6. **(PPO only) Federate the critic too** — see [GRPO vs PPO](#grpo-vs-ppo-gae) below.
7. **Disk hygiene.** `cleanup_round_checkpoints(...)` deletes the consumed per-client and
   aggregated FSDP shard dirs (keeping every `training.log` and the merged HF), so peak
   disk stays roughly one round's worth instead of growing unbounded. Gated by
   `cleanup_checkpoints` (on by default).
8. **Per-client metrics.** After each client, `run_client` calls into
   [`metrics_logger.py`](metrics_logger.py) to write `json_logs/metrics.json` and echo the
   per-step reward curve.

A `federated_summary.json` (per-round provenance + the val curve) is written to
`output_dir` at the end. Each subprocess is launched via `stream(...)`, which tees
combined stdout/stderr to the console (tagged) **and** to a per-stage log file.

### Environment services (`webshop` / `alfworld`)

`tinyguess` runs **in-process** (no service). For `webshop`/`alfworld`,
`start_webshop_services` / `start_alfworld_services` launch remote HTTP env services
**lazily each round, only for that round's selected clients** (`client_ids=selected`), so at
most `clients_per_round` services are alive at once (one service == one client's hidden
transition kernel / data shard). Each service is started with an env-var bridge
(`CLIENT_ID`, `CLIENT_NUM`, `PARTITION_STRATEGY`, the heterogeneity knobs, …) that selects
that client's heterogeneity variant, and the driver waits for each `/health` endpoint
(up to `service_health_timeout`). `run_client` points each client at its own service via
`WEBSHOP_SERVICE_URL` / `ALFWORLD_SERVICE_URL`. The round's per-client services are torn
down **per round, before aggregation**; only the shared unperturbed VAL service is started
once and persists for the whole run (stopped at the end).

### Round-threaded data seed

`run_client` sets `FEDAGENT_BASE_SEED = base_seed + round*100 + client_id` (read by the
dataset adapter). The round term makes each client re-draw goals from its **fixed** shard
every round (covering the shard over `total_rounds`); without it, every client would train
on the same goals each round. Stride 100 keeps the per-client offsets collision-free.

### Unperturbed global-model evaluation

If `val_env_spec` is set, `start_val_service` brings up **one shared, unperturbed**
validation service (`PARTITION_STRATEGY=""`/`uniform`, held-out val split) so every arm is
scored on the same fixed set. `eval_global(...)` runs a verl **val-only** pass
(`trainer.val_only=true`, `adv_estimator=grpo` regardless of the train algorithm — eval is
generate-and-score, never a critic, and `FEDPROX_MU` is stripped) on the aggregated global
model every `test_freq` rounds and on the final round; `val_before_train` also scores the
base model as the round-0 point. `summarize_val_dump` reads verl's validation JSONL dump
into `{n, success_rate, reward_mean}` (mean of `traj_success` / `score`). Eval failures
log a warning and never abort the run — it is measurement, not the loop. Val sampling uses
`val_temperature` (paper: 0.4).

---

## Baseline modes

The mode is derived from the config (no separate flag) in `run()`:

| Mode | Selected when | Behavior |
|---|---|---|
| **federated** | `total_clients > 1` and `local_client_id < 0` (default) | FedAvg across the selected clients each round. |
| **centralized** | `total_clients == 1` | One model on the pooled data; FedAvg of a single client is the identity, so the loop is just `total_rounds × epochs_per_round` of continued central training. |
| **local** | `local_client_id = k >= 0` | The paper's *Local Agent Training*: pin client `k` (its slice of the `total_clients`-way partition) every round and train it alone, no federation. |

`participating_client_ids` / `select_clients` honor `local_client_id` by pinning that one
client (and only its env service is launched).

---

## GRPO vs PPO (`gae`)

`adv_estimator` selects the RL algorithm:

- **`grpo`** (default): no critic. The driver FedAvgs and merges only the actor. GRPO's
  group size is set on the rollout (`actor_rollout_ref.rollout.n`, paper default **8**)
  via the agent config / `client_overrides`, not by a `run_fed` key.
- **`gae`** (PPO): `run_client` adds `algorithm.adv_estimator=gae` and loads the value
  model from `critic.model.path` (the **base model** on round 1, the previous round's
  **aggregated critic** thereafter). verl saves the critic to `global_step_K/critic`
  alongside the actor with the identical shard layout, so the driver runs the **same**
  `fedavg` + `merge_to_hf` machinery a second time (`kind="critic"`) and carries the merged
  critic (`aggregated/critic_hf`) into the next round. If `gae` is set but a client fails to
  emit a critic checkpoint, the round aborts with a clear error.

---

## Config-key reference (the `DEFAULTS` dict)

Every key below comes from `DEFAULTS` in `run_fed.py`; a YAML file is merged over these and
CLI flags override the result.

### Core loop

| Key | Default | Meaning |
|---|---|---|
| `model_path` | `""` | Base HF model for round 1; `""` auto-discovers a local Qwen2.5-0.5B-Instruct. |
| `output_dir` | `/tmp/xbb9020_fedagent_fed_tinyguess` | Root for all rounds, logs, and the summary. |
| `total_clients` | `2` | Number of clients `N` in the federation. |
| `clients_per_round` | `2` | Clients `M` sampled per round (all if `>= total_clients`). |
| `total_rounds` | `2` | Number of federated rounds `T`. |
| `epochs_per_round` | `1` | Local epochs `E` per client per round (`trainer.total_epochs`). |
| `base_seed` | `42` | Seed base for client selection and the round-threaded data seed. |
| `n_gpus_per_node` | `2` | GPUs per client training run (`trainer.n_gpus_per_node`; also FedAvg world size). |
| `total_training_steps` | `1` | Per-client-round step cap (smoke); `<= 0` emits `null` so verl runs the full `E` epochs. |
| `save_freq` | `1` | `trainer.save_freq` (checkpoint cadence within a client run). |
| `weights` | `""` | Comma-sep FedAvg weights; `""` is uniform. |
| `wait_between_clients` | `5` | Seconds to pause between sequential clients (let Ray/GPU release). |
| `client_overrides` | `[]` | Extra `key=value` Hydra overrides appended to every client run (rollout shape, batch sizes, …). |
| `adv_estimator` | `grpo` | `grpo` (no critic) or `gae` (PPO: federate the critic too). |
| `cleanup_checkpoints` | `True` | Delete consumed FSDP shards after each merge (keep HF + logs). |
| `custom_cls_path` | `data/agentic_dataset.py` | `data.custom_cls.path` (the dataset adapter). |
| `agent_config_path` | `config/agent.yaml` | Agent-loop config (`...rollout.agent.agent_loop_config_path`). |
| `env_spec` | `config/envs/tiny_guess.yaml` | Train/val data spec passed as `data.train_files`/`data.val_files`. |

### Environment selection & services

| Key | Default | Meaning |
|---|---|---|
| `env_kind` | `tinyguess` | `tinyguess` (in-process), `webshop`, or `alfworld` (remote per-client services). |
| `service_health_timeout` | `900` | Seconds to wait for each service `/health`. |
| `webshop_run_service` | `envs/webshop/service/run_service.sh` | Launcher for a WebShop service. |
| `webshop_base_port` | `8080` | Client `c`'s WebShop service listens on `webshop_base_port + c`. |
| `webshop_pool_size` | `8` | Env pool per WebShop service (must be `>= gen_batch`). |
| `search_return_n` | `200` | `WEBSHOP_SEARCH_RETURN_N`: BM25 top-K (paper=200). |
| `alfworld_run_service` | `envs/alfworld/service/run_service.sh` | Launcher for an ALFWorld service. |
| `alfworld_base_port` | `8200` | Client `c`'s ALFWorld service listens on `alfworld_base_port + c`. |
| `alfworld_pool_size` | `4` | TextWorld env pool per ALFWorld service (must be `>= gen_batch`). |
| `alfworld_train_eval` | `train` | ALFWorld game split: `train` / `eval_in_distribution` / `eval_out_of_distribution`. |
| `alfworld_task_types` | `""` | `""` = all 6 task types; else comma-sep IDs for the eval breakdown. |

### Heterogeneity

| Key | Default | Meaning |
|---|---|---|
| `partition_strategy` | `""` | `""` / `catalog_split`,`task_disjoint` (env) / `preference`,`coverage`,`hardness` (task) / `bm25_field_subset`,`bm25_reweight`,`lookalike`,`rank_wrapper` (env variants). |
| `env_div` | `0.7` | Catalog-split heterogeneity strength. |
| `keep_ratio` | `0.7` | Catalog-split distractor density. |
| `omega` | `0.5` | Preference (task-het) Dirichlet spread. |
| `size_std` | `1.0` | Coverage (task-het) Beta dispersion (ξ). |
| `success_std` | `1.0` | Hardness (task-het) Beta dispersion (ξ′). |
| `variant_n` | `0` | Env-variant arm count (bm25/lookalike/rank); `0` uses the function default. |
| `trajectories_file` | `""` | Hardness: required `task_id`→success-label file. |
| `min_goals_per_client` | `100` | Minimum goals/games assigned to each client. |

### Baselines

| Key | Default | Meaning |
|---|---|---|
| `local_client_id` | `-1` | `>= 0` selects the **local** baseline: train only this client of `total_clients`, no federation. |

(`total_clients == 1` selects **centralized**; the default `total_clients > 1` with
`local_client_id < 0` is **federated** — see [Baseline modes](#baseline-modes).)

### Evaluation

| Key | Default | Meaning |
|---|---|---|
| `val_env_spec` | `""` | `""` disables eval; else the UNPERTURBED val env-spec to score the global model. |
| `test_freq` | `5` | Eval the aggregated global model every `K` rounds (plus the final round). |
| `val_before_train` | `True` | Also eval the base model before round 1 (the round-0 point). |
| `val_temperature` | `0.4` | Val sampling temperature. |
| `webshop_val_port` | `8090` | Shared unperturbed WebShop val service port. |
| `alfworld_val_port` | `8290` | Shared unperturbed ALFWorld val service port. |
| `alfworld_val_split` | `eval_in_distribution` | ALFWorld val game split. |

### FedProx

| Key | Default | Meaning |
|---|---|---|
| `fedprox_mu` | `0.0` | `> 0` enables the client-side FedProx proximal term (else plain FedAvg). |

**How FedProx is wired:** when `fedprox_mu > 0`, `run_client` sets `FEDPROX_MU` in the
client's environment. The proximal term is injected by the repo-root
[`../../sitecustomize.py`](../../sitecustomize.py), which CPython auto-imports at startup in
every process on `PYTHONPATH` (the client **and** its Ray workers) and, gated on
`FEDPROX_MU`, patches `FSDPEngine.optimizer_step`. It is **not** a Ray
`runtime_env.worker_process_setup_hook` — that hook clobbered verl's per-worker
`CUDA_VISIBLE_DEVICES`, breaking GPU isolation. `eval_global` always strips `FEDPROX_MU`.

---

## CLI flags

`main()` parses flags that override the YAML (`load_cfg` merges `DEFAULTS` ← YAML ← flags):

| Flag | Overrides |
|---|---|
| `--config` | Path to the federated YAML (merged over `DEFAULTS`). |
| `--model-path` | `model_path` (base HF model for round 1). |
| `--output-dir` | `output_dir`. |
| `--rounds` | `total_rounds`. |
| `--clients` | `total_clients` (also clamps `clients_per_round`). |
| `--n-gpus` | `n_gpus_per_node` (e.g. `4` for a single 4-GPU run). |
| `--base-seed` | `base_seed` (seed sweeps). |
| `--port-base` | `webshop_base_port` (concurrent runs). |
| `--fedprox-mu` | `fedprox_mu` (`> 0` enables FedProx). |
| `--local-client-id` | `local_client_id` (local baseline). |

`load_cfg` also resolves package-relative paths (e.g. `config/envs/webshop_15.yaml`)
against the `fedagent/` package dir, so configs can use short paths.

---

## How to run

Run inside the `fedagent-verl08` conda env on a GPU node (the driver sets `PYTHONPATH` to
the repo root so `sitecustomize.py` and `fedagent` are importable in every subprocess).

```bash
conda activate fedagent-verl08

# TinyGuess smoke (in-process, fast):
python -m fedagent.fed.run_fed --config fedagent/config/examples/tinyguess_2cl_2rd.yaml

# WebShop, env-level heterogeneity (per-client services), single 4-GPU node:
python -m fedagent.fed.run_fed \
  --config fedagent/config/examples/webshop/scaled/catalog.yaml --n-gpus 4

# Baselines (same config family, different mode):
python -m fedagent.fed.run_fed --config fedagent/config/examples/webshop/scaled/centralized.yaml
python -m fedagent.fed.run_fed --config fedagent/config/examples/webshop/scaled/local.yaml \
  --local-client-id 0

# FedProx + a seed sweep entry:
python -m fedagent.fed.run_fed --config fedagent/config/examples/webshop/scaled/catalog.yaml \
  --fedprox-mu 0.1 --base-seed 7
```

Outputs land under `output_dir`: `round_<r>/client_<c>/{checkpoints,training.log,json_logs}`,
`round_<r>/aggregated/{checkpoints,hf,*.log}`, the per-service logs, and
`federated_summary.json`. See [`../EXPERIMENTS.md`](../EXPERIMENTS.md) for the curated
config matrix.

---

## `metrics_logger.py` — measurability without a verl fork

verl 0.8's stock console logger prints the full per-step metric dict to stdout (captured in
each client's `training.log`). Since the overlay does not fork verl, this module re-parses
those lines into the FedAgent plot/loader schema:

```json
[ {"step": <int>, "metrics": {"<key>": <float>, ...}}, ... ]
```

- **`parse_training_log(log_path)`** — regex-extracts each `step:<N> - k:v - k:v - …`
  line into a `{"step", "metrics"}` entry, unwrapping `np.float64(...)`/`tensor(...)`
  values and keeping only lines with `>= 5` parsed keys (filters stray `step:N` mentions).
- **`write_metrics_json(log_path, out_dir)`** — writes `<out_dir>/metrics.json`. `run_fed`
  calls this after each client round, producing `round_<r>/client_<c>/json_logs/metrics.json`.
- **`summarize(entries)`** — one-line per-step reward string (prefers
  `critic/rewards/mean`, then `critic/score/mean`), echoed to the console after each client.

It also runs standalone:

```bash
python -m fedagent.fed.metrics_logger path/to/training.log --out-dir path/to/json_logs
```
