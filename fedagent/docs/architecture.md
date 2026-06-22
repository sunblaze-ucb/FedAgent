# Architecture

FedAgent is **federated reinforcement learning for LLM agents**. This document explains how
the `fedagent/` package implements it as a **thin overlay on stock verl 0.8** — what runs
where, and how a federated round actually executes.

## Design principle: overlay, not fork

The original FedAgent forked verl-agent 0.3.1 and wove federated logic *into* the trainer.
This version imports **stock verl 0.8 as a library** and adds everything through verl's public
extension points — **no patched verl tree**:

| Extension point | What FedAgent plugs in |
|---|---|
| `data.custom_cls` | [`data/agentic_dataset.py`](../data/README.md) — emits env-spec rows instead of static text |
| agent-loop registry (`agent.yaml`) | [`agent_loops/`](../agent_loops/README.md) — `GymTextAgentLoop`, multi-turn rollout |
| Hydra `searchpath` | [`config/fedagent_ppo.yaml`](../config/README.md) — layered on verl's stock `ppo_trainer` |
| interpreter startup (`sitecustomize.py`) | FedProx proximal term, gated on `FEDPROX_MU` |
| process boundary (HTTP) | [`envs/webshop/service/`](../envs/webshop/service/README.md), [`envs/alfworld/service/`](../envs/alfworld/service/README.md) — remote envs |

The benefit: verl 0.8's trainer, FSDP engine, async agent-loop rollout, and model merger are
used **as-is**, so the framework tracks upstream without fork maintenance.

## Two planes

**Control plane** — [`fed/run_fed.py`](../fed/README.md). The federated round loop. It is
verl-agnostic: it never imports verl; a client is just a subprocess
(`python -m fedagent.main_ppo_fed`). It orchestrates subprocesses, FedAvg, and merging.

**In-framework hooks** — `envs/`, `agent_loops/`, `data/`, `fedprox.py`. These run *inside*
the verl client process, reached through the extension points above.

## Code map: `fedagent/` file → role

Every first-party file in the live overlay, grouped by subpackage. Each subpackage also has
its own README (linked) with code-level detail; this table is the one-screen index. There is
**no** legacy `core/` / `eval/` / `scripts/` control plane here — the entire federated loop is
[`fed/run_fed.py`](../fed/README.md) plus the in-process hooks below.

### `fed/` — control plane ([README](../fed/README.md))

| File | Role |
|---|---|
| `fed/run_fed.py` | The federated round loop. Verl-agnostic driver: launches one client **subprocess** per (client, round), starts/stops per-client + val env services, FedAvgs the FSDP shards, merges to HF, advances rounds, runs eval. Functions: `run`, `select_clients`, `run_client`, `fedavg`, `merge_to_hf`, `cleanup_round_checkpoints`, `eval_global`, `start_*_services`. |
| `fed/metrics_logger.py` | Parses each client's verl `training.log` stdout into `json_logs/metrics.json` in the FedAgent plot schema (`[{"step", "metrics"}]`). Restores measurability without forking verl's `Tracking`. |

### `envs/` — env contract + clients ([README](../envs/README.md))

| File | Role |
|---|---|
| `envs/base.py` | `BaseTextEnv` — the per-instance async env contract (`system_prompt` / `reset` / `step`), one env object per dataset row. Aligned with VAGEN's `GymBaseEnv`. |
| `envs/registry.py` | `ENV_REGISTRY` mapping the row's `env_name` → env class; `make_env(...)`. Registers `TinyGuess`, `WebShop`, `ALFWorld`. |
| `envs/tiny_guess.py` | `TinyGuessEnv` — dependency-free in-process guess-the-number env. Wiring smoke test, not part of the research suite. |
| `envs/webshop/webshop_env.py` | `WebShopEnv` — thin async **HTTP client** to the WebShop service. Ferries action text in, formats verl-agent `WEBSHOP_TEMPLATE` observations out. |
| `envs/webshop/service/server.py` | WebShop remote service (FastAPI). Pre-warms a pool of `WebAgentTextEnv`; serves `/create`·`/reset`·`/step`·`/close`; parses actions server-side with the original `webshop_projection`; reads heterogeneity `env_kwargs` from the environment. Runs in the `verl-agent-webshop` conda env. ([README](../envs/webshop/service/README.md)) |
| `envs/webshop/service/run_service.sh` | Launch script for the WebShop service (port, conda env, vendored `engine/` on path). |
| `envs/alfworld/alfworld_env.py` | `AlfworldEnv` — thin async **HTTP client** to the ALFWorld service. Mirrors `WebShopEnv`; uses verl-agent's `ALFWORLD_TEMPLATE_NO_HIS`. |
| `envs/alfworld/service/server.py` | ALFWorld remote service (FastAPI). Builds `AlfredTWEnv` once, pre-warms a pool of `batch_size=1` textworld envs; per-seed game selection via `env.seed(seed)`; parses actions with `alfworld_projection`. Runs in the `verl-agent-alfworld` conda env. ([README](../envs/alfworld/service/README.md)) |
| `envs/alfworld/service/run_service.sh` | Launch script for the ALFWorld service (port, conda env, `$ALFWORLD_DATA` / `$ALF_CONFIG`). |

### `agent_loops/` — rollout ([README](../agent_loops/README.md))

| File | Role |
|---|---|
| `agent_loops/gym_text_agent_loop.py` | `GymTextAgentLoop` (`@register("gym_text")`) — verl `AgentLoopBase` subclass that drives one `BaseTextEnv` per row on verl's native async seam (`reset → generate → decode → env.step → …`). Returns one concat `AgentLoopOutput` with a `response_mask` that is 1 on agent tokens, 0 on observation tokens, so PPO/GRPO trains only on actions. The verl-0.8 replacement for verl-agent's `TrajectoryCollector.multi_turn_loop`. |

### `data/` — dataset hook ([README](../data/README.md))

| File | Role |
|---|---|
| `data/agentic_dataset.py` | `AgenticDataset` — verl `data.custom_cls` that emits one row **per env instance** from an env-spec YAML (`name`/`n_envs`/`max_turns`/`agent_name`/`config`), each with a distinct seed. Non-tensor columns flow to `AgentLoop.run()` as kwargs. `_partition_specs` is the per-client heterogeneity seam (reads `PARTITION_STRATEGY`/`CLIENT_ID`/… → `hetero/`). |

### `hetero/` — heterogeneity constructions ([README](../hetero/README.md))

| File | Role |
|---|---|
| `hetero/webshop_task.py` | **Task-level** Preference (omega): a category-skewed (Dirichlet) goal distribution per client, full catalog. `preference_for_client(...) → goal_idxs`. |
| `hetero/webshop_coverage.py` | **Task-level** Coverage (xi): Beta-sized per-client goal counts with controlled cross-client overlap, full catalog. `coverage_for_client(...)`. |
| `hetero/webshop_hardness.py` | **Task-level** Hardness (xi'): easy-vs-hard skew from a precomputed per-task success file, full catalog. `hardness_for_client(...)` (requires a `trajectories_file`). |
| `hetero/webshop_catalog_split.py` | **Env-level** Variant 1 — Catalog Split: each client gets a disjoint product catalog + goal slice (the hidden-kernel divergence P_i). `load_webshop_data`, `catalog_split_for_client(...)`. |
| `hetero/webshop_env_variants.py` | **Env-level** Variants 2–5: Field-Subset Index, BM25 Reweighting, Lookalike Injection, Rank Wrapper. Emits the service `env_kwargs` overrides for each. |
| `hetero/_beta_sizing.py` | Shared Beta-distribution sizing primitives (`default_r`, `generate_client_sizes`, `assign_with_overlap`) used by Coverage/Hardness. |

> Each construction copies its core partition body **verbatim** from verl-agent's
> `partition_strategy.py` (with `base_seed=42`) so a client's shard is bit-identical to the
> 0.3.1 baseline; only the thin public API around it is new. See [heterogeneity.md](./heterogeneity.md).

### `config/` — Hydra configs ([README](../config/README.md))

| Path | Role |
|---|---|
| `config/fedagent_ppo.yaml` | The training config layered on verl's **stock** `ppo_trainer` (via `hydra.searchpath` → `$VERL_CFG`). Sets `adv_estimator`, `data.custom_cls`, batch sizes; machine paths come from the CLI. |
| `config/agent.yaml` | Agent-loop registry consumed by verl's `AgentLoopManager`: maps `agent_name: gym_text` → `GymTextAgentLoop._target_`. |
| `config/envs/*.yaml` | Env-spec files read by `AgenticDataset` (`tiny_guess`, `webshop_15`, `webshop_15_ppo`, `webshop_15_val`, `alfworld`, `alfworld_val`, …) — the per-run env pool + turn budget. |
| `config/fed_*.yaml` | Top-level **run configs** for `run_fed.py` (one per experiment: smoke, scaled WebShop arms, ALFWorld, centralized/local baselines, FedProx). |
| `config/paper/` | The paper matrix: `uniform/<model>/`, `task_heterogeneity/{grpo,ppo}/`, `env_heterogeneity/<variant>{,_ppo}/`, `decentralized/`. See [reproducing.md](./reproducing.md). |

### Top-level overlay modules

| File | Role |
|---|---|
| `main_ppo_fed.py` | The client entry: `python -m fedagent.main_ppo_fed`. Loads `config/fedagent_ppo.yaml` and runs verl's **stock** `run_ppo`; imports the agent-loop module so its `@register` fires. The verl-0.8 replacement for verl-agent's forked `verl/trainer/main_ppo_fed.py`. |
| `fedprox.py` | The FedProx proximal term as a one-method monkeypatch of `FSDPEngine.optimizer_step` (snapshot global weights `w_t` on first step, add `mu*(w - w_t)` thereafter). Enabled via `FEDPROX_MU`; no verl fork. |
| `EXPERIMENTS.md` | The running experiment log. |
| `README.md` | Package overview ([up one level](../README.md)). |

### Runtime dependencies (outside `fedagent/`)

| Path | Role |
|---|---|
| `envs/{webshop,alfworld}/engine/` | The **vendored WebShop/ALFWorld engines** (+ original `partition_strategy.py`, `*_projection` action parsers). `sys.path`-injected by the env services so the environment MDP is the *same code* the original FedAgent used — now carrying **no verl-agent dependency**. The trainer itself is **stock verl 0.8**. |
| `sitecustomize.py` (repo root) | Auto-imported by CPython at interpreter startup in every process on `PYTHONPATH` (client + Ray workers). Gated on `FEDPROX_MU`, it applies `fedprox.py`'s patch — deliberately **not** a Ray `runtime_env` hook (that clobbered per-worker `CUDA_VISIBLE_DEVICES`). |
| `tools/verl08_migration/aggregate_fedavg_fsdp.py` | The FedAvg core. Run under `torchrun --nproc_per_node=world_size`: each rank averages its own FSDP shard in place across clients and re-saves, byte-structurally identical to a verl checkpoint so the next round loads it unchanged. Shelled out to by `run_fed.py`'s `fedavg`. |

## The federated round loop

`run_fed.py` runs `T` rounds. Each round trains the selected clients as **separate
subprocesses**, FedAvgs their FSDP checkpoints, merges to a HuggingFace model, and the next
round starts from that merged model:

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

`model_1 = base model`; `model_r = round_{r-1}/aggregated/hf` for `r > 1`. PPO
(`adv_estimator=gae`) federates the **critic** the same way, in parallel with the actor.

The relevant functions in `run_fed.py`: `run` (driver), `select_clients`, `run_client`,
`fedavg`, `merge_to_hf`, `cleanup_round_checkpoints`, `eval_global`.

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

The env client (`envs/webshop/webshop_env.py`, `envs/alfworld/alfworld_env.py`) is a **thin
HTTP client**; the heavy WebShop/ALFWorld engine runs in the remote service. See
[envs/](../envs/README.md).

## End-to-end data flow

One trace from the driver down to weights, mapped onto the files above:

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

`metrics_logger.py` runs after each client to emit `json_logs/metrics.json`; `fedprox.py`
(via `sitecustomize.py`, gated on `FEDPROX_MU`) anchors the actor to `model_r` during the
local update.

## Remote env services (and why)

WebShop, ALFWorld, and the trainer have **mutually conflicting dependencies** (WebShop's
Java/pyserini/gym 0.24; ALFWorld's TextWorld/Fast-Downward/torchvision; verl 0.8). So each
environment runs as its **own HTTP service in its own conda env**, one service per client:

```
trainer (fedagent-verl08)  ──HTTP──>  client 0 service (verl-agent-webshop, :8080)
                           ──HTTP──>  client 1 service (verl-agent-webshop, :8081)
                                      ...
                           ──HTTP──>  shared unperturbed VAL service (:8090)
```

`run_fed.py` launches one service per participating client (`start_webshop_services` /
`start_alfworld_services`), waits for each `/health`, and tears them down at the end. The
services `sys.path`-inject the vendored engine from `fedagent/envs/<name>/engine/` — the **same
code the original FedAgent used**, so the environment MDP is unchanged (see
[migration.md](./migration.md)). This isolation is also why the service packages live at the
top level of `fedagent/`, not under `envs/`.

## Heterogeneity injection

`run_fed.py` passes the `partition_strategy` + its knobs to each client's service as env vars
(`PARTITION_STRATEGY`, `OMEGA`, `SIZE_STD`, `SUCCESS_STD`, `ENV_DIV`, `KEEP_RATIO`,
`VARIANT_N`, `CLIENT_ID`, `CLIENT_NUM`, …). The service calls [`hetero/`](../hetero/README.md)
to build *that client's* data shard from the real shuffled `server.goals`. Two levels:
environment (catalog) and task (goal distribution). See [heterogeneity.md](./heterogeneity.md).

## FedProx

When `fedprox_mu > 0`, `run_fed.py` sets `FEDPROX_MU` in the client env. The repo-root
`sitecustomize.py` runs at interpreter startup in **every** process (client + its Ray
workers) and, gated on that var, patches the FSDP optimizer step to add the proximal term.
It is deliberately **not** a Ray `runtime_env` hook (that clobbered verl's per-worker
`CUDA_VISIBLE_DEVICES`). `mu = 0` → plain FedAvg.

## Evaluation

A single **unperturbed** validation service (full env, held-out val split, no heterogeneity)
scores the **aggregated global model** every `test_freq` rounds — plus the base model at
round 0 (`val_before_train`) — at sampling temperature `val_temperature`. `eval_global` runs
a verl `val_only` pass and parses the round→success/reward curve into
`federated_summary.json`. A failed eval never aborts the run (it is measurement, not the loop).

## Outputs

Per run, under `output_dir/`: `round_*/client_*/training.log` + `json_logs/metrics.json`
(FedAgent plot format), `round_*/aggregated/hf` (the round's global model), the per-service
logs, and `federated_summary.json` (the round history + the unperturbed val curve). Consumed
FSDP shards are deleted after each merge (`cleanup_checkpoints`) to bound disk to ~one round.

## See also

- [running.md](./running.md) — how to run it (modes, GPUs, baselines, FedProx, eval)
- [configuration.md](./configuration.md) — every config key
- [reproducing.md](./reproducing.md) — the paper config matrix
- [migration.md](./migration.md) — what changed from verl-agent 0.3.1, and the fidelity record
