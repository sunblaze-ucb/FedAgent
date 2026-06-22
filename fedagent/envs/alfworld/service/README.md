# `envs/alfworld/service/` — one HTTP ALFWorld engine per federated client

A tiny [FastAPI](https://fastapi.tiangolo.com/) service that wraps the real
ALFWorld / TextWorld engine behind HTTP, so the verl-0.8 trainer can drive
embodied episodes without inheriting ALFWorld's conflicting dependency stack.
It is the ALFWorld twin of [`../../webshop/service/`](../../webshop/service/) and is
launched, one process per client, by [`../../../fed/run_fed.py`](../../../fed/run_fed.py).

## Why a separate service

ALFWorld needs the **TextWorld + Fast-Downward planning stack** plus `alfworld`,
`gymnasium`, and pinned `torch`/`torchvision` — none of which are compatible with
the trainer's `fedagent-verl08` env (see the [root README](../../../README.md)).
So the heavy engine runs in its **own conda env** (`verl-agent-alfworld`, holding
the vendored engine under [`../engine/`](../engine/), which `server.py`
injects onto `sys.path`), and the trainer talks to it over HTTP through the thin
client [`../alfworld_env.py`](../alfworld_env.py). Only that client is imported
trainer-side; importing this package never pulls the ALFWorld deps.

## Architecture: one service == one client's game shard

The federated abstraction is **one service process == one client's game shard /
hidden transition kernel `P_i`**. When `CLIENT_NUM > 1`, the whole env pool is
built from this client's slice of the **train** games (the env arm of the paper's
Input-Dynamics Asymmetry; eval splits stay full). `../../../fed/run_fed.py`'s
`start_alfworld_services()` launches one `run_service.sh` per client on
`alfworld_base_port + client_id`, sets `ALFWORLD_SERVICE_URL` per client, waits on
each `/health`, and tears them all down at the end of the run.

`run_service.sh` activates `verl-agent-alfworld`, exports `ALFWORLD_DATA`
(default `~/.cache/alfworld`, the PDDL + `game.tw-pddl` files fetched by
`alfworld-download`) and `PYTHONPATH`, then `exec`s:

```bash
uvicorn fedagent.envs.alfworld.service.server:app --host 0.0.0.0 --port "$PORT" --log-level warning
```

On startup (`_lifespan`) the service builds the `AlfredTWEnv` **once** (it walks
`$ALFWORLD_DATA` collecting solvable games — the slow part) and pre-warms a
**pool** of `POOL_SIZE` single-instance TextWorld gym envs
(`AlfredTWEnv.init_env(batch_size=1)`) so per-episode registration cost is paid
up front. Episodes then run as **borrow → reset → step\* → return** against the
pool. All TextWorld ops are serialized under one process-global lock
(`_TW_LOCK`): TextWorld's PDDL grammar parser (tatsu) is a shared mutable
singleton that corrupts under concurrent `reset`/`step`. Transitions are
millisecond-fast next to LLM generation, so the pool's real benefit (overlapping
rollouts) is preserved.

### Endpoints

| Method | Route | Handler | Purpose |
|---|---|---|---|
| `GET`  | `/health` | `health()` | Liveness + shard report (see below) |
| `POST` | `/create` | `create(Sid)` | Borrow a pooled env for `session_id` (blocks if exhausted) |
| `POST` | `/reset`  | `reset(ResetReq)` | `env.seed(seed)` then `env.reset()`; returns `obs`, `admissible_commands`, `gamefile` |
| `POST` | `/step`   | `step(StepReq)` | Parse text → `env.step`; returns `obs`, `reward`, `done`, `admissible_commands`, `success`, `is_action_valid` |
| `POST` | `/close`  | `close(Sid)` | Return the env to the pool for the next episode |

`/health` returns
`{ok, free, sessions, num_games, split, task_types, partition, client_id, client_num, alfworld_data}`
— so a caller (and `start_alfworld_services()`, which logs `partition` and
`num_games`) can confirm the shard before training.

Per-episode game selection is deterministic: TextWorld's `reset()` takes no game
argument (it advances a shuffled iterator), so `/reset` calls `env.seed(seed)`
first, mapping each `seed` value to a fixed game — the analog of WebShop's
per-seed goal selection. `/step` parses the model's action text **server-side**
with the original `alfworld_projection` (loaded in isolation), runs one PDDL
action, and computes the reward.

**Reward** is episode success only: `reward = 10.0 * float(won)` (the text-only
`compute_reward`), where `won` is the engine's `info["won"]`.

## Environment-variable bridge

Every knob is read from the environment in `server.py` (set per client by
`start_alfworld_services()`). `ALFWORLD_PORT` is consumed by `run_service.sh`.

| Variable | Default | Meaning |
|---|---|---|
| `ALFWORLD_PORT` | `8081` | Port `uvicorn` binds (read in `run_service.sh`). |
| `ALFWORLD_POOL_SIZE` | `4` | Number of pre-warmed TextWorld envs (concurrency cap; must be ≥ gen batch). |
| `ALFWORLD_DATA` | `~/.cache/alfworld` | Game-data root; expanded inside config paths (exported by `run_service.sh`). |
| `ALF_CONFIG` | verl-agent's bundled `config_tw.yaml` | ALFWorld config (env type `AlfredTWEnv`, max 50 steps). |
| `ALFWORLD_TRAIN_EVAL` | `train` | Game split: `train` \| `eval_in_distribution` \| `eval_out_of_distribution`. |
| `ALFWORLD_TASK_TYPES` | `""` (all 6) | Comma-separated task-type IDs to keep (the 6-way eval breakdown); applied via the engine's native `env.task_types` filter. |
| `PARTITION_STRATEGY` | `uniform` | Game-shard strategy for this client (see below). |
| `CLIENT_ID` | unset → `None` | This client's 0-based id; with `CLIENT_NUM` selects the train shard. |
| `CLIENT_NUM` | unset → `None` | Total clients; `>1` shards the train split, `1`/unset = full set. |
| `MIN_GOALS_PER_CLIENT` | `100` | Minimum games per client (passed as `min_games_per_client`). |
| `OMEGA` | `""` | `preference` spread ω, forwarded only when strategy is `preference`. |
| `SIZE_STD` | `""` | `coverage` set-size dispersion, forwarded only when strategy is `coverage`. |
| `SUCCESS_STD` | `""` | `hardness` success-count dispersion, forwarded only when strategy is `hardness`. |
| `TRAJECTORIES_FILE` | `""` | `hardness` task_id→success labels file, forwarded only when strategy is `hardness`. |

Note `ALFWORLD_SERVICE_URL` is read **trainer-side** by `../alfworld_env.py`,
not by this service.

## Game-shard logic

`_build_base_env()` loads `ALF_CONFIG`, optionally narrows
`config["env"]["task_types"]` to `ALFWORLD_TASK_TYPES`, then constructs
`AlfredTWEnv` with `train_eval`, `client_id`, `client_num`,
`partition_strategy`, `min_games_per_client`, and the strategy-specific kwargs
from `_partition_kwargs()`. The engine collects solvable games from the chosen
split and — for the **train** split with `client_num > 1` — keeps only this
client's partition; eval splits always use the full dataset. `num_games` (the
resulting shard size) is reported on `/health`.

`_partition_kwargs()` forwards only the kwargs the chosen strategy expects (the
upstream partition functions reject unexpected kwargs):

| `PARTITION_STRATEGY` | Level | Extra kwargs forwarded |
|---|---|---|
| `uniform` | — | none (even split) |
| `env_disjoint` | environment | none |
| `preference` | task | `omega` (from `OMEGA`) |
| `coverage` | task | `size_std` (from `SIZE_STD`) |
| `hardness` | task | `success_std`, `trajectories_file` |

This is **narrower than WebShop**: the supported set is exactly what the engine's
`partition_dataset` accepts for ALFWorld (`uniform`, `preference`, `coverage`,
`hardness`, `env_disjoint`). WebShop's catalog-split and BM25 / lookalike / rank
transition variants do **not** apply here (see [`../../../hetero/`](../../../hetero/) for the
WebShop-side constructions).

The **six ALFWorld task types** (used by `ALFWORLD_TASK_TYPES` and by the
`preference` partition's category axis) are:

| ID | Task type |
|---|---|
| 1 | `pick_and_place_simple` (Pick) |
| 2 | `look_at_obj_in_light` (Look) |
| 3 | `pick_clean_then_place_in_recep` (Clean) |
| 4 | `pick_heat_then_place_in_recep` (Heat) |
| 5 | `pick_cool_then_place_in_recep` (Cool) |
| 6 | `pick_two_obj_and_place` (Pick2) |

## Running standalone

The service is normally launched by `../../../fed/run_fed.py`, but it runs on its own:

```bash
# One client's shard of an 8-client uniform train split, on port 8200.
conda activate verl-agent-alfworld
ALFWORLD_PORT=8200 ALFWORLD_POOL_SIZE=4 \
ALFWORLD_TRAIN_EVAL=train \
PARTITION_STRATEGY=uniform CLIENT_ID=0 CLIENT_NUM=8 MIN_GOALS_PER_CLIENT=100 \
ALFWORLD_DATA="$HOME/.cache/alfworld" \
bash fedagent/envs/alfworld/service/run_service.sh

# Confirm the shard, then drive an episode.
curl -s localhost:8200/health        # -> {... "num_games": N, "split": "train", "partition": "uniform" ...}
```

## Files

- `server.py` — FastAPI app: pool lifecycle, env-var bridge, `/health` `/create` `/reset` `/step` `/close`.
- `run_service.sh` — activates `verl-agent-alfworld`, exports `ALFWORLD_DATA` / `PYTHONPATH`, `exec`s `uvicorn`.
- `__init__.py` — package marker; never imported trainer-side (only the sibling `../alfworld_env.py` client is), so the trainer env never pulls ALFWorld deps.

## See also

- [`../../../README.md`](../../../README.md) — project overview and two-conda-env setup.
- [`../alfworld_env.py`](../alfworld_env.py) — the trainer-side HTTP client (`AlfworldEnv`).
- [`../../webshop/service/`](../../webshop/service/) — the WebShop service this mirrors.
- [`../../../hetero/`](../../../hetero/) — WebShop heterogeneity constructions (ALFWorld uses the engine's native partitioner).
- [`../../../fed/run_fed.py`](../../../fed/run_fed.py) — federated runner; `start_alfworld_services()` launches one service per client.
