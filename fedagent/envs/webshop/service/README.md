# `envs/webshop/service/` — one HTTP WebShop env per federated client

A small [FastAPI](https://fastapi.tiangolo.com/) service that wraps the heavy,
in-process WebShop gym env (`WebAgentTextEnv`) behind a handful of HTTP routes.
The federated runner ([`../../../fed/run_fed.py`](../../../fed/run_fed.py)) launches **one
service per client**, and each service builds its *entire* env pool with that
client's heterogeneity variant. So the design invariant is literally:

> **one service == one client's environment == one hidden transition kernel `P_i`.**

The trainer never imports WebShop. It talks to this service through the thin
async client [`../webshop_env.py`](../webshop_env.py) (`WebShopEnv`), which
reads `WEBSHOP_SERVICE_URL` to find its client's service. See the top-level
[`../../../README.md`](../../../README.md) for the project, and
[`../../../docs/heterogeneity.md`](../../../docs/heterogeneity.md) for the two-level
heterogeneity taxonomy this service realizes.

---

## Why a separate service (and a separate conda env)

WebShop needs a Java/Lucene search stack (`pyserini`/`pyjnius`) plus
`gym 0.24` / `numpy 1.26` / `torch 2.6`, which **hard-conflict** with the verl-0.8
trainer env. `run_service.sh` therefore activates the WebShop-specific env and
launches `server.py` under `uvicorn`:

```bash
# run_service.sh (verbatim shape)
source /software/miniconda3/4.10.3/etc/profile.d/conda.sh
conda activate verl-agent-webshop          # the WebShop env (gym 0.24 / pyserini / JDK / Lucene)
export PYTHONPATH="$REPO_ROOT:$PYTHONPATH"  # so `import fedagent.envs.webshop.service.server` resolves
exec uvicorn fedagent.envs.webshop.service.server:app --host 0.0.0.0 --port "${WEBSHOP_PORT:-8080}" --log-level warning
```

The package's [`__init__.py`](__init__.py) is kept import-clean (no WebShop deps
at import time) precisely so the trainer env can import the *client*
([`../webshop_env.py`](../webshop_env.py)) without dragging in WebShop's conflicting stack.

At startup `server.py` `sys.path`-injects the vendored WebShop engine from
`../engine/webshop/` and loads the original action parser
(`webshop_projection`) **in isolation** — bypassing the `agent_system` package
`__init__` (which would pull verl-0.3.1 / torch). The lifespan handler then
pre-warms a **pool** of `WebAgentTextEnv` instances (`gym.make` is ~26 s each, JVM
+ index startup) so episodes never pay that cost.

---

## Endpoints

Episodes flow `borrow → reset → step* → return`, reusing a pooled env each time.

| Method | Route | Handler | Purpose |
|---|---|---|---|
| `GET`  | `/health` | `health()` | Readiness + this client's shard summary (below). |
| `POST` | `/create` | `create(Sid)` | Borrow an env from the pool (waits if exhausted); bind it to `session_id`. |
| `POST` | `/reset`  | `reset(ResetReq)` | Pick a goal for `session_id` from this client's shard via `seed`; return `obs`, `available_actions`, `goal_id`. |
| `POST` | `/step`   | `step(StepReq)` | Server-side `webshop_projection([text])` → `env.step`; return `obs`, sparse `reward`, `task_score`, `done`, `success`, `is_action_valid`. |
| `POST` | `/close`  | `close(Sid)` | Return the env to the pool for the next episode. |

`/health` returns exactly:

```json
{ "ok": true, "free": <pooled envs>, "sessions": <active>,
  "split": "train|val", "client_id": 0, "partition": "none|<strategy>",
  "catalog_size": <n or null>, "goal_slice": <n or null>,
  "env_variant_keys": [<keys> or null] }
```

`run_fed.py`'s `start_webshop_services()` polls `/health` after launch and logs
`partition` and `catalog_size` once each service is up.

---

## Environment-variable bridge

The service is configured **entirely** through environment variables, set per
client by `start_webshop_services()`. All are read in `server.py`; the search
top-K is read by the underlying engine (see the last section).

| Variable | Meaning | Default |
|---|---|---|
| `WEBSHOP_PORT` | Port `uvicorn` binds (read in `run_service.sh`). | `8080` |
| `WEBSHOP_POOL_SIZE` | Number of pre-warmed `WebAgentTextEnv` in the pool (must be ≥ gen batch). | `4` |
| `WEBSHOP_NUM_GOALS` | Size of the full goal pool (uniform-sampling bound). | `6910` |
| `WEBSHOP_SEARCH_RETURN_N` | BM25 top-K (read by the WebShop engine, not `server.py`). | `50` engine / `200` runner |
| `PARTITION_STRATEGY` | Heterogeneity strategy (see table below); empty ⇒ uniform/unperturbed. | `""` |
| `CLIENT_ID` | This client's id (selects its shard / variant). | `0` |
| `CLIENT_NUM` | Total clients (partition denominator). | `1` |
| `ENV_DIV` | Catalog-Split heterogeneity strength (`catalog_split`/`task_disjoint`). | `0.7` |
| `KEEP_RATIO` | Catalog-Split distractor density. | `0.7` |
| `OMEGA` | Preference Dirichlet spread (`preference`). | `0.5` |
| `SIZE_STD` | Coverage Beta dispersion ξ (`coverage`). | `1.0` |
| `SUCCESS_STD` | Hardness Beta dispersion ξ′ (`hardness`). | `1.0` |
| `VARIANT_N` | Number of env variants in the pool for `bm25_*`/`lookalike`/`rank_wrapper`; empty ⇒ each fn's default. | `""` |
| `TRAJECTORIES_FILE` | Hardness only: REQUIRED `task_id → success` labels file. | `""` |
| `MIN_GOALS_PER_CLIENT` | Floor on goals per client (all goal partitions). | `100` |
| `HOLDOUT_FILE` | Optional Catalog-Split distractor-holdout JSON (`{"asins": [...]}`). | unset |
| `WEBSHOP_SPLIT` | `train` ⇒ goals `[VAL_SIZE:]`; `val` ⇒ shared unperturbed held-out `[0:VAL_SIZE]`. | `train` |
| `WEBSHOP_VAL_SIZE` | Size of the held-out val slice. | `500` |
| `FEDAGENT_LOG_GOAL_ID` | When set, `/reset` also returns each goal's `task_id` (hardness-labelling pass only). | unset |

---

## Goal / data-shard logic

Every pooled env shares the same `env.unwrapped.server.goals` — a **seed-42
reproducible shuffle** of the catalog's goals (the catalog filter does not
perturb that RNG, GPU-verified), so all envs and all clients see an identical
goal order. The split is positional:

- **train** (`WEBSHOP_SPLIT=train`, default): goals `[VAL_SIZE:]` (val holdout
  is offset out so a uniform/centralized client never leaks val goals).
- **val** (`WEBSHOP_SPLIT=val`): the shared **unperturbed** validation env —
  ignores any partition and draws from goals `[0:VAL_SIZE]` on the full catalog,
  so every arm is scored on the same fixed set.

`PARTITION_STRATEGY` then selects this client's shard from those *real shuffled
goals* by calling [`../../../hetero/`](../../../hetero/):

| Strategy | Level | `../../../hetero/` call | Effect |
|---|---|---|---|
| `catalog_split` | env | `catalog_split_for_client` | Disjoint goal slice **+** disjoint catalog (`catalog_filter_asins`). |
| `task_disjoint` | task | `catalog_split_for_client` | Same disjoint slice, **full** catalog (clean ablation of the env effect). |
| `preference` | task | `preference_for_client` | Dirichlet over goal `category` (ω). |
| `coverage` | task | `coverage_for_client` | Beta-sized overlapping slices (ξ). |
| `hardness` | task | `hardness_for_client` | Beta-skewed easy/hard goals (ξ′, needs `TRAJECTORIES_FILE`). |
| `bm25_field_subset` | env | `bm25_variant_for_client(...fields_only)` | Variant 2 — field-subset BM25 index. |
| `bm25_reweight` | env | `bm25_variant_for_client` | Variant 3 — BM25 `k1`/`b` reweighting. |
| `lookalike` | env | `lookalike_injection_for_client` | Variant 4 — adversarial lookalike products. |
| `rank_wrapper` | env | `rank_wrapper_for_client` | Variant 5 — shuffled/inverted/partial-random ranking. |

`catalog_split`/`task_disjoint` and the env-variant strategies are resolved at
**import** (their indices are order-independent). The content-dependent task
strategies (`preference`/`coverage`/`hardness`) are **deferred** to the lifespan
handler (`_compute_task_partition`) and computed from `env.server.goals` once the
pool is warm, so the goal served at index *i* really carries the
category/size/hardness the partition selected. Env-variant strategies leave the
goal split uniform and inject their `env_kwargs` (`bm25_in_memory_config`,
`extra_products`, `search_engine_variant`) into every `gym.make` (deep-copied per
env, since the engine mutates product dicts in place).

**Goal selection per reset.** `/reset` maps its `seed` to a session id:
`val` ⇒ `seed % VAL_SIZE`; a partitioned client ⇒
`CLIENT_GOAL_IDXS[Random(seed).randrange(len)]` (full seed entropy so the
round-threaded base seed varies goals across rounds rather than collapsing onto
the first few); otherwise the uniform train pool `VAL_SIZE + seed % (NUM_GOALS −
VAL_SIZE)`.

**Reward shape (sparse).** `/step` mirrors verl-agent's `envs.py` exactly: the
env's graded score in `[0,1]` is returned only as `task_score` (diagnostic),
while the training `reward` is binary **`{0, 10}`** — `10` iff the episode ends
with a perfect match (`done and score == 1.0`), else `0`. Each step also reports
`is_action_valid`; the small **per-invalid-action penalty** (`coef × #invalid`,
default `0.1`) is applied downstream by the agent loop
([`../../../agent_loops/gym_text_agent_loop.py`](../../../agent_loops/gym_text_agent_loop.py)),
not by the service.

---

## `WEBSHOP_SEARCH_RETURN_N` — why it matters

This knob is **not** read in `server.py`; it is read by the vendored WebShop
engine (`engine.py`: `SEARCH_RETURN_N = int(os.environ.get('WEBSHOP_SEARCH_RETURN_N', 50))`),
which the service process inherits. It is the BM25 **top-K** that a `search[...]`
returns before per-client filtering. The engine then drops any ASIN not in this
client's `catalog_filter_asins`, so under env-level heterogeneity the legacy
default **`50` can filter out the target item entirely** (no reward signal). The
paper — and the runner default in `run_fed.py` (`search_return_n: 200`, exported
as `WEBSHOP_SEARCH_RETURN_N`) — uses **`200`** so the target survives filtering.

---

## Example launch (standalone)

Normally `../../../fed/run_fed.py` launches and tears these down for you. To run one
service by hand (e.g. a `catalog_split` client `1` of `4`):

```bash
WEBSHOP_PORT=8081 WEBSHOP_POOL_SIZE=8 WEBSHOP_SEARCH_RETURN_N=200 \
PARTITION_STRATEGY=catalog_split CLIENT_ID=1 CLIENT_NUM=4 \
ENV_DIV=0.7 KEEP_RATIO=0.7 MIN_GOALS_PER_CLIENT=100 \
bash fedagent/envs/webshop/service/run_service.sh

# in another shell, once /health is up:
curl -s localhost:8081/health   # -> {"ok":true,..,"partition":"catalog_split","catalog_size":...}
```

---

## See also

- [`../../../README.md`](../../../README.md) — FedAgent overview.
- [`../webshop_env.py`](../webshop_env.py) — the trainer-side HTTP client.
- [`../../../hetero/`](../../../hetero/) — the partition functions this service calls.
- [`../../../fed/run_fed.py`](../../../fed/run_fed.py) — `start_webshop_services()` launches one per client.
- [`../../../docs/heterogeneity.md`](../../../docs/heterogeneity.md) — the two-level taxonomy.
