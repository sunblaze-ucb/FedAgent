# `hetero/`: FedAgent's two-level heterogeneity suite

The scientific core of FedAgent. This package builds **per-client data shards** that
inject controlled heterogeneity into the federated WebShop loop, separated into two
structurally different channels so the paper's headline claim can be *measured* rather
than assumed: federated agent RL is **robust** to **task-level** heterogeneity (the
task descriptor is observable in the prompt) but **worst-case non-robust** to
**environment-level** heterogeneity (the transition kernel is hidden). See
[`../docs/heterogeneity.md`](../docs/heterogeneity.md) for the conceptual
treatment (the *Input-Dynamics Asymmetry*) and [`../README.md`](../README.md) for how
this package sits on stock verl 0.8.

Every partition function here is **copied verbatim** from verl-agent's
`agent_system/environments/partition_strategy.py` (the "science red line": exact copy,
`base_seed=42`, deterministic per-`client_id` assignment) so each client's slice is
bit-identical to the 0.3.1 baseline. The only *additions* are the thin public
`*_for_client(...)` wrappers each module exposes for the verl-0.8 WebShop remote
service ([`../envs/webshop/service/`](../envs/webshop/service/)).

> **Exception: `hardness_partition` is corrected, not verbatim.** The upstream body
> had an assembly bug: the step-2 success-shortfall was recomputed *after* the
> step-1 picks were removed from the easy bucket, so it always fired and double-drew
> an extra `|Y_i|` items from the **unsuccessful** pool, then filled from a **mixed**
> pool. That floored each client's success rate at the global rate `g` and cut the
> realized inter-client difficulty spread `Delta^2_hard` to ~25-59% of the intended
> `C_h/(xi'+1)` with a drifting mean. It now implements the paper's `HardnessPartition`
> **literally**: `Y_i` is placed on the easy pool by `assign_with_overlap` (the box's
> `Y_i <- CoveragePartition(Y, ..., xi', r)`: full easy-pool coverage + exact cross-client
> replica budget, vs ~6-9% orphaned by an independent draw), then `X_i = Y_i ∪ F_i` fills
> the remainder **only** from the hard pool. So `rho_i = |Y_i|/L` and the D1/D3 guarantees
> hold. Beta sizing, seeds, and `base_seed=42` are unchanged. See
> [`../docs/bugfixes.md`](../docs/bugfixes.md).

## The two-level taxonomy

| Strategy (`PARTITION_STRATEGY`) | Level | Entry function | Knob (env var) | What it perturbs |
|---|---|---|---|---|
| `catalog_split` | **Environment** | `catalog_split_for_client` | `env_div`, `keep_ratio` (`ENV_DIV`, `KEEP_RATIO`) | Disjoint per-client **catalog** (hidden kernel `P_i`) **+** disjoint goal slice |
| `task_disjoint` | **Task** | `catalog_split_for_client` | `env_div`, `keep_ratio` | The **same** disjoint goal slice but the **full** catalog (the env-effect ablation) |
| `preference` | **Task** | `preference_for_client` | `omega` (`OMEGA`) | Dirichlet-skewed goal **category** distribution; full catalog |
| `coverage` | **Task** | `coverage_for_client` | `size_std` (`SIZE_STD`) | Unequal, overlapping goal-count **coverage**; full catalog |
| `hardness` | **Task** | `hardness_for_client` | `success_std` (`SUCCESS_STD`) | Easy/hard goal mix from success labels; full catalog |
| `bm25_field_subset` | **Environment** (variant 2) | `bm25_variant_for_client` | `variant_n` (`VARIANT_N`) | Search **index** field subset (`bm25_in_memory_config`) |
| `bm25_reweight` | **Environment** (variant 3) | `bm25_variant_for_client` | `variant_n` | BM25 **scoring** (`k1`/`b`) reweighting (`bm25_in_memory_config`) |
| `lookalike` | **Environment** (variant 4) | `lookalike_injection_for_client` | `variant_n` | Adversarial lookalike products (`extra_products`) |
| `rank_wrapper` | **Environment** (variant 5) | `rank_wrapper_for_client` | `variant_n` | Search-engine **ranking** wrapper (`search_engine_variant`) |

**Environment-level** strategies perturb the transition kernel / catalog: the agent
never observes the change directly, only through successor states. **Task-level**
strategies perturb the *goal distribution* over a **shared, unperturbed** environment
(full catalog), and the goal is observable in the prompt. `catalog_split` vs
`task_disjoint` is the clean ablation: identical goal slice, differing **only** by the
catalog filter, so any divergence isolates the environment effect.

> Naming caution (carried from the source): the `_v4`/`_v5` tags inside
> `webshop_catalog_split.py` are **implementation-revision numbers of paper Variant 1
> (Catalog Split)**, *not* the env-variant arms 4/5 (`lookalike` / `rank_wrapper`)
> in `webshop_env_variants.py`.

## Task-level partitioners operate on the *real* shuffled goals

This is the load-bearing correctness invariant. The three task-level wrappers
(`preference_for_client` / `coverage_for_client` / `hardness_for_client`) take an
`env_goals` argument and partition the env's **actual** `server.goals` list at
runtime, the seed-42 shuffled goal dicts (each carrying `category`, `asin`,
`goal_options`, `instruction_text`). The original verl-agent partitions exactly this
list (`partition_dataset(data=goals, ...)` then `goals.index(goal)`), so the served
goal at index *i* carries the property the partition selected. The WebShop service
**defers** these partitions to its `_lifespan` and calls them with `env_goals=
env.server.goals` once the env pool is warmed (the catalog filter does not perturb the
shuffle RNG, GPU-verified).

The `data_dir` branch (which reconstructs goals from `items_shuffle_1000.json` /
`items_ins_v2_1000.json` via `_generate_goal_asins_for_partition`) is an **offline
test fallback only** and is **not order-faithful**; for `hardness` it yields asin-only
dicts whose `task_id == asin`, which will not match an options-hash-labelled
trajectories file.

## The strategies

### `webshop_catalog_split.py`: environment level (Variant 1)

- `catalog_split_for_client(client_id, client_num, *, env_div=0.7, keep_ratio=0.7, min_goals_per_client=100, holdout_file=None, base_seed=42, data_dir=None) -> (catalog_asins, client_goal_idxs)`
- Wraps the verbatim `_distractor_disjoint_partition_webshop_v5`. Cuts a ~100-goal
  slice (via uniform partition), then builds a disjoint catalog: the client's target
  ASINs are always kept; distractors are chosen by a per-ASIN blend
  `e = (1-env_div)·u + env_div·v` keeping the top `keep_ratio·|pool|`. `env_div`
  controls cross-client catalog divergence; `keep_ratio` controls distractor density.
- `task_disjoint` reuses this function and **discards the catalog** (full catalog kept,
  goal slice retained), see `../envs/webshop/service/server.py`.
- Also exports `load_webshop_data(data_dir=None)` and `_generate_goal_asins_for_partition(...)`,
  reused by the task-level modules' offline fallback.

### `webshop_task.py`: task level: **Preference** (knob `omega`)

- `preference_for_client(client_id, client_num, *, omega=0.5, min_goals_per_client=100, start_idx=500, env_goals=None, data_dir=None) -> List[int]`
- Wraps verbatim `_preference_partition_generic`: draws `q_i ~ Dir(pi·(1-omega)/omega)`
  then `counts ~ Multinomial(L; q_i)`. `E[q_i]=pi` exactly; category spread grows with
  `omega`. Partitions the train pool `goals[start_idx:]` by `category`; returns absolute
  goal indices.

### `webshop_coverage.py`: task level: **Coverage** (knob `size_std`)

- `coverage_for_client(client_id, client_num, *, size_std, min_goals_per_client=100, start_idx=500, env_goals=None, data_dir=None) -> List[int]`
- Wraps verbatim `coverage_partition`. Draws each client's goal **count** from a Beta
  distribution (`size_std` forwarded as `dispersion_s`) and hands out goals with
  cross-client overlap so the union covers the pool while individual slices are unequal.
  Uses all three `_beta_sizing` helpers.

### `webshop_hardness.py`: task level: **Hardness** (knob `success_std`, needs a labels file)

- `hardness_for_client(client_id, client_num, *, success_std, trajectories_file, min_goals_per_client=100, start_idx=500, env_goals=None, data_dir=None) -> List[int]`
- Wraps the **corrected** `hardness_partition` (see the exception note at the top).
  Reads per-task success labels from `trajectories_file` (`task_id -> success`), buckets
  goals into high/low success, and a Beta distribution (`success_std`) sets each client's
  count of easy ("success") goals `|Y_i|`; the remainder of the fixed quota `L` is filled
  **only from the hard (low-success) bucket**, so `rho_i = |Y_i|/L`. `task_id` is derived
  as `f"{asin}_{md5(goal_options)}"`
  (with `instruction_text` / bare-`asin` fallbacks), the **same** formula the labelling
  pass records, so the lookup resolves only against real goal dicts.
- **`trajectories_file` is required**: there is no usable default in this package, so
  the wrapper raises if it is missing/absent (the verbatim body also raises
  `FileNotFoundError`). The module-level `path_cfg` stand-in exists solely so the
  verbatim body imports cleanly; its default-path branch is never taken here.

### `webshop_env_variants.py`: environment level: variant arms 2–5

Ports paper Variants 2–5; each `*_for_client` returns the **env_kwargs fragment** the
service merges into `gym.make` (no goal partition, the task split stays uniform):

- `bm25_variant_for_client(client_id, client_num, *, N=4, variant_pool=None, ...) -> {'bm25_in_memory_config': {...}}`,
  one function serves **both** `bm25_reweight` (Variant 3, default `k1`/`b` pool) and
  `bm25_field_subset` (Variant 2, `variant_pool='fields_only'`, set via the
  `BM25_VARIANT_POOL` env var). SimServer routes through `InMemoryBM25Searcher`.
- `lookalike_injection_for_client(client_id, client_num, *, N=2, project_root=None, ...) -> {'extra_products': [...]}`,
  Variant 4: injects adversarial lookalike products (price/color/...) appended to the
  catalog before indexing.
- `rank_wrapper_for_client(client_id, client_num, *, N=4, ...) -> {'search_engine_variant': {...}}`,
  Variant 5: swaps the search-engine **type** (`bm25_default` / `bm25_shuffle` /
  `bm25_invert` / `bm25_partial`), with a per-client `seed`.

Each client is deterministically assigned one of `N` variants via
`RandomState(base_seed + client_id)`. `N` defaults match the service:
`bm25_*`→4, `lookalike`→2, `rank_wrapper`→4 (overridable with `VARIANT_N`).

## `_beta_sizing.py`: shared Beta-sizing helpers

Verbatim helpers used by the Beta-based task partitioners:

- `default_r(N, C, low, center, high)`: overlap coefficient (assignments / samples),
  clipped to the feasible band.
- `generate_client_sizes(C, low, center, high, dispersion_s, target_sum, rng)`: draws
  `C` per-client sizes from a Beta reparameterized by mean + `dispersion_s`, rescaled to
  `target_sum` and integer-rounded to match the sum exactly.
- `assign_with_overlap(N, sizes, r, rng)`: hands samples to clients with cross-client
  overlap (`Coverage` only).

**Coverage** uses all three; **Hardness** uses `default_r` + `generate_client_sizes`
(to size the per-client easy-goal counts).

## End-to-end invocation

The federated runner [`../fed/run_fed.py`](../fed/run_fed.py) starts **one WebShop
remote service per client** and passes `PARTITION_STRATEGY` plus the knob env vars into
each service process (`start_webshop_services`). The service
([`../envs/webshop/service/server.py`](../envs/webshop/service/server.py)) reads those env vars and
calls the matching `*_for_client` function to build that client's shard:

```
run_fed.py ──env vars──▶ envs/webshop/service/server.py ──▶ hetero/*_for_client(...) ──▶ shard
  (per-client service)        (one client's env)              (this package)
```

- `catalog_split` / `task_disjoint` → computed at **import** (`catalog_split_for_client`;
  contiguous index range, order-independent). `catalog_split` keeps the catalog filter;
  `task_disjoint` drops it.
- `preference` / `coverage` / `hardness` → **deferred** to runtime and computed from
  `env.server.goals` (content-dependent; see invariant above).
- `bm25_field_subset` / `bm25_reweight` / `lookalike` / `rank_wrapper` → env_kwargs
  fragment merged into `gym.make`; goals stay uniform.

**Env vars consumed** (set by `run_fed.py`, read by `server.py` / these modules):
`PARTITION_STRATEGY`, `CLIENT_ID`, `CLIENT_NUM`, `ENV_DIV`, `KEEP_RATIO`, `OMEGA`,
`SIZE_STD`, `SUCCESS_STD`, `VARIANT_N`, `TRAJECTORIES_FILE`, `MIN_GOALS_PER_CLIENT`
(plus `HOLDOUT_FILE` and `BM25_VARIANT_POOL`, read directly by the service / variant
pool selector). The shared unperturbed **validation** service runs with
`PARTITION_STRATEGY=""` so every arm is scored on the same fixed held-out goals.
