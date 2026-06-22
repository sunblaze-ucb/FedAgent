# Heterogeneity

FedAgent's scientific core is a **two-level heterogeneity suite**: a set of
partition constructions that inject *controlled, measurable* statistical
difference across the federated clients, split into two structurally distinct
channels so the paper's headline claim can be **measured** rather than assumed —
federated agent RL is **robust** to **task-level** heterogeneity (the task
descriptor is in the prompt, so a single aggregated policy can condition on it)
but **worst-case non-robust** to **environment-level** heterogeneity (the
transition kernel is hidden, so the policy only senses it through successor
states). This is the **Input-Dynamics Asymmetry**.

A FedAgent client trains on a task-augmented MDP: each episode draws a task
descriptor `tau ~ D_tau` and rolls out under a transition kernel `P`. Cross-client
heterogeneity can enter through **either** channel, and the entire suite is
organized to keep them separable:

- A **task descriptor `tau`** enters the policy through its *input channel* — it
  is literally part of the prompt. The policy can read it, so `tau` is
  **observable**, and FedAvg over task-heterogeneous clients converges to a
  policy that does well on the *union* of task distributions. This is the
  **robust** case.
- A **transition kernel `P`** is implicit in the dynamics. The policy never sees
  `P`; it only senses it through successor states. `P` is **not observable**, so a
  perturbation of `P` can drive each client's optimal policy `pi*_i` apart and
  break naive aggregation. This is the **worst-case non-robust** case.

This guide is both the construction reference (the verified partition math for
each arm, so a researcher can reconstruct it) and the operator's view (the exact
`run_fed` knob and paper config family for each arm). For the module layout and
the env-var bridge read [`../hetero/README.md`](../hetero/README.md); for the
full `run_fed` field reference read [`./configuration.md`](./configuration.md);
for the config-to-figure mapping read [`./reproducing.md`](./reproducing.md).

**Contents**

1. [The two levels](#the-two-levels)
2. [Task-level constructions (observable `tau`)](#task-level-constructions-observable-tau)
3. [Environment-level constructions (hidden `P`)](#environment-level-constructions-hidden-p)
4. [The asymmetric-robustness spectrum (stable -> degrade -> collapse)](#the-asymmetric-robustness-spectrum-stable---degrade---collapse)
5. [Arm -> knob -> paper-config map](#arm---knob---paper-config-map)
6. [Selecting an arm](#selecting-an-arm)
7. [Cross-cutting invariants](#cross-cutting-invariants)

---

## The two levels

| Level | Channel perturbed | Observable to policy? | WebShop arms | ALFWorld arms |
|---|---|---|---|---|
| **Task** | goal distribution `D_tau` over a **shared, unperturbed** environment | **yes** (goal is in the prompt) | `preference`, `coverage`, `hardness`, plus the `task_disjoint` ablation | `preference`, `coverage`, `hardness` |
| **Environment** | **transition kernel `P` / catalog** (the retrieval pipeline) | **no** (only via successor states) | `catalog_split` + 4 retrieval-pipeline variants (`bm25_field_subset`, `bm25_reweight`, `lookalike`, `rank_wrapper`) | *(none — WebShop-specific)* |

Throughout the **task-level** sweep the transition kernel is held fixed (every
client searches the **full 1000-product catalog**); throughout the
**environment-level** sweep the task split is held **uniform** (100 goals/client,
no goal skew). So divergence in each sweep is attributable to that level alone.
Every arm — both levels — is scored on the **same shared unperturbed validation
service** (`WEBSHOP_SPLIT=val`, which ignores `PARTITION_STRATEGY`), which is what
makes the cross-arm curves directly comparable.

`task_disjoint` is the clean ablation that isolates the environment effect: it
reuses Catalog Split's goal-slice math to serve the **same** disjoint goal slice
as `catalog_split` for each client, but over the **full** catalog (it sets
`CATALOG_ASINS = None`). Any divergence between `catalog_split` and
`task_disjoint` at matched `env_div`/`keep_ratio` is therefore due to the hidden
catalog perturbation, not the goal split.

All construction code lives in [`../hetero/`](../hetero/), one module per arm.
Each module copies its partition body **verbatim** from the original verl-agent
`partition_strategy.py` (the *science red line*: `base_seed = 42` hardcoded,
exact copy, no paraphrasing) and adds only a thin `*_for_client(...)` wrapper for
the verl-0.8 remote env service. The shared Beta-sizing primitives
(`default_r` / `generate_client_sizes` / `assign_with_overlap`) live in
[`../hetero/_beta_sizing.py`](../hetero/_beta_sizing.py).

---

## Task-level constructions (observable `tau`)

Clients share one environment tuple (full catalog, fixed `P`) but differ in their
per-client goal distribution `D_tau_i`. Three operationally separable sub-types,
each governed by a **single dispersion knob** so one axis can be moved without
disturbing the other two. Each sub-type has a WebShop and an ALFWorld backend
(ALFWorld derives a goal's category/difficulty from its task-file path rather than
a product field).

| Paper name | Question | `run_fed` knob | Verbatim partition | Endpoints (near-uniform -> extreme) |
|---|---|---|---|---|
| **Preference** | *what kind of task?* | `omega` (`OMEGA`) | `_preference_partition_generic` | `0.01` -> `0.99` |
| **Coverage** | *how many tasks?* | `size_std` (`SIZE_STD`), the paper's `xi` | `coverage_partition` | `256` -> `1` |
| **Hardness** | *how hard are the tasks?* | `success_std` (`SUCCESS_STD`), the paper's `xi'` (+ `trajectories_file`) | `hardness_partition` | `256` -> `1` |

> **Knob-direction caveat.** `size_std`/`success_std` are named like standard
> deviations but are forwarded as the **Beta concentration** `dispersion_s`
> (`generate_client_sizes`). They set spread *inversely*: the **large** endpoint
> (`256`) is near-uniform; the **small** endpoint (`1`) is the extreme imbalance.
> `omega` runs the natural way (`0.01` near-uniform, `0.99` extreme).

### Preference — Dirichlet skew over goal categories

[`../hetero/webshop_task.py`](../hetero/webshop_task.py),
`_preference_partition_generic`. Each client's category mixture is drawn from a
Dirichlet centered on the global category marginal `pi`, then per-category goal
counts are drawn by multinomial sampling:

```python
omega = float(np.clip(omega, 1e-3, 1 - 1e-3))   # clipped into (1e-3, 1-1e-3)
# pi = smoothed global category marginal (raw counts + eps_smooth=0.01, renormalized)
alpha_vec = pi * ((1.0 - omega) / omega)        # concentration scales as (1-omega)/omega
q = rng.dirichlet(alpha_vec); q = q / q.sum()   # this client's per-category probs; E[q] = pi
counts = rng.multinomial(min_samples_per_client, q)   # per-category goal counts
```

`E[q] = pi` exactly, so the **global** category mixture is preserved in
expectation while the **per-client** mixture varies. As `omega -> 0` the
concentration `alpha` grows without bound and every client converges to `pi`
(near-IID); as `omega -> 1`, `alpha -> 0` and each client collapses toward a
one-hot vertex (one client ~ one category). The per-client RNG is
`RandomState(42 + client_id)`; the `fashion` category is sub-sampled to 20%
(`fashion_sample_ratio`) before the marginal is computed; a capacity-overflow
redistribution loop and a top-up pass guarantee each client reaches
`min_goals_per_client`. Legacy configs that pass only `tau` are aliased to
`omega` (`omega` wins if both present); this `tau` is the *old name of the
Preference knob*, unrelated to the MDP task descriptor `tau`.

### Coverage — Beta-dispersed pool sizes with overlap

[`../hetero/webshop_coverage.py`](../hetero/webshop_coverage.py),
`coverage_partition`, on top of
[`../hetero/_beta_sizing.py`](../hetero/_beta_sizing.py). Each client's **pool
size** is drawn from a Beta distribution; goals are then handed out with
controlled cross-client overlap so the **union** of client pools covers the goal
pool. This changes the *spread* of how many goals each client sees without
changing the per-client mean or the global mixture. The sizing band is fixed:

```python
center = 500; low = min_samples_per_client; high = 1000   # per-client size band
rng = np.random.default_rng(42)                            # fixed -> every client agrees
r = default_r(total_size, client_num, low, center, high)   # overlap coefficient r
target_sum = int(round(r * total_size))                    # total assignments to hand out
client_sizes = generate_client_sizes(C=client_num, low=low, center=center,
                                      high=high, dispersion_s=size_std,
                                      target_sum=target_sum, rng=rng)
client_sets, k = assign_with_overlap(total_size, client_sizes, r, rng)
```

Inside `generate_client_sizes`, the Beta is reparameterized by a mean
`mu = (center - low)/(high - low)` and concentration `s = max(dispersion_s, 2e-3)`,
i.e. `Beta(mu*s, (1-mu)*s)`; samples are rescaled to hit `target_sum`, clipped to
`[low, high]`, and rounded with largest-remainder + corrective passes so the
integer sum is exact. `default_r` computes `r = clip(C*center/N, C*low/N,
C*high/N)` — the average per-client size over the pool, clipped to the feasible
band. `assign_with_overlap` replicates each goal `k ~ {floor(r), ceil(r)}` times
and hands the copies to distinct clients with probability proportional to
remaining capacity (plus a shortfall top-up).

> **Note (vs the original prose):** the `overlap_ratio=1.3` argument in
> `coverage_partition`'s signature is an **unused legacy default** — the overlap
> actually applied is the computed `r` from `default_r`, *not* a fixed `1.3`.
> Larger `size_std` -> tighter Beta -> nearly equal pool sizes; `size_std=1` ->
> heavy-tailed sizes -> extreme coverage imbalance.

### Hardness — Beta-skewed easy/hard mix over success labels

[`../hetero/webshop_hardness.py`](../hetero/webshop_hardness.py),
`hardness_partition`. Goals are first bucketed by a **per-task success label**
read from a `trajectories_file` (`task_id -> success`, produced by rolling a
reference policy over the whole catalog); a Beta then sets each client's count of
"success" (easy) goals, and the remainder of its fixed quota is filled with
random goals. The number of goals per client stays constant — only the
*difficulty mix* shifts:

```python
rng = np.random.default_rng(42)
center = min_samples_per_client // 2          # center = half the per-client quota
low = 0; high = min_samples_per_client        # success-count band: 0 .. min_goals
r = default_r(total_size, client_num, low, center, high)
target_sum = int(round(r * total_size))
success_counts = generate_client_sizes(C=client_num, low=low, center=center,
                                        high=high, dispersion_s=success_std,
                                        target_sum=target_sum, rng=rng)
# bucket goals by label, then fill each client: success_counts[client_id] easy
# goals from high_success_data, remainder random, capped at max_samples_per_client
```

Goals are bucketed `high_success` (label `True`) vs `low_success` (label `False`
or unknown); the client draws `success_counts[client_id]` from the easy bucket,
backfills from the hard bucket if short, then fills the rest of its quota
randomly. Larger `success_std` -> uniform difficulty across clients; `success_std=1`
-> extreme (some clients see almost only solvable goals, others almost only hard
ones).

> **Shipped input.** The original **trained-checkpoint** labels ship in
> `data/hardness/` (WebShop + ALFWorld, full train pool), so the Hardness configs run
> as-is; `hardness_for_client` still raises `FileNotFoundError` for a bad path. Labels
> depend on the reference policy, so regenerate **per backbone** with a **trained**
> checkpoint (the base/zero-shot model collapses the easy/hard split) via
> `tools/verl08_migration/gen_hardness_trajectories.py`, which writes labels keyed
> on the **exact** `task_id` formula the partitioner uses:
> ```python
> # webshop_hardness.py: task_id derivation (per goal dict)
> options_hash = int(hashlib.md5(str(sorted(item['goal_options'].items())).encode()).hexdigest(), 16)
> task_id = f"{asin}_{abs(options_hash)}"        # human-goal fallback: md5(instruction_text)
> ```
> The lookup resolves **only** against the env's real goal dicts (which carry
> `goal_options` / `instruction_text`); the offline `data_dir` fallback yields
> asin-only ids and will **not** match an options-hash labels file.

By construction each axis offers target control of its own dispersion, expected
invariance of the other two measures, preservation of the global mixture in
expectation, and joint configurability — so the three can be combined or varied
one at a time.

---

## Environment-level constructions (hidden `P`)

Clients share the (uniform) task split but differ in their **transition kernel
`P_i`**. WebShop's retrieval pipeline factors into **four stages**, and the five
environment variants perturb across them:

1. **content** — *what is in the catalog* (the products search can return);
2. **encoding** — *how a product becomes indexed text* (which fields feed BM25);
3. **matching** — *how a query is scored* (the BM25 ranking function);
4. **rendering** — *how the ranked page is presented* to the agent.

| Variant (paper) | Stage(s) | `run_fed` strategy | Verbatim constructor | Config dir |
|---|---|---|---|---|
| **Catalog Split** (Variant 1) | content | `catalog_split` | `_distractor_disjoint_partition_webshop_v5` | `env_heterogeneity/catalog_split/` |
| **Field-Subset Index** (Variant 2) | encoding | `bm25_field_subset` | `_bm25_variant_partition_webshop` (`fields_only` pool) | `env_heterogeneity/field_subset_index/` |
| **BM25 Reweighting** (Variant 3) | matching | `bm25_reweight` | `_bm25_variant_partition_webshop` (default pool) | `env_heterogeneity/bm25_reweighting/` |
| **Lookalike Injection** (Variant 4) | content + matching | `lookalike` | `_lookalike_injection_partition_webshop` | `env_heterogeneity/lookalike_injection/` |
| **Rank Wrapper** (Variant 5) | rendering | `rank_wrapper` | `_rank_wrapper_partition_webshop` | `env_heterogeneity/rank_wrapper/` |

> **Naming caution.** The Catalog-Split helper's `_v4`/`_v5` suffix is an
> **implementation-revision number of paper Variant 1**, *not* paper Variant 4
> (Lookalike) or Variant 5 (Rank Wrapper). The current `catalog_split` key is
> served by the `_v5` revision (per-client ~100-goal slice, per-client target
> floor); the superseded `_v4` revision (`distractor_disjoint`) shared
> `goals[500:]` and is unused by any reported result.

Catalog Split partitions the **goal set** and returns per-client `goal_idxs`;
Variants 2-5 keep the task split **uniform** and return only an `env_kwargs`
fragment merged into `gym.make`. Per-client variant assignment for Variants 2-5
is deterministic by `client_id` (`RandomState(42 + client_id)`, `chosen =
pool[rng.randint(N)]`), so the same client keeps the same variant across rounds —
required for FedAvg to average comparable policies.

### Catalog Split (Variant 1, content)

[`../hetero/webshop_catalog_split.py`](../hetero/webshop_catalog_split.py),
`_distractor_disjoint_partition_webshop_v5`. Each client gets a protected
per-client floor of **target** ASINs (so every client can still complete its own
goals) plus a per-client **distractor** pool drawn so the catalogs diverge. The
optimal "search -> click -> buy" behavior is unchanged by *which* extra products
are present, so `pi*` stays largely invariant — this is the mildest env
perturbation. The divergence math, knobbed by `env_div` and `keep_ratio`:

```python
# one shared u and one per-client v per ASIN (keyed by ASIN string, not pool index)
asin_to_u = {a: RandomState(base_seed).random()              for a in all_asins}      # shared
asin_to_v = {a: RandomState(base_seed + 1000*client_id).random() for a in all_asins}  # per-client
u = u[distractor_pool]; v = v[distractor_pool]
e = (1.0 - env_div) * u + env_div * v        # env_div=0 -> identical ranking u across clients
n_keep = int(round(keep_ratio * D))          # D = |distractor_pool| ~ 920 (per-client)
include_distractors = [distractor_pool[i] for i in np.argsort(e)[:n_keep]]
catalog_asins = sorted(client_target_asins | set(include_distractors))
```

`env_div in [0,1]` is the **catalog-divergence strength**: at `env_div=0` every
client ranks distractors by the *shared* `u` and keeps the same top-`n_keep`
(catalogs maximally overlap, the near-homogeneous floor); at `env_div=1` each
client ranks by its *own* `v` and the catalogs diverge maximally. `keep_ratio`
sets the **distractor density** (fraction of the ~920-item per-client pool kept).
The keying-by-ASIN-string detail is load-bearing: each client's `distractor_pool`
has different content, so the same ASIN must read the same `u` across clients.
This per-client target floor (vs the legacy full-target floor) widens the
pairwise-Jaccard range and strengthens the `env_div` signal while keeping the
task split consistent with the main experiment.

### Field-Subset Index (Variant 2, encoding) and BM25 Reweighting (Variant 3, matching)

[`../hetero/webshop_env_variants.py`](../hetero/webshop_env_variants.py),
**one** function `_bm25_variant_partition_webshop` serves both — the pool is
selected by `variant_pool` (`"fields_only"` -> Variant 2; default -> Variant 3).
Each client is deterministically assigned one `{fields, k1, b}` config, threaded
into `env_kwargs['bm25_in_memory_config']`; the catalog, goals, and reward are
identical across clients — only the search transition `T(s'|s,a)` differs.

- **Field-Subset Index** (`BM25_VARIANTS_FIELDS_ONLY`): all variants share
  `k1=1.2, b=0.75`; only the **indexed field subset** differs, so the same query
  ranks products differently and the agent must learn per-client query crafting.
  The `N=4` sweep is `full {name,Title,description,features,BulletPoints}`,
  `name {name,Title}`, `desc {description}`, `bullets {BulletPoints}` (entries
  5-8 — `features`, `name_bullets`, `desc_features`, `no_name` — extend it for
  `N=8`).
- **BM25 Reweighting** (`BM25_VARIANTS_DEFAULT`): all variants index the **full**
  field set but use **extreme `(k1, b)` corners** that reshape TF saturation and
  length normalization. The `N=4` sweep is `(1.2, 0.75)` (default), `(1.2, 0.00)`,
  `(0.3, 0.75)`, `(5.0, 0.75)` (entries 5-8 — `(0.1,0.75)`, `(1.2,1.00)`,
  `(2.0,0.50)`, `(0.3,0.00)` — extend it for `N=8`).

### Lookalike Injection (Variant 4, content + matching)

[`../hetero/webshop_env_variants.py`](../hetero/webshop_env_variants.py),
`_lookalike_injection_partition_webshop`. The strongest content attack: each
client gets a per-client set of synthetic **lookalike products** appended to the
base 1000-product catalog (`env_kwargs['extra_products']`), tuned to fool BM25
ranking **and** to defeat one specific reward subterm, so the agent is forced to
check a particular attribute (price, color, ...) to filter the fakes. Because
different clients attack different attributes, their optimal policies diverge
*structurally*. The default `N=2` covers the two reward-validated attacks
(`v_price`, `v_color`); `N=4` adds `v_size`, `v_price_color`. JSON ships under
`data/env_heterogeneity/lookalike_data/` (paths resolved against `PROJECT_ROOT`,
exported by the runner).

### Rank Wrapper (Variant 5, rendering)

[`../hetero/webshop_env_variants.py`](../hetero/webshop_env_variants.py),
`_rank_wrapper_partition_webshop`. Each client's results are post-processed by a
different **wrapper** over the same BM25 base (`env_kwargs['search_engine_variant']`),
breaking any "trust the top position" heuristic while preserving the reward
gradient (the target stays reachable in the candidate set). The `N=4` pool
(`SEARCH_ENGINE_VARIANTS_DEFAULT`): `v_bm25_default` (control),
`v_shuffled_topk` (shuffle the top 50), `v_inverted_topk` (reverse the top-K),
`v_partial_random` (50% of queries return random). A per-client `seed =
base_seed + client_id` makes the shuffle/random behavior differ across clients
sharing a variant.

---

## The asymmetric-robustness spectrum (stable -> degrade -> collapse)

The two levels sit on opposite ends of a **robustness spectrum**, and within each
level the knob value moves an arm along it. This is the paper's central
observation made operational: *task-level heterogeneity is robust; env-level
heterogeneity is worst-case non-robust.*

| Regime | What FedAvg does | Where it lives |
|---|---|---|
| **stable** | the single aggregated policy transfers across clients almost losslessly | **all task-level arms** (`tau` observable); env-level **near-homogeneous floors** (`catalog_split` at `env_div=0.0`) |
| **degrade** | aggregation still helps but the global policy loses accuracy; divergence real but recoverable | env-level arms at **moderate** strength (`catalog_split` `env_div ~ 0.3-0.7`; `bm25_field_subset` / `bm25_reweight`) |
| **collapse** | clients' optimal policies `pi*_i` diverge structurally; naive FedAvg breaks down under GRPO | env-level **worst-case attacks** at full strength (`catalog_split` `env_div=1.0`; `lookalike`, `rank_wrapper`) |

Key facts this spectrum encodes:

- **Task-level arms stay stable across their whole sweep.** Even at the extreme
  endpoints (`omega=0.99`, `size_std=1`, `success_std=1`) the goal descriptor is
  in the prompt, so the aggregated policy reads each client's `tau` and serves the
  union. Task-het arms do **not** reach *collapse*.
- **Env-level severity is ordered by stage and strength.** *Content* perturbation
  (`catalog_split`) leaves `pi*` largely invariant and degrades gracefully; its
  `env_div` knob slides it from the near-homogeneous floor (`0.0`, *stable*)
  through *degrade* (`0.3`, `0.7`) to *collapse* (`1.0`). The *encoding*/*matching*
  variants (`bm25_field_subset`, `bm25_reweight`) sit in *degrade* (real but
  recoverable). The two strongest attacks — *content+matching* (`lookalike`) and
  *rendering* (`rank_wrapper`) — reach *collapse* under GRPO.
- **GRPO -> PPO rescue.** The two collapse-regime attacks (`lookalike`,
  `rank_wrapper`) that break naive GRPO aggregation are *rescued back toward the
  degrade regime under PPO* — the critic absorbs the hidden-dynamics variance. This
  is why every env-het config has a `*_ppo` sibling: the paired runs produce the
  GRPO-vs-PPO halves of the env-heterogeneity figure. The
  `task_disjoint`/`catalog_split` pair is the matched control that attributes any
  collapse to the **hidden catalog**, not the goal split.

---

## Arm -> knob -> paper-config map

Every arm is selected by the single `run_fed` key `partition_strategy` plus that
strategy's knob; `run_fed.py` exports them as env vars to each per-client remote
env service, which dispatches to the matching `*_for_client` constructor. Paper
values are the endpoints actually present under
`config/paper/{task,env}_heterogeneity/`. See
[`./configuration.md`](./configuration.md) for the full field table.

| `partition_strategy` | Level | Env(s) | Knob(s) (`run_fed`) -> env var | Paper values | Config family |
|---|---|---|---|---|---|
| `preference` | task | WebShop, ALFWorld | `omega` -> `OMEGA` | `{0.01, 0.99}` | `task_heterogeneity/{grpo,ppo}/{webshop,alfworld}/...preference_omega-*` |
| `coverage` | task | WebShop, ALFWorld | `size_std` -> `SIZE_STD` | `{256, 1}` | `..._coverage_std-*` |
| `hardness` | task | WebShop, ALFWorld | `success_std` -> `SUCCESS_STD` (+ `trajectories_file` -> `TRAJECTORIES_FILE`) | `{256, 1}` | `..._hardness_success_std-*` |
| `task_disjoint` | task | WebShop | `env_div`, `keep_ratio` -> `ENV_DIV`, `KEEP_RATIO` | matches `catalog_split` | env-effect ablation against `catalog_split` |
| `catalog_split` | environment | WebShop | `env_div`, `keep_ratio` -> `ENV_DIV`, `KEEP_RATIO` | `env_div in {0.0, 0.3, 0.7, 1.0}`, `keep_ratio 0.7` | `env_heterogeneity/catalog_split[_ppo]/...div-*_keep-0.7` |
| `bm25_field_subset` | environment | WebShop | `variant_n` -> `VARIANT_N` | `{4, 8}` | `env_heterogeneity/field_subset_index[_ppo]/...field_subset_index_N-*` |
| `bm25_reweight` | environment | WebShop | `variant_n` -> `VARIANT_N` | `{4, 8}` | `env_heterogeneity/bm25_reweighting[_ppo]/...bm25_reweighting_N-*` |
| `lookalike` | environment | WebShop | `variant_n` -> `VARIANT_N` | `{2, 4}` | `env_heterogeneity/lookalike_injection[_ppo]/...lookalike_injection_N-*` |
| `rank_wrapper` | environment | WebShop | `variant_n` -> `VARIANT_N` | `4` | `env_heterogeneity/rank_wrapper[_ppo]/...rank_wrapper_N-*` |

Notes:

- The env-variant **config directory names** (`field_subset_index`,
  `bm25_reweighting`, `lookalike_injection`) differ from the `partition_strategy`
  **values** (`bm25_field_subset`, `bm25_reweight`, `lookalike`). The strategy
  value is what the service dispatches on.
- `variant_n` is the number of variants in the per-client pool (passed as `N`). A
  value of `0` means "use the constructor default" (4 for bm25/rank, 2 for
  lookalike); the paper configs set it explicitly.
- The multi-point sweeps (`catalog_split` 4-point `env_div`; bm25/field-subset
  `N in {4,8}`; lookalike `N in {2,4}`) exist only in the GRPO directories; each
  `*_ppo` sibling holds a single config (the most-divergent sweep point) for the
  GRPO-vs-PPO comparison.
- **ALFWorld** gets only the env-agnostic task-level subset (`preference` /
  `coverage` / `hardness`); there is no `env_heterogeneity/` ALFWorld arm because
  the WebShop variants perturb WebShop's retrieval pipeline specifically and do
  not transfer.

---

## Selecting an arm

Set `partition_strategy` and the strategy's knob(s) in the `run_fed` YAML, then
launch with `python -m fedagent.fed.run_fed --config <yaml>`. Examples copied
from the paper config tree.

**Catalog Split** (env level), the four-point `env_div` sweep at `keep_ratio: 0.7`:

```yaml
env_kind: webshop
search_return_n: 200          # env-het perturbs the catalog -> paper top-K (>= 100 required)
partition_strategy: catalog_split
env_div: 0.7                  # 0.0 stable floor -> 1.0 collapse
keep_ratio: 0.7
```

**Preference** (task level), the extreme endpoint:

```yaml
env_kind: webshop
partition_strategy: preference
omega: 0.99                   # 0.01 = near-uniform, 0.99 = extreme
```

**Hardness** (task level) needs a per-backbone success-labels file:

```yaml
env_kind: webshop
partition_strategy: hardness
success_std: 1                # 256 = uniform difficulty, 1 = extreme
trajectories_file: data/hardness/qwen2.5-1.5b_webshop_trajectories.json
```

An **env-variant** arm is just a strategy name plus `variant_n`:

```yaml
env_kind: webshop
search_return_n: 200
partition_strategy: bm25_reweight   # or bm25_field_subset / lookalike / rank_wrapper
variant_n: 4
```

The env-variant arms set `search_return_n: 200` (the runner aborts under 100):
raising the BM25 top-K keeps the rendered result page full after aggressive
per-client filtering so a target is never silently dropped. The task-level arms
leave it at the engine default (50), matching the non-het baselines.

---

## Cross-cutting invariants

These hold across **every** arm and are what make the federated runs reproducible
and the cross-arm curves comparable.

- **Seed-42 science red line.** Every partition body is copied verbatim with
  `base_seed = 42` hardcoded. Shared randomness uses `RandomState(42)` /
  `default_rng(42)`; per-client randomness uses `RandomState(42 + client_id)` (or
  `42 + 1000*client_id` for Catalog Split's per-client `v`). Assignment is
  therefore **deterministic by `client_id`** and stable across rounds.

- **Unperturbed validation.** Regardless of the training perturbation, the
  aggregated global model is scored on the shared **unperturbed** val service
  (`WEBSHOP_SPLIT=val`, which ignores `PARTITION_STRATEGY`; the held-out
  `goals[0:VAL_SIZE]` over the full Lucene index / full 1000-product catalog). The
  ALFWorld val service is the analogous full-game-set service
  (`PARTITION_STRATEGY=uniform`). This is the single yardstick every arm is graded
  on.

- **Content-dependent task partitions are deferred to runtime.** The three
  task-level partitioners (`preference` / `coverage` / `hardness`) pick goals by
  **content** (category / size / hardness), so the WebShop service defers them and
  partitions the env's **real, seed-42-shuffled `server.goals`** once the env pool
  is warmed (`_compute_task_partition` from `env.server.goals`) — the served goal
  at index *i* then carries exactly the property the partition selected. The
  environment-level arms are safe to compute at import: `catalog_split` /
  `task_disjoint` use an order-independent contiguous index range, and the
  bm25/lookalike/rank variants keep the goal split uniform and only return an
  `env_kwargs` fragment. Do **not** swap the deferred task partitions onto a
  reconstructed-goal list — the offline fallback is not order-faithful and will
  not match a real labels file.

---

See [`../hetero/README.md`](../hetero/README.md) for the module layout and the
config -> env-var -> constructor bridge, [`./configuration.md`](./configuration.md)
for the complete `run_fed` field reference, and
[`./reproducing.md`](./reproducing.md) for the full config-to-figure mapping.
