# Reproducing the paper

This is the per-experiment reproduction guide for the FedAgent **verl-0.8
overlay** — the thin layer that re-runs the paper's config matrix on
**unmodified verl 0.8**. Every cell of the matrix is a single YAML under
[`../config/paper/`](../config/paper/), and every cell runs with one command:

```bash
python -m fedagent.fed.run_fed --config fedagent/config/paper/<...>.yaml
```

Read this together with [`../config/README.md`](../config/README.md) (the four
config types and the `paper/` naming convention), [`../fed/README.md`](../fed/README.md)
(the runner internals — the round loop, FedAvg, baselines, eval),
[`./heterogeneity.md`](./heterogeneity.md) (the two-level heterogeneity suite the
het arms instantiate), and [`./running.md`](./running.md) (the hardware knobs and
CLI overrides). This is **scientific-equivalence** reproduction, not bit-identical
— see [the fidelity note](#scientific-equivalence-not-bit-identical).

---

## Prerequisites

- **Conda env `fedagent-verl08`** (py3.12, stock verl 0.8). Activate it first;
  `run_fed` sets `PYTHONPATH` to the repo root so `fedagent` and the root
  `sitecustomize.py` (FedProx) are importable in every subprocess it spawns.
- **A 4-GPU node.** The `paper/` configs pin `n_gpus_per_node: 4` (FSDP world
  size 4); `--n-gpus` overrides it.
- **The env service env.** WebShop and ALFWorld arms talk to one remote HTTP
  service per client; `run_fed` **launches the services itself**, but their conda
  env / data must be installed and on PATH. `tinyguess` runs in-process.
- **Models.** Each config sets `model_path` to an HF id
  (e.g. `Qwen/Qwen2.5-1.5B-Instruct`) which auto-downloads. On an offline cluster
  pass `--model-path <local snapshot>` to point at a pre-fetched directory.

> ### ℹ️ The Hardness arm uses a shipped reference-labels file
>
> The **task-heterogeneity Hardness** configs (`p-hardness_success_std-*`) are the
> **only** cells with a required external input: each references a
> `trajectories_file` — a `task_id`→success-label map from a reference policy.
> **These ship in `data/hardness/`** (the original **trained-checkpoint** labels —
> Qwen2.5-1.5B fine-tuned, full train pool: WebShop 6,402 goals / 27.8 % easy,
> ALFWorld 3,553 games / 59.4 % easy), so the Hardness cells run out of the box.
>
> The eight Hardness configs reference exactly two paths —
> `data/hardness/qwen2.5-1.5b_webshop_trajectories.json` and
> `data/hardness/qwen2.5-1.5b_alfworld_trajectories.json` (the het backbone is
> Qwen2.5-1.5B for both envs). To regenerate with a different backbone, use a
> **trained** checkpoint as the reference (NOT the base instruct model — zero-shot
> strictly succeeds on only ~1.4 % of WebShop goals, which collapses the easy/hard
> split):
>
> ```bash
> python -m tools.verl08_migration.gen_hardness_trajectories \
>   --config fedagent/config/fed_webshop_scaled_hardness.yaml \
>   --model  <trained Qwen2.5-1.5B checkpoint> --num-goals 6410 \
>   --output fedagent/data/hardness/qwen2.5-1.5b_webshop_trajectories.json
> ```
>
> (ALFWorld labels come from the original verl-agent inference pipeline.) The
> schema is `{"trajectories": [{"task_info": {"task_id": ...}, "traj_info":
> {"success": ...}}, ...]}`. See
> [`../data/hardness/README.md`](../data/hardness/README.md) and
> [`./heterogeneity.md`](./heterogeneity.md#the-hardness-labels-file).

- **ALFWorld arms** drive episodes at `max_turns: 50` (the original
  `max_steps=50`) paired with a **widened context window**
  (`rollout.max_model_len=16384`, `response_length=8192`). This is flagged
  **GPU-VERIFY** in `config/envs/alfworld.yaml`: confirm no OOM / prompt
  truncation at 50 turns on your GPUs, and raise `max_model_len` if verbose rooms
  truncate before `done`.

---

## The one-command run pattern

Every command below is the full invocation; the only thing that changes is the
config path. CLI flags (`--rounds --clients --n-gpus --base-seed --fedprox-mu
--local-client-id --model-path`) override the YAML.

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

The federation protocol is baked into the `paper/` configs and matches the paper:
**N = 100** clients (`total_clients`), **M = 2** sampled per round
(`clients_per_round`), **T = 70** rounds (`total_rounds`), **E = 3** local epochs
(`epochs_per_round`). Each round trains the selected clients from the previous
round's merged FedAvg model, re-aggregates, and (every `test_freq` rounds) scores
the global model on the shared unperturbed val set.

---

## The experiment matrix

176 configs total under `config/paper/`, mirroring the original paper structure.
The **main table is the 4-backbone uniform sweep across WebShop + ALFWorld**; the
heterogeneity and decentralized families are run on a single backbone
(Qwen2.5-1.5B-Instruct).

| Family | Config dir | Backs (paper artifact) | Backbones | Count |
|---|---|---|---|---|
| **[Uniform (main)](#1-uniform-the-main-table)** | `uniform/<Model>/{main,main_seed1,main_seed2}/{grpo,ppo}/` | Main table **FedAgent** rows + the training-dynamics curve | 4 | (in 112) |
| **[Uniform (baselines)](#1-uniform-the-main-table)** | `uniform/<Model>/{centralized,local_client1-3}/{grpo,ppo}/` | Main table **Centralized** + **Local Agent** rows | 4 | (in 112) |
| **[Env heterogeneity](#2-environment-level-heterogeneity-the-worst-case-non-robust-study)** | `env_heterogeneity/<strategy>[_ppo]/` | The env-variant figure (worst-case non-robust) | Qwen2.5-1.5B | 16 |
| **[Task heterogeneity](#3-task-level-heterogeneity-the-robust-study)** | `task_heterogeneity/{grpo,ppo}/{webshop,alfworld}/` | The task-het figure, 6 sub-type × benchmark panels (robust) | Qwen2.5-1.5B | 24 |
| **[Decentralized](#4-decentralized-protocol-ablations)** | `decentralized/{ep_per_round_change,samples_change,selected_cl_change}/{grpo,ppo}/` | The protocol-sensitivity ablation figure | Qwen2.5-1.5B | 24 |

The `uniform/` family (112) is the four backbones — `Qwen2.5-1.5B-Instruct`,
`Qwen2.5-3B-Instruct`, `Qwen2.5-7B-Instruct`, `Llama-3.2-3B-Instruct` — each with
7 run kinds (`main`, `main_seed1`, `main_seed2`, `centralized`, `local_client1-3`)
× `{grpo, ppo}` × `{webshop, alfworld}`. The het / decentralized families are
Qwen2.5-1.5B only. `config/paper/` mirrors the original tree's structure and
naming; contents are verl-0.8 `run_fed` configs, regenerable with
`tools/verl08_migration/gen_paper_configs.py` (see
[`../config/README.md`](../config/README.md)).

Every `paper/` filename is self-describing; the trailing `p-*` token is the only
thing that changes within a heterogeneity sweep:

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

**Backs:** the headline table — the **FedAgent**, **Centralized**, and **Local
Agent** rows, for all four backbones, on both benchmarks, under both algorithms —
plus the training-dynamics validation-success curve (FedAgent vs Centralized on
Qwen2.5-1.5B).

### Layout

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

Each leaf holds exactly two configs, one per benchmark (`fed_webshop_*.yaml`,
`fed_alfworld_*.yaml`).

### Row → config mapping

| Table row | Config subdir | Federation shape (filename) | Mode selected by |
|---|---|---|---|
| **FedAgent** | `main/`, `main_seed1/`, `main_seed2/` | `total-100_cl-per-rd-2_rd-70_ep-per-cl-3` | `total_clients: 100` (FedAvg) |
| **Centralized** | `centralized/` | `total-1_cl-per-rd-1_rd-70_ep-per-cl-3` | `total_clients: 1` |
| **Local Agent** | `local_client{1,2,3}/` | `total-100_cl-per-rd-1_rd-70_ep-per-cl-3` | `local_client_id ≥ 0` |

All three rows hold the **total optimization budget fixed at T·E = 70·3 = 210
local epochs** so the comparison is compute-matched: Centralized trains on the
pooled data (FedAvg of a single client is the identity), Local Agent pins one
client's shard every round with no federation, and FedAgent distributes the same
210 epochs across `M = 2` clients over `T = 70` rounds with FedAvg between rounds.
The three Local indices **21 / 42 / 84** are the goal shards those configs pin
under the deterministic partition.

> **Why T = 70 for the baselines (not 1 round × 210 epochs)?** The original ran
> the baselines as a single 210-epoch round. In this overlay goal variety is
> drawn **per round** (the round-threaded data seed
> `FEDAGENT_BASE_SEED = base_seed + round*100 + client_id` re-draws each client's
> goals every round), so a 1-round baseline would repeat one goal draw. Keeping
> **70 rounds** reproduces the same goal coverage at the same 210-epoch budget;
> the per-round FedAvg of one client/shard is a no-op. See
> [`../fed/README.md`](../fed/README.md#baseline-modes).

### Run

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

### Notes

- **Backbones:** swap the `<Model>` directory to reproduce another block of the
  table; the model is pinned at `model_path` inside each config (offline:
  `--model-path <snapshot>`).
- **GRPO vs PPO:** the `grpo/` configs back the main text (GRPO, group size
  **G = 8**); the sibling `ppo/` configs back the PPO appendix and set
  `adv_estimator: gae` so the **critic is federated alongside the actor**.
- **Training-dynamics curve:** built from the Qwen2.5-1.5B `main/grpo` and
  `centralized/grpo` `val_curve`s (see [Outputs](#outputs)).

---

## 2. Environment-level heterogeneity: the worst-case-non-robust study

**Backs:** the WebShop env-variant figure (GRPO and PPO side by side).
Environment-level heterogeneity enters through the **transition kernel / catalog**
— the policy only senses it through successor states, *not* from the prompt — so
the federated objective is **worst-case non-robust** to it (the paper's negative
result). The task partition is held **uniform** across every env-level run, so any
divergence is attributable to the transition perturbation alone. **WebShop only**
(ALFWorld has no catalog/search to perturb), Qwen2.5-1.5B only. WebShop's
search/transition pipeline factors into four stages, and the five strategies
perturb across them.

| Strategy (dir) | Pipeline stage | Knob | Sweep points (GRPO) | PPO sibling |
|---|---|---|---|---|
| `catalog_split/` | content | `env_div`, `keep_ratio` | `div ∈ {0.0, 0.3, 0.7, 1.0}`, `keep 0.7` | `div 1.0` only |
| `field_subset_index/` | encoding | `variant_n` | `N ∈ {4, 8}` | `N 4` only |
| `bm25_reweighting/` | matching | `variant_n` | `N ∈ {4, 8}` | `N 4` only |
| `lookalike_injection/` | content + matching | `variant_n` | `N ∈ {2, 4}` | `N 4` only |
| `rank_wrapper/` | rendering | `variant_n` | `N 4` | `N 4` |

That is 11 GRPO + 5 PPO = **16** configs. Note the asymmetry: the GRPO
directories sweep multiple points, but every `*_ppo` directory holds only the
**single most-divergent point** used for the GRPO-vs-PPO contrast — do not expect
a full PPO sweep. The directory/filename token (e.g. `bm25_reweighting`,
`field_subset_index`) mirrors the original paper name; the value `run_fed`
actually consumes is the short strategy id (`bm25_reweight`, `bm25_field_subset`,
`lookalike`, `rank_wrapper`, `catalog_split`).

These arms set `search_return_n: 200` (the paper's BM25 top-K) because perturbing
the catalog/search would otherwise drop targets out of reach; the uniform, task-
het, decentralized, and baseline WebShop runs use the engine default `50`, which
is what matches the original non-het numbers.

### Run

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

### Notes

- **Validation is always on the UNPERTURBED WebShop environment** (`val_env_spec`
  forces perturbation kwargs off), so the metric isolates post-aggregation
  generalization, not per-client overfitting.
- To reproduce a figure point, run **both** the GRPO config and its `*_ppo`
  sibling, 3 seeds each.
- See [`./heterogeneity.md`](./heterogeneity.md#the-arms) for the per-stage
  construction and the per-client variant-assignment seeding.

---

## 3. Task-level heterogeneity: the robust study

**Backs:** the task-het figure — **6 panels**, one per (sub-type × benchmark):
**Preference**, **Coverage**, **Hardness**, each on WebShop and ALFWorld. Task-
level heterogeneity enters the policy **through the prompt** (the task descriptor
is observable), so the federated objective is **robust** to it (the paper's
positive result). The base federation shape is identical to the uniform main run;
**only the partition strategy differs.** Qwen2.5-1.5B, both benchmarks, both
algorithms.

### Layout

```
config/paper/task_heterogeneity/
  grpo/ {webshop,alfworld}/        # 6 configs each
  ppo/  {webshop,alfworld}/        # 6 configs each      = 24
```

Each leaf holds the two endpoints of each sub-type:

| Sub-type | Strategy | Filename token | Endpoints (near-uniform → extreme) |
|---|---|---|---|
| **Preference** | `preference` | `p-preference_omega-*` | `omega 0.01` → `omega 0.99` |
| **Coverage** | `coverage` | `p-coverage_std-*` | `std 256` → `std 1` |
| **Hardness** | `hardness` | `p-hardness_success_std-*` | `success_std 256` → `success_std 1` |

> **Naming note.** Use `omega` for the Preference knob (env var `OMEGA`); the code
> still accepts a legacy alias `tau`/`TAU` for the same Dirichlet spread, which is
> **unrelated** to the paper's symbol $\tau$ (the observable task descriptor).
> Prefer `omega` everywhere. See [`./heterogeneity.md`](./heterogeneity.md).

> **Hardness needs the labels file** — generate
> `data/hardness/qwen2.5-1.5b_<env>_trajectories.json` first; see the
> [prerequisites callout](#️-the-hardness-arm-needs-a-generated-labels-file-absent-by-design).
> The other two sub-types need no external input.

### Run

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

To reproduce a panel, run **both** endpoints of the relevant sub-type for the
relevant benchmark, 3 seeds each; the PPO appendix variant uses the `ppo/`
sibling. The partition is realized by `partition_dataset(strategy, ...)`, selected
via `partition_strategy` in the config.

---

## 4. Decentralized: protocol ablations

**Backs:** the protocol-sensitivity ablation figure. These sweeps justify the
default `(M = 2, E = 3, |X_i| = 100)` by varying **one** federation knob at a time
on the homogeneous (`p-uniform`) baseline, holding the optimization budget
comparable. Qwen2.5-1.5B, both benchmarks, both algorithms.

### Layout and what each sweep varies

```
config/paper/decentralized/
  selected_cl_change/  {grpo,ppo}/   # vary M (clients sampled per round)
  ep_per_round_change/ {grpo,ppo}/   # vary E (local epochs), T scaled to hold ~210 epochs
  samples_change/      {grpo,ppo}/   # vary |X_i| (tasks per client)
```

| Sweep | Knob | Points present (the baseline point lives in `uniform/`) |
|---|---|---|
| `selected_cl_change/` | `M` (`cl-per-rd`) | `cl-per-rd-1`, `cl-per-rd-4` (the `M=2` point is the uniform main run) |
| `ep_per_round_change/` | `E × T` | `rd-210_ep-1`, `rd-42_ep-5` (T scaled inversely with E to keep ~210 epochs; `E=3/T=70` is baseline) |
| `samples_change/` | `\|X_i\|` (`min-goals`) | `min-goals-500`, `min-goals-1000` (the `100` point is the uniform main run) |

Each leaf holds the WebShop and ALFWorld variants of its points: 3 sweeps × 2
points × 2 envs × 2 algos = **24** configs.

### Run

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

### Notes

- The **baseline point** of each sweep (`M=2`, `E=3`, `|X_i|=100`) is *not*
  duplicated here — it is the corresponding `uniform/Qwen2.5-1.5B-Instruct/main`
  run from [§1](#1-uniform-the-main-table).
- `ep_per_round_change/` scales `total_rounds` inversely with `epochs_per_round`
  to hold the total local-epoch budget near 210, isolating the round/epoch
  trade-off rather than total compute.
- PPO counterparts live in the `ppo/` siblings.

---

## Three-seed replication

The main table reports three seeds. They are already **separate configs** —
`main`, `main_seed1`, `main_seed2` — differing only in `base_seed`:

| Run kind | `base_seed` |
|---|---|
| `main` | 42 |
| `main_seed1` | 21 |
| `main_seed2` | 84 |

`base_seed` drives both the per-round client selection and the round-threaded
data seed (`FEDAGENT_BASE_SEED = base_seed + round*100 + client_id`), so the three
runs explore distinct client schedules and goal draws. You can also reproduce a
seed by overriding any base config on the CLI:

```bash
python -m fedagent.fed.run_fed --config <main config> --base-seed 21   # == main_seed1
```

The het / decentralized families ship a single seed (`base_seed: 42`); rerun with
`--base-seed 21` / `--base-seed 84` for their 3-seed error bars.

---

## Baselines (centralized & local) vs federated

The runner derives the mode from the config — there is no separate flag (see
[`../fed/README.md`](../fed/README.md#baseline-modes)):

| Mode | Selected by | Behavior |
|---|---|---|
| **federated** | `total_clients: 100` (default) | FedAvg across the 2 sampled clients each round. |
| **centralized** | `total_clients: 1` | One model on the pooled data; FedAvg of a single client is the identity, so the run is continued central training. |
| **local** | `local_client_id >= 0` (`clients_per_round: 1`) | The paper's *Local Agent Training*: pin one client's data shard every round, no federation. |

The three local configs pin distinct clients of the 100-way partition:

| Config dir | `local_client_id` |
|---|---|
| `local_client1/` | 21 |
| `local_client2/` | 42 |
| `local_client3/` | 84 |

`--local-client-id` overrides it for any base config.

**Epoch budget.** Both baselines run **T = 70 × E = 3 = 210 epochs**, matching the
federated arms' total. The original paper ran the baselines as 1 round × 210
epochs; in this overlay the per-round FedAvg of a single client/shard is a no-op,
but **goal variety is drawn per round** (the round-threaded data seed re-draws
each client's goals every round), so the runner keeps **70 rounds** to reproduce
that variety — same total epochs, same goal coverage.

---

## Compute budget

Approximately **1,800 H100 GPU-hours** total across all reported experiments.
Per-config (single seed) estimates, on the default 4 × H100 node:

| Benchmark × algorithm | Wall-clock (4 × H100) | GPU-hours / config |
|---|---|---|
| WebShop GRPO  | ~24 h | ~93 |
| WebShop PPO   | ~29 h | ~117 |
| ALFWorld GRPO | ~29 h | ~117 |
| ALFWorld PPO  | ~35 h | ~140 |

GPU-hours = wall-clock × 4 GPUs. Multiply by **3 seeds** for each reported
mean ± std cell, and by the number of sweep points / backbones in a given figure
or table block. To shrink cost while developing, drop the group size
(`gen_paper_configs.py --group-size 2` regenerates a cheap-smoke matrix) or run on
fewer GPUs with `--n-gpus`; see [`./running.md`](./running.md).

---

## Outputs

Each run writes everything under the config's `output_dir`:

- **`federated_summary.json`** — per-round provenance (clients selected, the model
  each round started from, aggregated actor + HF paths, the critic chain for PPO)
  plus the `mode`, `partition_strategy`, final model, and the **`val_curve`**.
- **Per-round logs** — `round_<r>/client_<c>/training.log`,
  `round_<r>/aggregated/{aggregate,merge}_*.log`, and the per-service logs
  (`webshop_service_client<c>.log` / `alfworld_service_client<c>.log`).
- **`round_<r>/client_<c>/json_logs/metrics.json`** — each client's `training.log`
  re-parsed into the FedAgent plot schema (`[{"step", "metrics"}, ...]`).
- **The unperturbed val success curve** — `eval_global` scores the aggregated
  global model on the shared unperturbed val service every `test_freq: 5` rounds
  (plus the final round), with `val_before_train: true` adding the base model as
  the round-0 point and `val_temperature: 0.4`. The curve lands in
  `federated_summary.json` (`val_curve`) and the round-`r` eval dumps live in
  `round_<r>/eval/`.

`tools/verl08_migration/summarize_fed_run.py` post-processes a run directory.

> **Disk.** Consumed FSDP shards are deleted after each merge
> (`cleanup_checkpoints`, on by default), keeping every `training.log` and the
> merged HF; peak disk stays roughly one round's worth.

---

## Scientific-equivalence, not bit-identical

This overlay reproduces the paper's **science** — the same federation protocol
(N/M/T/E), the same algorithms (GRPO G = 8, PPO/GAE with a federated critic), the
same heterogeneity construction, and the same unperturbed-val measurement — on
**stock verl 0.8** with no trainer fork. It is **not** bit-for-bit identical to
the original verl-agent 0.3.1 stack (different rollout engine, FSDP checkpoint
layout, and RNG threading). For the full fidelity record — what is preserved,
what changed, and why — see [`./migration.md`](./migration.md).
