# FedAgent → verl 0.8 — Experiment Log

Lab notebook for the FedAgent (federated agent-RL) migration from vendored verl-agent
0.3.1 to **verl 0.8.0**, on branch `migrate/verl-0.8.0`. Records every run: setup,
result, status, artifacts.

> **Living document** — the status table, observed-reward data, and the roadmap below are
> updated as runs complete and the plan evolves. Last sections: completed runs (top) →
> in-flight → **planned roadmap** (bottom).

## Setup
- **Package**: `fedagent/` (thin overlay — imports verl 0.8 as a library, stock
  `RayPPOTrainer` + native async `AgentLoopManager`, no trainer fork).
- **Trainer env**: conda `fedagent-verl08` (py3.12, torch 2.8.0+cu128, vllm 0.11, flash-attn 2.7.4).
- **WebShop env**: conda `verl-agent-webshop` (gym 0.24 / pyserini / Java), runs as a remote HTTP service.
- **Models** (local, offline): Qwen2.5-{0.5B,1.5B,3B,7B}-Instruct, Llama-3.2-3B-Instruct.
- **Test GPU**: SLURM job `4542895` on `cs-gpu/qgpu3021` (4×H100 80GB); `srun --jobid=4542895 --overlap <cmd>`.
- **Fidelity bar**: SCIENTIFIC EQUIVALENCE (reproduce conclusions, not bit-identical curves). The
  ORIGINAL FedAgent used **GRPO** (`adv_estimator=grpo`, `env.rollout.n=8`; see
  `config/uniform/*/main/grpo/*.yaml` -- NOT GiGPO, despite the verl-agent fork being the GiGPO repo).
  The migration keeps the SAME algorithm (GRPO, group size G=8) and only swaps the ROLLOUT MECHANISM
  to verl 0.8's native concat multi-turn agent loop (the fork's per-turn rollout machinery is not ported).
- **Note**: federated checkpoints live on the COMPUTE node's `/tmp` — inspect via `srun --overlap`.
  The recurring `DataLoader worker ... Killed` traceback is a benign teardown artifact (every client exits 0).

## Status summary

| # | Experiment | Setup | Result | Status |
|---|---|---|---|---|
| 0a | Checkpoint round-trip + matched-PG FedAvg (spike) | 2×H100, synthetic + real shards | FedAvg-exact, resume OK | ✅ |
| 0b | Custom AgentLoop on STOCK verl 0.8 (spike) | TinyText, 2×H100 | rollout→GRPO→update→ckpt ×2 | ✅ |
| 1 | TinyGuess smoke through `fedagent` package | Qwen0.5B, 2×H100 | canonical FSDP ckpt ×2 | ✅ |
| 2 | WebShop remote service + GRPO smoke | Qwen0.5B, 6-turn | real episodes; reward≈0 (small model) | ✅ |
| 6 | Matched-PG FedAvg on REAL verl-0.8 ckpts | 2 real Qwen ckpts | max\|got−avg\|=0.0 | ✅ |
| 7a | **Federated loop — TinyGuess 2 client × 2 round** | Qwen0.5B | loop CLOSED (R2 from R1 aggregate), exit 0 | ✅ |
| 7b | **Federated WebShop Catalog-Split 2×2** | Qwen0.5B, env-het | catalogs 762/750, loop CLOSED | ✅ |
| 4v | Catalog-Split determinism (CPU) | numpy 1.26 & 2.2 | c0=762, c1=750, Jaccard 0.62, identical | ✅ |
| P | **Signal probe** — 1.5B / 15-turn WebShop | 1 client, 4 steps | reward 0.026→0.147 (max 0.857) | ✅ |
| A | Scaled env-het (catalog_split) | 1.5B/15-turn, 2×4 | R1/2/3 mean 0.137/0.138/0.093 (flat); within-round peaks ~0.26 | ⚠ OOM at R4 (ran concurrent w/ B) |
| B | Scaled task-het (task_disjoint) | 1.5B/15-turn, 2×4 | R1/2 mean 0.156/0.124 | ⚠ OOM at R3 (ran concurrent w/ A) |
| Fx | FedProx hook test | 1.5B, μ=0.1, 1×2 | `[fedprox] enabled` in every worker | ✅ |
| **C4-homog** | 8-round IID baseline (compounding) | 1.5B/15-turn, 2×4, `--n-gpus 4`, qgpu3021 | R1 0.109→R8 0.164, slope **+0.0090**/rd | ✅ |
| **C4-envhet** | 8-round catalog_split (compounding) | 1.5B/15-turn, 2×4, `--n-gpus 4`, qgpu3016 | R1 0.110→R8 0.125, slope **+0.0023**/rd | ✅ |
| **C4-task** | 8-round task_disjoint (compounding) | 1.5B/15-turn, 2×4, `--n-gpus 4`, qgpu3003 | R1 0.130→R8 0.159, slope **+0.0035**/rd | ✅ |
| Pt | **Heterogeneity suite ported + wired** (workflow) | Coverage/Hardness, env Variants 2–5, ALFWorld svc | 8/8 fns byte-identical; all branches verified | ✅ |
| ALF | **ALFWorld service smoke** (CPU, standalone) | verl-agent-alfworld, pool 1, 3553 train games | /health→/create→/reset(38 acts)→/step OK | ✅ |
| ALFf | ALFWorld **federated plumbing** wired in run_fed | env_kind=alfworld, per-client svc, registry | config valid; GPU run pending | ✅(code) |

Legend update: ⚠ = crashed/partial.

Legend: ✅ done · ▶ running · ⏳ queued.

---

## Validation spikes (Phase 0)

**0a — checkpoint round-trip + matched-PG FedAvg.** `tools/verl08_migration/phase0a_ckpt_roundtrip.py`.
Established that verl-0.8 FSDP shards are averaged under a matched-world-size `torchrun` PG
(`aggregate_fedavg_fsdp.py`). FedAvg numerically exact; round-trips through verl's own loader.

**0b — custom AgentLoop on stock trainer.** `tools/verl08_migration/run_phase0b.sh`.
Proved a custom `AgentLoopBase` runs the full GRPO loop on STOCK `verl.trainer.main_ppo` (no fork) —
the seam the whole overlay rests on.

## Package smokes (Phase 1–2)

**1 — TinyGuess.** `fedagent/scripts/run_smoke.sh`. `python -m fedagent.main_ppo_fed` + Hydra config +
`custom_cls` dataset + registered AgentLoop → rollout→GRPO→update→checkpoint ×2, canonical
`global_step_N/actor/model_world_size_2_rank_*.pt` layout.

**2 — WebShop.** `fedagent/scripts/run_webshop_smoke.sh`. Remote service (FastAPI in `verl-agent-webshop`,
pre-warmed env pool, server-side action parsing) + thin HTTP client. Real 6-turn episodes → GRPO → ckpt.
Reward ≈ 0 (Qwen0.5B can't shop in 6 truncated turns) — plumbing proven, signal deferred to a capable model.

## Federated loop (Phase 6/7)

**7a — TinyGuess 2×2.** `fedagent/scripts/run_tinyguess_fed_smoke.sh`. The closed loop:
per-(client,round) `main_ppo_fed` → matched-PG FedAvg → `model_merger` (FSDP→HF) → round 2 trains from
the round-1 aggregate. `federated_summary.json`: round 2 `started_from` = round-1 aggregated HF. Exit 0.

**7b — WebShop Catalog-Split 2×2.** `fedagent/scripts/run_webshop_fed_smoke.sh`. Each client a DISJOINT
product catalog via its own service (client 0 = 762 ASINs @:8080, client 1 = 750 @:8081); FedAvg across
the divergent envs; round 2 re-enters from the aggregate. Proves the env-heterogeneity arm composes with
the federated loop on the real env. (Qwen0.5B → reward weak; mechanism proof.)

## Heterogeneity + signal (Phase 4 / scaling)

**4v — Catalog-Split determinism.** `fedagent/hetero/webshop_catalog_split.py` (verbatim port of the
`_distractor_disjoint_partition_webshop_v5` partition). Verified per-client catalogs are distinct
(client0=762, client1=750 ASINs, Jaccard 0.62), goal slices disjoint, and IDENTICAL under numpy 1.26
(WebShop env) and 2.2 (trainer env) — `RandomState` is version-stable.

**P — signal probe.** `config/examples/webshop/probe_signal.yaml`. Qwen2.5-1.5B, 15-turn WebShop, 1 client,
4 steps, full catalog. **critic/rewards/mean = 0.026 → 0.036 → 0.104 → 0.147** (max → 0.857). First
nonzero, *rising* reward — the env produces a learnable signal with a capable model + 15-turn budget.

## In-flight: the A/B/C decomposition (Phase 8-lite)

Qwen2.5-1.5B, 15-turn WebShop, 2 clients × 4 rounds, GRPO, FedAvg. Three conditions that differ by one
factor each so the asymmetry decomposes cleanly:

| run | config | goals | catalog | isolates |
|---|---|---|---|---|
| **A** | `examples/webshop/scaled/catalog.yaml` (catalog_split) | disjoint | disjoint | env + task heterogeneity |
| **B** | `examples/webshop/scaled/task.yaml` (task_disjoint) | disjoint | FULL | task heterogeneity only |
| **C** | `examples/webshop/scaled/homog.yaml` (partition="") | shared | FULL | IID baseline |

→ **A − B = pure environment-heterogeneity effect**, **B − C = pure task-heterogeneity effect**, both
under FedAvg. Expected (Input-Dynamics Asymmetry): A−B negative (env-het hurts), B−C ≈ 0 (task-het robust).
Analyze with `tools/verl08_migration/summarize_fed_run.py A=… B=… C=…`.

**Fx — FedProx hook test.** `config/examples/webshop/fedprox_test.yaml` (μ=0.1). Verifies the non-fork FedProx
injection (`fedagent/fedprox.py` patches `FSDPEngine.optimizer_step` via the Ray
`worker_process_setup_hook`). Look for `[fedprox] enabled: proximal mu=0.1` in the worker log.

## Results: 8-round 3-way asymmetry (Phase 8-lite, 2026-06-20)

The first **8-round** federated comparison (Qwen2.5-1.5B, 15-turn WebShop, 2 clients,
4 steps/round, GRPO+FedAvg, `--n-gpus 4`, one run per node). `critic/rewards/mean`,
per round = mean-over-clients of the round's step-averaged reward:

| round | envhet (catalog_split) | task (task_disjoint) | homog (IID) |
|---|---|---|---|
| 1 | 0.110 | 0.130 | 0.109 |
| 2 | 0.139 | 0.157 | 0.086 |
| 3 | 0.106 | 0.115 | 0.134 |
| 4 | 0.098 | 0.187 | 0.122 |
| 5 | 0.165 | 0.126 | 0.153 |
| 6 | 0.149 | 0.120 | 0.158 |
| 7 | 0.117 | 0.185 | 0.139 |
| 8 | 0.125 | 0.159 | 0.164 |
| **slope/rd** | **+0.0023** | **+0.0035** | **+0.0090** |
| **mean** | 0.126 | 0.147 | 0.133 |

**Finding (directional, suggestive — not yet conclusive).** Both by **compounding slope**
(homog +0.0090 > task +0.0035 > **envhet +0.0023**) and by mean reward, the
**env-heterogeneity arm (catalog_split) is the distinct loser**; the task-heterogeneity arm
(task_disjoint) tracks the IID baseline (B−C ≈ 0, noisy). envhet's total 8-round gain
(+0.015) is ~¼ of the IID baseline's (+0.055). This is **directionally consistent with the
Input-Dynamics Asymmetry**: env (transition-kernel P_i) heterogeneity degrades FedAvg while
task (goal-distribution) heterogeneity is ~robust. It is also the **first multi-round signal**
— the earlier 4-round runs were flat.

**Honest caveats**: **1 seed**, **4 steps/round** ⇒ per-round reward is noisy (the A−B/B−C
*trajectory* is not readable round-by-round; only the aggregate slope/mean ordering is). The
absolute gaps sit within the noise band. This is **suggestive, not significant** — paper-scale
(E×T≈210) × **3 seeds** + unperturbed validation are needed for a firm claim (Phase 8-full).
`task` here is the `task_disjoint` stand-in; the faithful Preference/Coverage task arm is a
follow-up (Coverage run launched). Reproduce: `tools/verl08_migration/collect_fed_logs.sh`.

## Observed reward data (critic/rewards/mean per step)

- **Probe** (1.5B, full catalog): `0.026, 0.036, 0.104, 0.147`.
- **Run B** (task_disjoint): R1 c0 `0.100, 0.222, 0.128, 0.182`; R1 c1 `0.220, 0.038, 0.182, 0.179`;
  R2 c0 `0.102, 0.135, 0.114, 0.185`; R2 c1 `0.031, 0.134, 0.165, 0.121`.
- **Run A** (catalog_split): pending post-hoc parse (ran on the pre-logger runner).

> Small-scale (4 steps/round) → noisy; the A−B / B−C deltas are the signal, not absolute values.
> Magnitude study (3 seeds, E×T=210, multi-model, unperturbed val) is Phase 8-full.

## Findings & lessons (so far)

- **Signal is real**: with Qwen2.5-1.5B @ 15 turns, `critic/rewards/mean` is nonzero and
  rises *within* a round (probe 0.026→0.147; scaled within-round peaks ~0.26–0.28). With
  Qwen0.5B @ 6 turns it was ~0 — capable model + turn budget matter.
- **No compounding at tiny budget**: across federated rounds at 4 steps/round the round-mean
  reward is FLAT/noisy (A 0.137/0.138/0.093; B 0.156/0.124), so the env-vs-task asymmetry is
  NOT visible at 16 steps. Each round re-enters from the aggregate with a FRESH optimizer
  (faithful to FedAgent), so durable progress needs many more steps × rounds. **The asymmetry
  MAGNITUDE requires ~paper budget (E×T≈210)** — a dedicated large run, not a debugging-window run.
- **OOM lesson (memory sizing)**: the SLURM job has a **900GB** memory cgroup; each WebShop
  service env is **≈14GB** (in-memory catalog/index). So pool=16 ×2 services ≈ **32 envs ≈448GB
  per run** (safe), but **two concurrent runs ≈64 envs ≈896GB → OOM** (killed a worker → both
  runs died via NCCL heartbeat). ⇒ **Run ONE WebShop run at a time** (use `--n-gpus 4` for 4-GPU
  single runs), or drop pool to ~8 for two concurrent runs.
- **Clean ablation still holds**: A−B (catalog_split vs task_disjoint, identical goal slices)
  isolates the env effect; needs adequate budget to read.

## Infra verified (2026-06-20, later)

- **FedProx hook WORKS (non-fork)**: `[fedprox] enabled: proximal mu=0.1 (FSDPEngine.optimizer_step
  patched)` printed in every actor worker via the Ray `worker_process_setup_hook`
  (`+ray_kwargs.ray_init.runtime_env.worker_process_setup_hook=fedagent.fedprox.worker_setup`,
  gated on `FEDPROX_MU`). Confirmed the agent-loop workers are SEPARATE processes from the
  actor-engine worker, so the runtime_env hook (not an import) is required.
- **Robust GPU recipe**: run ONE WebShop run at a time with `--n-gpus 4` (single Ray cluster,
  all 4 GPUs, 32 envs ≈448GB). This avoids BOTH the concurrent-run OOM (64 envs > 900GB) AND
  the `Duplicate GPU detected` race (two `CUDA_VISIBLE_DEVICES`-split Ray clusters racing at
  init). Verified: `--n-gpus 4` reaches step:1 with no Duplicate-GPU error.
- **ALFWorld feasible**: conda env `verl-agent-alfworld` imports alfworld+textworld; game data
  at `~/.cache/alfworld/{json_2.1.1,logic,detectors}` (+ `/projects/b1222/userdata/canyu/.cache/alfworld`).
  Ready to port as a remote service (Phase 3).

## Heterogeneity suite ported + wired (2026-06-20, workflow `wr6zr0to1`)

A 3-agent ultracode workflow ported the remaining FedAgent heterogeneity variants + the
ALFWorld env into the `fedagent/` overlay, **all science partition functions copied
byte-for-byte** from verl-agent's `partition_strategy.py` (now vendored at `fedagent/envs/alfworld/engine/.../partition_strategy.py`; AST-verified:
8/8 functions identical to source). Then wired + independently smoke-tested:

- **Task-level variants** (full catalog, goal-distribution skew only):
  - `fedagent/hetero/webshop_coverage.py` — `coverage_for_client(...)` (Beta-sized goal
    coverage ξ). Verified: client0=876 / client1=124 goals, overlap 0.
  - `fedagent/hetero/webshop_hardness.py` — `hardness_for_client(...)` (Beta easy/hard ξ′);
    requires a `TRAJECTORIES_FILE` (task_id→success); raises cleanly if missing/empty.
  - shared `fedagent/hetero/_beta_sizing.py` (`generate_client_sizes` / `assign_with_overlap`).
- **Env-level (transition) variants 2–5** (`fedagent/hetero/webshop_env_variants.py`,
  return gym kwargs merged into `gym.make`; uniform goals): `bm25_field_subset` (V2,
  fields=['description']), `bm25_reweight` (V3, k1=0.3), `lookalike` (V4, +1000 products),
  `rank_wrapper` (V5, type=bm25_invert). Confirmed the vendored `WebAgentTextEnv-v0`
  **already consumes** `bm25_in_memory_config` / `extra_products` / `search_engine_variant`
  (same path as the proven `catalog_filter_asins`).
- **Service wiring**: `envs/webshop/service/server.py` now branches on `PARTITION_STRATEGY ∈
  {coverage, hardness, bm25_field_subset, bm25_reweight, lookalike, rank_wrapper}` (in
  addition to catalog_split / task_disjoint / preference), merges variant kwargs in
  `_make_env`, and reports `goal_slice` / `env_variant_keys` on `/health`.
- **ALFWorld** remote service (`fedagent/envs/alfworld/service/`, runs in `verl-agent-alfworld`)
  + thin client `fedagent/envs/alfworld.py`, **registered as `ALFWorld`** in the env
  registry (resolves in the trainer env). Mirrors the WebShop service one-to-one; per-turn
  obs body + reward (10·won) reproduced verbatim from verl-agent. Sharding handled inside
  the vendored `AlfredTWEnv` via the forwarded `CLIENT_*` env vars.
  - **Service smoke PASSED (CPU, standalone)**: built `AlfredTWEnv` (collected 8810 →
    3553 solvable train games), warmed pool=1, then `/health` → `/create` → `/reset`
    (real TextWorld obs `-= Welcome to TextWorld, ALFRED! =-`, **38 admissible actions**)
    → `/step "go to cabinet 1"` (`reward=0, valid=1`, next obs `You arrive at cabinet 1.
    The cabinet 1 is closed.`, 40 next actions). The 2nd env is functional in the overlay.
  - **Federated plumbing wired** in `run_fed.py`: `env_kind=alfworld` →
    `start_alfworld_services` (per-client `ALFWORLD_PORT`/`POOL_SIZE`/`TRAIN_EVAL`/`CLIENT_*`)
    + per-client `ALFWORLD_SERVICE_URL` + generic `stop_services`. New `config/envs/alfworld.yaml`
    (max_turns 20) + `config/examples/alfworld/smoke.yaml` (2×2, n_gpus 4). Config validated
    offline (service URLs, gen_batch=pool=8, registry); GPU federated run pending a free node.
  - **GPU federated run — 2nd-env loop PROVEN (2026-06-20).** After two GPU bugs found+fixed,
    the federated ALFWorld loop **trained 2 full GRPO steps end-to-end** (service → concat
    `gym_text` agent loop → GRPO → `update_actor`; num_turns mean ~19/20, response ~3000 tok).
    `critic/rewards/mean = 0` (Qwen2.5-1.5B can't *solve* ALFWorld at this budget — expected;
    the mechanism is what's proven). Bugs: **(1)** pool=8 raced textworld's shared module-level
    PDDL parser (`tatsu` `IndexError: pop from empty list`) → process-global `_TW_LOCK`
    serializing `reset`/`step`; **(2)** batch shape (`data size must be divisible by
    force_group_size*micro_batch`) → matched the proven WebShop recipe (train_batch 8 / pool 16).
    Remaining: vLLM `EngineCore died` at step 3 = **memory/stability tuning** (20-turn concat
    stresses the KV cache), not a loop bug; needs lower `max_model_len`/shorter turns for a full run.

**FedProx under verl 0.8 — injection finding (2026-06-20).** The proximal patch is correct and
the hook *fires* (`[fedprox] enabled: proximal mu=0.1` in every actor worker), BUT injecting it
via `ray_kwargs.ray_init.runtime_env.worker_process_setup_hook` **breaks verl's per-worker GPU
isolation** → `Duplicate GPU detected: rank N and rank 0 both on CUDA device 42000`. Confirmed
it's the cluster-level `runtime_env`, not the patch: (a) c4 runs on the same nodes work without
the hook; (b) importing `FSDPEngine` is **CUDA-safe** (`torch.cuda.is_initialized()` stays
False). ⇒ FedProx **parked**; candidate fix = `sitecustomize.py` (Python-startup injection, no
`runtime_env`), pending verification Ray workers execute it. Freed GPU → env-het generality runs.

**Env-het generality + faithful task arm (2026-06-20).** Testing whether env-het-hurts
generalizes beyond catalog_split, and whether a *faithful* task variant tracks IID:
- **Coverage** (faithful task-het, Beta goal coverage, full catalog, 4rd): mean **0.118** ≈
  homog's first-4 **0.113** ⇒ task-het tracks the IID baseline (confirms the robust side with a
  faithful variant, not just the task_disjoint stand-in).
- **rank_wrapper** (env Variant 5, 4rd): mean **0.076** ≪ homog 0.113 ⇒ a 2nd env-het pattern
  underperforms IID (consistent with env-het-hurts).
- **bm25_reweight** (env Variant 3): trained with **real reward** (R1 mean 0.134, max 0.70) but
  the run hit `No space left on device` at R2 (compute /tmp full) — inconclusive (2 rounds);
  rerun with disk hygiene pending.
- **lookalike** (env Variant 4): **data-format bug** — injected products have a list field where
  the WebShop engine's `load_products` calls `.split()` (`AttributeError: 'list' object has no
  attribute 'split'`); parked pending a format fix.

> **Caveat (confound):** absolute-reward gaps for the env variants don't isolate the *federation*
> effect from per-client task difficulty (e.g. rank-perturbation makes retrieval harder
> regardless of FedAvg). The clean test remains the catalog_split **A−B** (same goal slice). The
> generality runs add breadth, not a controlled federation isolation.

**Infra: disk hygiene (2026-06-20).** Each 8-round run wrote ~**367 GB** of FSDP checkpoints to
the compute node's `/tmp` (save-per-round × clients × shards) → `/tmp` hit 100% and killed runs
mid-flight. Fix: `run_fed.cleanup_round_checkpoints` deletes the consumed per-client + aggregated
**actor shards after each merge** (keeps every `training.log` + the aggregated HF), capping peak
disk at ~one round. Plus a relaunch wrapper that `ray stop`s + clears stale procs (the failed
FedProx attempts had left a Ray cluster → `Duplicate GPU detected` on relaunch).

**Seed robustness — seed-43 REPLICATES the env effect (2026-06-20).** Re-ran the 3-way at
**base_seed 43** (8 rounds, disk hygiene on). The env arm vs IID, both seeds:

| | envhet slope | homog slope | envhet mean | homog mean |
|---|---|---|---|---|
| seed 42 | +0.0023 | +0.0090 | 0.126 | 0.133 |
| seed 43 | +0.0023 | +0.0052 | 0.123 | 0.144 |

**env-het compounds slower than IID *and* has lower mean reward in BOTH seeds** (envhet's slope
is +0.0023 in both). The env-het-underperforms finding **replicates across 2 seeds** — materially
strengthening the headline beyond the single-seed caveat. (seed-43 `task_disjoint` arm still
running → full seed-43 3-way decomposition + the env-het generality at 8 rounds, rank/bm25,
pending those runs.)

All modules `py_compile` clean under `fedagent-verl08`; registry resolves
`{ALFWorld, TinyGuess, WebShop}`. This makes roadmap rows #4, #5 **code-complete** and #7
**service-validated + plumbing-complete** (GPU runs pending).

## Planned experiments (roadmap)

Priority order; each row updated to ✅ with results as it lands. "Cost" = rough GPU-hours on 2×H100.

| Pri | Experiment | Depends on | Cost | Purpose |
|---|---|---|---|---|
| 1 | **Finish A/B/C (seed 1)** + `summarize_fed_run` | A/B running, C queued | ~0.5h | first env-vs-task asymmetry decomposition |
| 2 | **FedProx hook test** (Fx) → if OK, **FedProx A/B** (μ sweep) | fedprox.py wired | ~2h | FedAvg-vs-FedProx baseline under env-het |
| 3 | **A/B/C × 3 seeds** (`--base-seed 42/43/44`) | #1 signal seen | ~3h | seed robustness of the asymmetry |
| 4 | **Faithful task variants**: Preference(ω)✅wired, Coverage(ξ)✅wired, Hardness(ξ')✅wired | **CODE DONE** (run pending) | ~3h | canonical task arm (vs the task_disjoint stand-in) |
| 5 | **Other env variants**: Field-Subset, BM25 Reweight, Lookalike, Rank Wrapper | **CODE DONE** (ported+wired+verified; run pending) | ~4h | full env-het suite (Patterns B/C/D) |
| 6 | **Baselines**: Local (1 client), Centralized (1 client, all data) | configs only | ~2h | FedAgent vs Centralized vs Local |
| 7 | **ALFWorld** env_disjoint | **SERVICE SMOKE ✅ + fed plumbing DONE** (federated GPU run pending) | ~4h | 2nd environment, env-het generality |
| 8 | **PPO** variants of the key conditions | config (verl native PPO) | ~3h | GRPO + PPO coverage |
| 9 | **Phase 8-full**: 3 seeds × E×T=210 × {1.5B,3B,7B} × {GRPO,PPO}, unperturbed val | all above + ops resume | large (needs a dedicated allocation) | full reproduction vs 0.3.1 |

Open design points to resolve as we go:
- **Unperturbed validation** (science red line): val service must use `catalog_filter_asins=None`; wire when enabling `test_freq>0`.
- **Seeds**: also vary verl's training seed (not just `FEDAGENT_BASE_SEED`) for true seed independence.
- **Rollout-mechanism equivalence**: the algorithm is GRPO on BOTH sides (G=8 now matched); the only
  difference is per-turn (fork) vs concat multi-turn (verl 0.8) rollout. Confirm the asymmetry conclusions
  hold under the concat mechanism (validated implicitly by #1/#9). (NOT a GiGPO->GRPO switch -- the original
  was already GRPO.)
- **Ops for long runs**: `run_fed` resume / checkpoint-rotation before the 70-round Phase-8 campaign.

## Reproduction audit + fix round (2026-06-21)

Exhaustive feature-completeness audit (overlay vs 0.3.1 source) found the overlay **ran** but
diverged from the paper recipe on 8 science-critical points. All fixed + GPU-verified:

- **B1 save_freq** `-1` → `100000` (gen_paper_configs + examples/alfworld/paper): `-1` NEVER saves →
  FedAvg would average nothing. Regenerated 56 paper configs (0 stale `save_freq:-1`).
- **B2 SEARCH_RETURN_N** default 50 → **200** (run_fed DEFAULTS + both service starters + hardness
  generator): env-het needs ≥100 (filtered catalogs drop targets under top-50 BM25).
- **B4 GRPO objective** restored in **base** `fedagent_ppo.yaml` (`use_kl_loss=true,
  kl_loss_coef=0.01, kl_loss_type=low_var_kl, entropy_coeff=0.001`) — compose-verified to
  propagate to every GRPO arm.
- **B5 prompt tokenization**: agent loop now uses a NON-truncating `_tokenize_chat` (the stock
  `apply_chat_template` left-truncates to prompt_length=2048, silently dropping multi-turn obs).
- **B7 reward**: WebShop service returns **sparse {0,10}** (dense kept as `task_score`) + agent-loop
  **invalid-action penalty** (coef 0.1) — both reproduced from verl-agent envs.py.
- **B8 val specs**: WebShop 500 / ALFWorld 140 (+274 via the by-task-type eval tool).
- **ALFWorld het wiring**: run_fed forwards OMEGA/SIZE_STD/SUCCESS_STD/TRAJECTORIES_FILE →
  service `_partition_kwargs()` → AlfredTWEnv → partition_dataset.

- **B3 (the crux) — WebShop task-het goal mapping.** The original partitions the env's ACTUAL
  `server.goals` (the seed-42 *shuffled* list) and maps back via `goals.index()`, so the served
  goal at index *i* carries the category/size/hardness the partition selected. The overlay was
  computing indices from `_generate_goal_asins_for_partition` (the **unshuffled** order), so for
  the CONTENT-dependent arms (preference/coverage/hardness) the served goals' properties were
  scrambled → the het signal was broken. **Fix**: defer those partitions to `_lifespan` and
  compute `CLIENT_GOAL_IDXS` from `env.server.goals` (refactored `*_for_client(env_goals=...)`);
  goal-id logging now uses the real options-hash task_id (matches `hardness_partition`).
  - GPU-verified the lynchpin: `server.goals` is reproducible run-to-run and **identical** whether
    or not a `catalog_filter_asins` is applied (goals come from the full pool; the filter is pure
    list ops, no RNG) — so catalog_split/task_disjoint (contiguous index RANGE from raw products)
    were **already equivalent** (an earlier "shuffle-reproduction" fix was correctly reverted).
  - GPU-verified equivalence to the original on the SAME real `server.goals`: preference
    `idx_set_equal=True`; coverage/hardness `selected_goal_multiset_equal=True` (idx values differ
    only where duplicate goals map to their true slot vs `.index()` first-match — same goals
    served). Service wiring confirmed end-to-end (defer at import → runtime partition; omega=0.5 →
    92% top-category skew on real goals).

Verification scripts: `_scratch/gpu_verify/check_b3_*.py` (not committed). Scope: 12 paper configs
(preference/coverage/hardness ×4) + 3 scaled task-het configs depend on B3; catalog_split (14),
task_disjoint (2), variants (18), homog "" (4) are unaffected.

### Group-size alignment + local-training recipe (2026-06-21, round 2)

Established the ORIGINAL FedAgent base RL is **GRPO with G=8** (`config/uniform/*/main/grpo/*.yaml`:
`adv_estimator=grpo`, `env.rollout.n=8`, `train_batch_size=8`, `ppo_mini_batch_size=64`) — **not
GiGPO** (the verl-agent fork is the GiGPO repo, but the federated experiments used GRPO/PPO; zero
`gigpo` configs in the tree). A second ultracode audit (10 dims) + a deep code-mechanism audit then
found the migration's local-training recipe diverged. Fixed + verified:

- **G=8 alignment**: generator default `rollout.n=8` (was 2). `ppo_mini_batch_size` stays **8 prompts**
  because stock verl-0.8 multiplies it by `rollout.n` (`others/verl/.../ray_trainer.py:1311`) →
  8×8=64 seq = full batch = 1 update/rollout (== original's `mini=64`). `--group-size` flag added.
- **B-A `total_training_steps`**: base `fedagent_ppo.yaml` had a leaked smoke value `2`; run_fed only
  overrode when `>0`, so EVERY paper run capped at 2 steps/round. Fixed: base → `null` + run_fed emits
  `=null` when `<=0` → verl uses `len(dataloader)*total_epochs` (full E epochs). **GPU-verified**: smoke
  logs `Total training steps: 2` (= 1 step/epoch × E=2), not the leaked 2-cap nor 0.
- **B-C `n_envs`**: train env-specs sized to the original `train_data_size` (GRPO 8 / PPO 64 / ALFWorld
  8) so `len(dataloader)=1` → E updates/round (was 32/16 → 12/6). New `webshop_15_ppo.yaml` (n_envs=64).
  **GPU-verified**: smoke logs `Size of train dataloader: 1`.
- **B-G + B-G2 round-threading**: the original threads round into the data seed
  (`main_ppo_fed.py:274`); the migration didn't, AND the service's `CLIENT_GOAL_IDXS[seed % len]`
  annihilated the round term mod a ~100 shard → each client saw the SAME goals every round. Fixed:
  run_fed `FEDAGENT_BASE_SEED = base + round*100 + client` + service picks via
  `random.Random(seed).randrange(len)` (full entropy, == original `RandomState(fed_seed).choice`).
  Verified offline: a client now covers **100/100** of its shard over 70 rounds (was 8).
- **B-D Local baseline**: was `catalog_split env_div=1.0` (max env perturbation) → fixed to
  `partition_strategy=""` (full-catalog IID, == original `uniform_single`).
- **B-E PPO batch**: PPO arms now `train_batch_size=64` + `ppo_mini_batch_size=64` + the PPO env-spec
  (was reusing the GRPO 8).

Still open (coverage/secondary, tracked): ALFWorld 50-step budget (B-B: needs `max_model_len` raise +
GPU verify; ALFWorld 1.5B is secondary), and the config-emission breadth (multi-backbone 3B/7B/Llama,
ALFWorld matrix, decentralized M/E-T/min_goals ablations) — additive generator work, no run_fed change.

## Reproduce

```bash
# on the GPU node (srun --jobid=<JID> --overlap):
bash fedagent/scripts/run_smoke.sh                         # TinyGuess package smoke
bash fedagent/scripts/run_tinyguess_fed_smoke.sh           # federated loop (TinyGuess 2x2)
bash fedagent/scripts/run_webshop_fed_smoke.sh CFG         # federated WebShop (CFG = a fed config)
#   CFG ∈ config/examples/webshop/{2cl_catalog_split, probe_signal, fedprox_test}.yaml
#       or config/examples/webshop/scaled/{catalog, task, homog}.yaml
# extra args forwarded to run_fed, e.g.:  ... CFG --base-seed 43 --output-dir /tmp/run_s43 --port-base 8090
python tools/verl08_migration/summarize_fed_run.py A=/tmp/...scaled_env B=/tmp/...scaled_task C=/tmp/...scaled_homog
```
