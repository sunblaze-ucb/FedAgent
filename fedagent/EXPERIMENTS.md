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
- **Fidelity bar**: SCIENTIFIC EQUIVALENCE (reproduce conclusions, not bit-identical curves). Uses
  verl 0.8 native concat multi-turn + GRPO instead of the old forked GiGPO (to be validated in Phase 8).
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
| C | Homogeneous (IID) baseline | 1.5B/15-turn, 2×4 | — | ⏳ queued (run single, not concurrent) |
| Fx | FedProx hook test | 1.5B, μ=0.1, 1×2 | — | ▶ verifying |

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

**P — signal probe.** `config/fed_webshop_probe_signal.yaml`. Qwen2.5-1.5B, 15-turn WebShop, 1 client,
4 steps, full catalog. **critic/rewards/mean = 0.026 → 0.036 → 0.104 → 0.147** (max → 0.857). First
nonzero, *rising* reward — the env produces a learnable signal with a capable model + 15-turn budget.

## In-flight: the A/B/C decomposition (Phase 8-lite)

Qwen2.5-1.5B, 15-turn WebShop, 2 clients × 4 rounds, GRPO, FedAvg. Three conditions that differ by one
factor each so the asymmetry decomposes cleanly:

| run | config | goals | catalog | isolates |
|---|---|---|---|---|
| **A** | `fed_webshop_scaled_catalog.yaml` (catalog_split) | disjoint | disjoint | env + task heterogeneity |
| **B** | `fed_webshop_scaled_task.yaml` (task_disjoint) | disjoint | FULL | task heterogeneity only |
| **C** | `fed_webshop_scaled_homog.yaml` (partition="") | shared | FULL | IID baseline |

→ **A − B = pure environment-heterogeneity effect**, **B − C = pure task-heterogeneity effect**, both
under FedAvg. Expected (Input-Dynamics Asymmetry): A−B negative (env-het hurts), B−C ≈ 0 (task-het robust).
Analyze with `tools/verl08_migration/summarize_fed_run.py A=… B=… C=…`.

**Fx — FedProx hook test.** `config/fed_webshop_fedprox_test.yaml` (μ=0.1). Verifies the non-fork FedProx
injection (`fedagent/fedprox.py` patches `FSDPEngine.optimizer_step` via the Ray
`worker_process_setup_hook`). Look for `[fedprox] enabled: proximal mu=0.1` in the worker log.

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

## Planned experiments (roadmap)

Priority order; each row updated to ✅ with results as it lands. "Cost" = rough GPU-hours on 2×H100.

| Pri | Experiment | Depends on | Cost | Purpose |
|---|---|---|---|---|
| 1 | **Finish A/B/C (seed 1)** + `summarize_fed_run` | A/B running, C queued | ~0.5h | first env-vs-task asymmetry decomposition |
| 2 | **FedProx hook test** (Fx) → if OK, **FedProx A/B** (μ sweep) | fedprox.py wired | ~2h | FedAvg-vs-FedProx baseline under env-het |
| 3 | **A/B/C × 3 seeds** (`--base-seed 42/43/44`) | #1 signal seen | ~3h | seed robustness of the asymmetry |
| 4 | **Faithful task variants**: Preference(ω), Coverage(ξ), Hardness(ξ') | port to `_partition_specs` | ~3h | canonical task arm (vs the task_disjoint stand-in) |
| 5 | **Other env variants**: Field-Subset, BM25 Reweight, Lookalike, Rank Wrapper | port partition fns + service kwargs | ~4h | full env-het suite (Patterns B/C/D) |
| 6 | **Baselines**: Local (1 client), Centralized (1 client, all data) | configs only | ~2h | FedAgent vs Centralized vs Local |
| 7 | **ALFWorld** env_disjoint | ALFWorld remote service (Phase 3) | ~4h | 2nd environment, env-het generality |
| 8 | **PPO** variants of the key conditions | config (verl native PPO) | ~3h | GRPO + PPO coverage |
| 9 | **Phase 8-full**: 3 seeds × E×T=210 × {1.5B,3B,7B} × {GRPO,PPO}, unperturbed val | all above + ops resume | large (needs a dedicated allocation) | full reproduction vs 0.3.1 |

Open design points to resolve as we go:
- **Unperturbed validation** (science red line): val service must use `catalog_filter_asins=None`; wire when enabling `test_freq>0`.
- **Seeds**: also vary verl's training seed (not just `FEDAGENT_BASE_SEED`) for true seed independence.
- **GiGPO equivalence**: confirm the asymmetry conclusions hold under native concat-GRPO (validated implicitly by #1/#9).
- **Ops for long runs**: `run_fed` resume / checkpoint-rotation before the 70-round Phase-8 campaign.

## Reproduce

```bash
# on the GPU node (srun --jobid=<JID> --overlap):
bash fedagent/scripts/run_smoke.sh                         # TinyGuess package smoke
bash fedagent/scripts/run_tinyguess_fed_smoke.sh           # federated loop (TinyGuess 2x2)
bash fedagent/scripts/run_webshop_fed_smoke.sh CFG         # federated WebShop (CFG = a fed config)
#   CFG ∈ config/{fed_webshop_2cl_catalog_split, fed_webshop_scaled_catalog,
#                 fed_webshop_scaled_task, fed_webshop_scaled_homog,
#                 fed_webshop_probe_signal, fed_webshop_fedprox_test}.yaml
# extra args forwarded to run_fed, e.g.:  ... CFG --base-seed 43 --output-dir /tmp/run_s43 --port-base 8090
python tools/verl08_migration/summarize_fed_run.py A=/tmp/...scaled_env B=/tmp/...scaled_task C=/tmp/...scaled_homog
```
