# ALFWorld benchmark configs

The ALFWorld acceleration-economics probes. Full rationale + results:
[`fedagent/docs/alfworld_testing.md`](https://github.com/sunblaze-ucb/FedAgent/tree/migrate/verl-0.8.0/fedagent/docs/alfworld_testing.md) (migrate/verl-0.8.0 branch). All 1.5B
(Qwen2.5-1.5B-Instruct), `response_length=4096`, ALFWorld service in conda env `verl-agent-alfworld`.
Output → gitignored `runs/`.

> `§` citations in the table below refer to the [full archived acceleration analysis](https://github.com/sunblaze-ucb/FedAgent/tree/migrate/verl-0.8.0/fedagent/docs/acceleration.md);
> main's [`fedagent/docs/acceleration.md`](../../../../fedagent/docs/acceleration.md) is the how-&-why summary.

| config(s) | experiment | result |
|---|---|---|
| `alf_smoke.yaml` | de-risk: 0.5B, 1 client/round, eval off — confirm service (8810-game load ~3 min) + federated loop close end-to-end | GREEN (rc=0) |
| `alf_em_{inline,parallel,shared,worker}.yaml` | **eval-mode sweep** — 2 client × 2 round, eval every round, 48-game val (`alfworld_val_48.yaml`); same methodology as the WebShop sweep | **worker 3509s < parallel 3620s < shared 4560s < inline 4738s** — eval-decoupled {worker,parallel} beat eval-coupled by ~25–30% |
| `alf_scale_{g1,g2,g4}.yaml` | **GPU-scaling** — 1 client/round, 1 step, eval off; isolate `timing_s/step` at 1/2/4 GPU | step 534 / 387 / 298 s; **gen FLAT (228→219, env-bound) while update_actor scales (140→43)**; 1-GPU +38%/step |
| `alf_conc_{A,B}.yaml` | **Tier-1 concurrency** — 2 independent training jobs on GPUs {0,1}+{2,3}, both weight-syncing, to stress the ZMQ `VERL_RAY_JOB_ID` fix on ALFWorld's 2-service load | **PASS** — both rc=0, no deadlock (A 392s, B 473s) |
| `alf_scale_g{4,1}_r8.yaml`, `alf_scale_g1_r{1,4,8}n1.yaml` | **Tier-1 replica sharding** (`alfworld_replicas`) — K-sweep + pool control + 4/1-GPU components (incl. the 1×H100/8-core node) | gen **217.5→65.8→61.8** (K1/4/8); pool irrelevant (control 217.5); 4-GPU step **298→127.6** (−57%); 1-GPU 534→350–359 (−33%, K=4 enough on 8 cores) |
| `alf_em_worker_r8.yaml` | **end-to-end A/B**: the worker baseline config + `alfworld_replicas: 8` (train+val services) | **3509 → 2412 s (−31%)**, steps −65%, val healthy |
| `alfworld_val_48.yaml` | 48-of-140 `valid_seen` val spec used by the eval-mode sweep (big enough to surface "shared throttles", small enough for a 4-mode sweep) | — |
| `alf_scale_g4_r8{dyn,fused}.yaml` | **Tier-2 within-step probes** on the K=8 4-GPU base: `use_dynamic_bsz` / fused kernels | dyn **+11%** (step 127.6→141.7 — refuted, FLOP-bound); fused wash (+2%) — acceleration.md §10.2 |
| `alf_scale_g1_paper_r{1,4}.yaml` | **1×H100 at the real paper caps** (2048/512/2560), K=1 / K=4 (`../run_g1_paper_stack.sh`) | r1: step **514 s** (gen 212/olp 52/ref 105/upd 139) vs 534 s at 4096-caps — cap choice immaterial (env-bound); EXPERIMENTS.md 2026-07-02 |
| `alf_t2_{base,cache,scope,feval,all}.yaml` | **Tier-2 knob arms on the real config**: same-config replicate + manifest cache + `service_scope: run` + `final_eval_mode: worker` + all-four (`../run_t2_stack.sh`) | replicate = noise floor **9.293e-5**; walls **−18/−16/−11/−24%**, every arm ≤ floor — acceleration.md §10.1 |
| `paper_alf_wiring_r8.yaml` | **first full paper-config run** (uniform/1.5B/grpo, 100-client partition, 2 rounds, adopted stack) | rc=0, **3719 s** (1766 + 1375 + 578 cold final eval); val 0.0429→**0.1143** (n=140) |
| `paper_alf_combo{,_lanes}.yaml` | **the final recipe** (r8 + 4×Tier-2) ± `parallel_clients: 2` | combo **3202 s** (steady round **762 s** vs 1125; hot final eval 389); lanes 3136 s = **wash** → not adopted; 70-round **16.7 h** — acceleration.md §10.4 |

**Drivers** (in `runs/` — gitignored, transient): `run_alf_evalmode.sh` (committed, in `../`) ran the
sweep; `run_rerun.sh` / `run_alf_scale.sh` / `run_alf_conc.sh` ran the scaling + concurrency + the
durable rerun; `../run_t2_stack.sh`, `../run_g1_paper_stack.sh`, `../run_paper2r_stack.sh` (committed)
ran the Tier-2 arms, the 1-GPU paper probes and the wiring/combo stack. Full-task offline
per-task-type eval: `tools/verl08_migration/eval_alfworld_by_tasktype.py`.

> **Key cross-env finding.** ALFWorld *flips* the WebShop eval-mode ranking (`parallel<worker<inline<shared`):
> worker overtakes parallel (cross-round cold-start amortization pays off on a heavy eval) and inline
> becomes slowest (its per-round eval-engine re-spin dominates). See alfworld_testing.md §6 (migrate/verl-0.8.0 branch).
