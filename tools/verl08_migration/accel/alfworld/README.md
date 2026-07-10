# ALFWorld benchmark configs

The ALFWorld acceleration-economics probes. Full rationale + results:
[`fedagent/docs/alfworld_testing.md`](https://github.com/sunblaze-ucb/FedAgent/tree/migrate/verl-0.8.0/fedagent/docs/alfworld_testing.md) (migrate/verl-0.8.0 branch). All 1.5B
(Qwen2.5-1.5B-Instruct), `response_length=4096`, ALFWorld service in conda env `verl-agent-alfworld`.
Output → gitignored `runs/`.

| config(s) | experiment | result |
|---|---|---|
| `alf_smoke.yaml` | de-risk: 0.5B, 1 client/round, eval off — confirm service (8810-game load ~3 min) + federated loop close end-to-end | GREEN (rc=0) |
| `alf_em_{inline,parallel,shared,worker}.yaml` | **eval-mode sweep** — 2 client × 2 round, eval every round, 48-game val (`alfworld_val_48.yaml`); same methodology as the WebShop sweep | **worker 3509s < parallel 3620s < shared 4560s < inline 4738s** — eval-decoupled {worker,parallel} beat eval-coupled by ~25–30% |
| `alf_scale_{g1,g2,g4}.yaml` | **GPU-scaling** — 1 client/round, 1 step, eval off; isolate `timing_s/step` at 1/2/4 GPU | step 534 / 387 / 298 s; **gen FLAT (228→219, env-bound) while update_actor scales (140→43)**; 1-GPU +38%/step |
| `alf_conc_{A,B}.yaml` | **Tier-1 concurrency** — 2 independent training jobs on GPUs {0,1}+{2,3}, both weight-syncing, to stress the ZMQ `VERL_RAY_JOB_ID` fix on ALFWorld's 2-service load | **PASS** — both rc=0, no deadlock (A 392s, B 473s) |
| `alf_scale_g{4,1}_r8.yaml`, `alf_scale_g1_r{1,4,8}n1.yaml` | **Tier-1 replica sharding** (`alfworld_replicas`) — K-sweep + pool control + 4/1-GPU components (incl. the 1×H100/8-core node) | gen **217.5→65.8→61.8** (K1/4/8); pool irrelevant (control 217.5); 4-GPU step **298→127.6** (−57%); 1-GPU 534→350–359 (−33%, K=4 enough on 8 cores) |
| `alf_em_worker_r8.yaml` | **end-to-end A/B**: the worker baseline config + `alfworld_replicas: 8` (train+val services) | **3509 → 2412 s (−31%)**, steps −65%, val healthy |
| `alfworld_val_48.yaml` | 48-of-140 `valid_seen` val spec used by the eval-mode sweep (big enough to surface "shared throttles", small enough for a 4-mode sweep) | — |

**Drivers** (in `runs/` — gitignored, transient): `run_alf_evalmode.sh` (committed, in `../`) ran the
sweep; `run_rerun.sh` / `run_alf_scale.sh` / `run_alf_conc.sh` ran the scaling + concurrency + the
durable rerun. Full-task offline per-task-type eval: `tools/verl08_migration/eval_alfworld_by_tasktype.py`.

> **Key cross-env finding.** ALFWorld *flips* the WebShop eval-mode ranking (`parallel<worker<inline<shared`):
> worker overtakes parallel (cross-round cold-start amortization pays off on a heavy eval) and inline
> becomes slowest (its per-round eval-engine re-spin dominates). See alfworld_testing.md §6 (migrate/verl-0.8.0 branch).
