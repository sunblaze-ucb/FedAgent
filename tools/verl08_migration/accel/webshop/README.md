# WebShop benchmark configs

WebShop acceleration experiments (Qwen2.5-1.5B-Instruct, paper settings). Numbers:
[`acceleration_results.md`](../../../../fedagent/docs/acceleration_results.md);
analysis: [`acceleration.md`](../../../../fedagent/docs/acceleration.md). Shared val specs
`../webshop_val_64.yaml`, `../webshop_val_tiny.yaml` (referenced by absolute path). Output → gitignored.

| config(s) | experiment | doc |
|---|---|---|
| `ws_eval_inline.yaml`, `ws_eval_parallel.yaml`, `paper_ws_mode_{inline,parallel,shared,worker}.yaml`, `paper_ws_modeB_{inline,shared,worker}.yaml` | **eval-mode sweep** (inline/parallel/shared/worker) — n=500: parallel 2493 < worker 2637 < inline 3090 < shared 3316 | §7.4 |
| `ws_xround_{parallel,shared,worker,worker_eager,val}.yaml`, `ws_clean_worker.yaml` | cross-round persistence (`cross_round: true`) × eval-mode | §7.2 / §7.4 |
| `ws_clientend.yaml`, `ws_clientend15.yaml`, `ws_clientend_worker.yaml`, `ws_clientend15_worker.yaml` | client-end eval "circles" (per-client val marks) | §7.4 |
| `webshop_prewarm_on.yaml`, `webshop_prewarm_off.yaml` | lever #2 — env-service pre-warm (benefit ≈ 0 for homogeneous WebShop) | §Lever #2 |
| `ws_scale_g1.yaml`, `ws_scale_g1b.yaml`, `ws_scale_g4.yaml` | **first WebShop `timing_s` decomposition** (1 step, eval off; g1=1×H100 node, g1b=1 GPU on the 4-GPU node, g4=4 GPU) | **GPU-compute-bound** (74% @1 GPU); gen flat-ish 54.6→44.1; per-step 1-GPU penalty **2.41×** (corrects the wall-based 1.37×) | acceleration.md §9.1 |
| `ws_scale_g4_p64.yaml`, `ws_scale_g4_p64r4.yaml` | gen levers: pool 16→64 alone vs + `webshop_replicas: 4` | pool-only **hurts** (gen 44.1→50.1, GIL); +replicas → step 93.4→**82.2 (−12%)** | acceleration.md §9.1 |
| `ws_route.yaml`, `ws_ab_subproc.yaml`, `ws_ab_xround.yaml`, `paper_ws_grpo15_wiring.yaml` | per-client service routing, subprocess vs persistent A/B, GRPO wiring | §7.3 |

Drivers: `../run_evalmode.sh`, `../run_paper_modes.sh`, `../run_paper_modesB.sh`, `../run_paper_4card.sh`,
`../run_ws_smoke.sh` (historical — reference the retired `_scratch/accel/` base; see `../README.md`).
