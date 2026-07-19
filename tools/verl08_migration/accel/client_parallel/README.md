# Client-parallel (#3) layout configs

Lever #3, "parallel clients within a round": run `clients_per_round` clients concurrently on disjoint
GPU sets on one node. `A`/`B` = the two concurrent client jobs. Analysis:
[archive acceleration.md §Lever #3 / §7.7](https://github.com/sunblaze-ucb/FedAgent/tree/migrate/verl-0.8.0/fedagent/docs/acceleration.md). Eval spec for the worker
variants: `../p3_eval_2gpu.yaml` (top-level, referenced by absolute path). Output → gitignored.

| config(s) | experiment | result | doc |
|---|---|---|---|
| `p3_1gpu_A.yaml`, `p3_1gpu_B.yaml` | "2 train on 1 GPU + 2-GPU hidden eval" layout: t1(1)=995s, eval 407s hidden | correctness OK but **not** the fast path; surfaced the 3-job ZMQ deadlock + fix | §7.7 |
| `p3_2gpu_A.yaml`, `p3_2gpu_B.yaml`, `p3_4gpu.yaml` | GPU-scaling + 2-client concurrency: t1(4)=558, t1(2)=725, #3 2×2=727s (−35%) | sub-linear scaling; #3 coexists on 4 GPU | §Lever #3 |
| `p3_2gpu_worker_A.yaml`, `p3_2gpu_worker_B.yaml` | #3 + `eval_mode=worker` (hot-engine eval) | **845s/round, the recommended single-node fast path** | §7.7 |

Drivers: `../run_p3.sh`, `../run_p3_baseline.sh`, `../run_worker3.sh`, `../run_complete.sh`,
`../run_2x1gpu.sh` (historical, reference the retired `_scratch/accel/` base; see `../README.md`).

> The ZMQ weight-transfer deadlock this layout first exposed (every isolated Ray cluster picks the same
> first job_id `01000000` → same `/tmp` socket) is fixed via `VERL_RAY_JOB_ID`, and **re-confirmed on
> ALFWorld** by `../alfworld/alf_conc_{A,B}.yaml`.
