# Proof-of-concept / GPU-validation source

Source-only (configs / verify scripts / custom code) from the exploratory PoC + GPU-validation runs,
moved out of the gitignored `_scratch/` (their **GB of run output was not committed**). One-off
scaffolding behind some "GPU-validated" doc claims — reference, not maintained.

- **`windowed/`** — windowed-vs-concat rollout PoC (`agent_windowed.yaml`, `alf_grpo_{concat,windowed}_rmode.yaml`, `verify_*.sh`). Backs the windowed-default work ([archive acceleration.md §7.5](https://github.com/sunblaze-ucb/FedAgent/tree/migrate/verl-0.8.0/fedagent/docs/acceleration.md)).
- **`gpu_verify/`** — GPU-validation sweep (86 files): ALFWorld GRPO/PPO context-length / `gpu_memory_utilization` / offload / heterogeneity smokes (`alf_grpo_ctx10240.yaml`, `alf_ppo_offload_off.yaml`, `het_alfworld_coverage_4gpu.yaml`, `verify_*.sh`, …).
- **`bounded/`** — bounded-rollout PoC: custom `bounded_rollout.py` / `bounded_worker.py` + ALFWorld bounded/unbounded smokes (exploratory).

Same path caveat as [`../accel/README.md`](../accel/README.md): these reference their original
`_scratch/` run paths; outputs are regenerable and were never committed.
