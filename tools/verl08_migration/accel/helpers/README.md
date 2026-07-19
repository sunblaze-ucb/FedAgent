# Helpers

Standalone scripts used by the acceleration drivers. Output → gitignored.

| script | purpose |
|---|---|
| `standalone_eval.py` | Faithful standalone val pass, drives `run_fed._build_eval` → the same verl val-only path the loop uses. Used by `../run_complete.sh` to measure hidden-eval cost. Usage: `python standalone_eval.py <eval_spec.yaml> <gpu_ids>`. |
| `cmp_hf.py` | Tensor-by-tensor HF safetensors compare (cross-mode weight equivalence, e.g. persistent vs subprocess aggregated checkpoints). |
| `critic_diag.py` | PPO critic-reload diagnostics (verifies the critic state survives cross-round persistence). |
