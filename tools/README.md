# `tools/`

Maintained operator tooling for the verl-0.8 FedAgent overlay. Everything here is run **by
hand against the repo**; code the training loop itself needs lives inside `fedagent/` (the
FedAvg aggregator run_fed shells out to each round is
[`fedagent/fed/aggregate_fedavg_fsdp.py`](../fedagent/fed/aggregate_fedavg_fsdp.py)).

| script | purpose |
|---|---|
| [`gen_paper_configs.py`](gen_paper_configs.py) | regenerates the 176-config paper matrix under `fedagent/config/paper/` (`--accel` for the accelerated twins) |
| [`gen_hardness_trajectories.py`](gen_hardness_trajectories.py) | generates the `data/hardness/*.json` task-difficulty labels the hardness heterogeneity arm requires |
| [`eval_alfworld_by_tasktype.py`](eval_alfworld_by_tasktype.py) | per-task-type success breakdown on the ALFWorld validation split |
| [`gen_alfworld_manifest.py`](gen_alfworld_manifest.py) | generates / verifies `data/alfworld_games/*.json`, the authoritative per-split game lists the ALFWorld engine reads instead of walking `$ALFWORLD_DATA`. `--check` compares a data root against the shipped manifests and exits non-zero on drift (run it after any `alfworld-download`). Regenerating **changes which games every ALFWorld run uses** — see [revision.md](../fedagent/docs/revision.md). Stdlib only: runs wherever the data is |
| [`summarize_fed_run.py`](summarize_fed_run.py) | post-processes a run directory (round/client metrics, asymmetry/compounding summary). Reads whichever metrics layout the run wrote: per-client `json_logs/`, the persistent path's shared per-round `json_logs/` (clients split on the step reset; cross-round logs are cumulative, so the round's own clients are the last segments), per-lane `json_logs_lane*/`, or the raw `client_*/training.log` |
| [`rebuild_summary.py`](rebuild_summary.py) | rebuilds `federated_summary.json` from a run's on-disk artifacts when the run died before `run_fed`'s teardown could write it (every figure is drawn from that file). Same fold-in as the teardown — val curve, client circles, per-round train metrics — via the shared reader `fedagent/fed/eval_dumps.py`; output is marked `"_reconstructed": true` |
| [`collect_fed_logs.sh`](collect_fed_logs.sh) | gathers per-node `training.log`s into one stage dir, then runs the summarizer |
| [`plot_training_dynamics.py`](plot_training_dynamics.py) | training-dynamics figures — one call renders the default set of 4: the paper pair `val/task_score` + `val/success_rate`, each plain AND `_with_clients`. Scale/label are per metric (`success_rate` ×100 `(%)`; `task_score` ×100 `(0-100)` — a score, **not** a percentage; `--raw` for stored [0,1]). `val/reward_mean` is deliberately not in the set: the reward is the binarized {0,10} success signal, so it duplicates success_rate ×10. Reads `federated_summary.json` (verl-0.8 runs) or the legacy `round_*/client_*/json_logs/metrics.json` |

[`setup/`](setup/) holds install-time assets: `build_fa.sh` (flash-attn source build for
old-glibc nodes) and [`patches/`](setup/patches/) (the 2-line verl weight-transfer patch
applied during [installation](../fedagent/docs/installation.md)).

History lives on branches, not here: the verl-0.8 migration's A/B experiment matrix,
verification smokes and diagnostics (the former `tools/verl08_migration/{accel,poc,archived_diagnostics}`)
are preserved on `migrate/verl-0.8.0`; the original verl-agent-0.3.1 tooling
(`run_federated.py`, `aggregation/`, `monitor/`, …) on `paper-reproduce-verl-agent`.
