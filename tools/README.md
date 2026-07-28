# `tools/`

Maintained operator tooling for the verl-0.8 FedAgent overlay. Everything here is run **by
hand against the repo**; code the training loop itself needs lives inside `fedagent/` (the
FedAvg aggregator run_fed shells out to each round is
[`fedagent/fed/aggregate_fedavg_fsdp.py`](../fedagent/fed/aggregate_fedavg_fsdp.py)).

| script | purpose |
|---|---|
| [`gen_paper_configs.py`](gen_paper_configs.py) | regenerates the 176-config paper matrix under `fedagent/config/paper/` (`--accel` for the accelerated twins) |
| [`gen_hardness_trajectories.py`](gen_hardness_trajectories.py) | generates the WebShop `data/hardness/*.json` task-difficulty labels the hardness heterogeneity arm requires |
| [`gen_alfworld_hardness_trajectories.py`](gen_alfworld_hardness_trajectories.py) | the ALFWorld twin: rolls the reference model over the 3,553 train games (greedy windowed episodes via an unperturbed train-split service) and records per-game `task_id -> success` labels |
| [`eval_alfworld_by_tasktype.py`](eval_alfworld_by_tasktype.py) | per-task-type success breakdown on the ALFWorld validation split — default `--mode single` is the paper-table estimator (ONE pass grouped by the per-row `task_type` tag; types pool to All); `--mode per-type-passes` is the legacy 7-pass high-sample variant (different quantity) |
| [`gen_alfworld_manifest.py`](gen_alfworld_manifest.py) | generates / verifies `data/alfworld_games/*.json`, the authoritative per-split game lists the ALFWorld engine reads instead of walking `$ALFWORLD_DATA`. `--check` compares a data root against the shipped manifests and exits non-zero on drift (run it after any `alfworld-download`). Regenerating **changes which games every ALFWorld run uses** — see [revision.md](../fedagent/docs/revision.md). Stdlib only: runs wherever the data is |
| [`verify_train_val_disjoint.py`](verify_train_val_disjoint.py) | proves train never touches val: WebShop analytically through the REAL shard function (`fedagent/hetero/webshop_uniform.py` with `val_size` = the service's `WEBSHOP_VAL_SIZE`), ALFWorld empirically (trial-id walk over `$ALFWORLD_DATA`, `--check-content` adds SHA1). Ported 2026-07-28 from the original experiment repo; the port imports the shipped partition code instead of mirroring it, so the check cannot drift |
| [`summarize_fed_run.py`](summarize_fed_run.py) | post-processes a run directory (round/client metrics, asymmetry/compounding summary). Reads whichever metrics layout the run wrote: per-client `json_logs/`, the persistent path's shared per-round `json_logs/` (clients split on the step reset; cross-round logs are cumulative, so the round's own clients are the last segments), per-lane `json_logs_lane*/`, or the raw `client_*/training.log` |
| [`rebuild_summary.py`](rebuild_summary.py) | rebuilds `federated_summary.json` from a run's on-disk artifacts when the run died before `run_fed`'s teardown could write it (every figure is drawn from that file). Same fold-in as the teardown — val curve, client circles, per-round train metrics — via the shared reader `fedagent/fed/eval_dumps.py`; output is marked `"_reconstructed": true` |
| [`collect_fed_logs.sh`](collect_fed_logs.sh) | gathers per-node `training.log`s into one stage dir, then runs the summarizer |
| [`plot_training_dynamics.py`](plot_training_dynamics.py) | training-dynamics figures — one call renders the default set of 4: the paper pair `val/task_score` + `val/success_rate`, each plain AND `_with_clients`. Scale/label are per metric (`success_rate` ×100 `(%)`; `task_score` ×100 `(0-100)` — a score, **not** a percentage; `--raw` for stored [0,1]). `val/reward_mean` is deliberately not in the set: the reward is the binarized {0,10} success signal, so it duplicates success_rate ×10. Reads `federated_summary.json` (verl-0.8 runs) or the legacy `round_*/client_*/json_logs/metrics.json` |

## [`env_heterogeneity/`](env_heterogeneity/) — provenance of the shipped env-het assets

Deterministic generators/probes behind the committed `data/env_heterogeneity/` artifacts.
Each one either reproduces its committed output exactly or says why it can't:

| script | artifact | notes |
|---|---|---|
| [`synthesize_lookalike.py`](env_heterogeneity/synthesize_lookalike.py) | `data/env_heterogeneity/lookalike_data/` (V4 Lookalike pools) | recovered original script (SEED=99999), paths adapted; regeneration is content-identical to the shipped pools |
| [`gen_holdout_webshop.py`](env_heterogeneity/gen_holdout_webshop.py) | `data/env_heterogeneity/holdout_webshop_v1.json` | ported 2026-07-28; `--check` re-verifies (30 distractor ASINs, seed 99999, byte-stable) |
| [`gen_holdout_alfworld.py`](env_heterogeneity/gen_holdout_alfworld.py) | `data/env_heterogeneity/holdout_alfworld_v1.json` | ported 2026-07-28; `--check` re-verifies against the local `$ALFWORLD_DATA` walk (8 scenes / 264 trials; guard the dataset itself with `gen_alfworld_manifest.py --check`) |
| [`probe_bm25_effective_fields.py`](env_heterogeneity/probe_bm25_effective_fields.py) | measured V2/V3 divergence stats quoted in the paper appendix | service-faithful re-measurement (normalized products, verbatim Searcher). Replaces the original repo's `probe_bm25_real_queries.py`, which probed RAW products — a configuration the service never runs (see docs/heterogeneity.md) |

[`setup/`](setup/) holds install-time assets: `build_fa.sh` (flash-attn source build for
old-glibc nodes) and [`patches/`](setup/patches/) (the 2-line verl weight-transfer patch
applied during [installation](../fedagent/docs/installation.md)).

## Legacy tooling inventory (what was ported, what wasn't, and why)

The original experiment repo (`federated_agent/tools/`) and the intermediate
`paper-reproduce-verl-agent` branch carry ~40 more scripts. Audited 2026-07-28; disposition:

**Ported** (above): `verify_train_val_disjoint.py`, `gen_holdout_webshop.py`,
`gen_holdout_alfworld.py`, `synthesize_lookalike.py`.

**Superseded by this overlay** — not ported because a maintained successor exists here:

| legacy | successor |
|---|---|
| `aggregation/` (`verl_fsdp_aggregation.py`, `check/verify_aggregation.py`, DTensor fix-ups) | [`fedagent/fed/aggregate_fedavg_fsdp.py`](../fedagent/fed/aggregate_fedavg_fsdp.py) (`--phase verify` re-loads shards and exits non-zero on FAIL) |
| `monitor/` (out-of-band checkpoint-cleanup daemon) | run_fed's in-loop `cleanup_round_checkpoints` + the `keep_client_hf_rounds` rolling window |
| `env_heterogeneity/probe_bm25_*.py` | `probe_bm25_effective_fields.py` (the old probes measured raw products the service never serves) |
| `env_heterogeneity/plot_aggregated_curves.py`, `viz_sweep_summary*.py` | `summarize_fed_run.py` + `plot_training_dynamics.py` |
| `viz_{webshop,alfworld}_partition.py` | ALFWorld: the vendored engine emits the partition figures itself into the run dir (per round, per client). WebShop: the service logs each client's catalog/goal-shard composition at startup, and the partition math is covered by unit tests — the standalone figure scripts drove the OLD launcher's layout |
| `audit_yaml_consistency.py`, `generate_{uniform,decentralized,…}_configs.py` | `gen_paper_configs.py` — configs are *generated*, so consistency is by construction |
| `heterogeneity_test/` partition simulators | unit tests under [`tests/`](../tests/) (`test_hardness_fill.py`, `test_catalog_split_runtime.py`, …) |
| `eval/` offline checkpoint-eval harness (`evaluate.sh`, `batch_*_eval.sh`, `merge_trajectories.py`, `view_results.py`) | run_fed's in-loop `eval_global` / final eval + `eval_alfworld_by_tasktype.py`. The legacy harness drives verl-agent-0.3.1 entrypoints and only makes sense on that branch |

**Not ported — legacy-stack ops with nothing to run against here** (preserved on the
`paper-reproduce-verl-agent` branch / the original repo): the devbox sweep orchestration
(`sweep_controller_*.sh`, `respawn_controllers.sh`, relaunch/cleanup helpers), `smoke/`
batch lists, `misc/` session helpers, `mem_sampler.sh`, and the old launcher pair
`run_federated.py` + `resolve_paths.py`. They shell out to the legacy repo's entrypoints
and directory layout; porting them would mean porting the old stack.

History lives on branches, not here: the verl-0.8 migration's A/B experiment matrix,
verification smokes and diagnostics (the former `tools/verl08_migration/{accel,poc,archived_diagnostics}`)
are preserved on `migrate/verl-0.8.0`; the original verl-agent-0.3.1 tooling
(`run_federated.py`, `aggregation/`, `monitor/`, …) on `paper-reproduce-verl-agent`.
