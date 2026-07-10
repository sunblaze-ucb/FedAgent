# Acceleration & env benchmarks — configs, drivers, helpers

Reproducibility artifacts behind the acceleration + ALFWorld workstreams
([`fedagent/docs/acceleration.md`](../../../fedagent/docs/acceleration.md),
[`acceleration_results.md`](https://github.com/sunblaze-ucb/FedAgent/tree/migrate/verl-0.8.0/fedagent/docs/acceleration_results.md),
[`alfworld_testing.md`](https://github.com/sunblaze-ucb/FedAgent/tree/migrate/verl-0.8.0/fedagent/docs/alfworld_testing.md) — both on the migrate/verl-0.8.0 branch). These are **one-off experiment
scaffolding**, not maintained APIs — kept in-repo (out of the retired, gitignored `_scratch/`) so the
GPU-validated results stay reproducible.

## Layout

```
accel/
├── webshop/          WebShop configs — eval-mode sweep, cross-round, client-end marks, prewarm
├── alfworld/         ALFWorld configs — eval-mode sweep, GPU-scaling, concurrency, smoke (+ val spec)
├── client_parallel/  #3 "parallel clients within a round" layout configs (p3_*)
├── dev/              equivalence / persistence dev smokes (persistent #4, cross-round, PPO, TinyGuess)
├── helpers/          standalone eval + weight-compare + critic-diag scripts
├── run_*.sh          drivers (entry points; run from repo root)
└── *_val_*.yaml, p3_eval_2gpu.yaml   shared val/eval specs (referenced by configs via absolute path)
```

Each subfolder has its own `README.md` mapping its configs → the result they produced → the doc section.

| subfolder | what | doc |
|---|---|---|
| [`webshop/`](./webshop/) | WebShop eval-mode sweep (inline/parallel/shared/worker), cross-round persistence, client-end eval marks, lever-#2 prewarm | acceleration.md §7.4 / §Lever #2 |
| [`alfworld/`](./alfworld/) | **ALFWorld** eval-mode sweep, GPU-scaling (g1/g2/g4), 2-job concurrency, de-risk smoke | alfworld_testing.md §6 (migrate/verl-0.8.0 branch) |
| [`client_parallel/`](./client_parallel/) | #3 client-parallel layout (1/2/4-GPU, +worker eval) | acceleration.md §Lever #3 / §7.7 |
| [`dev/`](./dev/) | persistent-trainer (#4) vs subprocess equivalence, cross-round, PPO critic-reload, TinyGuess/windowed smokes | acceleration.md §7.1 / §7.2 |
| [`helpers/`](./helpers/) | `standalone_eval.py`, `cmp_hf.py`, `critic_diag.py` | — |

## Drivers (`run_*.sh`)

Entry points, run from the repo root. The **live** drivers reference configs at their subfolder
paths (`accel/<env>/…`): `run_alf_evalmode.sh` (ALFWorld eval-mode sweep), `run_t2_stack.sh`
(Tier-2 knob arms + the final combos), `run_g1_paper_stack.sh` (1×H100 paper-geometry probes),
`run_ws_r4.sh` (WebShop paper wiring), `run_paper2r_stack.sh` (2-round paper wiring/combo stack),
`run_oso_probe.sh` (`one_step_off` probe). The **historical WebShop/#3 drivers**
(`run_p3.sh`, `run_evalmode.sh`, `run_complete.sh`, `run_2x1gpu.sh`, `run_paper_*.sh`, `run_worker3.sh`,
`run_ws_smoke.sh`, `run_xround_*.sh`, …) were authored against the **retired `_scratch/accel/` base** for
both config input and (gitignored) output, and are kept **for provenance** — to re-run one, resolve its
config from the appropriate subfolder (`accel/<env>/<cfg>.yaml`) and point output at any gitignored dir.
The numbers they produced are recorded in the acceleration docs.

## Outputs
Run **output** (checkpoints, dumps, per-config logs) is gitignored — sent to `runs/` or any scratch
location, never committed. Source lives here; outputs do not. (See [[no-scratch-dir]] rationale in the
acceleration docs.)
