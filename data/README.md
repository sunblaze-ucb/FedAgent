# Data

What ships on this branch, and where the rest comes from:

**WebShop catalog, vendored in-tree.** The small catalog backing the paper's
`use_small` code path is committed beside the engine at
`fedagent/envs/webshop/engine/webshop/data/`
(`items_shuffle_1000.json`, `items_ins_v2_1000.json`, `items_human_ins.json`); the env
service loads it from there, no download step. The full ~5.2 GB catalog
(`items_shuffle.json`, `items_ins_v2.json`) is only needed for full-catalog runs; drop it
into the same directory if you need it (it is gitignored).

**ALFWorld game files, one-time download.** Populate `$ALFWORLD_DATA` (default
`~/.cache/alfworld`) from the ALFWorld env's conda environment:

```bash
alfworld-download -f
```

Export the **same** `ALFWORLD_DATA` when launching training. Full recipe:
[`../fedagent/docs/installation.md`](../fedagent/docs/installation.md) §3.

**Bundled env-heterogeneity data (`env_heterogeneity/`).** Small, derived data files used
by the *environment-level* heterogeneity experiments are committed here (they are inputs,
not run configs; the run configs live under `fedagent/config/paper/env_heterogeneity/`):

- `lookalike_data/lookalike_v_{price,color,size,price_color}.json`, pre-synthesized
  lookalike / distractor product pools injected by the `lookalike_injection*` runs.
- `holdout_{webshop,alfworld}_v1.json`, env-level OOD holdout sets for the
  `catalog_split` / scene-disjoint runs (forwarded to the env services via
  `HOLDOUT_FILE`). These paths are resolved relative to the repo root; launch
  `run_fed` from there.

The original generator scripts for the holdout files live on the
**paper-reproduce branch** (`tools/env_heterogeneity/gen_holdout_{webshop,alfworld}.py`
there); the shipped v1 artifacts are the ones the paper used.

**Hardness labels** (task-level heterogeneity reference trajectories) are a separate
bundle under [`../fedagent/data/hardness/`](../fedagent/data/hardness/README.md).

These are **derived from the WebShop and ALFWorld benchmark data (both MIT)**, see
[`../NOTICE`](../NOTICE) for attribution.
