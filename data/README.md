# Data

The WebShop and ALFWorld environment data are **not** bundled with this
repository, they are public but large (the full WebShop catalog alone is
~5.2 GB). Fetch them with the top-level downloader:

```bash
bash ../download_data.sh           # both environments
bash ../download_data.sh --webshop # WebShop only
```

**Shipped small variants.** Three small WebShop files are committed and back the
`webshop.use_small: true` code path (used by the default / smoke configs):

- `items_shuffle_1000.json`
- `items_ins_v2_1000.json`
- `items_human_ins.json`

The full catalog (`items_shuffle.json`, `items_ins_v2.json`) and the ALFWorld
game cache are only required for full-scale runs and are gitignored.

**Bundled env-heterogeneity data (`env_heterogeneity/`).** Small, derived data
files used by the *environment-level* heterogeneity experiments are committed here
(they are inputs, not run configs, the run configs live under
`config/env_heterogeneity/`):

- `lookalike_data/lookalike_v_{price,color,size,price_color}.json`, pre-synthesized
  lookalike / distractor product pools injected by the `lookalike_injection*` runs.
- `holdout_{webshop,alfworld}_v1.json`, env-level OOD holdout sets for the
  `catalog_split` / scene-disjoint runs (regenerate with
  `tools/env_heterogeneity/gen_holdout_{webshop,alfworld}.py`).

These are **derived from the WebShop and ALFWorld benchmark data (both MIT)**: see
[`../NOTICE`](../NOTICE) for attribution.

**Where each artifact lands.** WebShop product data (the shipped `*_1000.json` /
`items_human_ins.json` small files, and the full `items_shuffle.json` /
`items_ins_v2.json` catalog if you fetch it) lives in
`fedagent/envs/webshop/engine/webshop/data/`, which is where the env loads it from. ALFWorld game files land in `$ALFWORLD_DATA`
(default `~/.cache/alfworld`); export the **same** `ALFWORLD_DATA` for both
`download_data.sh` and training. The bundled `env_heterogeneity/` files above stay
where they are committed under `data/`.
