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

**ALFWorld game manifests (`alfworld_games/`).** The authoritative per-split game lists —
`train.json` (3553), `eval_in_distribution.json` (140 = `valid_seen`),
`eval_out_of_distribution.json` (134 = `valid_unseen`) — each a sorted list of split-relative
`game.tw-pddl` paths plus a sha256 of that list.

They exist because the download above is *not* the whole story: `game.tw-pddl` and its
`solvable` flag are produced by ALFWorld's preprocessing, so a directory walk collects
whatever that step happened to produce on a given machine (here, 47 of the 187 eligible
`valid_seen` trials have no game file at all). Everything downstream is positional — client
shards, `games[seed]`, the validation set — so a machine that preprocessed further would
silently train and evaluate on different games. The manifest makes the task set a repo asset:
the engine reads it instead of walking, a listed game missing on disk aborts the service, and
extra games on disk are ignored by construction.

Verify a fresh `$ALFWORLD_DATA` against them (writes nothing, non-zero exit on drift):

```bash
python tools/gen_alfworld_manifest.py --data $ALFWORLD_DATA/json_2.1.1 --check
```

Regenerating them **changes which games every run trains and evaluates on** — a deliberate
act, logged in [`../fedagent/docs/revision.md`](../fedagent/docs/revision.md). Mechanism:
[`../fedagent/envs/alfworld/game_manifest.py`](../fedagent/envs/alfworld/game_manifest.py).

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

**Hardness labels** (task-level heterogeneity reference trajectories) ship right here
under [`hardness/`](hardness/README.md).

These are **derived from the WebShop and ALFWorld benchmark data (both MIT)**, see
[`../NOTICE`](../NOTICE) for attribution.
