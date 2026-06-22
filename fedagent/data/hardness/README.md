# Hardness trajectories (task-heterogeneity *Hardness* / paper symbol ξ′)

The **Hardness** task-heterogeneity arm partitions train goals into *easy* / *hard*
by a per-goal success label from a **reference policy**, then Beta-allocates the easy
goals across clients (dispersion = `success_std` = ξ′). That partition
([`../../hetero/webshop_hardness.py`](../../hetero/webshop_hardness.py), ported verbatim
from the original verl-agent) **requires** a labels file — there is no usable default.

This folder holds those labels.

| file | env | reference policy | coverage | easy rate |
|---|---|---|---|---|
| `qwen2.5-1.5b_webshop_trajectories.json` | WebShop | Qwen2.5-1.5B-Instruct, zero-shot, greedy | 2,498 goals | 36 (1.4 %) |
| `qwen2.5-1.5b_alfworld_trajectories.json` | ALFWorld | — | **not generated** | — |

## Schema

```json
{ "trajectories": [
    { "task_info": { "task_id": "<asin>_<md5(goal_options)>" },
      "traj_info": { "success": false } },
    ...
] }
```

`task_id` is computed by the **exact** formula `hardness_partition` keys on
(`f"{asin}_{abs(md5(sorted(goal_options.items())))}"`, else asin+instruction hash, else
asin). The labelling service derives it from the env's real `server.goals` via the same
function, so labels match the partition **by construction** — no asin-vs-options-hash drift.
`success` is a **strict binary**: `True` iff the episode ended with a *perfect* WebShop
match (`done and dense_score == 1.0`), mirroring the original verl-agent reward
(`envs.py:32-40`) and the partition loader (which reads `traj_info.success` as a bool).

## How the WebShop file was generated

```bash
python -m tools.verl08_migration.gen_hardness_trajectories \
    --config fedagent/config/fed_webshop_scaled_hardness.yaml \
    --model  <Qwen2.5-1.5B-Instruct snapshot> \
    --num-goals 2500 \
    --output fedagent/data/hardness/qwen2.5-1.5b_webshop_trajectories.json --n-gpus 4
```

The generator ([`../../../tools/verl08_migration/gen_hardness_trajectories.py`](../../../tools/verl08_migration/gen_hardness_trajectories.py))
reuses the overlay end to end: it starts a WebShop service on the **train** split / full
unperturbed catalog with `FEDAGENT_LOG_GOAL_ID=1`, runs a `val_only` pass of the reference
model (one greedy trajectory per goal), and aggregates per-`task_id` success into the file
above. 2,500 goals were labelled (contiguous train goals `[500:3000]`); 2 collided on
`task_id` → 2,498 unique.

## Caveat — sparse easy pool (read before reproducing the Hardness arm)

The 1.4 % easy rate is the **true strict-success rate of the zero-shot Qwen2.5-1.5B
reference** on WebShop — it is *faithful*, not a bug (WebShop's dense graded score is
computed but the label, like the original, is strict perfect-match success). But 36 easy
goals is a **small easy pool**: at the paper's 100-client scale the Beta allocation has
little room to spread, so the *magnitude* of the Hardness heterogeneity signal will be
muted. The arm still runs and `success_std=1` vs `256` still produce different allocations.

To strengthen the signal, regenerate with any of:
- **More coverage** — `--num-goals 6410` (the whole train pool; ~80 min on 4×H100).
- **A stronger reference** — pass a fine-tuned / few-shot checkpoint via `--model`; a
  reference that succeeds on a larger fraction yields a more balanced easy/hard split.

Both keep the schema identical, so no config change is needed — just overwrite the file.

## ALFWorld — not yet generated

The 4 ALFWorld Hardness configs reference `qwen2.5-1.5b_alfworld_trajectories.json`, but the
generator is **WebShop-only** (it drives a WebShop service and keys on `asin`). ALFWorld
labels need a separate reference pass keyed on the game id; that generator does not exist
yet, so the ALFWorld Hardness arm remains blocked. The configs are marked accordingly.
