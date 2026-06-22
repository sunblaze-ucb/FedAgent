# Hardness trajectories (task-heterogeneity *Hardness* / paper symbol ξ′)

The **Hardness** task-heterogeneity arm partitions train goals into *easy* / *hard*
by a per-goal success label from a **reference policy**, then Beta-allocates the easy
goals across clients (dispersion = `success_std` = ξ′). The partition
([`../../hetero/webshop_hardness.py`](../../hetero/webshop_hardness.py) for WebShop;
the ALFWorld branch of the vendored
[`partition_strategy.py`](../../envs/alfworld/engine/agent_system/environments/partition_strategy.py))
**requires** a labels file — there is no usable default.

These are the **original FedAgent reference labels**, produced by the paper's **trained
checkpoint** (a Qwen2.5-1.5B policy fine-tuned on each benchmark — *not* zero-shot), via
the original verl-agent inference pipeline (`scripts/inference/run_{webshop,alfworld}_inference.sh`,
Sept 2025), and copied verbatim from the original `output/inference/` summaries.

| file | env | reference | coverage | easy rate |
|---|---|---|---|---|
| `qwen2.5-1.5b_webshop_trajectories.json` | WebShop | trained Qwen2.5-1.5B | 6,402 goals (full train pool) | 1,780 (27.8 %) |
| `qwen2.5-1.5b_alfworld_trajectories.json` | ALFWorld | trained Qwen2.5-1.5B | 3,553 games (full train pool) | 2,112 (59.4 %) |

## Schema

```json
{ "metadata": { ... },
  "trajectories": [
    { "task_info": { "task_id": "<key>" }, "traj_info": { "success": false } },
    ...
] }
```

The partition reads only `trajectories` (the `metadata` block records provenance and is
ignored). `success` is a **strict binary** (the episode achieved the benchmark's success
condition). `task_id` matches the partition's keying **by construction** — both the labels
and the partition come from the same verl-agent code:
- **WebShop**: `f"{asin}_{abs(md5(sorted(goal_options.items())))}"` (e.g.
  `B07WMMYB6G_18488311…`).
- **ALFWorld**: `f"alfworld_{task_type_dir}_{trial_dir}_game"` (e.g.
  `alfworld_pick_clean_then_place_in_recep-Plate-None-DiningTable-19_trial_T2019…_game`).

## Regenerating

The labels depend on the reference policy, so regenerate per backbone if you change it:
- **WebShop**: the overlay ships a generator — run it with a **trained** checkpoint as the
  reference (NOT the base instruct model; zero-shot Qwen2.5-1.5B strictly succeeds on only
  ~1.4 % of goals, which collapses the easy/hard split):
  ```bash
  python -m tools.verl08_migration.gen_hardness_trajectories \
      --config fedagent/config/fed_webshop_scaled_hardness.yaml \
      --model <trained Qwen2.5-1.5B checkpoint> --num-goals 6410 \
      --output fedagent/data/hardness/qwen2.5-1.5b_webshop_trajectories.json
  ```
- **ALFWorld**: there is no overlay-native generator; the shipped labels come from the
  original verl-agent inference pipeline (`run_alfworld_inference.sh` over the train split).
  Regenerating requires that pipeline (or a port of it) with a trained checkpoint.

Both keep the schema identical, so no config change is needed — just overwrite the file.
