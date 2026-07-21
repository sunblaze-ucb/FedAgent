# Hardness trajectories (task-heterogeneity *Hardness* / paper symbol ξ′)

The **Hardness** task-heterogeneity arm partitions train goals into *easy* / *hard*
by a per-goal success label from a **reference policy**, then Beta-allocates the easy
goals across clients (dispersion = `success_std` = ξ′). The partition
([`../../hetero/webshop_hardness.py`](../../hetero/webshop_hardness.py) for WebShop;
the ALFWorld branch of the vendored
[`partition_strategy.py`](../../envs/alfworld/engine/agent_system/environments/partition_strategy.py))
**requires** a labels file; there is no usable default.

These are the **original FedAgent reference labels**, produced by the paper's **trained
checkpoint** (a Qwen2.5-1.5B policy fine-tuned on each benchmark, *not* zero-shot), via
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
condition). `task_id` matches the partition's keying **by construction**; both the labels
and the partition come from the same verl-agent code:
- **WebShop**: `f"{asin}_{abs(md5(sorted(goal_options.items())))}"` (e.g.
  `B07WMMYB6G_18488311…`).
- **ALFWorld**: `f"alfworld_{task_type_dir}_{trial_dir}_game"` (e.g.
  `alfworld_pick_clean_then_place_in_recep-Plate-None-DiningTable-19_trial_T2019…_game`).

## Regenerating

The labels depend on the reference policy, so regenerate per backbone if you change it:
- **WebShop**: the overlay ships a generator, run it with a **trained** checkpoint as the
  reference (NOT the base instruct model; zero-shot Qwen2.5-1.5B strictly succeeds on only
  ~1.4 % of goals, which collapses the easy/hard split). Two protocol requirements:
  1. **Rollout mode must match the reference's training/eval mode.** The generator injects
     the run_fed rollout mode itself (`rollout_mode` in the passed YAML, DEFAULT `windowed` =
     the faithful paper rollout). A windowed-trained checkpoint rolled out in CONCAT mode is
     out-of-distribution: measured 1.6 % vs 22.7 % strict success on the same 128 train goals
     (greedy) — the symptom looks like a broken checkpoint when it isn't.
  2. **Use the reference's rollout budgets** — a paper-style config (windowed per-turn budgets
     `prompt 4096 / response 512 / max_model_len 4608`, `search_return_n: 50`), NOT the smoke
     config's concat budgets:
  ```bash
  python -m tools.verl08_migration.gen_hardness_trajectories \
      --config fedagent/config/paper/task_heterogeneity/grpo/webshop/fed_webshop_grpo_total-100_cl-per-rd-2_rd-70_ep-per-cl-3_min-goals-per-cl-100_p-hardness_success_std-1.yaml \
      --model <trained Qwen2.5-1.5B checkpoint> --num-goals 6410 \
      --output fedagent/data/hardness/qwen2.5-1.5b_webshop_trajectories.json
  ```
  For chunked / resumable generation (shared or preemptible GPUs): the dataset honors
  `FEDAGENT_SEED_OFFSET` (additive per-row seed shift; the TRAIN service maps
  `goal = VAL_SIZE + seed % pool`), so offset K + `--num-goals N` labels the disjoint window
  `goals[500+K : 500+K+N]`; run windows back to back and merge the outputs.
- **ALFWorld**: there is no overlay-native generator; the shipped labels come from the
  original verl-agent inference pipeline (`run_alfworld_inference.sh` over the train split).
  Regenerating requires that pipeline (or a port of it) with a trained checkpoint.

Both keep the schema identical, so no config change is needed, just overwrite the file.
