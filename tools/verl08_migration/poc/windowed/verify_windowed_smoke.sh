#!/bin/bash
# WINDOWED (faithful) GRPO ALFWorld Stage-1 smoke. Validates per-turn windowed rollout runs
# end-to-end (structure) with stock grpo. Files are in the tracked fedagent/ pkg (no PYTHONPATH).
set +e
REPO=/gpfs/projects/b1222/userdata/canyu/kangyu/fedagent; cd "$REPO" || exit 2
for __c in "$CONDA_PREFIX_1" "$HOME/miniconda3" "$HOME/anaconda3" /opt/conda /software/miniconda3/4.10.3; do [ -f "$__c/etc/profile.d/conda.sh" ] && { . "$__c/etc/profile.d/conda.sh"; break; }; done; conda activate fedagent-verl08
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
G=_scratch/windowed_poc
echo "[WINDOWED start $(date)] host=$(hostname)"
pkill -f uvicorn 2>/dev/null; pkill -f main_ppo_fed 2>/dev/null; pkill -f service.server 2>/dev/null
for p in $(seq 41600 41800); do fuser -k "$p/tcp" 2>/dev/null; done; sleep 6
rm -rf "$G/runs/grpo_windowed_smoke" 2>/dev/null
timeout 3600 python -m fedagent.fed.run_fed --config "$G/alf_grpo_windowed_smoke.yaml" --rounds 1 --clients 2 \
  --model-path /projects/b1222/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306 \
  --output-dir "$G/runs/grpo_windowed_smoke" > "$G/vwindowed_smoke.log" 2>&1
echo "[WINDOWED exit=$? $(date)]"
grep -iE "FEDERATED LOOP CLOSED|global_step:[0-9]|out of memory|Traceback|AssertionError|Error:|run_episode_windowed|prompt_length/mean|response_length/mean|num_turns/mean|most recent" "$G/vwindowed_smoke.log" | grep -vE "atexit|dump_compile|WARNING" | tail -16
