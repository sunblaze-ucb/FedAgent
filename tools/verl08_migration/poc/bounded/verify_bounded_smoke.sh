#!/bin/bash
# Bounded (per-turn, sliding-window) GRPO ALFWorld smoke. POC validation: does the bounded
# manager/worker run end-to-end (batch-contract OK?) + does windowed gen reduce cost?
# Runs the parked POC from _scratch via PYTHONPATH (no files in the tracked tree).
set +e
REPO=/gpfs/projects/b1222/userdata/canyu/kangyu/fedagent; cd "$REPO" || exit 2
for __c in "$CONDA_PREFIX_1" "$HOME/miniconda3" "$HOME/anaconda3" /opt/conda /software/miniconda3/4.10.3; do [ -f "$__c/etc/profile.d/conda.sh" ] && { . "$__c/etc/profile.d/conda.sh"; break; }; done; conda activate fedagent-verl08
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$REPO/_scratch/bounded_poc:$PYTHONPATH"
G=_scratch/bounded_poc
echo "[BOUNDED start $(date)] host=$(hostname) PYTHONPATH=$PYTHONPATH"
pkill -f uvicorn 2>/dev/null; pkill -f main_ppo_fed 2>/dev/null; pkill -f service.server 2>/dev/null
for p in $(seq 41000 41200); do fuser -k "$p/tcp" 2>/dev/null; done; sleep 6
rm -rf "$G/runs/grpo_bounded_smoke" 2>/dev/null
timeout 3600 python -m fedagent.fed.run_fed --config "$G/alf_grpo_bounded_smoke.yaml" --rounds 1 --clients 2 \
  --model-path /projects/b1222/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306 \
  --output-dir "$G/runs/grpo_bounded_smoke" > "$G/vbounded_smoke.log" 2>&1
echo "[BOUNDED exit=$? $(date)]"
grep -iE "FEDERATED LOOP CLOSED|global_step:[0-9]|out of memory|Traceback|AssertionError|Error:|run_episode_bounded|timing_s/gen:|prompt_length/mean|response_length/mean|num_turns/mean" "$G/vbounded_smoke.log" | grep -vE "atexit|dump_compile|WARNING" | tail -18
