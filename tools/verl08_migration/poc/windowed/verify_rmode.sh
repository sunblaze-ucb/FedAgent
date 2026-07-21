#!/bin/bash
# Parameterized rollout_mode smoke driver.  args: CONFIG OUTDIR LOG PLO PHI
# Validates the rollout_mode switch (windowed auto-injects the manager; concat does not).
set +e
REPO=/gpfs/projects/b1222/userdata/canyu/kangyu/fedagent; cd "$REPO" || exit 2
for __c in "$CONDA_PREFIX_1" "$HOME/miniconda3" "$HOME/anaconda3" /opt/conda /software/miniconda3/4.10.3; do [ -f "$__c/etc/profile.d/conda.sh" ] && { . "$__c/etc/profile.d/conda.sh"; break; }; done; conda activate fedagent-verl08
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
CONFIG="$1"; OUTDIR="$2"; LOG="$3"; PLO="$4"; PHI="$5"
echo "[RMODE start $(date)] host=$(hostname) config=$(basename "$CONFIG")"
pkill -f uvicorn 2>/dev/null; pkill -f main_ppo_fed 2>/dev/null; pkill -f service.server 2>/dev/null
for p in $(seq "$PLO" "$PHI"); do fuser -k "$p/tcp" 2>/dev/null; done; sleep 6
rm -rf "$OUTDIR" 2>/dev/null
timeout 3600 python -m fedagent.fed.run_fed --config "$CONFIG" --rounds 1 --clients 2 \
  --model-path /projects/b1222/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306 \
  --output-dir "$OUTDIR" > "$LOG" 2>&1
echo "[RMODE exit=$? $(date)] $(basename "$CONFIG")"
grep -iE "FEDERATED LOOP CLOSED|global_step:[0-9]|\[windowed\] train batch|out of memory|Traceback|AssertionError|Got torch.Size|prompt_length/mean|num_turns/mean|val-core/alfworld|most recent" "$LOG" | grep -vE "atexit|dump_compile|WARNING" | tail -18
