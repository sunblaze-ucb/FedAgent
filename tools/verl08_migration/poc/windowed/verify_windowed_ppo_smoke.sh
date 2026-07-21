#!/bin/bash
# WINDOWED (faithful) PPO ALFWorld smoke. Verifies the per-turn windowed batch flows through the
# CRITIC path (gae) too. Launch ONLY after the GRPO smoke has finished (pkill below would kill it).
set +e
REPO=/gpfs/projects/b1222/userdata/canyu/kangyu/fedagent; cd "$REPO" || exit 2
for __c in "$CONDA_PREFIX_1" "$HOME/miniconda3" "$HOME/anaconda3" /opt/conda /software/miniconda3/4.10.3; do [ -f "$__c/etc/profile.d/conda.sh" ] && { . "$__c/etc/profile.d/conda.sh"; break; }; done; conda activate fedagent-verl08
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
G=_scratch/windowed_poc
echo "[WINDOWED-PPO start $(date)] host=$(hostname)"
pkill -f uvicorn 2>/dev/null; pkill -f main_ppo_fed 2>/dev/null; pkill -f service.server 2>/dev/null
for p in $(seq 41800 42000); do fuser -k "$p/tcp" 2>/dev/null; done; sleep 6
rm -rf "$G/runs/ppo_windowed_smoke" 2>/dev/null
timeout 3600 python -m fedagent.fed.run_fed --config "$G/alf_ppo_windowed_smoke.yaml" --rounds 1 --clients 2 \
  --model-path /projects/b1222/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306 \
  --output-dir "$G/runs/ppo_windowed_smoke" > "$G/vwindowed_ppo_smoke.log" 2>&1
echo "[WINDOWED-PPO exit=$? $(date)]"
grep -iE "FEDERATED LOOP CLOSED|global_step:[0-9]|\[windowed\] train batch|out of memory|Traceback|AssertionError|Got torch.Size|prompt_length/mean|num_turns/mean|FedAvg.*critic|most recent" "$G/vwindowed_ppo_smoke.log" | grep -vE "atexit|dump_compile|WARNING" | tail -16
