#!/bin/bash
# EXP A: ALFWorld PPO with optimizer_offload=FALSE (actor+critic), gpu_mem=0.7.
# Tests the faithful free fix: original verl-agent 0.3.1 PPO ran offload=false; the new
# config flipped it to true (a memory workaround for the long concat context). Question:
# does offload=false OOM on 80GB H100 at max_model_len=16384, and how much does it cut
# update_actor/update_critic? Compare update_* timings vs v3 (offload=true) on qgpu3022.
set +e
REPO=/gpfs/projects/b1222/userdata/canyu/kangyu/fedagent; cd "$REPO" || exit 2
for __c in "$CONDA_PREFIX_1" "$HOME/miniconda3" "$HOME/anaconda3" /opt/conda /software/miniconda3/4.10.3; do [ -f "$__c/etc/profile.d/conda.sh" ] && { . "$__c/etc/profile.d/conda.sh"; break; }; done; conda activate fedagent-verl08
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
G=_scratch/gpu_verify; RUNS=$REPO/$G/runs; mkdir -p "$RUNS"
echo "[OFFLOFF start $(date)] host=$(hostname)"
pkill -f uvicorn 2>/dev/null; pkill -f main_ppo_fed 2>/dev/null; pkill -f service.server 2>/dev/null
for p in $(seq 40128 40328); do fuser -k "$p/tcp" 2>/dev/null; done; sleep 6
rm -rf "$RUNS/alf_ppo_offload_off" 2>/dev/null
timeout 9000 python -m fedagent.fed.run_fed --config "$G/alf_ppo_offload_off.yaml" --rounds 1 --clients 2 \
  --model-path /projects/b1222/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306 \
  --output-dir "$RUNS/alf_ppo_offload_off" > "$G/valf_ppo_offload_off.log" 2>&1
echo "[OFFLOFF exit=$? $(date)]"
grep -iE "FEDERATED LOOP CLOSED|global_step:[0-9]|out of memory|OutOfMemory|CUDA error|ReadTimeout|FAILED|timing_s/step" "$G/valf_ppo_offload_off.log" | grep -vE "atexit|dump_compile" | tail -10
