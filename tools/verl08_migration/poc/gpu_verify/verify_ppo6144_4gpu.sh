#!/bin/bash
set +e
REPO=/gpfs/projects/b1222/userdata/canyu/kangyu/fedagent; cd "$REPO" || exit 2
for __c in "$CONDA_PREFIX_1" "$HOME/miniconda3" "$HOME/anaconda3" /opt/conda /software/miniconda3/4.10.3; do [ -f "$__c/etc/profile.d/conda.sh" ] && { . "$__c/etc/profile.d/conda.sh"; break; }; done; conda activate fedagent-verl08
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
G=_scratch/gpu_verify; RUNS=$REPO/_scratch/gpu_verify/runs; mkdir -p "$RUNS"
echo "[PPO6144_4GPU start $(date)] host=$(hostname)"
pkill -f uvicorn 2>/dev/null; pkill -f main_ppo_fed 2>/dev/null; for p in $(seq 9770 9810); do fuser -k "$p/tcp" 2>/dev/null; done; sleep 6
timeout 4000 python -m fedagent.fed.run_fed --config "$G/webshop_ppo6144_4gpu.yaml" --output-dir "$RUNS/ppo6144_4gpu" > "$G/vppo6144_4gpu.log" 2>&1
echo "[PPO6144_4GPU exit=$? $(date)]"
grep -iE "FEDERATED LOOP CLOSED|FedAvg critic round 1 OK|model_world_size_4|response_length/clip_ratio|max_memory_reserved|CUDA out of memory|OutOfMemory|killed by signal" "$G/vppo6144_4gpu.log" | tail -8
