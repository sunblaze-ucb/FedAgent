#!/bin/bash
# F6 memory smoke: does WebShop PPO at response_length=6144 (== GRPO) fit + close on GPU?
set +e
REPO=/gpfs/projects/b1222/userdata/canyu/kangyu/fedagent; cd "$REPO" || exit 2
for __c in "$CONDA_PREFIX_1" "$HOME/miniconda3" "$HOME/anaconda3" /opt/conda /software/miniconda3/4.10.3; do [ -f "$__c/etc/profile.d/conda.sh" ] && { . "$__c/etc/profile.d/conda.sh"; break; }; done; conda activate fedagent-verl08
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
G=_scratch/gpu_verify; RUNS=$REPO/_scratch/gpu_verify/runs; mkdir -p "$RUNS"
echo "[PPO6144 start $(date)] host=$(hostname)"
pkill -f uvicorn 2>/dev/null; pkill -f main_ppo_fed 2>/dev/null; for p in $(seq 9740 9780); do fuser -k "$p/tcp" 2>/dev/null; done; sleep 6
timeout 2400 python -m fedagent.fed.run_fed --config "$G/het_webshop_catalog_ppo_6144.yaml" --output-dir "$RUNS/ppo6144" > "$G/vppo6144.log" 2>&1
echo "[PPO6144 exit=$? $(date)]"
grep -iE "FEDERATED LOOP CLOSED|FedAvg critic|response_length/clip_ratio|OutOfMemory|killed by signal|CUDA out of memory|Error:" "$G/vppo6144.log" | tail -8
