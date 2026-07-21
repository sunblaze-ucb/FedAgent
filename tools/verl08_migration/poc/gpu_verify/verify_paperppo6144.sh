#!/bin/bash
set +e
REPO=/gpfs/projects/b1222/userdata/canyu/kangyu/fedagent; cd "$REPO" || exit 2
for __c in "$CONDA_PREFIX_1" "$HOME/miniconda3" "$HOME/anaconda3" /opt/conda /software/miniconda3/4.10.3; do [ -f "$__c/etc/profile.d/conda.sh" ] && { . "$__c/etc/profile.d/conda.sh"; break; }; done; conda activate fedagent-verl08
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
G=_scratch/gpu_verify; RUNS=$REPO/$G/runs; mkdir -p "$RUNS"
echo "[PAPERPPO6144 start $(date)] host=$(hostname)"
pkill -f uvicorn 2>/dev/null; pkill -f main_ppo_fed 2>/dev/null; for p in $(seq 9800 9840); do fuser -k "$p/tcp" 2>/dev/null; done; sleep 6
# real shipped paper PPO config, smoke-sized via --rounds 1 --clients 2, GPFS output, clean port, 1.5B model via --model-path local snapshot
timeout 4000 python -m fedagent.fed.run_fed --config "fedagent/config/paper/uniform/Qwen2.5-1.5B-Instruct/main/ppo/fed_webshop_ppo_total-100_cl-per-rd-2_rd-70_ep-per-cl-3_min-goals-per-cl-100_p-uniform.yaml"   --rounds 1 --clients 2 --port-base 9800   --model-path /projects/b1222/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306   --output-dir "$RUNS/paperppo6144" > "$G/vpaperppo6144.log" 2>&1
echo "[PAPERPPO6144 exit=$? $(date)]"
grep -iE "FEDERATED LOOP CLOSED|FedAvg critic round 1 OK|response_length=6144|response_length/clip_ratio|CUDA out of memory|killed by signal|Error:" "$G/vpaperppo6144.log" | tail -8
