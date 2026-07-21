#!/bin/bash
# Robust-fix test: ALFWorld PPO at memory-SAFE pool=16 (32 envs) WITH block-on-/create (no read
# timeout while waiting for a pooled env). If this CLOSES at pool=16, the blocking borrow decouples
# correctness from pool size -- pool then only sets speed, not crash-avoidance.
set +e
REPO=/gpfs/projects/b1222/userdata/canyu/kangyu/fedagent; cd "$REPO" || exit 2
for __c in "$CONDA_PREFIX_1" "$HOME/miniconda3" "$HOME/anaconda3" /opt/conda /software/miniconda3/4.10.3; do [ -f "$__c/etc/profile.d/conda.sh" ] && { . "$__c/etc/profile.d/conda.sh"; break; }; done; conda activate fedagent-verl08
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
G=_scratch/gpu_verify; RUNS=$REPO/$G/runs; mkdir -p "$RUNS"
echo "[ALF16BLK2 start $(date)] host=$(hostname)"
pkill -f uvicorn 2>/dev/null; pkill -f main_ppo_fed 2>/dev/null; pkill -f service.server 2>/dev/null
for p in $(seq 8200 8300); do fuser -k "$p/tcp" 2>/dev/null; done; sleep 6
rm -rf "$RUNS/alf_ppo_pool16_block" 2>/dev/null
timeout 9000 python -m fedagent.fed.run_fed --config "$G/alf_ppo_pool16_block.yaml" \
  --rounds 1 --clients 2 \
  --model-path /projects/b1222/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306 \
  --output-dir "$RUNS/alf_ppo_pool16_block" > "$G/valf_ppo_pool16_block_v2.log" 2>&1
echo "[ALF16BLK2 exit=$? $(date)]"
grep -iE "FEDERATED LOOP CLOSED|client [01] round 1 OK|FedAvg|warmed [0-9]+ envs|ReadTimeout|out of memory|Killed|FAILED" "$G/valf_ppo_pool16_block_v2.log" | grep -vE "atexit|dump_compile" | tail -10
