#!/bin/bash
# Training-half of the user's "2 client × 1 GPU + 2 GPU eval" idea:
# two clients train IN PARALLEL, A pinned to GPU 0, B to GPU 1, eval OFF (isolate the
# 1-GPU training cost + confirm world_size=1 fits memory at gpu_memory_utilization=0.5).
# Durable: per-client logs + a barrier file on GPFS so the wall-clock survives a session teardown.
# (no `set -u`: conda's deactivate scripts reference unbound vars and would abort under nounset)
REPO=/gpfs/projects/b1222/userdata/canyu/kangyu/fedagent
for __c in "$CONDA_PREFIX_1" "$HOME/miniconda3" "$HOME/anaconda3" /opt/conda /software/miniconda3/4.10.3; do [ -f "$__c/etc/profile.d/conda.sh" ] && { . "$__c/etc/profile.d/conda.sh"; break; }; done
conda activate fedagent-verl08
cd "$REPO"
BAR=_scratch/accel/p3_1gpu_barrier.log
: > "$BAR"
t0=$(date +%s)
echo "START 2x1gpu t0=$t0 host=$(hostname)" >> "$BAR"

run_client () {  # $1=cfg basename  $2=gpu  $3=tag
  local CFG=$1 GPU=$2 TAG=$3
  export CUDA_VISIBLE_DEVICES=$GPU
  export RAY_TMPDIR=/tmp/ray_$TAG
  mkdir -p "$RAY_TMPDIR"
  local OUT; OUT=$(grep -E "^output_dir:" _scratch/accel/$CFG.yaml | awk '{print $2}')
  rm -rf "$OUT"
  local c0=$(date +%s)
  echo "[$TAG] START gpu=$GPU $(date +%T)" >> "$BAR"
  python -u -m fedagent.fed.run_fed --config _scratch/accel/$CFG.yaml > _scratch/accel/$CFG.log 2>&1
  local rc=$?
  echo "[$TAG] rc=$rc wall=$(($(date +%s)-c0))s $(date +%T)" >> "$BAR"
}

run_client p3_1gpu_A 0 A1g & pA=$!
run_client p3_1gpu_B 1 B1g & pB=$!
wait $pA; wait $pB
echo "BARRIER wall=$(($(date +%s)-t0))s" >> "$BAR"
echo "=== DONE ===" >> "$BAR"
