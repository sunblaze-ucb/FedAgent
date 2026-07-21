#!/bin/bash
# COMPLETE "2 client x 1 GPU + 2 GPU eval" test (the user's full idea = #3 client-parallel  x  #1 eval-parallel-train):
#   A trains on GPU 0, B trains on GPU 1 (eval OFF), E evals base 1.5B on GPU 2,3 -- all concurrent.
# Measures: (1) 4-GPU coexistence of THREE jobs (2 train + 1 eval), (2) whether eval HIDES under the
# 1-GPU training (eval wall < train barrier => round costs ~= t1(1) => faster than any serial layout).
# Durable: per-job logs + a barrier file on GPFS so the numbers survive a session teardown.
# (no `set -u`: conda's deactivate scripts reference unbound vars and would abort under nounset)
REPO=/gpfs/projects/b1222/userdata/canyu/kangyu/fedagent
for __c in "$CONDA_PREFIX_1" "$HOME/miniconda3" "$HOME/anaconda3" /opt/conda /software/miniconda3/4.10.3; do [ -f "$__c/etc/profile.d/conda.sh" ] && { . "$__c/etc/profile.d/conda.sh"; break; }; done
conda activate fedagent-verl08
cd "$REPO"
# Hardened (review fix b): unique RUN_ID so a re-run never clobbers a prior run's logs/barrier;
# EXIT trap that records the driver's own rc; fail-fast wait (below) that captures each child's rc.
RUN_ID=$(date +%H%M%S)-$$
BAR=_scratch/accel/p3_complete_${RUN_ID}.barrier
ln -sf "p3_complete_${RUN_ID}.barrier" _scratch/accel/p3_complete_latest.barrier
: > "$BAR"
trap 'echo "[trap] driver EXIT rc=$? at $(date +%T)" >> "'"$BAR"'"' EXIT
t0=$(date +%s)
echo "START complete RUN_ID=$RUN_ID t0=$t0 host=$(hostname)" >> "$BAR"

train_client () {  # $1=cfg basename  $2=gpu  $3=tag
  local CFG=$1 GPU=$2 TAG=$3
  export CUDA_VISIBLE_DEVICES=$GPU
  export RAY_TMPDIR=/tmp/ray_$TAG
  mkdir -p "$RAY_TMPDIR"
  local OUT; OUT=$(grep -E "^output_dir:" _scratch/accel/$CFG.yaml | awk '{print $2}')
  rm -rf "$OUT"
  local c0=$(date +%s)
  echo "[$TAG] TRAIN START gpu=$GPU $(date +%T)" >> "$BAR"
  python -u -m fedagent.fed.run_fed --config _scratch/accel/$CFG.yaml > _scratch/accel/${CFG}.${RUN_ID}.log 2>&1
  echo "[$TAG] TRAIN rc=$? wall=$(($(date +%s)-c0))s $(date +%T)" >> "$BAR"
}

eval_job () {  # eval pinned to GPU 2,3
  export CUDA_VISIBLE_DEVICES=2,3
  export RAY_TMPDIR=/tmp/ray_E
  mkdir -p "$RAY_TMPDIR"
  rm -rf _scratch/accel/p3_eval_2gpu_out
  local c0=$(date +%s)
  echo "[E] EVAL START gpu=2,3 $(date +%T)" >> "$BAR"
  python -u _scratch/accel/standalone_eval.py _scratch/accel/p3_eval_2gpu.yaml 2,3 > _scratch/accel/p3_eval.${RUN_ID}.log 2>&1
  echo "[E] EVAL rc=$? wall=$(($(date +%s)-c0))s $(date +%T)" >> "$BAR"
}

train_client p3_1gpu_A 0 A1g & pA=$!
train_client p3_1gpu_B 1 B1g & pB=$!
eval_job & pE=$!
rcA=0; rcB=0; rcE=0
wait "$pA" || rcA=$?
wait "$pB" || rcB=$?
wait "$pE" || rcE=$?
echo "BARRIER wall=$(($(date +%s)-t0))s rcA=$rcA rcB=$rcB rcE=$rcE" >> "$BAR"
echo "=== DONE ===" >> "$BAR"
