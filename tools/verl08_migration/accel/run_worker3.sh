#!/bin/bash
# #3 + worker-eval fast-path test: 2 clients each on 2 GPUs (A:0,1  B:2,3), persistent trainer +
# eval_mode=worker (hot-engine eval, no eval cold-start). Measures the round wall-clock to compare
# against the 1-GPU layout's 995s. Also re-validates the VERL_RAY_JOB_ID fix on the PERSISTENT path
# (two concurrent persistent workers must not collide on the /tmp weight-transfer socket).
#
# Hardened (review fix b): unique RUN_ID so a re-run never clobbers a prior run's logs/barrier;
# EXIT trap that records the driver's own rc; fail-fast wait that captures each child's rc instead
# of silently empty-waiting. (no `set -u`: conda deactivate scripts reference unbound vars.)
REPO=/gpfs/projects/b1222/userdata/canyu/kangyu/fedagent
for __c in "$CONDA_PREFIX_1" "$HOME/miniconda3" "$HOME/anaconda3" /opt/conda /software/miniconda3/4.10.3; do [ -f "$__c/etc/profile.d/conda.sh" ] && { . "$__c/etc/profile.d/conda.sh"; break; }; done
conda activate fedagent-verl08
cd "$REPO"
RUN_ID=$(date +%H%M%S)-$$
BAR=_scratch/accel/p3_worker3_${RUN_ID}.barrier
ln -sf "p3_worker3_${RUN_ID}.barrier" _scratch/accel/p3_worker3_latest.barrier
: > "$BAR"
trap 'echo "[trap] driver EXIT rc=$? at $(date +%T)" >> "'"$BAR"'"' EXIT
t0=$(date +%s)
echo "START worker3 RUN_ID=$RUN_ID t0=$t0 host=$(hostname)" >> "$BAR"

run2 () {  # $1=cfg basename  $2=CUDA_VISIBLE_DEVICES  $3=tag
  local CFG=$1 GPUS=$2 TAG=$3
  export CUDA_VISIBLE_DEVICES=$GPUS
  export RAY_TMPDIR=/tmp/ray_$TAG
  mkdir -p "$RAY_TMPDIR"
  local OUT; OUT=$(grep -E "^output_dir:" _scratch/accel/$CFG.yaml | awk '{print $2}')
  rm -rf "$OUT"
  local c0=$(date +%s)
  echo "[$TAG] START gpus=$GPUS $(date +%T)" >> "$BAR"
  python -u -m fedagent.fed.run_fed --config _scratch/accel/$CFG.yaml > _scratch/accel/${CFG}.${RUN_ID}.log 2>&1
  local rc=$?
  echo "[$TAG] rc=$rc wall=$(($(date +%s)-c0))s $(date +%T)" >> "$BAR"
  return $rc
}

run2 p3_2gpu_worker_A 0,1 W3A & pA=$!
run2 p3_2gpu_worker_B 2,3 W3B & pB=$!
rcA=0; rcB=0
wait "$pA" || rcA=$?
wait "$pB" || rcB=$?
echo "BARRIER wall=$(($(date +%s)-t0))s rcA=$rcA rcB=$rcB" >> "$BAR"
echo "=== DONE ===" >> "$BAR"
