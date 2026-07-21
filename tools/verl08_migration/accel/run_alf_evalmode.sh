#!/bin/bash
# ALFWorld eval-mode sweep: run inline / parallel / shared / worker SEQUENTIALLY (each uses the 4 GPUs;
# parallel splits 2 train + 2 eval internally), time each, write a barrier with the ranking. 1.5B,
# 2 client × 2 round, eval every round, 48-game val. Output -> gitignored runs/alf_em/ (NOT _scratch).
# Hardened: unique RUN_ID + EXIT trap (review fix b). (no `set -u`: conda deactivate is not nounset-clean)
REPO=/gpfs/projects/b1222/userdata/canyu/kangyu/fedagent
for __c in "$CONDA_PREFIX_1" "$HOME/miniconda3" "$HOME/anaconda3" /opt/conda /software/miniconda3/4.10.3; do [ -f "$__c/etc/profile.d/conda.sh" ] && { . "$__c/etc/profile.d/conda.sh"; break; }; done
conda activate fedagent-verl08
cd "$REPO"
RUN_ID=$(date +%H%M%S)-$$
mkdir -p runs/alf_em
BAR=runs/alf_em/sweep_${RUN_ID}.barrier
ln -sf "sweep_${RUN_ID}.barrier" runs/alf_em/latest.barrier
: > "$BAR"
trap 'echo "[trap] driver EXIT rc=$? $(date +%T)" >> "'"$BAR"'"' EXIT
export CUDA_VISIBLE_DEVICES=0,1,2,3
t0=$(date +%s); echo "START alf_em RUN_ID=$RUN_ID $(date +%T) host=$(hostname)" >> "$BAR"
for mode in inline parallel shared worker; do
  export RAY_TMPDIR=/tmp/ray_alfem_$mode; mkdir -p "$RAY_TMPDIR"
  rm -rf runs/alf_em/$mode
  c0=$(date +%s); echo "[$mode] START $(date +%T)" >> "$BAR"
  python -u -m fedagent.fed.run_fed --config tools/verl08_migration/accel/alfworld/alf_em_$mode.yaml > runs/alf_em/$mode.log 2>&1
  echo "[$mode] rc=$? wall=$(($(date +%s)-c0))s $(date +%T)" >> "$BAR"
done
echo "BARRIER total=$(($(date +%s)-t0))s" >> "$BAR"
echo "=== DONE ===" >> "$BAR"
