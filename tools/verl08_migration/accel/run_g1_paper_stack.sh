#!/bin/bash
# 1xH100 paper-geometry probe stack (2026-07-02) — fills the idle 1-card allocation while the
# 4-card node runs the paper wiring tests. Sequence (each 1 client x 1 round x 1 step, eval off):
#   1. alf g1 PAPER caps K=1  (control: 1-GPU penalty at true paper geometry)
#   2. alf g1 PAPER caps K=4  (the 1xH100 recipe arm)
#   3. ws  g1 off-arm rerun   (fresh same-day baseline + noise bar vs the recorded 225.2s)
#   4. ws  g1 FUSED on-arm    (is -6.5% bigger where GPU-compute is 74% of the step?)
#   5. repeats: alf K=4, ws off, ws fused (run-to-run noise bars) until walltime clips.
# Timings land in each run's log + <out>/timing_s/; dumped into the barrier after each run so
# repeats can safely rm -rf the output dir. Run inside a FOREGROUND srun --overlap step.
# (no `set -u`: conda deactivate is not nounset-clean)
REPO=/gpfs/projects/b1222/userdata/canyu/kangyu/fedagent
source /software/miniconda3/4.10.3/etc/profile.d/conda.sh 2>/dev/null
conda activate fedagent-verl08
cd "$REPO"
RUN_ID=$(date +%H%M%S)-$$
mkdir -p runs/g1_paper
BAR=runs/g1_paper/stack_${RUN_ID}.barrier
ln -sf "stack_${RUN_ID}.barrier" runs/g1_paper/latest.barrier
: > "$BAR"
trap 'echo "[trap] driver EXIT rc=$? $(date +%T)" >> "'"$BAR"'"' EXIT
export CUDA_VISIBLE_DEVICES=0
t0=$(date +%s); echo "START g1_paper RUN_ID=$RUN_ID $(date +%T) host=$(hostname)" >> "$BAR"

run_one() {  # $1=tag  $2=config  $3=output_dir
  export RAY_TMPDIR=/tmp/ray_g1p_$1; mkdir -p "$RAY_TMPDIR"
  rm -rf "$3"
  local c0=$(date +%s); echo "[$1] START $(date +%T)" >> "$BAR"
  python -u -m fedagent.fed.run_fed --config "$2" > "runs/g1_paper/$1.log" 2>&1
  echo "[$1] rc=$? wall=$(($(date +%s)-c0))s $(date +%T)" >> "$BAR"
  if [ -d "$3/timing_s" ]; then
    for f in "$3"/timing_s/*; do echo "  [$1] $(basename "$f")=$(cat "$f")" >> "$BAR"; done
  fi
}

A=tools/verl08_migration/accel
run_one alf_r1_a   "$A/alfworld/alf_scale_g1_paper_r1.yaml" runs/alf_scale/g1_paper_r1
run_one alf_r4_a   "$A/alfworld/alf_scale_g1_paper_r4.yaml" runs/alf_scale/g1_paper_r4
run_one ws_off_a   "$A/webshop/ws_scale_g1_rep.yaml"        runs/ws_scale/g1_rep
run_one ws_fused_a "$A/webshop/ws_scale_g1_fused.yaml"      runs/ws_scale/g1_fused
# noise-bar repeats (walltime clips whatever doesn't fit)
run_one alf_r4_b   "$A/alfworld/alf_scale_g1_paper_r4.yaml" runs/alf_scale/g1_paper_r4
run_one ws_off_b   "$A/webshop/ws_scale_g1_rep.yaml"        runs/ws_scale/g1_rep
run_one ws_fused_b "$A/webshop/ws_scale_g1_fused.yaml"      runs/ws_scale/g1_fused
run_one alf_r1_b   "$A/alfworld/alf_scale_g1_paper_r1.yaml" runs/alf_scale/g1_paper_r1

echo "BARRIER total=$(($(date +%s)-t0))s" >> "$BAR"
echo "=== G1 DONE ===" >> "$BAR"
