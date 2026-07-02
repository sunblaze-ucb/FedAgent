#!/bin/bash
# Paper-config 2-round wiring tests WITH the adopted accel stack (2026-07-02), chained after the
# fused equivalence A/B frees the GPUs:
#   0. wait for runs/fused_ab/latest.barrier "=== DONE ===" (cap 45 min) + GPU drain (cap 10 min)
#   1. ALFWorld REAL paper config + cross_round/worker + replicas=8 -> runs/paper_alf/wiring_r8
#      (first-ever paper-config run on ALFWorld; per-unit costs for 70-round feasibility)
#   2. WebShop paper wiring re-run + replicas=4 -> runs/paper_ws/wiring_r4
#      (A/B vs old wiring: cold+base-eval 707s / round 496s / eval 630s)
# ALFWorld first: it is the never-measured arm and has the wider variance; WebShop's headline
# lands early even if walltime clips its tail. Run inside a FOREGROUND srun --overlap step.
# (no `set -u`: conda deactivate is not nounset-clean)
REPO=/gpfs/projects/b1222/userdata/canyu/kangyu/fedagent
source /software/miniconda3/4.10.3/etc/profile.d/conda.sh 2>/dev/null
conda activate fedagent-verl08
cd "$REPO"
RUN_ID=$(date +%H%M%S)-$$
mkdir -p runs/paper2r runs/paper_alf runs/paper_ws
BAR=runs/paper2r/stack_${RUN_ID}.barrier
ln -sf "stack_${RUN_ID}.barrier" runs/paper2r/latest.barrier
: > "$BAR"
trap 'echo "[trap] driver EXIT rc=$? $(date +%T)" >> "'"$BAR"'"' EXIT
t0=$(date +%s); echo "START paper2r RUN_ID=$RUN_ID $(date +%T) host=$(hostname)" >> "$BAR"

# --- 0. wait for the fused A/B to finish and release the GPUs ---
for i in $(seq 1 90); do
  grep -q "=== DONE ===" runs/fused_ab/latest.barrier 2>/dev/null && break
  sleep 30
done
grep -q "=== DONE ===" runs/fused_ab/latest.barrier 2>/dev/null \
  && echo "[wait] fused A/B done $(date +%T)" >> "$BAR" \
  || echo "[wait] TIMEOUT waiting for fused A/B (proceeding to drain check) $(date +%T)" >> "$BAR"
for i in $(seq 1 20); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | sort -rn | head -1)
  [ "${used:-99999}" -lt 2000 ] && break
  sleep 30
done
if [ "${used:-99999}" -ge 2000 ]; then
  echo "ABORT: GPUs still busy (max used=${used} MiB) $(date +%T)" >> "$BAR"
  exit 1
fi
echo "[wait] GPUs drained (max used=${used} MiB), settling 60s $(date +%T)" >> "$BAR"
sleep 60
export CUDA_VISIBLE_DEVICES=0,1,2,3

# --- 1. ALFWorld real paper config + accel stack ---
export RAY_TMPDIR=/tmp/ray_paper_alf; mkdir -p "$RAY_TMPDIR"
rm -rf runs/paper_alf/wiring_r8
c0=$(date +%s); echo "[alf] START $(date +%T)" >> "$BAR"
python -u -m fedagent.fed.run_fed --config tools/verl08_migration/accel/alfworld/paper_alf_wiring_r8.yaml > runs/paper_alf/wiring_r8.log 2>&1
echo "[alf] rc=$? wall=$(($(date +%s)-c0))s $(date +%T)" >> "$BAR"

# --- 2. WebShop paper wiring + replicas=4 ---
export RAY_TMPDIR=/tmp/ray_paper_ws; mkdir -p "$RAY_TMPDIR"
rm -rf runs/paper_ws/wiring_r4
c0=$(date +%s); echo "[ws] START $(date +%T)" >> "$BAR"
python -u -m fedagent.fed.run_fed --config tools/verl08_migration/accel/webshop/paper_ws_wiring_r4.yaml > runs/paper_ws/wiring_r4.log 2>&1
echo "[ws] rc=$? wall=$(($(date +%s)-c0))s $(date +%T)" >> "$BAR"

echo "BARRIER total=$(($(date +%s)-t0))s" >> "$BAR"
echo "=== STACK DONE ===" >> "$BAR"
