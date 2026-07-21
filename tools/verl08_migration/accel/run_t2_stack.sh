#!/bin/bash
# Tier-2 / lanes / one_step_off VALIDATION STACK (2026-07-02): runs after the WS r4 wiring
# frees the GPUs. Phases:
#   1. ALFWorld Tier-2 A/B suite (paper caps, 48-game val): base, cache, scope, feval, all,
#      cache_b (re-run: manifest now warm from arm 2 -> round-1 savings too)
#   2. equivalence compares (each arm vs base, round_2 aggregated actor, bar <= 1e-4)
#   3. TinyGuess cross-round equivalence arms: base, hfexport, lanes (+ compares)
#   4. OPTIMAL-COMBO paper-config 2-round runs: alf_combo, ws_combo, alf_combo_lanes,
#      ws_combo_lanes (the user-required paper-config measurements of the final stack)
#   5. one_step_off timing probe (ADDITIONAL OPTION, off-policy -- timing only)
# Failures don't stop the stack (rc recorded; compares skip missing dirs).
# Run inside a FOREGROUND srun --overlap step. (no set -u: conda not nounset-clean)
REPO=/gpfs/projects/b1222/userdata/canyu/kangyu/fedagent
for __c in "$CONDA_PREFIX_1" "$HOME/miniconda3" "$HOME/anaconda3" /opt/conda /software/miniconda3/4.10.3; do [ -f "$__c/etc/profile.d/conda.sh" ] && { . "$__c/etc/profile.d/conda.sh"; break; }; done
conda activate fedagent-verl08
cd "$REPO"
RUN_ID=$(date +%H%M%S)-$$
mkdir -p runs/t2_stack
BAR=runs/t2_stack/stack_${RUN_ID}.barrier
ln -sf "stack_${RUN_ID}.barrier" runs/t2_stack/latest.barrier
: > "$BAR"
trap 'echo "[trap] driver EXIT rc=$? $(date +%T)" >> "'"$BAR"'"' EXIT
t0=$(date +%s); echo "START t2_stack RUN_ID=$RUN_ID $(date +%T) host=$(hostname)" >> "$BAR"

# --- 0. wait for the WS r4 wiring to finish + GPUs to drain -------------------------------
for i in $(seq 1 120); do
  grep -q "=== WS_R4 DONE ===" runs/paper_ws/latest.barrier 2>/dev/null && break
  sleep 30
done
for i in $(seq 1 20); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | sort -rn | head -1)
  [ "${used:-99999}" -lt 2000 ] && break
  sleep 30
done
if [ "${used:-99999}" -ge 2000 ]; then
  echo "ABORT: GPUs still busy (max used=${used} MiB) $(date +%T)" >> "$BAR"; exit 1
fi
echo "[wait] WS r4 done + GPUs drained; settling 60s $(date +%T)" >> "$BAR"
sleep 60
export CUDA_VISIBLE_DEVICES=0,1,2,3
A=tools/verl08_migration/accel

run_one() {  # $1=tag $2=config $3=output_dir
  export RAY_TMPDIR=/tmp/ray_t2_$1; mkdir -p "$RAY_TMPDIR"
  rm -rf "$3"
  local c0=$(date +%s); echo "[$1] START $(date +%T)" >> "$BAR"
  python -u -m fedagent.fed.run_fed --config "$2" > "runs/t2_stack/$1.log" 2>&1
  echo "[$1] rc=$? wall=$(($(date +%s)-c0))s $(date +%T)" >> "$BAR"
}
cmp_agg() {  # $1=label $2=dirA $3=dirB   (round_2 aggregated actor shards)
  local a="$2/round_2/aggregated/checkpoints/global_step_0/actor"
  local b="$3/round_2/aggregated/checkpoints/global_step_0/actor"
  if [ -d "$a" ] && [ -d "$b" ]; then
    echo "[cmp:$1]" >> "$BAR"
    python tools/verl08_migration/compare_fsdp_checkpoints.py --a "$a" --b "$b" 2>/dev/null \
      | grep -E "OVERALL|VERDICT" >> "$BAR"
  else
    echo "[cmp:$1] SKIPPED (missing $a or $b)" >> "$BAR"
  fi
}

# --- 1+2. ALFWorld Tier-2 suite -----------------------------------------------------------
run_one alf_base  "$A/alfworld/alf_t2_base.yaml"  runs/t2_alf/base
run_one alf_cache "$A/alfworld/alf_t2_cache.yaml" runs/t2_alf/cache
run_one alf_scope "$A/alfworld/alf_t2_scope.yaml" runs/t2_alf/scope
run_one alf_feval "$A/alfworld/alf_t2_feval.yaml" runs/t2_alf/feval
run_one alf_all   "$A/alfworld/alf_t2_all.yaml"   runs/t2_alf/all
cmp_agg cache runs/t2_alf/base runs/t2_alf/cache
cmp_agg scope runs/t2_alf/base runs/t2_alf/scope
cmp_agg feval runs/t2_alf/base runs/t2_alf/feval
cmp_agg all   runs/t2_alf/base runs/t2_alf/all
# cache_b: manifest now warm -> round-1 service warm should collapse too
run_one alf_cache_b "$A/alfworld/alf_t2_cache.yaml" runs/t2_alf/cache

# --- 3. TinyGuess cross-round equivalence arms --------------------------------------------
run_one tg_base     "$A/dev/t2_base.yaml"     runs/t2_ab/base
run_one tg_hfexport "$A/dev/t2_hfexport.yaml" runs/t2_ab/hfexport
run_one tg_lanes    "$A/dev/t2_lanes.yaml"    runs/t2_ab/lanes
cmp_agg hfexport runs/t2_ab/base runs/t2_ab/hfexport
cmp_agg lanes    runs/t2_ab/base runs/t2_ab/lanes

# --- 4. OPTIMAL-COMBO paper-config 2-round measurements ------------------------------------
run_one alf_combo       "$A/alfworld/paper_alf_combo.yaml"       runs/paper_alf/combo
run_one ws_combo        "$A/webshop/paper_ws_combo.yaml"         runs/paper_ws/combo
run_one alf_combo_lanes "$A/alfworld/paper_alf_combo_lanes.yaml" runs/paper_alf/combo_lanes
run_one ws_combo_lanes  "$A/webshop/paper_ws_combo_lanes.yaml"   runs/paper_ws/combo_lanes

# --- 5. one_step_off timing probe (off-policy; timing only) --------------------------------
run_one oso_probe "$A/dev/oso_probe.yaml" runs/oso/probe

echo "BARRIER total=$(($(date +%s)-t0))s" >> "$BAR"
echo "=== T2 STACK DONE ===" >> "$BAR"
