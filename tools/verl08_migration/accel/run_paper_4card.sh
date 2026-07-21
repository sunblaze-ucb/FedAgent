#!/bin/bash
# 4-card comparison @ paper training settings (1.5B, G=8, webshop_15, n=500 val, 100-client part).
# ALL FOUR use 4 GPUs: B-{worker,inline,shared} train on 4; A-parallel = 2 train + 2 eval.
# Order: worker (validated) -> inline -> parallel -> shared (OOM-risk last so others still land).
REPO=/gpfs/projects/b1222/userdata/canyu/kangyu/fedagent
for __c in "$CONDA_PREFIX_1" "$HOME/miniconda3" "$HOME/anaconda3" /opt/conda /software/miniconda3/4.10.3; do [ -f "$__c/etc/profile.d/conda.sh" ] && { . "$__c/etc/profile.d/conda.sh"; break; }; done
conda activate fedagent-verl08
cd "$REPO"
echo "4CARD SWEEP START $(date +%s)"
run() {  # $1 = label, $2 = config basename
  local lbl=$1 CFG=$2
  local OUT=$(grep -E "^output_dir:" _scratch/accel/$CFG.yaml | awk '{print $2}')
  rm -rf "$OUT"
  local t0=$(date +%s); echo "=== $lbl START $t0 ==="
  python -u -m fedagent.fed.run_fed --config _scratch/accel/$CFG.yaml > _scratch/accel/$CFG.log 2>&1
  local rc=$?; local t1=$(date +%s); echo "=== $lbl rc=$rc wall=$((t1-t0))s END $t1 ==="
}
run "worker_4gpu"   paper_ws_modeB_worker
run "inline_4gpu"   paper_ws_modeB_inline
run "parallel_2+2"  paper_ws_mode_parallel
run "shared_4gpu"   paper_ws_modeB_shared
echo "4CARD SWEEP DONE $(date +%s)"
