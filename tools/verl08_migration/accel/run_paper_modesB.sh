#!/bin/bash
REPO=/gpfs/projects/b1222/userdata/canyu/kangyu/fedagent
for __c in "$CONDA_PREFIX_1" "$HOME/miniconda3" "$HOME/anaconda3" /opt/conda /software/miniconda3/4.10.3; do [ -f "$__c/etc/profile.d/conda.sh" ] && { . "$__c/etc/profile.d/conda.sh"; break; }; done
conda activate fedagent-verl08
cd "$REPO"
echo "PAPER-MODES-B SWEEP START $(date +%s)"
for M in inline shared worker; do
  CFG=paper_ws_modeB_$M
  OUT=$(grep -E "^output_dir:" _scratch/accel/$CFG.yaml | awk '{print $2}')
  rm -rf "$OUT"
  t0=$(date +%s); echo "=== MODEB $M START $t0 ==="
  python -u -m fedagent.fed.run_fed --config _scratch/accel/$CFG.yaml > _scratch/accel/$CFG.log 2>&1
  rc=$?; t1=$(date +%s); echo "=== MODEB $M rc=$rc wall=$((t1-t0))s END $t1 ==="
done
echo "PAPER-MODES-B SWEEP DONE $(date +%s)"
