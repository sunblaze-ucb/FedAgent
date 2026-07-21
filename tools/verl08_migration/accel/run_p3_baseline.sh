#!/bin/bash
# Clean SOLO baselines for #3 speed scaling: t1(2gpu) then t1(4gpu), one client, eval off, no contention.
REPO=/gpfs/projects/b1222/userdata/canyu/kangyu/fedagent
for __c in "$CONDA_PREFIX_1" "$HOME/miniconda3" "$HOME/anaconda3" /opt/conda /software/miniconda3/4.10.3; do [ -f "$__c/etc/profile.d/conda.sh" ] && { . "$__c/etc/profile.d/conda.sh"; break; }; done; conda activate fedagent-verl08; cd "$REPO"
for spec in "p3_2gpu_A:t1_2gpu_solo" "p3_4gpu:t1_4gpu_solo"; do
  CFG=${spec%%:*}; TAG=${spec##*:}
  OUT=$(grep -E "^output_dir:" _scratch/accel/$CFG.yaml | awk '{print $2}'); rm -rf "$OUT"
  t0=$(date +%s); echo "=== BASELINE $TAG ($CFG) START $t0 ==="
  python -u -m fedagent.fed.run_fed --config _scratch/accel/$CFG.yaml > _scratch/accel/${CFG}_solo.log 2>&1
  echo "=== BASELINE $TAG rc=$? wall=$(($(date +%s)-t0))s END ==="
done
echo "BASELINES DONE"
