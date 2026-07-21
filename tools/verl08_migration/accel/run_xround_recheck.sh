#!/bin/bash
REPO=/gpfs/projects/b1222/userdata/canyu/kangyu/fedagent
for __c in "$CONDA_PREFIX_1" "$HOME/miniconda3" "$HOME/anaconda3" /opt/conda /software/miniconda3/4.10.3; do [ -f "$__c/etc/profile.d/conda.sh" ] && { . "$__c/etc/profile.d/conda.sh"; break; }; done
conda activate fedagent-verl08
cd "$REPO"
rm -rf _scratch/accel/xround_recheck_out
sed 's#xround_full_out#xround_recheck_out#' _scratch/accel/xround_full.yaml > _scratch/accel/xround_recheck.yaml
echo "XROUND RECHECK START $(date +%s)"
python -u -m fedagent.fed.run_fed --config _scratch/accel/xround_recheck.yaml
echo "XROUND RECHECK rc=$? end=$(date +%s)"
grep -c "Started a local Ray instance" _scratch/accel/xround_recheck_out/round_*/persistent_training.log 2>/dev/null | head -1
ls -d _scratch/accel/xround_recheck_out/round_*/aggregated/checkpoints/global_step_0/actor 2>/dev/null
