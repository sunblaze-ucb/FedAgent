#!/bin/bash
# one_step_off timing probe re-run (after the hydra searchpath fix: fedagent_ppo split into
# body + thin primary so fedagent_one_step_off can layer the body). Off-policy — timing only.
# Run inside a FOREGROUND srun --overlap step.
REPO=/gpfs/projects/b1222/userdata/canyu/kangyu/fedagent
for __c in "$CONDA_PREFIX_1" "$HOME/miniconda3" "$HOME/anaconda3" /opt/conda /software/miniconda3/4.10.3; do [ -f "$__c/etc/profile.d/conda.sh" ] && { . "$__c/etc/profile.d/conda.sh"; break; }; done
conda activate fedagent-verl08
cd "$REPO"
BAR=runs/oso/probe2.barrier
mkdir -p runs/oso
: > "$BAR"
trap 'echo "[trap] driver EXIT rc=$? $(date +%T)" >> "'"$BAR"'"' EXIT
echo "START oso_probe2 $(date +%T) host=$(hostname)" >> "$BAR"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export RAY_TMPDIR=/tmp/ray_oso2; mkdir -p "$RAY_TMPDIR"
rm -rf runs/oso/probe
c0=$(date +%s)
python -u -m fedagent.fed.run_fed --config tools/verl08_migration/accel/dev/oso_probe.yaml \
  > runs/t2_stack/oso_probe2.log 2>&1
echo "[oso_probe2] rc=$? wall=$(($(date +%s)-c0))s $(date +%T)" >> "$BAR"
echo "=== OSO2 DONE ===" >> "$BAR"
