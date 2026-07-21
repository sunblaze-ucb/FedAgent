#!/bin/bash
# $1=config basename, $2=CUDA_VISIBLE_DEVICES, $3=tag. Pins GPUs + isolates Ray tmp, times the run.
CFG=$1; GPUS=$2; TAG=$3
REPO=/gpfs/projects/b1222/userdata/canyu/kangyu/fedagent
for __c in "$CONDA_PREFIX_1" "$HOME/miniconda3" "$HOME/anaconda3" /opt/conda /software/miniconda3/4.10.3; do [ -f "$__c/etc/profile.d/conda.sh" ] && { . "$__c/etc/profile.d/conda.sh"; break; }; done
conda activate fedagent-verl08
cd "$REPO"
export CUDA_VISIBLE_DEVICES=$GPUS
export RAY_TMPDIR=/tmp/ray_$TAG
mkdir -p "$RAY_TMPDIR"
OUT=$(grep -E "^output_dir:" _scratch/accel/$CFG.yaml | awk '{print $2}'); rm -rf "$OUT"
t0=$(date +%s); echo "P3[$TAG] cfg=$CFG CUDA=$GPUS START $t0"
python -u -m fedagent.fed.run_fed --config _scratch/accel/$CFG.yaml > _scratch/accel/$CFG.log 2>&1
echo "P3[$TAG] rc=$? wall=$(($(date +%s)-t0))s END"
