#!/bin/bash
# WS r4 wiring re-run (killed by walltime on 07-02 on the old allocation): the REAL WebShop paper
# config + webshop_replicas=4 vs the old-stack baseline (cold+base-eval 707s / round 496s /
# eval 630s). Single run; run inside a FOREGROUND srun --overlap step on the 2-day 4-card job.
# (no `set -u`: conda deactivate is not nounset-clean)
REPO=/gpfs/projects/b1222/userdata/canyu/kangyu/fedagent
for __c in "$CONDA_PREFIX_1" "$HOME/miniconda3" "$HOME/anaconda3" /opt/conda /software/miniconda3/4.10.3; do [ -f "$__c/etc/profile.d/conda.sh" ] && { . "$__c/etc/profile.d/conda.sh"; break; }; done
conda activate fedagent-verl08
cd "$REPO"
RUN_ID=$(date +%H%M%S)-$$
mkdir -p runs/paper_ws
BAR=runs/paper_ws/r4_${RUN_ID}.barrier
ln -sf "r4_${RUN_ID}.barrier" runs/paper_ws/latest.barrier
: > "$BAR"
trap 'echo "[trap] driver EXIT rc=$? $(date +%T)" >> "'"$BAR"'"' EXIT
export CUDA_VISIBLE_DEVICES=0,1,2,3
export RAY_TMPDIR=/tmp/ray_paper_ws; mkdir -p "$RAY_TMPDIR"
rm -rf runs/paper_ws/wiring_r4
t0=$(date +%s); echo "START ws_r4 RUN_ID=$RUN_ID $(date +%T) host=$(hostname)" >> "$BAR"
python -u -m fedagent.fed.run_fed --config tools/verl08_migration/accel/webshop/paper_ws_wiring_r4.yaml > runs/paper_ws/wiring_r4.log 2>&1
echo "[ws_r4] rc=$? wall=$(($(date +%s)-t0))s $(date +%T)" >> "$BAR"
echo "=== WS_R4 DONE ===" >> "$BAR"
