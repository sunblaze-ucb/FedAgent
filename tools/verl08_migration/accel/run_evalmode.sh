#!/bin/bash
REPO=/gpfs/projects/b1222/userdata/canyu/kangyu/fedagent
CFG=$1
for __c in "$CONDA_PREFIX_1" "$HOME/miniconda3" "$HOME/anaconda3" /opt/conda /software/miniconda3/4.10.3; do [ -f "$__c/etc/profile.d/conda.sh" ] && { . "$__c/etc/profile.d/conda.sh"; break; }; done
conda activate fedagent-verl08
cd "$REPO"
OUT=$(grep -E "^output_dir:" _scratch/accel/$CFG.yaml | awk '{print $2}')
rm -rf "$OUT"
echo "EVALMODE($CFG) START $(date +%s)"
python -u -m fedagent.fed.run_fed --config _scratch/accel/$CFG.yaml
echo "EVALMODE($CFG) rc=$? end=$(date +%s)"
echo "=== eval-mode log markers ==="
grep -hE "eval_mode=|Falling back to per-round|eval round [0-9].* async on GPU|round [0-9] VAL .unperturbed.|eval round [0-9] FAILED|FEDERATED LOOP CLOSED" "$OUT"/../*.log 2>/dev/null | head
grep -hE "eval_mode=parallel|eval_mode=shared|async on GPU|round [0-9] VAL|FEDERATED LOOP CLOSED|out of memory|desired GPU memory" "$OUT"/round_*/persistent_training.log 2>/dev/null | head
