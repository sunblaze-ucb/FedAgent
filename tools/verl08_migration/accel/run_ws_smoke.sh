#!/bin/bash
# $1 = config basename (no .yaml). Runs a webshop persistent smoke + prints routing evidence.
REPO=/gpfs/projects/b1222/userdata/canyu/kangyu/fedagent
CFG=$1
for __c in "$CONDA_PREFIX_1" "$HOME/miniconda3" "$HOME/anaconda3" /opt/conda /software/miniconda3/4.10.3; do [ -f "$__c/etc/profile.d/conda.sh" ] && { . "$__c/etc/profile.d/conda.sh"; break; }; done
conda activate fedagent-verl08
cd "$REPO"
OUT=$(grep -E "^output_dir:" _scratch/accel/$CFG.yaml | awk '{print $2}')
rm -rf "$OUT"
echo "WS SMOKE($CFG) START $(date +%s)"
python -u -m fedagent.fed.run_fed --config _scratch/accel/$CFG.yaml
echo "WS SMOKE($CFG) rc=$? end=$(date +%s)"
echo "=== routing evidence: worker 'route client' lines ==="
grep -hE "\[persistent\] route client" "$OUT"/round_*/persistent_training.log 2>/dev/null | sort -u
echo "=== per-client service logs got traffic? (request lines per service) ==="
for L in "$OUT"/webshop_service_client*.log; do
  [ -f "$L" ] && echo "  $(basename $L): $(grep -icE "POST|/reset|/step|GET /health|request" "$L") log lines, $(grep -icE "/reset|/step" "$L") reset/step"
done
echo "=== eval metrics (cross-round+val only) ==="
grep -hE "eval|val.*success|val/|aggregated.*val" "$OUT"/federated_summary.json 2>/dev/null | head
ls "$OUT"/round_*/aggregated/hf 2>/dev/null && echo "  aggregated models present"
