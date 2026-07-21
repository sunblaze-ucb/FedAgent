#!/bin/bash
# Cross-round full-loop smoke (lever #4 extended): ONE process spans 2 rounds. Closing rc=0 with
# 2 rounds aggregated proves the signal-file handshake (done/go/stop) + cross-round reset works.
# NO `set -u` (conda deactivate hook references an unbound var).
REPO=/gpfs/projects/b1222/userdata/canyu/kangyu/fedagent
for __c in "$CONDA_PREFIX_1" "$HOME/miniconda3" "$HOME/anaconda3" /opt/conda /software/miniconda3/4.10.3; do [ -f "$__c/etc/profile.d/conda.sh" ] && { . "$__c/etc/profile.d/conda.sh"; break; }; done
conda activate fedagent-verl08
cd "$REPO"
rm -rf _scratch/accel/xround_full_out
echo "XROUND FULL START $(date +%s)"
python -u -m fedagent.fed.run_fed --config _scratch/accel/xround_full.yaml
echo "XROUND FULL rc=$? end=$(date +%s)"
echo "=== cold-starts in the run (expect exactly ONE 'Started a local Ray instance') ==="
grep -c "Started a local Ray instance" _scratch/accel/xround_full_out/round_*/persistent_training.log 2>/dev/null
echo "=== signal handshake trace ==="
ls -1 _scratch/accel/xround_full_out/_xround/ 2>/dev/null
echo "=== final aggregated actors (both rounds) ==="
ls -d _scratch/accel/xround_full_out/round_*/aggregated/checkpoints/global_step_0/actor 2>/dev/null
