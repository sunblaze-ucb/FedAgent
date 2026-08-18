#!/bin/bash
# Consolidate federated training.logs from per-node compute /tmp -> shared GPFS, then
# run the asymmetry/compounding summary. The 3-way comparison runs on SEPARATE compute
# nodes (node-local /tmp), so the cross-condition decomposition needs the logs gathered
# into one place first. GPFS is mounted on the compute nodes, so each srun copies its
# node-local logs straight to the shared stage dir.
#
# Usage:
#   collect_fed_logs.sh STAGE LABEL:JOBID:OUTDIR [LABEL:JOBID:OUTDIR ...]
# Example (the 3-way asymmetry):
#   collect_fed_logs.sh /gpfs/.../fedagent/_stage/c4 \
#       envhet:<jobid>:outputs/fedagent_envhet_c4 \
#       task:<jobid>:outputs/fedagent_task_c4 \
#       homog:<jobid>:outputs/fedagent_homog_c4
# then it runs: summarize_fed_run.py envhet=STAGE/envhet task=STAGE/task homog=STAGE/homog \
#                                    --decomp=envhet,task,homog
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
STAGE="$1"; shift
mkdir -p "$STAGE"
declare -a LABELS=()
for spec in "$@"; do
  label="${spec%%:*}"; rest="${spec#*:}"; jobid="${rest%%:*}"; outdir="${rest#*:}"
  LABELS+=("$label")
  dest="$STAGE/$label"; mkdir -p "$dest"
  echo "[collect] $label  (job $jobid)  $outdir -> $dest"
  # copy round_*/client_*/training.log preserving structure, from that job's node-local /tmp
  srun --jobid="$jobid" --overlap -N1 -n1 bash -lc "
    cd '$outdir' 2>/dev/null || { echo '  (outdir missing on node)'; exit 0; }
    find round_* -name training.log -print 2>/dev/null | while read f; do
      mkdir -p '$dest/'\$(dirname \"\$f\"); cp -f \"\$f\" '$dest/'\"\$f\";
    done
    echo \"  copied \$(find round_* -name training.log 2>/dev/null | wc -l) logs\"
  " 2>/dev/null | grep -v "^----\|prologue\|PATH (in\|WORKDIR\|srun job\|Job ID\|Username\|Queue\|Account\|guaranteed\|variables\|not$\|same in\|run script" || true
done
# build the summarizer args + --decomp (assumes order env,task,iid if 3 labels)
args=""
for l in "${LABELS[@]}"; do args="$args $l=$STAGE/$l"; done
decomp=""
if [ "${#LABELS[@]}" -eq 3 ]; then decomp="--decomp=${LABELS[0]},${LABELS[1]},${LABELS[2]}"; fi
echo "[collect] running summarizer ..."
for __c in "$CONDA_PREFIX_1" "$HOME/miniconda3" "$HOME/anaconda3" /opt/conda /software/miniconda3/4.10.3; do [ -f "$__c/etc/profile.d/conda.sh" ] && { . "$__c/etc/profile.d/conda.sh"; break; }; done || true
conda activate fedagent-verl08 2>/dev/null || true
python "$HERE/summarize_fed_run.py" $args $decomp
