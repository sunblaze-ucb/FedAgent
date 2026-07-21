#!/bin/bash
# Verify the ALFWorld task-het fix: re-run preference + hardness (which crashed with
# "Invalid partition strategy") now that run_fed translates preference->category, hardness->hardiness.
# Expect: per-client services healthy, shards differ, LOOP CLOSED, no "Invalid partition". -> GPFS.
set +e
REPO=/gpfs/projects/b1222/userdata/canyu/kangyu/fedagent; cd "$REPO" || exit 2
for __c in "$CONDA_PREFIX_1" "$HOME/miniconda3" "$HOME/anaconda3" /opt/conda /software/miniconda3/4.10.3; do [ -f "$__c/etc/profile.d/conda.sh" ] && { . "$__c/etc/profile.d/conda.sh"; break; }; done; conda activate fedagent-verl08
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
G=_scratch/gpu_verify; RUNS=$REPO/_scratch/gpu_verify/runs; mkdir -p "$RUNS"
echo "[ALFHETFIX start $(date)] host=$(hostname)"
cleanup(){ pkill -f uvicorn 2>/dev/null; pkill -f main_ppo_fed 2>/dev/null; pkill -f service.server 2>/dev/null
  for p in $(seq 8900 8980); do fuser -k "$p/tcp" 2>/dev/null; done; sleep 6; }
run(){ local tag=$1 cfg=$2 outdir=$3; cleanup; echo "===== $tag START $(date) ====="
  timeout 3600 python -m fedagent.fed.run_fed --config "$G/$cfg" --output-dir "$RUNS/$tag" > "$G/vf_$tag.log" 2>&1
  echo "===== $tag exit=$? $(date) ====="
  grep -iE "FEDERATED LOOP CLOSED|ALFWorld service.*healthy|num_games|Invalid partition|Traceback|Error:" "$G/vf_$tag.log" | tail -6
  for c in 0 1; do grep -hiE "num_games|partition" "$RUNS/$tag"/alfworld_service_client$c.log 2>/dev/null | head -1; done; }
run alfhet_preference_fix2 het_alfworld_preference.yaml /gpfs/projects/b1222/userdata/canyu/kangyu/fedagent/_scratch/gpu_verify/runs/alfhet_preference_fix2
run alfhet_hardness_fix2   het_alfworld_hardness.yaml   /gpfs/projects/b1222/userdata/canyu/kangyu/fedagent/_scratch/gpu_verify/runs/alfhet_hardness_fix2
echo "[ALFHETFIX DONE $(date)]"
