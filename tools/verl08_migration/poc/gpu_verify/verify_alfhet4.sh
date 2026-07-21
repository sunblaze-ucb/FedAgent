#!/bin/bash
set +e; REPO=/gpfs/projects/b1222/userdata/canyu/kangyu/fedagent; cd "$REPO" || exit 2
for __c in "$CONDA_PREFIX_1" "$HOME/miniconda3" "$HOME/anaconda3" /opt/conda /software/miniconda3/4.10.3; do [ -f "$__c/etc/profile.d/conda.sh" ] && { . "$__c/etc/profile.d/conda.sh"; break; }; done; conda activate fedagent-verl08
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
G=_scratch/gpu_verify; RUNS=$REPO/_scratch/gpu_verify/runs
echo "[ALFHET4 start $(date)] host=$(hostname)"
cleanup(){ pkill -f uvicorn 2>/dev/null; pkill -f main_ppo_fed 2>/dev/null; pkill -f service.server 2>/dev/null; for p in $(seq 9700 9740); do fuser -k $p/tcp 2>/dev/null; done; sleep 6; }
run(){ local tag=$1 cfg=$2; cleanup; echo "===== $tag START $(date) ====="
  timeout 5000 python -m fedagent.fed.run_fed --config $G/$cfg --output-dir $RUNS/$tag > $G/v4h_$tag.log 2>&1
  echo "===== $tag exit=$? $(date) ====="
  grep -iE "FEDERATED LOOP CLOSED|partition_kwargs|partition=(preference|hardness)|num_games|Invalid partition|Missing required|Traceback|Error:" $G/v4h_$tag.log | tail -6
  for c in 0 1; do grep -hiE "num_games|partition" $RUNS/$tag/alfworld_service_client$c.log 2>/dev/null | head -1; done; }
run alfhet4_preference het_alfworld_preference_4gpu.yaml
run alfhet4_hardness   het_alfworld_hardness_4gpu.yaml
echo "[ALFHET4 DONE $(date)]"
