#!/bin/bash
# Detached 1-GPU verification -- fills the gaps the 4-GPU driver doesn't cover:
#   1) centralized run-mode (total_clients=1: FedAvg-of-1 == identity, continued training)
#   2) local run-mode (local_client_id=0: pin one client, NO federation -- paper "Local Agent")
#   3) ALFWorld GRPO federated FULL loop (the 2nd env: shard -> train -> FedAvg ws=1 -> merge -> round2)
# Modes 1-2 use tinyguess (in-process, fast); mode 3 is the heavy real-env loop. Sequential on 1 GPU.
set +e
REPO=/gpfs/projects/b1222/userdata/canyu/kangyu/fedagent
cd "$REPO" || exit 2
for __c in "$CONDA_PREFIX_1" "$HOME/miniconda3" "$HOME/anaconda3" /opt/conda /software/miniconda3/4.10.3; do [ -f "$__c/etc/profile.d/conda.sh" ] && { . "$__c/etc/profile.d/conda.sh"; break; }; done
conda activate fedagent-verl08
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
G=_scratch/gpu_verify
TG=fedagent/config/fed_tinyguess_2cl_2rd.yaml
echo "[1GPU ALL start $(date)] host=$(hostname)"
nvidia-smi --query-gpu=index,memory.total --format=csv,noheader | head

run(){ # tag  config  extra...
  local tag=$1; shift; local cfg=$1; shift
  pkill -f "envs.alfworld.service.server" 2>/dev/null; sleep 4
  echo "===== $tag START $(date) ====="
  timeout 3000 python -m fedagent.fed.run_fed --config "$cfg" "$@" > "$G/v1_$tag.log" 2>&1
  echo "===== $tag exit=$? $(date) ====="
  grep -iE "mode=|model_world_size|FEDERATED LOOP CLOSED|final aggregated|FedAvg.*OK|merge.*OK|Duplicate GPU|rc=|Traceback|Error" "$G/v1_$tag.log" | tail -16
}

# 1) centralized: total_clients=1 -> mode=centralized
run centralized $TG --clients 1 --rounds 2 --n-gpus 1
# 2) local: pin client 0 of 2 -> mode=local, no FedAvg
run local       $TG --local-client-id 0 --clients 2 --rounds 2 --n-gpus 1
# 3) ALFWorld federated full loop (2nd env), 0.5B, 1 GPU
run alfworld_fed $G/smoke_alfworld_1gpu.yaml --n-gpus 1
echo "[1GPU ALL DONE $(date)]"
