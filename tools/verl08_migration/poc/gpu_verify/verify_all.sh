#!/bin/bash
# Detached GPU verification of the ACTUAL paper training mode (4-GPU FSDP), 1.5B Qwen WebShop:
#   1) GRPO @ 4-GPU FSDP (mode parity vs the original)
#   2) PPO (gae) @ 4-GPU FSDP (critic federation -- previously unverified)
#   3) 0.5B WebShop + FedProx @ 4-GPU (verifies the deferred-patch fix; re-runs what hit Duplicate GPU)
# Sequential on the 4-GPU node. Launched detached (setsid nohup srun) so it survives the harness.
set +e
REPO=/gpfs/projects/b1222/userdata/canyu/kangyu/fedagent
cd "$REPO" || exit 2
for __c in "$CONDA_PREFIX_1" "$HOME/miniconda3" "$HOME/anaconda3" /opt/conda /software/miniconda3/4.10.3; do [ -f "$__c/etc/profile.d/conda.sh" ] && { . "$__c/etc/profile.d/conda.sh"; break; }; done
conda activate fedagent-verl08
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
G=_scratch/gpu_verify
M15=/projects/b1222/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306
echo "[ALL start $(date)] host=$(hostname)"
nvidia-smi --query-gpu=index,memory.total --format=csv,noheader | head

echo "===== PRECHECK: FedProx deferred-patch arms (mu=0.1) ====="
FEDPROX_MU=0.1 PYTHONPATH=$REPO python -c "pass" 2>&1 | grep -iE "fedprox|error|refus" | head

run(){ # tag  config  extra...
  local tag=$1; shift; local cfg=$1; shift
  pkill -f "envs.webshop.service.server" 2>/dev/null; sleep 4
  echo "===== $tag START $(date) ====="
  timeout 3000 python -m fedagent.fed.run_fed --config "$cfg" "$@" > "$G/v_$tag.log" 2>&1
  echo "===== $tag exit=$? $(date) ====="
  grep -iE "model_world_size|FEDERATED LOOP CLOSED|final aggregated|client [0-9] round.*OK|FedAvg (actor|critic).*OK|merge (actor|critic).*OK|Duplicate GPU|fedprox\] (enabled|deferred)|rc=" "$G/v_$tag.log" | tail -16
}

run grpo15_4gpu  $G/smoke_webshop_lazy.yaml --model-path "$M15" --n-gpus 4 --rounds 1 --clients 2 --fedprox-mu 0
run ppo15_4gpu   $G/smoke_webshop_ppo.yaml  --model-path "$M15" --n-gpus 4 --rounds 2
run fedprox_4gpu $G/smoke_webshop_lazy.yaml --n-gpus 4 --rounds 2 --clients 2 --fedprox-mu 0.1
echo "[ALL DONE $(date)]"
