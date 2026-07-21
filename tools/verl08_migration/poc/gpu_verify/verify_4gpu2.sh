#!/bin/bash
# qgpu3013 phase-2 (after GRPO/PPO/FedProx): fill the heavy 4-GPU gaps.
#   1) EVAL path (1.5B): val_env_spec + test_freq=1 -> val service scores the global model (paper metric)
#   2) ALFWorld GRPO 1.5B 4-GPU federated full loop (2nd env at paper size)
#   3) lazy client SAMPLING: WebShop 4cl / 2-per-round / 2rd, 1.5B, fedprox_mu=0.1 -> only the round's
#      2 selected services start (not all 4) + partial-teardown + 1.5B FedProx at sampling
set +e
REPO=/gpfs/projects/b1222/userdata/canyu/kangyu/fedagent
cd "$REPO" || exit 2
for __c in "$CONDA_PREFIX_1" "$HOME/miniconda3" "$HOME/anaconda3" /opt/conda /software/miniconda3/4.10.3; do [ -f "$__c/etc/profile.d/conda.sh" ] && { . "$__c/etc/profile.d/conda.sh"; break; }; done
conda activate fedagent-verl08
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
G=_scratch/gpu_verify
M15=/projects/b1222/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306
echo "[4GPU2 ALL start $(date)] host=$(hostname)"

run(){ local tag=$1; shift; local cfg=$1; shift
  pkill -f "envs.webshop.service.server" 2>/dev/null; pkill -f "envs.alfworld.service.server" 2>/dev/null; sleep 4
  echo "===== $tag START $(date) ====="
  timeout 4500 python -m fedagent.fed.run_fed --config "$cfg" "$@" > "$G/v2_$tag.log" 2>&1
  echo "===== $tag exit=$? $(date) ====="
  grep -iE "FEDERATED LOOP CLOSED|model_world_size|FedAvg.*OK|merge.*OK|VAL service|val/|success_rate|val_before_train|eval ON|selected|lazily|Duplicate GPU|fedprox\] enabled|Traceback|Error:" "$G/v2_$tag.log" | tail -16
}

run eval_path     $G/smoke_webshop_eval.yaml   --model-path "$M15"
run alfworld_4gpu $G/smoke_alfworld_4gpu.yaml
run lazy_sampling $G/smoke_webshop_lazy.yaml   --model-path "$M15"
echo "[4GPU2 ALL DONE $(date)]"
