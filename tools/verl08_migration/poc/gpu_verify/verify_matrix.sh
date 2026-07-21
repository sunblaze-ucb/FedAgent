#!/bin/bash
# RELEASE smoke matrix -> GPFS (persistent + login-node-visible; no node-local /tmp).
# Covers the reviewer's required mode list at 0.5B/1-GPU (the 1.5B 4-GPU GRPO/PPO run on qgpu3013):
#   federated (2rd, fedprox off) | centralized | local | eval-on | fedprox-on
# Each MUST reach "FEDERATED LOOP CLOSED". Robust pre-run cleanup (pkill uvicorn/trainer + free ports)
# so the in-process WebShop service can't collide on a stale port between sequential runs.
set +e
REPO=/gpfs/projects/b1222/userdata/canyu/kangyu/fedagent
cd "$REPO" || exit 2
for __c in "$CONDA_PREFIX_1" "$HOME/miniconda3" "$HOME/anaconda3" /opt/conda /software/miniconda3/4.10.3; do [ -f "$__c/etc/profile.d/conda.sh" ] && { . "$__c/etc/profile.d/conda.sh"; break; }; done
conda activate fedagent-verl08
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
G=_scratch/gpu_verify
RUNS=$REPO/_scratch/gpu_verify/runs        # GPFS (persistent, visible from login node)
mkdir -p "$RUNS"
LAZY=$G/smoke_webshop_lazy.yaml
EVAL=$G/smoke_webshop_eval.yaml
echo "[MATRIX start $(date)] host=$(hostname)  ->  $RUNS"

cleanup(){ pkill -f "uvicorn" 2>/dev/null; pkill -f "main_ppo_fed" 2>/dev/null; pkill -f "service.server" 2>/dev/null
  for p in $(seq 8400 8520) $(seq 10000 10520); do fuser -k "$p/tcp" 2>/dev/null; done; sleep 8; }

run(){ local tag=$1; shift
  cleanup
  echo "===== $tag START $(date) ====="
  timeout 4500 python -m fedagent.fed.run_fed "$@" --output-dir "$RUNS/$tag" > "$G/vm_$tag.log" 2>&1
  echo "===== $tag exit=$? $(date) ====="
  grep -iE "mode=|FEDERATED LOOP CLOSED|FedAvg.*OK|merge.*OK|VAL service|val success|success_rate|eval ON|fedprox\] enabled|Duplicate GPU|Traceback|Error:|rc=" "$G/vm_$tag.log" | tail -12
  # persistent summary artifact (GPFS)
  ls "$RUNS/$tag"/federated_summary.json 2>/dev/null && echo "summary: $RUNS/$tag/federated_summary.json"
}

run federated   --config $LAZY --clients 2 --rounds 2 --n-gpus 1 --fedprox-mu 0   --port-base 10000
run centralized --config $LAZY --clients 1 --rounds 2 --n-gpus 1 --fedprox-mu 0   --port-base 10100
run local       --config $LAZY --local-client-id 0 --clients 2 --rounds 2 --n-gpus 1 --fedprox-mu 0 --port-base 10200
run eval_on     --config $EVAL --n-gpus 1                                          # val ports baked in config (8400/8500)
run fedprox_on  --config $LAZY --clients 2 --rounds 2 --n-gpus 1 --fedprox-mu 0.1 --port-base 10400
echo "[MATRIX DONE $(date)]"
