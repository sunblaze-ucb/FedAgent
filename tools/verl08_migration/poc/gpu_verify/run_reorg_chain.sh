#!/bin/bash
# Reorg + family verification chain (NOT for commit; _scratch is gitignored).
# Runs 3 fast smokes sequentially, each total_training_steps=1 / G=2 / 2 clients x 2 rounds:
#   1) webshop GRPO + catalog_split  -> reorg runtime + env-het through the relocated service
#   2) alfworld GRPO uniform          -> 2nd env + relocated alfworld service + max_turns=50
#   3) webshop PPO (gae)              -> critic federation
# Writes pass/fail per smoke to reorg_logs/SUMMARY.txt. Each smoke is "PASS" iff run_fed
# prints "FEDERATED LOOP CLOSED".
cd /gpfs/projects/b1222/userdata/canyu/kangyu/fedagent
for __c in "$CONDA_PREFIX_1" "$HOME/miniconda3" "$HOME/anaconda3" /opt/conda /software/miniconda3/4.10.3; do [ -f "$__c/etc/profile.d/conda.sh" ] && { . "$__c/etc/profile.d/conda.sh"; break; }; done
conda activate fedagent-verl08   # NOTE: no `set -u` — conda's activate/deactivate hooks reference unbound vars
export CUDA_HOME=/hpc/software/cuda/cuda-12.1.0
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
export PYTHONUNBUFFERED=1

LOGDIR=_scratch/gpu_verify/reorg_logs
mkdir -p "$LOGDIR"
SUMMARY="$LOGDIR/SUMMARY.txt"
: > "$SUMMARY"
echo "[chain] host=$(hostname) start=$(date)" | tee -a "$SUMMARY"

cleanup() {
  ray stop --force >/dev/null 2>&1 || true
  # NOTE: new module paths -> match service.server + uvicorn (the old 'webshop_service.server' is gone)
  pkill -9 -f "main_ppo_fed|aggregate_fedavg_fsdp|model_merger|vllm|EngineCore|raylet|gcs_server|plasma|service\.server|uvicorn" 2>/dev/null || true
  sleep 6
}

run() {
  local name="$1" cfg="$2"
  cleanup
  echo "[chain] $(date) START $name ($cfg)" | tee -a "$SUMMARY"
  if python -m fedagent.fed.run_fed --config "$cfg" > "$LOGDIR/$name.log" 2>&1; then
    if grep -q "FEDERATED LOOP CLOSED" "$LOGDIR/$name.log"; then
      echo "[chain] $(date) PASS  $name" | tee -a "$SUMMARY"
    else
      echo "[chain] $(date) RAN-NO-CLOSE $name (rc=0 but no 'LOOP CLOSED' line)" | tee -a "$SUMMARY"
    fi
  else
    rc=$?
    echo "[chain] $(date) FAIL  $name (rc=$rc); tail:" | tee -a "$SUMMARY"
    tail -20 "$LOGDIR/$name.log" | sed 's/^/    /' | tee -a "$SUMMARY"
  fi
}

run reorg_webshop  _scratch/gpu_verify/reorg_webshop.yaml
run reorg_alfworld _scratch/gpu_verify/alfworld_smoke.yaml
run reorg_ppo      _scratch/gpu_verify/ppo_smoke.yaml
cleanup
echo "[chain] $(date) ALL DONE" | tee -a "$SUMMARY"
