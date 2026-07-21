#!/bin/bash
# Robustly relaunch a fed run on a node that may hold STALE Ray/vLLM state from a prior
# (failed) run -> otherwise the new Ray cluster collides ("Duplicate GPU detected").
# Each node runs ONE run at a time, so a full cleanup here is safe. arg: <config_abs>
cfg="$1"
REPO=/gpfs/projects/b1222/userdata/canyu/kangyu/fedagent
for __c in "$CONDA_PREFIX_1" "$HOME/miniconda3" "$HOME/anaconda3" /opt/conda /software/miniconda3/4.10.3; do [ -f "$__c/etc/profile.d/conda.sh" ] && { . "$__c/etc/profile.d/conda.sh"; break; }; done
conda activate fedagent-verl08 2>/dev/null
echo "[launch] cleaning stale Ray/vLLM/training procs on $(hostname) ..."
ray stop --force >/dev/null 2>&1 || true
pkill -9 -f "main_ppo_fed|aggregate_fedavg_fsdp|model_merger" 2>/dev/null || true
pkill -9 -f "raylet|gcs_server|ray::|plasma" 2>/dev/null || true
pkill -9 -f "vllm|EngineCore" 2>/dev/null || true
pkill -9 -f "webshop_service.server|alfworld_service.server" 2>/dev/null || true
sleep 8
for i in $(seq 1 72); do
  busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | awk '$1>2000{n++} END{print n+0}')
  if [ "${busy:-1}" -eq 0 ]; then echo "[launch] GPUs clean after ~$((i*5))s; launching $(basename $cfg)"; break; fi
  sleep 5
done
exec bash "$REPO/fedagent/scripts/run_webshop_fed_smoke.sh" "$@"
