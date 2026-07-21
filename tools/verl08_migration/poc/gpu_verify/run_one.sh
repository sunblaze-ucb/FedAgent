#!/bin/bash
# Clean stale Ray/vLLM state on this node, then run one fed config to completion.
cfg="$1"
cd /gpfs/projects/b1222/userdata/canyu/kangyu/fedagent
for __c in "$CONDA_PREFIX_1" "$HOME/miniconda3" "$HOME/anaconda3" /opt/conda /software/miniconda3/4.10.3; do [ -f "$__c/etc/profile.d/conda.sh" ] && { . "$__c/etc/profile.d/conda.sh"; break; }; done
conda activate fedagent-verl08
# vLLM 0.11 / deep_gemm needs CUDA_HOME (else _find_cuda_home() asserts at engine init).
# Matches the proven fedagent/scripts/run_webshop_fed_smoke.sh.
export CUDA_HOME=/hpc/software/cuda/cuda-12.1.0
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
ray stop --force >/dev/null 2>&1 || true
pkill -9 -f "main_ppo_fed|aggregate_fedavg_fsdp|model_merger|vllm|EngineCore|raylet|gcs_server|plasma|webshop_service.server|alfworld_service.server" 2>/dev/null || true
sleep 6
export PYTHONUNBUFFERED=1
echo "[run_one] $(hostname) CUDA_HOME=$CUDA_HOME starting $(basename $cfg) at $(date)"
exec python -m fedagent.fed.run_fed --config "$cfg"
