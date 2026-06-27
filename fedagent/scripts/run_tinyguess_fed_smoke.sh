#!/bin/bash
# Phase 6/7 smoke: the FedAgent FEDERATED CLOSED LOOP on stock verl 0.8.
#   2 clients x 2 rounds on TinyGuess, each client = `python -m fedagent.main_ppo_fed`,
#   clients' FSDP checkpoints FedAvg'd (matched-PG torchrun) -> merged to HF ->
#   round 2 trains from the aggregated model. Driven by fedagent.fed.run_fed.
# Run on the GPU node:  srun --jobid=<JID> --overlap bash fedagent/scripts/run_tinyguess_fed_smoke.sh
set -e
source /software/miniconda3/4.10.3/etc/profile.d/conda.sh
conda activate fedagent-verl08

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"        # .../fedagent/fedagent/scripts
PKG_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"            # .../fedagent/fedagent
REPO_ROOT="$(cd "$PKG_DIR/.." && pwd)"             # .../fedagent
export PYTHONPATH="$REPO_ROOT:$PYTHONPATH"          # so `import fedagent` works (driver + Ray workers)
export VERL_CFG="$(python -c 'import verl,os;print(os.path.join(os.path.dirname(verl.__file__),"trainer","config"))')"

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_USE_V1=1 VERL_LOGGING_LEVEL=WARN
export VLLM_USE_DEEP_GEMM=0 VLLM_SKIP_DEEP_GEMM_WARMUP=1
export CUDA_HOME=/hpc/software/cuda/cuda-12.1.0

CONFIG="${1:-$PKG_DIR/config/examples/tinyguess_2cl_2rd.yaml}"
echo "REPO_ROOT=$REPO_ROOT"
echo "VERL_CFG=$VERL_CFG"
echo "CONFIG=$CONFIG"
echo "host=$(hostname) ndev=$(python -c 'import torch;print(torch.cuda.device_count())')"

python -m fedagent.fed.run_fed --config "$CONFIG"

echo "===== federated tree ====="
OUT=$(python -c "from omegaconf import OmegaConf;import sys;print(OmegaConf.load('$CONFIG').get('output_dir','/tmp/xbb9020_fedagent_fed_tinyguess'))")
find "$OUT" -maxdepth 5 \( -name "federated_summary.json" -o -name "config.json" -o -name "*.safetensors" -o -name "model_world_size_*_rank_*.pt" -o -name "latest_checkpointed_iteration.txt" \) 2>/dev/null | sort
echo "===== DONE ====="
