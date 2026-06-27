#!/bin/bash
# FedAgent verl-0.8 FIRST REAL SCIENCE RUN: federated WebShop with ENV-LEVEL
# heterogeneity (Catalog Split, paper Variant 1). 2 clients x 2 rounds; each client
# gets a DISJOINT product catalog via its own remote WebShop service, FedAvg
# aggregates across the divergent envs, round 2 re-enters from the aggregate.
# fedagent.fed.run_fed launches/【tears down】 the per-client services itself.
# Run on the GPU node:  srun --jobid=<JID> --overlap bash fedagent/scripts/run_webshop_fed_smoke.sh
set -e
source /software/miniconda3/4.10.3/etc/profile.d/conda.sh
conda activate fedagent-verl08

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"        # .../fedagent/fedagent/scripts
PKG_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"            # .../fedagent/fedagent
REPO_ROOT="$(cd "$PKG_DIR/.." && pwd)"             # .../fedagent
export PYTHONPATH="$REPO_ROOT:$PYTHONPATH"          # `import fedagent` (driver, Ray workers, service)
export VERL_CFG="$(python -c 'import verl,os;print(os.path.join(os.path.dirname(verl.__file__),"trainer","config"))')"

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_USE_V1=1 VERL_LOGGING_LEVEL=WARN
export VLLM_USE_DEEP_GEMM=0 VLLM_SKIP_DEEP_GEMM_WARMUP=1
export CUDA_HOME=/hpc/software/cuda/cuda-12.1.0

CONFIG="${1:-$PKG_DIR/config/examples/webshop/2cl_catalog_split.yaml}"
echo "REPO_ROOT=$REPO_ROOT"
echo "VERL_CFG=$VERL_CFG"
echo "CONFIG=$CONFIG"
echo "host=$(hostname) ndev=$(python -c 'import torch;print(torch.cuda.device_count())')"

python -m fedagent.fed.run_fed --config "$CONFIG" "${@:2}"   # extra args (e.g. --base-seed 43 --output-dir ... --port-base 8086) forwarded

echo "===== federated tree ====="
OUT=$(python -c "from omegaconf import OmegaConf;print(OmegaConf.load('$CONFIG').get('output_dir','/tmp/xbb9020_fedagent_fed_webshop'))")
find "$OUT" -maxdepth 5 \( -name "federated_summary.json" -o -name "config.json" -o -name "*.safetensors" -o -name "model_world_size_*_rank_*.pt" \) 2>/dev/null | sort
echo "===== DONE ====="
