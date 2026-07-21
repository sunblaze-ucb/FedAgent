#!/bin/bash
# Generate WebShop hardness trajectories (reference-policy labels) on the GPU job.
# Reference model = Qwen2.5-1.5B-Instruct (the hardness arm's backbone).
set -e
for __c in "$CONDA_PREFIX_1" "$HOME/miniconda3" "$HOME/anaconda3" /opt/conda /software/miniconda3/4.10.3; do [ -f "$__c/etc/profile.d/conda.sh" ] && { . "$__c/etc/profile.d/conda.sh"; break; }; done
conda activate fedagent-verl08
export CUDA_HOME=/hpc/software/cuda/cuda-12.1.0
export VLLM_USE_DEEP_GEMM=0 VLLM_SKIP_DEEP_GEMM_WARMUP=1
export TOKENIZERS_PARALLELISM=false
cd /gpfs/projects/b1222/userdata/canyu/kangyu/fedagent

N="${1:-1500}"
OUT="${2:-/gpfs/projects/b1222/userdata/canyu/kangyu/fedagent/fedagent/data/hardness/qwen2.5-1.5b_webshop_trajectories.json}"
PORT="${3:-8097}"
MODEL=/projects/b1222/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306

echo "[gen-hardness] $(date) N=$N port=$PORT out=$OUT"
python -m tools.verl08_migration.gen_hardness_trajectories \
  --config fedagent/config/fed_webshop_scaled_hardness.yaml \
  --model "$MODEL" \
  --num-goals "$N" \
  --output "$OUT" \
  --port "$PORT" \
  --n-gpus 4
echo "[gen-hardness] $(date) DONE rc=$?"
