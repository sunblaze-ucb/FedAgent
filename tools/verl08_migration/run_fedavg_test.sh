#!/bin/bash
# Validate the generalized matched-PG FedAvg on REAL verl-0.8 checkpoints:
# average two genuinely-different real Qwen FSDP1 shards (global_step_1 + global_step_2
# from the Phase 1 TinyGuess run, as two "clients"), then load the result with verl's
# OWN FSDPCheckpointManager (what round r+1 does). Run on the GPU node via srun.
set -e
source /software/miniconda3/4.10.3/etc/profile.d/conda.sh
conda activate fedagent-verl08
cd "$(dirname "$0")"
export CUDA_HOME=/hpc/software/cuda/cuda-12.1.0 VLLM_USE_DEEP_GEMM=0 VERL_LOGGING_LEVEL=WARN

CKPT=/tmp/xbb9020_fedagent_phase1_ckpts
A="$CKPT/global_step_1/actor"
B="$CKPT/global_step_2/actor"
OUT=/tmp/xbb9020_fedavg_test/checkpoints/global_step_0/actor
rm -rf /tmp/xbb9020_fedavg_test
[ -d "$A" ] && [ -d "$B" ] || { echo "missing client ckpts ($A , $B)"; exit 1; }

echo "===== AGGREGATE (matched-PG FedAvg, ws=2) ====="
torchrun --nproc_per_node=2 aggregate_fedavg_fsdp.py --phase aggregate \
  --client-actor-dirs "$A,$B" --output-actor-dir "$OUT" --global-step 0

echo "===== VERIFY (matched-PG load-back + FedAvg correctness, ws=2) ====="
torchrun --nproc_per_node=2 aggregate_fedavg_fsdp.py --phase verify \
  --client-actor-dirs "$A,$B" --output-actor-dir "$OUT"

echo "===== aggregated tree ====="
find /tmp/xbb9020_fedavg_test -type f 2>/dev/null | sort
echo "===== FEDAVG TEST DONE ====="
