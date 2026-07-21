#!/bin/bash
# Phase 1 smoke: run the TinyGuess env THROUGH the new fedagent package
# (python -m fedagent.main_ppo_fed + Hydra config + custom_cls dataset + registered
# AgentLoop) on stock verl 0.8. Proves the package wiring end-to-end:
# rollout -> GRPO -> actor update -> checkpoint, on 2 H100s.
# Run on the GPU node:  srun --jobid=<JID> --overlap bash fedagent/scripts/run_smoke.sh
set -e
for __c in "$CONDA_PREFIX_1" "$HOME/miniconda3" "$HOME/anaconda3" /opt/conda /software/miniconda3/4.10.3; do [ -f "$__c/etc/profile.d/conda.sh" ] && { . "$__c/etc/profile.d/conda.sh"; break; }; done
conda activate fedagent-verl08

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"        # .../fedagent/fedagent/scripts
PKG_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"            # .../fedagent/fedagent  (the package)
REPO_ROOT="$(cd "$PKG_DIR/.." && pwd)"             # .../fedagent          (repo root)
export PYTHONPATH="$REPO_ROOT:$PYTHONPATH"          # so `import fedagent` works (driver + Ray workers)
# verl's stock ppo_trainer base config dir, for hydra.searchpath in fedagent_ppo.yaml
export VERL_CFG="$(python -c 'import verl,os;print(os.path.join(os.path.dirname(verl.__file__),"trainer","config"))')"

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_USE_V1=1 VERL_LOGGING_LEVEL=WARN
# deep_gemm (Hopper GEMM) asserts a CUDA toolkit; disable + point CUDA_HOME at the module.
export VLLM_USE_DEEP_GEMM=0 VLLM_SKIP_DEEP_GEMM_WARMUP=1
export CUDA_HOME=/hpc/software/cuda/cuda-12.1.0

# locate a local Qwen2.5-0.5B-Instruct snapshot (offline)
MODEL=""
for base in /projects/b1222/.cache/huggingface ~/.cache/huggingface; do
  cand=$(ls -d "$base"/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/*/ 2>/dev/null | head -1)
  [ -n "$cand" ] && MODEL="$cand" && break
done
[ -z "$MODEL" ] && { echo "No local Qwen2.5-0.5B-Instruct snapshot found"; exit 1; }
MODEL="${MODEL%/}"   # verl copy_to_local rejects a trailing slash

CKPT=/tmp/xbb9020_fedagent_phase1_ckpts
rm -rf "$CKPT"

echo "REPO_ROOT=$REPO_ROOT"
echo "VERL_CFG=$VERL_CFG"
echo "MODEL=$MODEL"
echo "host=$(hostname) ndev=$(python -c 'import torch;print(torch.cuda.device_count())')"

python -m fedagent.main_ppo_fed \
  data.train_files="$PKG_DIR/config/envs/tiny_guess.yaml" \
  data.val_files="$PKG_DIR/config/envs/tiny_guess.yaml" \
  data.custom_cls.path="$PKG_DIR/data/agentic_dataset.py" \
  actor_rollout_ref.model.path="$MODEL" \
  +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
  actor_rollout_ref.rollout.agent.agent_loop_config_path="$PKG_DIR/config/agent.yaml" \
  trainer.default_local_dir="$CKPT" 2>&1

echo "===== ckpt tree ====="
find "$CKPT" -maxdepth 4 -type f \( -name "*.pt" -o -name "*.json" -o -name "*.txt" \) 2>/dev/null | sort | head -40
echo "===== DONE ====="
