#!/bin/bash
# Phase 0(b) async agent-loop spike: drive TinyGuessEnv via a custom AgentLoop on
# stock verl 0.8 (vllm async rollout) + GRPO, 2 steps, on 2 H100s.
# Run on the GPU node via: srun --jobid=<JID> --overlap bash run_phase0b.sh
set -e
source /software/miniconda3/4.10.3/etc/profile.d/conda.sh
conda activate fedagent-verl08

HERE="$(cd "$(dirname "$0")" && pwd)"
FED_ROOT="$(cd "$HERE/../.." && pwd)"
export PYTHONPATH="$HERE:$FED_ROOT:$PYTHONPATH"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_USE_V1=1
export VERL_LOGGING_LEVEL=WARN
# deep_gemm (Hopper GEMM opt) asserts a CUDA toolkit / CUDA_HOME, absent on this node.
# Disable it (not needed for bf16) + point CUDA_HOME at the cuda-12.1 module as a fallback.
export VLLM_USE_DEEP_GEMM=0
export VLLM_SKIP_DEEP_GEMM_WARMUP=1
export CUDA_HOME=/hpc/software/cuda/cuda-12.1.0

# locate a local Qwen2.5-0.5B-Instruct snapshot
MODEL=""
for base in /projects/b1222/.cache/huggingface ~/.cache/huggingface; do
  cand=$(ls -d "$base"/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/*/ 2>/dev/null | head -1)
  [ -n "$cand" ] && MODEL="$cand" && break
done
[ -z "$MODEL" ] && { echo "No local Qwen2.5-0.5B-Instruct snapshot found"; exit 1; }
MODEL="${MODEL%/}"   # strip trailing slash (verl copy_to_local rejects it)
echo "MODEL=$MODEL"
echo "host=$(hostname) ndev=$(python -c 'import torch;print(torch.cuda.device_count())')"

CKPT=/tmp/xbb9020_phase0b_ckpts
rm -rf "$CKPT"
cd "$HERE"

python -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files="$HERE/tiny_envs.yaml" \
  data.val_files="$HERE/tiny_envs.yaml" \
  data.custom_cls.path="$HERE/tiny_guess_dataset.py" \
  data.custom_cls.name=TinyGuessDataset \
  data.train_batch_size=8 \
  data.val_batch_size=8 \
  data.max_prompt_length=512 \
  data.max_response_length=512 \
  actor_rollout_ref.model.path="$MODEL" \
  +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=8 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.multi_turn.enable=True \
  actor_rollout_ref.rollout.n=4 \
  actor_rollout_ref.rollout.prompt_length=512 \
  actor_rollout_ref.rollout.response_length=512 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.rollout.agent.agent_loop_config_path="$HERE/agent.yaml" \
  actor_rollout_ref.rollout.agent.default_agent_loop=gym_text \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  algorithm.use_kl_in_reward=False \
  reward_model.enable=False \
  trainer.n_gpus_per_node=2 \
  trainer.nnodes=1 \
  trainer.total_epochs=1 \
  trainer.total_training_steps=2 \
  trainer.save_freq=1 \
  trainer.test_freq=-1 \
  trainer.val_before_train=False \
  trainer.logger=[console] \
  trainer.project_name=phase0b \
  trainer.experiment_name=tinyguess \
  trainer.default_local_dir="$CKPT" 2>&1

echo "===== ckpt tree ====="
find "$CKPT" -maxdepth 4 -name "*.pt" -o -name "latest_checkpointed_iteration.txt" 2>/dev/null | head
echo "===== DONE ====="
