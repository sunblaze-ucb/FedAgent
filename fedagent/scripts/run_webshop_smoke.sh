#!/bin/bash
# Phase 2 smoke: drive the REAL WebShop env through the fedagent package on stock verl 0.8.
#   1. start the WebShop remote service (verl-agent-webshop env, background, pre-warms a pool)
#   2. wait for it to be healthy
#   3. run a short GRPO training on WebShop through `python -m fedagent.main_ppo_fed`
#      (trainer env), small batch so concurrency <= pool size
#   4. tear the service down; print the checkpoint tree
# Run on the GPU node:  srun --jobid=<JID> --overlap bash fedagent/scripts/run_webshop_smoke.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"        # .../fedagent/fedagent/scripts
PKG_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"            # .../fedagent/fedagent
REPO_ROOT="$(cd "$PKG_DIR/.." && pwd)"             # .../fedagent
PORT="${WEBSHOP_PORT:-8080}"
# pool must be >= gen batch (concurrency): train_batch_size(4) * rollout.n(2) = 8
export WEBSHOP_POOL_SIZE="${WEBSHOP_POOL_SIZE:-8}"

for __c in "$CONDA_PREFIX_1" "$HOME/miniconda3" "$HOME/anaconda3" /opt/conda /software/miniconda3/4.10.3; do [ -f "$__c/etc/profile.d/conda.sh" ] && { . "$__c/etc/profile.d/conda.sh"; break; }; done
conda activate fedagent-verl08                      # trainer env (parent shell: health check + training)

# --- 1. start the WebShop service in its own env (background) ---
SVC_LOG=/tmp/fedagent_webshop_service.log
rm -f "$SVC_LOG"
WEBSHOP_PORT="$PORT" WEBSHOP_POOL_SIZE="$WEBSHOP_POOL_SIZE" \
  bash "$PKG_DIR/envs/webshop/service/run_service.sh" > "$SVC_LOG" 2>&1 &
SVC_PID=$!
cleanup() { kill "$SVC_PID" 2>/dev/null; pkill -f "fedagent.envs.webshop.service.server" 2>/dev/null; }
trap cleanup EXIT

# --- 2. wait for health (warmup of the env pool can take a couple of minutes) ---
echo "[smoke] waiting for WebShop service on :$PORT (pool=$WEBSHOP_POOL_SIZE) ..."
UP=0
for i in $(seq 1 120); do
  if python -c "import urllib.request,sys; urllib.request.urlopen('http://localhost:$PORT/health',timeout=2)" 2>/dev/null; then
    echo "[smoke] service healthy after ~$((i*3))s"; UP=1; break
  fi
  if ! kill -0 "$SVC_PID" 2>/dev/null; then echo "[smoke] SERVICE DIED; log:"; tail -40 "$SVC_LOG"; exit 1; fi
  sleep 3
done
[ "$UP" = 1 ] || { echo "[smoke] service never came up; log:"; tail -40 "$SVC_LOG"; exit 1; }

# --- 3. run training through the package (trainer env) ---
export PYTHONPATH="$REPO_ROOT:$PYTHONPATH"
export VERL_CFG="$(python -c 'import verl,os;print(os.path.join(os.path.dirname(verl.__file__),"trainer","config"))')"
export WEBSHOP_SERVICE_URL="http://localhost:$PORT"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_USE_V1=1 VERL_LOGGING_LEVEL=WARN
export VLLM_USE_DEEP_GEMM=0 VLLM_SKIP_DEEP_GEMM_WARMUP=1
export CUDA_HOME=/hpc/software/cuda/cuda-12.1.0

MODEL=""
for base in /projects/b1222/.cache/huggingface ~/.cache/huggingface; do
  cand=$(ls -d "$base"/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/*/ 2>/dev/null | head -1)
  [ -n "$cand" ] && MODEL="$cand" && break
done
[ -z "$MODEL" ] && { echo "No local Qwen2.5-0.5B-Instruct snapshot"; exit 1; }
MODEL="${MODEL%/}"

CKPT="$REPO_ROOT/outputs/fedagent_webshop_ckpts"
rm -rf "$CKPT"
echo "[smoke] MODEL=$MODEL  WEBSHOP_SERVICE_URL=$WEBSHOP_SERVICE_URL"

python -m fedagent.main_ppo_fed \
  data.train_files="$PKG_DIR/config/envs/webshop.yaml" \
  data.val_files="$PKG_DIR/config/envs/webshop.yaml" \
  data.custom_cls.path="$PKG_DIR/data/agentic_dataset.py" \
  data.train_batch_size=4 \
  data.max_prompt_length=2048 \
  data.max_response_length=1024 \
  actor_rollout_ref.model.path="$MODEL" \
  +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
  actor_rollout_ref.actor.ppo_mini_batch_size=4 \
  actor_rollout_ref.rollout.n=2 \
  actor_rollout_ref.rollout.prompt_length=2048 \
  actor_rollout_ref.rollout.response_length=1024 \
  actor_rollout_ref.rollout.agent.agent_loop_config_path="$PKG_DIR/config/agent.yaml" \
  trainer.default_local_dir="$CKPT" 2>&1

echo "===== ckpt tree ====="
find "$CKPT" -maxdepth 4 -type f \( -name "*.pt" -o -name "*.json" -o -name "*.txt" \) 2>/dev/null | sort | head -40
echo "===== DONE ====="
