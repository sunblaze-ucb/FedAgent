#!/bin/bash
# Verify federated-critic reload: 2-round WebShop PPO (gae). Round 2 must train FROM
# round_1/aggregated/critic_hf (and actor hf). The one path the 1-round storm runs don't cover.
set +e
REPO=/gpfs/projects/b1222/userdata/canyu/kangyu/fedagent; cd "$REPO" || exit 2
for __c in "$CONDA_PREFIX_1" "$HOME/miniconda3" "$HOME/anaconda3" /opt/conda /software/miniconda3/4.10.3; do [ -f "$__c/etc/profile.d/conda.sh" ] && { . "$__c/etc/profile.d/conda.sh"; break; }; done; conda activate fedagent-verl08
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
G=_scratch/gpu_verify; RUNS=$REPO/$G/runs; mkdir -p "$RUNS"
echo "[PPO2RD start $(date)] host=$(hostname)"
pkill -f uvicorn 2>/dev/null; pkill -f main_ppo_fed 2>/dev/null; pkill -f service.server 2>/dev/null
for p in $(seq 9780 9820); do fuser -k "$p/tcp" 2>/dev/null; done; sleep 6
timeout 5400 python -m fedagent.fed.run_fed --config "$G/ppo2round_critic.yaml" \
  --port-base 9780 \
  --model-path /projects/b1222/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306 \
  --output-dir "$RUNS/ppo2round_critic" > "$G/vppo2round_critic.log" 2>&1
echo "[PPO2RD exit=$? $(date)]"
echo "=== round-2 client cmd: does it load round_1 aggregated actor + critic? ==="
grep -oE "actor_rollout_ref.model.path=[^ ]*round_1/aggregated[^ ]*|critic.model.path=[^ ]*round_1/aggregated[^ ]*" "$G/vppo2round_critic.log" | sort -u
echo "=== terminal ==="
grep -iE "FEDERATED LOOP CLOSED|client [01] round 2 OK|FedAvg critic round 2 OK|Traceback|Error loading|size mismatch|KeyError|RuntimeError|FAILED" "$G/vppo2round_critic.log" | tail -10
