#!/bin/bash
# 4th matrix corner on final code: ALFWorld GRPO (env_spec=alfworld.yaml n_envs=8, batch=8).
# Confirms the alfworld GRPO config + alfworld service + _post path close the federated loop.
# ALFWorld ports default to 8200/8201 + val 8290 (--port-base only overrides webshop); different
# node from the ALFWorld-PPO run so no clash.
set +e
REPO=/gpfs/projects/b1222/userdata/canyu/kangyu/fedagent; cd "$REPO" || exit 2
for __c in "$CONDA_PREFIX_1" "$HOME/miniconda3" "$HOME/anaconda3" /opt/conda /software/miniconda3/4.10.3; do [ -f "$__c/etc/profile.d/conda.sh" ] && { . "$__c/etc/profile.d/conda.sh"; break; }; done; conda activate fedagent-verl08
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
G=_scratch/gpu_verify; RUNS=$REPO/$G/runs; mkdir -p "$RUNS"
echo "[ALFGRPO start $(date)] host=$(hostname)"
pkill -f uvicorn 2>/dev/null; pkill -f main_ppo_fed 2>/dev/null; pkill -f service.server 2>/dev/null
for p in $(seq 8200 8300); do fuser -k "$p/tcp" 2>/dev/null; done; sleep 6
CFG=fedagent/config/paper/uniform/Qwen2.5-1.5B-Instruct/main/grpo/fed_alfworld_grpo_total-100_cl-per-rd-2_rd-70_ep-per-cl-3_min-goals-per-cl-100_p-uniform.yaml
timeout 6000 python -m fedagent.fed.run_fed --config "$CFG" \
  --rounds 1 --clients 2 \
  --model-path /projects/b1222/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306 \
  --output-dir "$RUNS/alf_grpo_smoke" > "$G/valf_grpo_smoke.log" 2>&1
echo "[ALFGRPO exit=$? $(date)]"
grep -iE "FEDERATED LOOP CLOSED|client [01] round 1 OK|FedAvg.*round 1 OK|Train dataloader is empty|ReadError|TransportError|Traceback|FAILED" "$G/valf_grpo_smoke.log" | grep -vE "atexit|dump_compile" | tail -8
