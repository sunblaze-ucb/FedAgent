#!/bin/bash
# Verify the CORRECTED storm patch on the SYMMETRIC ALFWorld side: real paper ALFWorld PPO
# config at full batch=64 (= 64*8 = 512 episodes) hitting alfworld_pool_size=8 -- a TIGHTER
# storm than WebShop's pool=16, with longer response_length=8192 / max_model_len=16384.
# Same corrected client/server: idempotent /create, /step fail-fast (no retry), raise_for_status,
# retry only on /create+/reset. Must reach FEDERATED LOOP CLOSED with 0 ReadError/TransportError
# surfaced to the trainer.  NOTE: run_fed --port-base only overrides webshop ports, so ALFWorld
# services use their default band 8200/8201 + val 8290 (cleaned below). Different node from the
# WebShop run, so no cross-node port clash.
set +e
REPO=/gpfs/projects/b1222/userdata/canyu/kangyu/fedagent; cd "$REPO" || exit 2
for __c in "$CONDA_PREFIX_1" "$HOME/miniconda3" "$HOME/anaconda3" /opt/conda /software/miniconda3/4.10.3; do [ -f "$__c/etc/profile.d/conda.sh" ] && { . "$__c/etc/profile.d/conda.sh"; break; }; done; conda activate fedagent-verl08
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
G=_scratch/gpu_verify; RUNS=$REPO/$G/runs; mkdir -p "$RUNS"
echo "[PAPERALFPPO start $(date)] host=$(hostname)"
pkill -f uvicorn 2>/dev/null; pkill -f main_ppo_fed 2>/dev/null; pkill -f service.server 2>/dev/null
for p in $(seq 8200 8300); do fuser -k "$p/tcp" 2>/dev/null; done; sleep 6
PAPERCFG=fedagent/config/paper/uniform/Qwen2.5-1.5B-Instruct/main/ppo/fed_alfworld_ppo_total-100_cl-per-rd-2_rd-70_ep-per-cl-3_min-goals-per-cl-100_p-uniform.yaml
timeout 9000 python -m fedagent.fed.run_fed --config "$PAPERCFG" \
  --rounds 1 --clients 2 \
  --model-path /projects/b1222/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306 \
  --output-dir "$RUNS/paperalf_ppo" > "$G/vpaperalf_ppo.log" 2>&1
echo "[PAPERALFPPO exit=$? $(date)]"
grep -iE "FEDERATED LOOP CLOSED|FedAvg critic round 1 OK|client [01] round 1 OK|httpx|ReadError|TransportError|HTTPStatusError|FAILED|Traceback|pool" "$G/vpaperalf_ppo.log" | tail -10
