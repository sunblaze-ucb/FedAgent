#!/bin/bash
# qgpu3022 (freed after ALFHETFIX): extend het+PPO to a 2nd het axis (coverage = task-het).
# Confirms critic federation (FedAvg+merge of critic FSDP shards) holds under task-het, not just
# catalog_split (env-het). Evidence: per-client coverage shards differ + FedAvg/merge critic OK + LOOP CLOSED.
set +e
REPO=/gpfs/projects/b1222/userdata/canyu/kangyu/fedagent; cd "$REPO" || exit 2
for __c in "$CONDA_PREFIX_1" "$HOME/miniconda3" "$HOME/anaconda3" /opt/conda /software/miniconda3/4.10.3; do [ -f "$__c/etc/profile.d/conda.sh" ] && { . "$__c/etc/profile.d/conda.sh"; break; }; done; conda activate fedagent-verl08
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
G=_scratch/gpu_verify; RUNS=$REPO/_scratch/gpu_verify/runs; mkdir -p "$RUNS"
echo "[HETPPO2 start $(date)] host=$(hostname)"
cleanup(){ pkill -f uvicorn 2>/dev/null; pkill -f main_ppo_fed 2>/dev/null; pkill -f service.server 2>/dev/null
  for p in $(seq 9640 9680); do fuser -k "$p/tcp" 2>/dev/null; done; sleep 6; }
cleanup
echo "===== hetppo2_coverage START $(date) ====="
timeout 3600 python -m fedagent.fed.run_fed --config "$G/het_webshop_coverage_ppo.yaml" \
  --output-dir "$RUNS/hetppo2_coverage" > "$G/vhp2_coverage.log" 2>&1
echo "===== hetppo2_coverage exit=$? $(date) ====="
grep -iE "FEDERATED LOOP CLOSED|FedAvg critic round|merge critic round|federating the critic|Traceback|Error:" "$G/vhp2_coverage.log" | tail -6
echo "--- per-client coverage shard (client0 vs client1, must differ) ---"
for c in 0 1; do grep -hiE "num_goals|goal_idxs|catalog_size|coverage|Client $c" "$RUNS/hetppo2_coverage"/webshop_service_client$c.log 2>/dev/null | head -2; done
echo "[HETPPO2 DONE $(date)]"
