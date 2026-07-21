#!/bin/bash
# REMAINING 7 WebShop heterogeneity strategies -> GPFS (catalog_split + preference done earlier).
# Each 0.5B/1-GPU/2cl/1rd. Evidence = per-client service shard must DIFFER between clients.
#   ENV-het: task_disjoint | TASK-het: coverage, hardness | TRANSITION-het: bm25_field_subset,
#   bm25_reweight, lookalike, rank_wrapper
set +e
REPO=/gpfs/projects/b1222/userdata/canyu/kangyu/fedagent
cd "$REPO" || exit 2
for __c in "$CONDA_PREFIX_1" "$HOME/miniconda3" "$HOME/anaconda3" /opt/conda /software/miniconda3/4.10.3; do [ -f "$__c/etc/profile.d/conda.sh" ] && { . "$__c/etc/profile.d/conda.sh"; break; }; done
conda activate fedagent-verl08
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
G=_scratch/gpu_verify
RUNS=$REPO/_scratch/gpu_verify/runs
mkdir -p "$RUNS"
echo "[HET2 ALL start $(date)] host=$(hostname)"

cleanup(){ pkill -f "uvicorn" 2>/dev/null; pkill -f "main_ppo_fed" 2>/dev/null; pkill -f "service.server" 2>/dev/null
  for p in $(seq 9200 9320); do fuser -k "$p/tcp" 2>/dev/null; done; sleep 6; }

run(){ local tag=$1
  cleanup
  echo "===== HET $tag START $(date) ====="
  timeout 2400 python -m fedagent.fed.run_fed --config "$G/het_webshop_$tag.yaml" --output-dir "$RUNS/het_$tag" > "$G/vh_$tag.log" 2>&1
  echo "===== HET $tag exit=$? $(date) ====="
  grep -iE "FEDERATED LOOP CLOSED|response_length/mean|Traceback|Error:|unknown PARTITION" "$G/vh_$tag.log" | tail -3
  echo "--- PER-CLIENT SHARD (client0 vs client1, must differ) ---"
  grep -hiE "healthy on|webshop-service\]" "$RUNS/het_$tag"/webshop_service_client0.log 2>/dev/null | grep -iE "catalog_size|num_goals|variant|goal_idxs|client 0" | head -2
  grep -hiE "healthy on|webshop-service\]" "$RUNS/het_$tag"/webshop_service_client1.log 2>/dev/null | grep -iE "catalog_size|num_goals|variant|goal_idxs|client 1" | head -2
}

for tag in task_disjoint coverage hardness bm25_field_subset bm25_reweight lookalike rank_wrapper; do
  run "$tag"
done
echo "[HET2 ALL DONE $(date)]"
