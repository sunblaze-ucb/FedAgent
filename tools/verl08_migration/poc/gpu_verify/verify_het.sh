#!/bin/bash
# Detached 1-GPU HETEROGENEITY verification (qgpu3022) -- the paper's core mechanism, untested by
# the homogeneous smokes. Confirms two clients get DIFFERENT shards AND train on them:
#   1) catalog_split (ENV-het: disjoint goals + disjoint catalog == hidden P_i)
#   2) preference    (TASK-het: content-dependent goal partition, omega skew)
# Evidence = per-client "[webshop-service] <strategy> client i/N" lines must DIFFER between clients.
set +e
REPO=/gpfs/projects/b1222/userdata/canyu/kangyu/fedagent
cd "$REPO" || exit 2
for __c in "$CONDA_PREFIX_1" "$HOME/miniconda3" "$HOME/anaconda3" /opt/conda /software/miniconda3/4.10.3; do [ -f "$__c/etc/profile.d/conda.sh" ] && { . "$__c/etc/profile.d/conda.sh"; break; }; done
conda activate fedagent-verl08
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
G=_scratch/gpu_verify
echo "[HET ALL start $(date)] host=$(hostname)"
nvidia-smi --query-gpu=index,memory.total --format=csv,noheader | head

run(){ # tag  config  outdir
  local tag=$1 cfg=$2 outdir=$3
  pkill -f "envs.webshop.service.server" 2>/dev/null; sleep 4
  echo "===== $tag START $(date) ====="
  timeout 2400 python -m fedagent.fed.run_fed --config "$cfg" > "$G/vh_$tag.log" 2>&1
  echo "===== $tag exit=$? $(date) ====="
  echo "--- training milestones ---"
  grep -iE "FEDERATED LOOP CLOSED|model_world_size|FedAvg.*OK|merge.*OK|response_length/mean|Traceback|Error" "$G/vh_$tag.log" | tail -8
  echo "--- PER-CLIENT PARTITION EVIDENCE (must differ) ---"
  grep -h "webshop-service\]" "$outdir"/webshop_service_client0.log 2>/dev/null | grep -iE "client 0/|goal|catalog|slice|idx" | head -4
  echo "   ---- vs ----"
  grep -h "webshop-service\]" "$outdir"/webshop_service_client1.log 2>/dev/null | grep -iE "client 1/|goal|catalog|slice|idx" | head -4
}

run catalog    $G/het_webshop_catalog.yaml    /tmp/xbb9020_het_catalog
run preference $G/het_webshop_preference.yaml  /tmp/xbb9020_het_preference
echo "[HET ALL DONE $(date)]"
