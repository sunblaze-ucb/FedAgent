#!/bin/bash
# qgpu3022: WebShop env-het DOSE-RESPONSE control + re-run the flaked preference -> GPFS.
# div0.0 (clients ~same catalog) vs div0.7 (762 vs 750) vs div1.0 (max) proves env_div controls het.
set +e
REPO=/gpfs/projects/b1222/userdata/canyu/kangyu/fedagent; cd "$REPO" || exit 2
for __c in "$CONDA_PREFIX_1" "$HOME/miniconda3" "$HOME/anaconda3" /opt/conda /software/miniconda3/4.10.3; do [ -f "$__c/etc/profile.d/conda.sh" ] && { . "$__c/etc/profile.d/conda.sh"; break; }; done; conda activate fedagent-verl08
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
G=_scratch/gpu_verify; RUNS=$REPO/_scratch/gpu_verify/runs; mkdir -p "$RUNS"
echo "[DOSE start $(date)] host=$(hostname)"
cleanup(){ pkill -f uvicorn 2>/dev/null; pkill -f main_ppo_fed 2>/dev/null; pkill -f service.server 2>/dev/null
  for p in $(seq 9100 9120) $(seq 9400 9440); do fuser -k "$p/tcp" 2>/dev/null; done; sleep 6; }
run(){ local tag=$1 cfg=$2; cleanup; echo "===== $tag START $(date) ====="
  timeout 2400 python -m fedagent.fed.run_fed --config "$G/$cfg" --output-dir "$RUNS/$tag" > "$G/vd_$tag.log" 2>&1
  echo "===== $tag exit=$? $(date) ====="
  grep -iE "FEDERATED LOOP CLOSED|catalog_size|goal_idxs|Traceback|Error:" "$G/vd_$tag.log" | tail -3
  for c in 0 1; do grep -hiE "catalog_size|goal_idxs" "$RUNS/$tag"/webshop_service_client$c.log 2>/dev/null | head -1; done; }
run cat_div0 het_webshop_catalog_div0.yaml
run cat_div1 het_webshop_catalog_div1.yaml
run pref_rerun het_webshop_preference.yaml
echo "[DOSE DONE $(date)]"
