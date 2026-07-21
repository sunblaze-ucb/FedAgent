#!/bin/bash
set +e; REPO=/gpfs/projects/b1222/userdata/canyu/kangyu/fedagent; cd "$REPO" || exit 2
for __c in "$CONDA_PREFIX_1" "$HOME/miniconda3" "$HOME/anaconda3" /opt/conda /software/miniconda3/4.10.3; do [ -f "$__c/etc/profile.d/conda.sh" ] && { . "$__c/etc/profile.d/conda.sh"; break; }; done; conda activate fedagent-verl08
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
G=_scratch/gpu_verify; RUNS=$REPO/_scratch/gpu_verify/runs
pkill -f uvicorn 2>/dev/null; pkill -f main_ppo_fed 2>/dev/null; for p in $(seq 9600 9620); do fuser -k $p/tcp 2>/dev/null; done; sleep 6
echo "===== hetppo_catalog START $(date) ====="
timeout 3000 python -m fedagent.fed.run_fed --config $G/het_webshop_catalog_ppo.yaml --n-gpus 1 --output-dir $RUNS/hetppo_catalog > $G/vhp_catalog.log 2>&1
echo "===== hetppo_catalog exit=$? $(date) ====="
grep -iE "federating the critic|FedAvg (actor|critic).*OK|merge (actor|critic).*OK|FEDERATED LOOP CLOSED|catalog_size|Traceback|Error:" $G/vhp_catalog.log | tail -8
