#!/bin/bash
set +e; REPO=/gpfs/projects/b1222/userdata/canyu/kangyu/fedagent; cd "$REPO" || exit 2
for __c in "$CONDA_PREFIX_1" "$HOME/miniconda3" "$HOME/anaconda3" /opt/conda /software/miniconda3/4.10.3; do [ -f "$__c/etc/profile.d/conda.sh" ] && { . "$__c/etc/profile.d/conda.sh"; break; }; done; conda activate fedagent-verl08
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
G=_scratch/gpu_verify; RUNS=$REPO/_scratch/gpu_verify/runs
pkill -f uvicorn 2>/dev/null; pkill -f main_ppo_fed 2>/dev/null; for p in $(seq 9500 9520); do fuser -k $p/tcp 2>/dev/null; done; sleep 6
echo "===== pref_p9500 START $(date) ====="
timeout 2400 python -m fedagent.fed.run_fed --config $G/het_webshop_preference_p9500.yaml --output-dir $RUNS/pref_p9500 > $G/vpp_pref9500.log 2>&1
echo "===== pref_p9500 exit=$? $(date) ====="
grep -iE "FEDERATED LOOP CLOSED|goal_idxs|Address already|DIED|Traceback" $G/vpp_pref9500.log | tail -4
