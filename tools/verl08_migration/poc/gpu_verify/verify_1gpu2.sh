#!/bin/bash
# qgpu3012 phase-2 (after centralized/local/alfworld_fed): ALFWorld TASK-heterogeneity (the 2nd env's
# partitioning, never tested). preference(omega=0.99) + hardness(success_std=256, alfworld labels).
# Evidence = per-client ALFWorld service num_games / game-set must DIFFER between clients.
set +e
REPO=/gpfs/projects/b1222/userdata/canyu/kangyu/fedagent
cd "$REPO" || exit 2
for __c in "$CONDA_PREFIX_1" "$HOME/miniconda3" "$HOME/anaconda3" /opt/conda /software/miniconda3/4.10.3; do [ -f "$__c/etc/profile.d/conda.sh" ] && { . "$__c/etc/profile.d/conda.sh"; break; }; done
conda activate fedagent-verl08
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
G=_scratch/gpu_verify
echo "[1GPU2 ALL start $(date)] host=$(hostname)"

run(){ local tag=$1 cfg=$2 outdir=$3
  pkill -f "envs.alfworld.service.server" 2>/dev/null; sleep 4
  echo "===== $tag START $(date) ====="
  timeout 3600 python -m fedagent.fed.run_fed --config "$cfg" > "$G/v2_$tag.log" 2>&1
  echo "===== $tag exit=$? $(date) ====="
  grep -iE "FEDERATED LOOP CLOSED|model_world_size|ALFWorld service.*healthy|num_games|partition=|Traceback|Error:" "$G/v2_$tag.log" | tail -8
  echo "--- per-client game-shard (client0 vs client1) ---"
  grep -hiE "num_games|partition" "$outdir"/alfworld_service_client0.log 2>/dev/null | head -2
  grep -hiE "num_games|partition" "$outdir"/alfworld_service_client1.log 2>/dev/null | head -2
}

# first RE-RUN the ALFWorld federated full loop (phase-1 failed on the n_envs<batch smoke-spec bug,
# now fixed: alfworld_smoke_spec.yaml n_envs=8) -- this is the real "2nd env federated loop" check.
run alfworld_fed_retry $G/smoke_alfworld_1gpu.yaml      /tmp/xbb9020_alfworld_1gpu
run alfworld_het_pref  $G/het_alfworld_preference.yaml  /tmp/xbb9020_alfworld_het_pref
run alfworld_het_hard  $G/het_alfworld_hardness.yaml    /tmp/xbb9020_alfworld_het_hard
echo "[1GPU2 ALL DONE $(date)]"
