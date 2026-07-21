#!/bin/bash
# qgpu3013 (4-GPU, was idle): ALFWorld at paper size + bound the task-het blast radius.
#   1) uniform 1.5B 4-GPU federated 2rd  (heavy 2nd-env paper-mode loop)
#   2) coverage     (task-het; name exists in installed alfworld -> should WORK)
#   3) env_disjoint (env-het;  name exists in installed alfworld -> should WORK)
# If 2+3 close but preference/hardness crashed, the blast radius = {preference,hardness} only.
set +e
REPO=/gpfs/projects/b1222/userdata/canyu/kangyu/fedagent
cd "$REPO" || exit 2
for __c in "$CONDA_PREFIX_1" "$HOME/miniconda3" "$HOME/anaconda3" /opt/conda /software/miniconda3/4.10.3; do [ -f "$__c/etc/profile.d/conda.sh" ] && { . "$__c/etc/profile.d/conda.sh"; break; }; done
conda activate fedagent-verl08
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
G=_scratch/gpu_verify
RUNS=$REPO/_scratch/gpu_verify/runs
mkdir -p "$RUNS"
echo "[ALF4 ALL start $(date)] host=$(hostname)"
nvidia-smi --query-gpu=index,memory.total --format=csv,noheader | head

cleanup(){ pkill -f "uvicorn" 2>/dev/null; pkill -f "main_ppo_fed" 2>/dev/null; pkill -f "service.server" 2>/dev/null
  for p in $(seq 8750 8890); do fuser -k "$p/tcp" 2>/dev/null; done; sleep 6; }

run(){ local tag=$1 cfg=$2
  cleanup
  echo "===== $tag START $(date) ====="
  timeout 5000 python -m fedagent.fed.run_fed --config "$cfg" --output-dir "$RUNS/$tag" > "$G/v4_$tag.log" 2>&1
  echo "===== $tag exit=$? $(date) ====="
  grep -iE "FEDERATED LOOP CLOSED|model_world_size_4|ALFWorld service.*healthy|num_games|Invalid partition|Traceback|Error:" "$G/v4_$tag.log" | tail -8
}

run alfworld_uniform_4gpu     $G/smoke_alfworld_4gpu.yaml
run alfworld_coverage_4gpu    $G/het_alfworld_coverage_4gpu.yaml
run alfworld_envdisjoint_4gpu $G/het_alfworld_envdisjoint_4gpu.yaml
echo "[ALF4 ALL DONE $(date)]"
