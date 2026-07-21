#!/bin/bash
# GPU verification of the review fixes + the deferred ALFWorld budget. Runs on the allocated
# node via srun --overlap. Logs land in _scratch/gpu_verify/ (gitignored).
set +e
REPO=/gpfs/projects/b1222/userdata/canyu/kangyu/fedagent
cd "$REPO" || exit 2
for __c in "$CONDA_PREFIX_1" "$HOME/miniconda3" "$HOME/anaconda3" /opt/conda /software/miniconda3/4.10.3; do [ -f "$__c/etc/profile.d/conda.sh" ] && { . "$__c/etc/profile.d/conda.sh"; break; }; done
conda activate fedagent-verl08
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
G=_scratch/gpu_verify
echo "[start $(date)] host=$(hostname)"

echo "===== UNIT #3: dataset fail-fast (bad path raises; empty -> TinyGuess) ====="
python - <<'PY' 2>&1 | tail -4
from fedagent.data.agentic_dataset import AgenticDataset
try:
    AgenticDataset._load_specs("/nonexistent/x.yaml"); print("FAIL#3: no raise on bad path")
except FileNotFoundError: print("OK#3: bad path -> FileNotFoundError")
s = AgenticDataset._load_specs("")
print("OK#3: empty -> TinyGuess" if s and s[0]["name"] == "TinyGuess" else "FAIL#3: empty path")
PY

echo "===== UNIT #5: FedProx fail-closed happy path (verl present -> patch applies) ====="
FEDPROX_MU=0.1 PYTHONPATH=$REPO python -c "pass" 2>&1 | grep -iE "fedprox|error|refus" || echo "WARN#5: no [fedprox] log seen"

echo "===== SMOKE A: WebShop lazy per-round services + loop + dataset + FedProx ====="
timeout 2400 python -m fedagent.fed.run_fed --config $G/smoke_webshop_lazy.yaml > $G/smoke_webshop_lazy.log 2>&1
echo "[webshop exit=$? $(date)]"

echo "===== SMOKE B: ALFWorld max_turns=50 + max_model_len=16384 OOM check ====="
timeout 3000 python -m fedagent.fed.run_fed --config $G/smoke_alfworld_oom.yaml > $G/smoke_alfworld_oom.log 2>&1
echo "[alfworld exit=$? $(date)]"
echo "[done $(date)]"
