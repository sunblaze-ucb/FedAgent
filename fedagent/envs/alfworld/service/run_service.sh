#!/bin/bash
# Launch the ALFWorld remote env service in the verl-agent-alfworld conda env.
# Env vars: ALFWORLD_PORT (default 8081), ALFWORLD_POOL_SIZE (default 4),
#           ALFWORLD_DATA (game data root; default ~/.cache/alfworld),
#           ALF_CONFIG (config_tw.yaml; default verl-agent bundled config),
#           ALFWORLD_TRAIN_EVAL (train|eval_in_distribution|eval_out_of_distribution).
set -e
# Locate conda robustly: honor an explicit override, else derive from `conda`/CONDA_EXE,
# else fall back to common install prefixes (portable across machines).
if [ -n "$CONDA_PROFILE" ] && [ -f "$CONDA_PROFILE" ]; then
    source "$CONDA_PROFILE"
elif [ -n "$CONDA_EXE" ] && [ -f "$(dirname "$(dirname "$CONDA_EXE")")/etc/profile.d/conda.sh" ]; then
    source "$(dirname "$(dirname "$CONDA_EXE")")/etc/profile.d/conda.sh"
elif command -v conda >/dev/null 2>&1; then
    source "$(dirname "$(dirname "$(command -v conda)")")/etc/profile.d/conda.sh"
else
    for c in "$HOME/miniconda3" "$HOME/anaconda3" /opt/conda /software/miniconda3/4.10.3; do
        if [ -f "$c/etc/profile.d/conda.sh" ]; then source "$c/etc/profile.d/conda.sh"; break; fi
    done
fi
conda activate "${ALFWORLD_CONDA_ENV:-verl-agent-alfworld}"

HERE="$(cd "$(dirname "$0")" && pwd)"              # .../fedagent/fedagent/envs/alfworld/service
REPO_ROOT="$(cd "$HERE/../../../.." && pwd)"        # .../fedagent (repo root)
export PYTHONPATH="$REPO_ROOT:$PYTHONPATH"          # so `import fedagent.envs.alfworld.service.server` resolves
# config_tw.yaml's data paths are '$ALFWORLD_DATA/json_2.1.1/...' (expanded via os.path.expandvars
# inside AlfredTWEnv), and logic/detector paths likewise -> ALFWORLD_DATA must be exported.
export ALFWORLD_DATA="${ALFWORLD_DATA:-$HOME/.cache/alfworld}"
PORT="${ALFWORLD_PORT:-8081}"
export ALFWORLD_POOL_SIZE="${ALFWORLD_POOL_SIZE:-4}"

echo "[alfworld-service] python=$(which python) port=$PORT pool=$ALFWORLD_POOL_SIZE root=$REPO_ROOT data=$ALFWORLD_DATA"
exec uvicorn fedagent.envs.alfworld.service.server:app --host 0.0.0.0 --port "$PORT" --log-level warning
