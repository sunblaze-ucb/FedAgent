#!/bin/bash
# Launch the WebShop remote env service in the verl-agent-webshop conda env.
# Env vars: WEBSHOP_PORT (default 8080), WEBSHOP_POOL_SIZE (default 4).
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
conda activate "${WEBSHOP_CONDA_ENV:-verl-agent-webshop}"

HERE="$(cd "$(dirname "$0")" && pwd)"              # .../fedagent/fedagent/envs/webshop/service
REPO_ROOT="$(cd "$HERE/../../../.." && pwd)"        # .../fedagent (repo root)
export PYTHONPATH="$REPO_ROOT:$PYTHONPATH"          # so `import fedagent.envs.webshop.service.server` resolves
PORT="${WEBSHOP_PORT:-8080}"
export WEBSHOP_POOL_SIZE="${WEBSHOP_POOL_SIZE:-4}"

echo "[webshop-service] python=$(which python) port=$PORT pool=$WEBSHOP_POOL_SIZE root=$REPO_ROOT"
exec uvicorn fedagent.envs.webshop.service.server:app --host 0.0.0.0 --port "$PORT" --log-level warning
