#!/bin/bash
# Phase 2 de-risk: validate the WebShop service + client round-trip end-to-end
# (real WebShop env behind HTTP), WITHOUT the trainer/GPU. Starts the service in the
# verl-agent-webshop env, then drives create/reset/step/close via the WebShopEnv client.
set -e
REPO_ROOT=/gpfs/projects/b1222/userdata/canyu/kangyu/fedagent
PORT="${WEBSHOP_PORT:-8091}"

source /software/miniconda3/4.10.3/etc/profile.d/conda.sh
conda activate fedagent-verl08   # client side (httpx)

WEBSHOP_PORT="$PORT" WEBSHOP_POOL_SIZE=2 \
  bash "$REPO_ROOT/fedagent/webshop_service/run_service.sh" > /tmp/ws_svc_test.log 2>&1 &
SVC=$!
cleanup() { kill "$SVC" 2>/dev/null; pkill -f "fedagent.webshop_service.server" 2>/dev/null; }
trap cleanup EXIT

echo "[test] waiting for service on :$PORT ..."
UP=0
for i in $(seq 1 90); do
  if python -c "import urllib.request;urllib.request.urlopen('http://localhost:$PORT/health',timeout=2)" 2>/dev/null; then
    echo "[test] healthy after ~$((i*3))s"; UP=1; break
  fi
  kill -0 "$SVC" 2>/dev/null || { echo "[test] SERVICE DIED:"; tail -40 /tmp/ws_svc_test.log; exit 1; }
  sleep 3
done
[ "$UP" = 1 ] || { echo "[test] never came up:"; tail -40 /tmp/ws_svc_test.log; exit 1; }

PYTHONPATH="$REPO_ROOT:$PYTHONPATH" WEBSHOP_SERVICE_URL="http://localhost:$PORT" python - <<'PY'
import asyncio, os
from fedagent.envs.webshop import WebShopEnv

async def main():
    e = WebShopEnv({"service_url": os.environ["WEBSHOP_SERVICE_URL"]})
    sp = await e.system_prompt()
    print("SYS (first 90):", sp["obs_str"][:90])
    obs, _ = await e.reset(seed=0)
    print("=== RESET obs (first 320) ===\n", obs["obs_str"][:320])
    o, r, d, i = await e.step("<think>find a shirt</think><action>search[shirt]</action>")
    print("=== STEP obs (first 320) ===\n", o["obs_str"][:320])
    print("reward", r, "done", d, "info", i)
    await e.close()
    print("WEBSHOP_SERVICE_TEST_OK")

asyncio.run(main())
PY
echo "===== SERVICE TEST DONE ====="
