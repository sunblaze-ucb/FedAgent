"""Phase 2 de-risk: confirm WebShop runs standalone in the verl-agent-webshop env
(Lucene index loads, reset/step work) BEFORE building the remote service + client.
Mirrors envs.py:WebshopWorker's import incantation.
"""
import os
import sys
import time
import traceback

WS = (
    "/gpfs/projects/b1222/userdata/canyu/kangyu/fedagent/third_party/verl-agent/"
    "agent_system/environments/env_package/webshop/webshop"
)
sys.path.append(WS)


def _unpack_reset(r):
    return (r[0], r[1]) if isinstance(r, tuple) else (r, {})


def main():
    import gym
    from web_agent_site.envs import WebAgentTextEnv  # noqa: F401  (registers the gym id)

    t0 = time.time()
    env = gym.make("WebAgentTextEnv-v0", observation_mode="text", num_products=None, seed=0)
    print(f"[ok] gym.make in {time.time()-t0:.1f}s")

    obs, info = _unpack_reset(env.reset(session=0))
    print("=== RESET obs (first 500 chars) ===")
    print((obs or "")[:500])
    try:
        acts = env.get_available_actions()
        print("=== available_actions ===", acts)
    except Exception as e:
        print("get_available_actions err:", e)

    t1 = time.time()
    obs, reward, done, info = env.step("search[shirt]")
    print(f"=== after search[shirt] in {time.time()-t1:.1f}s (first 500) ===")
    print((obs or "")[:500])
    print("reward", reward, "done", done, "info_keys", sorted((info or {}).keys()))
    print("WEBSHOP_SMOKE_OK")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        print("WEBSHOP_SMOKE_FAIL")
        sys.exit(1)
