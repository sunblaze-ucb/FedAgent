#!/usr/bin/env python
"""E2 stage 1 -- roll the reference policy over the standard WebShop eval set and record
the (s,a) stream that defines the occupancy measure d^pi for the delta_eff estimate.

Companion to `measure_delta_occ.py` (stage 2), which replays every recorded (s,a) in each
client's environment backend.

Protocol (paper H.7 evaluation, windowed rollout):
  * service  : WEBSHOP_SPLIT=val -- seeds 0..N-1 map to held-out goals[0:N]; N=64 is the paper's
               val_data_size (`config/envs/webshop_15_val.yaml`, n_envs=64).
  * rollout  : windowed mode (FEDAGENT_HISTORY_LENGTH=2) -- the mode every FedAgent checkpoint
               was trained and evaluated in; concat mode is OOD and collapses to zero-shot.
  * decoding : greedy (temperature 0) so the occupancy is exactly reproducible.
  * budget   : max_turns=15.

Two occupancy modes:
  * SIMPLIFIED (default, `--partition-strategy ''`): one rollout on the UNPERTURBED env; the same
    occupancy is used to evaluate every client pair. Matches H.7's common-basis logic.
  * FAITHFUL (`--partition-strategy <arm> --client-id i`, or a `--sweep` file): the same eval
    goals rolled out INSIDE client i's env M_i, so the trace carries d^pi_{M_i} -- the occupancy
    delta_eff(pi; i, j) is literally defined over (Appendix L). The service's val branch draws
    goals[0:VAL_SIZE] regardless of the partition, so only the KERNEL changes; the task
    distribution is held fixed. A `--sweep` file rolls many client configs through ONE vLLM
    process (the engine init dominates a 64-goal pass otherwise).

The prompt is built with the SHIPPED client helpers (`_fmt_actions`, `_extract_task`,
`_format_obs`, `build_webshop_obs`) rather than a local copy, so the rollout the occupancy is
sampled from is the same one the eval harness runs. We talk to the service over raw httpx
instead of through `WebShopEnv` because we need the RAW page text and the SERVER-SIDE PROJECTED
action per step, neither of which `WebShopEnv` surfaces, and stage 2 replays the projected
action verbatim.

Runs in `fedagent-verl08` (vLLM); the service subprocess runs in `verl-agent-webshop`.
`VLLM_USE_FLASHINFER_SAMPLER=0` is required on this box (the flashinfer sampler JIT fails).

Usage:
    python -m tools.env_heterogeneity.rollout_reference_occupancy \
        --model /path/to/uniform/r65_final --out e2_delta_occ/reference_rollout.json
    python -m tools.env_heterogeneity.rollout_reference_occupancy \
        --model ... --sweep e2_delta_occ/faithful_sweep.json --out-dir e2_delta_occ/faithful
"""
import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from uuid import uuid4

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

RUN_SERVICE = REPO_ROOT / "fedagent" / "envs" / "webshop" / "service" / "run_service.sh"


def wait_health(port: int, timeout: float = 2400.0) -> dict:
    url = f"http://localhost:{port}/health"
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                return json.loads(r.read())
        except (urllib.error.URLError, OSError):
            time.sleep(3)
    raise SystemExit(f"webshop service on :{port} never became healthy")


def start_service(cfg, port, pool_size, search_return_n, log_path):
    """Boot one WebShop service. `cfg` selects the client kernel (empty -> unperturbed)."""
    env = os.environ.copy()
    env.update({
        "PYTHONPATH": f"{REPO_ROOT}:{env.get('PYTHONPATH', '')}".rstrip(":"),
        "WEBSHOP_PORT": str(port),
        "WEBSHOP_POOL_SIZE": str(pool_size),
        "WEBSHOP_SEARCH_RETURN_N": str(search_return_n),
        "WEBSHOP_SPLIT": "val",              # held-out goals[0:VAL_SIZE], partition-independent
        "PARTITION_STRATEGY": cfg.get("partition_strategy", "") or "",
        "CLIENT_ID": str(cfg.get("client_id", 0)),
        "CLIENT_NUM": str(cfg.get("client_num", 100) if cfg.get("partition_strategy") else 1),
        "MIN_GOALS_PER_CLIENT": str(cfg.get("min_goals_per_client", 100)),
        "FEDAGENT_LOG_GOAL_ID": "1",         # /reset also returns each goal's task_id
    })
    for key, name in (("variant_n", "VARIANT_N"), ("env_div", "ENV_DIV"),
                      ("keep_ratio", "KEEP_RATIO")):
        if cfg.get(key) is not None:
            env[name] = str(cfg[key])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[e2] starting service :{port} cfg={cfg or 'UNPERTURBED'} (log {log_path})", flush=True)
    proc = subprocess.Popen(["bash", str(RUN_SERVICE)], env=env,
                            stdout=open(log_path, "w"), stderr=subprocess.STDOUT)
    return proc


def stop_service(proc):
    if proc is None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        proc.kill()


def rollout_all(llm, sp, base_url, num_goals, max_turns, helpers):
    import httpx
    (_extract_task, _fmt_actions, _format_obs, build_webshop_obs, webshop_projection) = helpers
    episodes = []
    client = httpx.Client(base_url=base_url, timeout=300.0)
    try:
        for seed in range(num_goals):
            sid = uuid4().hex
            client.post("/create", json={"session_id": sid}).raise_for_status()
            task, goal_id, turns = "", None, []
            try:
                r = client.post("/reset", json={"session_id": sid, "seed": seed})
                r.raise_for_status()
                d = r.json()
                raw = d.get("obs") or ""
                task = _extract_task(raw)
                obs_txt = _format_obs(raw, task)
                memory = []
                prompt = build_webshop_obs(
                    task=task, memory=memory, current_obs=obs_txt,
                    available_str=_fmt_actions(d.get("available_actions", {})),
                    history_length=2, init=True)
                goal_id = d.get("goal_id")
                pre_obs = obs_txt
                for t in range(max_turns):
                    text = llm.chat([[{"role": "user", "content": prompt}]], sp,
                                    use_tqdm=False)[0].outputs[0].text
                    # project locally too, so the trace records exactly what stage 2 replays
                    projected = webshop_projection([text])[0][0]
                    r = client.post("/step", json={"session_id": sid, "text": text, "step_id": t})
                    r.raise_for_status()
                    d = r.json()
                    assert d.get("action") == projected, (
                        f"local projection {projected!r} != server {d.get('action')!r}")
                    raw = d.get("obs") or ""
                    obs_txt = _format_obs(raw, task)
                    turns.append({
                        "turn": t,
                        "prompt": prompt,
                        "model_output": text,
                        "action": projected,       # what env.step() actually received
                        "obs_raw_after": raw,      # stage 2's replay-fidelity check
                        "reward": float(d.get("reward", 0.0)),
                        "task_score": d.get("task_score"),
                        "done": bool(d.get("done", False)),
                        "success": bool(d.get("success", False)),
                        "is_action_valid": bool(d.get("is_action_valid", True)),
                    })
                    memory.append({"text_obs": pre_obs, "action": d.get("action", text)})
                    pre_obs = obs_txt
                    prompt = build_webshop_obs(
                        task=task, memory=memory, current_obs=obs_txt,
                        available_str=_fmt_actions(d.get("available_actions", {})),
                        history_length=2, init=False)
                    if d.get("done"):
                        break
            finally:
                try:
                    client.post("/close", json={"session_id": sid})
                except Exception:
                    pass
            episodes.append({
                "seed": seed,
                "goal_index": seed % 500,   # val split: /reset uses seed % WEBSHOP_VAL_SIZE
                "goal_id": goal_id,
                "task": task,
                "n_turns": len(turns),
                "success": bool(turns[-1]["success"]) if turns else False,
                "task_score": turns[-1]["task_score"] if turns else None,
                "turns": turns,
            })
            print(f"[e2] seed={seed:4d} turns={len(turns):2d} "
                  f"succ={episodes[-1]['success']} score={episodes[-1]['task_score']}", flush=True)
    finally:
        client.close()
    return episodes


def main():
    ap = argparse.ArgumentParser(description="E2 stage 1: reference-policy occupancy rollout")
    ap.add_argument("--model", required=True, help="reference HF checkpoint (uniform GRPO r65_final)")
    ap.add_argument("--out", default=None, help="output JSON trace (single-config mode)")
    ap.add_argument("--out-dir", default=None, help="output directory (--sweep mode)")
    ap.add_argument("--sweep", default=None,
                    help="JSON list of client configs; each entry needs a 'tag' plus any of "
                         "partition_strategy/client_id/client_num/variant_n/env_div/keep_ratio")
    ap.add_argument("--num-goals", type=int, default=64,
                    help="val goals; 64 = the paper val_data_size, 500 = whole held-out pool")
    ap.add_argument("--port", type=int, default=8123)
    ap.add_argument("--pool-size", type=int, default=4)
    ap.add_argument("--search-return-n", type=int, default=200,
                    help="WEBSHOP_SEARCH_RETURN_N; 200 = the env-heterogeneity arms' executed "
                         "protocol and the value their client backends run at")
    ap.add_argument("--max-turns", type=int, default=15)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--gpu-mem-util", type=float, default=0.35)
    ap.add_argument("--partition-strategy", default="")
    ap.add_argument("--client-id", type=int, default=0)
    ap.add_argument("--client-num", type=int, default=100)
    ap.add_argument("--variant-n", type=int, default=None)
    ap.add_argument("--env-div", type=float, default=None)
    ap.add_argument("--keep-ratio", type=float, default=None)
    args = ap.parse_args()

    if args.sweep:
        configs = json.loads(Path(args.sweep).read_text())
        if not args.out_dir:
            raise SystemExit("--sweep requires --out-dir")
        out_dir = Path(args.out_dir)
    else:
        if not args.out:
            raise SystemExit("--out is required without --sweep")
        configs = [{
            "tag": "reference",
            "partition_strategy": args.partition_strategy,
            "client_id": args.client_id,
            "client_num": args.client_num,
            "variant_n": args.variant_n,
            "env_div": args.env_div,
            "keep_ratio": args.keep_ratio,
        }]
        out_dir = Path(args.out).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    from vllm import LLM, SamplingParams
    from fedagent.envs.webshop.webshop_env import _extract_task, _fmt_actions, _format_obs
    from fedagent.envs.legacy_prompts import build_webshop_obs
    sys.path.insert(0, str(REPO_ROOT / "fedagent" / "envs" / "webshop" / "engine"))
    from projection import webshop_projection  # noqa: E402  (stdlib-only module)
    helpers = (_extract_task, _fmt_actions, _format_obs, build_webshop_obs, webshop_projection)

    llm = LLM(model=args.model, dtype="bfloat16", gpu_memory_utilization=args.gpu_mem_util,
              max_model_len=4608, enforce_eager=True, disable_log_stats=True)
    sp = SamplingParams(temperature=args.temperature, max_tokens=512)

    manifest = {}
    for k, cfg in enumerate(configs):
        tag = cfg.get("tag", f"cfg{k}")
        out_path = Path(args.out) if (args.out and not args.sweep) else out_dir / f"{tag}.json"
        if out_path.exists():
            print(f"[e2] {tag}: {out_path} exists, skipping", flush=True)
            manifest[tag] = str(out_path)
            continue
        port = args.port + (k % 40)
        svc = start_service(cfg, port, args.pool_size, args.search_return_n,
                            out_dir / f"service_{tag}.log")
        try:
            health = wait_health(port)
            print(f"[e2] {tag}: service healthy: {health}", flush=True)
            t0 = time.time()
            episodes = rollout_all(llm, sp, f"http://localhost:{port}",
                                   args.num_goals, args.max_turns, helpers)
            meta = {
                "model": args.model,
                "tag": tag,
                "num_goals": args.num_goals,
                "max_turns": args.max_turns,
                "temperature": args.temperature,
                "search_return_n": args.search_return_n,
                "rollout_mode": "windowed(history_length=2)",
                "split": "val",
                "occupancy": "faithful(M_i)" if cfg.get("partition_strategy") else "simplified(standard env)",
                "client_config": cfg,
                "service_health": health,
                "n_state_action_pairs": sum(e["n_turns"] for e in episodes),
                "n_success": sum(1 for e in episodes if e["success"]),
                "wall_seconds": round(time.time() - t0, 1),
            }
            out_path.write_text(json.dumps({"meta": meta, "episodes": episodes},
                                           ensure_ascii=False))
            manifest[tag] = str(out_path)
            print(f"[e2] {tag}: wrote {out_path}  N(s,a)={meta['n_state_action_pairs']}  "
                  f"success={meta['n_success']}/{args.num_goals}", flush=True)
        finally:
            stop_service(svc)

    if args.sweep:
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
        print(f"[e2] wrote {out_dir/'manifest.json'} ({len(manifest)} traces)", flush=True)


if __name__ == "__main__":
    main()
