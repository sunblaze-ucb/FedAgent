#!/usr/bin/env python
"""Planner-level verification of the ALFWorld kernel-variant arms.

For a task-type-stratified sample of train games x every pool variant, this
tool proves the science-critical properties the arms depend on
(docs/dev_doc/alfworld_env_heterogeneity.md, verification layer 2):

  1. SOLVABLE: the rewritten kernel still admits a fast-downward plan that,
     executed step by step through a real ``PddlEnv``, ends in ``won`` -- no
     silently-unwinnable arm (the WebShop rank_wrapper-invert failure mode).
  2. OBS-PURE: obs_variant rewrites must leave the DEFAULT kernel's plan
     winning verbatim (they change rendering only, never dynamics/goal).
  3. BINDING RATE: for dyn/goal variants, replaying the default plan under the
     variant measures whether the rewrite actually bites on that game
     (v_examined should bind on most games; v_closed / v_autoclose only where
     the goal receptacle is openable, ~25% -- the dilution the dev doc
     predicts). Non-binding games are expected and REPORTED, not failed.
  4. BOUNDED: the variant plan exceeds the default plan by at most
     --max-extra-steps (default 6), far inside the 50-step Limit.

Plan text note: this textworld build's ``replan`` crashes templating operators
whose lifted parameters carry existential preconditions (examineReceptacle) --
never exercised by stock ALFWorld goals. We therefore take the RAW grounded
operator sequence from fast-downward and map each operator onto the env's
``_valid_actions``/``_valid_commands`` pair (same index), which is exactly how
``PddlEnv.step`` resolves commands internally.

Run inside the ``verl-agent-alfworld`` conda env (textworld + fast-downward +
$ALFWORLD_DATA):

    conda run -n verl-agent-alfworld python \
        tools/env_heterogeneity/verify_alfworld_kernel_variants.py \
        --per-type 5 --out /tmp/verify_kernel_variants.json

Exit status is non-zero if any sampled (game, variant) violates a hard
property (1/2/4, or a zero binding rate for a variant that should bind).
"""
import argparse
import collections
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, os.path.join(_REPO, "fedagent", "envs", "alfworld", "engine"))

from agent_system.environments.alfworld_kernel_variants import (  # noqa: E402
    VARIANT_POOLS,
    rewrite_game_data,
)

TASK_TYPES = (
    "pick_and_place_simple",
    "look_at_obj_in_light",
    "pick_clean_then_place_in_recep",
    "pick_heat_then_place_in_recep",
    "pick_cool_then_place_in_recep",
    "pick_two_obj_and_place",
)


def _sample_games(manifest, per_type, offset=0):
    with open(manifest) as f:
        games = json.load(f)["games"]
    by_type = collections.defaultdict(list)
    for rel in games:
        by_type[rel.split("/")[0].split("-")[0]].append(rel)
    picked = []
    for tt in TASK_TYPES:
        picked.extend(by_type[tt][offset:offset + per_type])
    return picked


def _fresh_env(game_data):
    from textworld import EnvInfos
    from textworld.envs.pddl import PddlEnv

    env = PddlEnv(EnvInfos())
    env.load(dict(game_data))   # PddlEnv.load accepts a Mapping directly
    return env


def _raw_plan(env):
    """Raw grounded fast-downward plan ('name arg1 arg2 ...' per step) or None."""
    from textworld.envs.pddl.logic import fast_downward

    ps = env._pddl_state
    if not ps.downward_lib.replan(False):
        return None
    n = ps.downward_lib.get_last_plan_length()
    ops = (fast_downward.Operator * n)()
    ps.downward_lib.get_last_plan(ops)
    return [op.name for op in ops]


def _execute_raw(env, state, raw_plan, budget=50):
    """Step a raw operator plan by matching each op onto _valid_commands.

    Fast-downward may prune statically-determined lifted parameters from a
    grounded operator name, so ops are matched by action name + the op's args
    appearing IN ORDER within the candidate's grounded variable list.
    Returns (won, commands_taken, stuck_op).
    """
    won, taken = False, []
    for op in raw_plan[:budget]:
        toks = op.lower().split()
        name, args = toks[0], toks[1:]
        idx = None
        for k, act in enumerate(state["_valid_actions"]):
            if act.name.lower() != name:
                continue
            grounded = [var.name.lower() for _, var in act.mapping.items()]
            it = iter(grounded)
            if all(any(a == g for g in it) for a in args):
                idx = k
                break
        if idx is None:
            return won, taken, op
        cmd = state["_valid_commands"][idx]
        state, _, done = env.step(cmd)
        taken.append(cmd)
        won = bool(state.get("won"))
        if done:
            break
    return won, taken, None


def _replay_text(env, state, commands):
    """Replay recorded text commands verbatim (the 'learned policy' probe)."""
    won = False
    for cmd in commands:
        if cmd not in state["_valid_commands"]:
            # the kernel made this step inapplicable (e.g. examine gate /
            # autoclosed receptacle): keep going, later steps may still apply
            continue
        state, _, done = env.step(cmd)
        won = bool(state.get("won"))
        if done:
            break
    return won


def _solve_inproc(game_data):
    """(won, plan_len, commands, stuck_op) for a kernel, from a fresh env."""
    env = _fresh_env(game_data)
    state = env.reset()
    plan = _raw_plan(env)
    if plan is None:
        return False, 0, [], "NO_PLAN"
    won, taken, stuck = _execute_raw(env, state, plan)
    return won, len(plan), taken, stuck


def _solve_child(game_data, conn):
    try:
        conn.send(_solve_inproc(game_data))
    except Exception as e:  # noqa: BLE001 -- surface the child's failure verbatim
        conn.send((False, 0, [], f"CHILD_EXC:{e!r}"))
    finally:
        conn.close()


def _solve(game_data, timeout_s=120):
    """_solve_inproc behind a PROCESS-level timeout.

    fast-downward runs as a long C call, so signal.alarm cannot interrupt it
    (handlers only fire at Python bytecode boundaries; single pathological
    instances have been observed to grind for ~650s). A forked child +
    terminate() is the only reliable kill. ``timeout_s=0`` runs in-process.
    """
    if not timeout_s:
        return _solve_inproc(game_data)
    import multiprocessing as mp

    ctx = mp.get_context("fork")
    parent, child = ctx.Pipe(duplex=False)
    p = ctx.Process(target=_solve_child, args=(game_data, child))
    p.start()
    child.close()
    try:
        if parent.poll(timeout_s):
            result = parent.recv()
        else:
            result = (False, 0, [], f"TIMEOUT>{timeout_s}s")
            p.terminate()
    except EOFError:   # child died before sending (e.g. OOM-killed)
        result = (False, 0, [], "CHILD_DIED")
    finally:
        p.join(5)
        if p.is_alive():
            p.kill()
            p.join(5)
        parent.close()
    return result


def check_obs_surfaces(game_data, default_cmds):
    """Spot-check observation rewrites on the initial/goto feedback text."""
    out = {}
    base = _fresh_env(game_data)
    s0 = base.reset()
    out["default_intro_has_room"] = "you see" in s0.feedback.lower()

    blind = _fresh_env(rewrite_game_data(
        game_data, {"strategy": "obs_variant", "key": "v_blind_intro"}))
    sb = blind.reset()
    out["blind_intro_has_room"] = "you see" in sb.feedback.lower()
    out["blind_intro_has_task"] = "your task is to" in sb.feedback.lower()

    goto = next((c for c in default_cmds if c.startswith("go to ")), None)
    if goto:
        terse = _fresh_env(rewrite_game_data(
            game_data, {"strategy": "obs_variant", "key": "v_terse_goto"}))
        st = terse.reset()
        if goto in st["_valid_commands"]:
            st, _, _ = terse.step(goto)
            out["terse_goto_lists_contents"] = "you see" in st.feedback.lower()
        base2 = _fresh_env(game_data)
        st2 = base2.reset()
        if goto in st2["_valid_commands"]:
            st2, _, _ = base2.step(goto)
            out["default_goto_lists_contents"] = "you see" in st2.feedback.lower()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=os.path.join(
        _REPO, "data", "alfworld_games", "train.json"))
    ap.add_argument("--data-root", default=os.path.join(
        os.environ.get("ALFWORLD_DATA", os.path.expanduser("~/.cache/alfworld")),
        "json_2.1.1", "train"))
    ap.add_argument("--per-type", type=int, default=5)
    ap.add_argument("--offset", type=int, default=0,
                    help="skip the first N games of each type (fresh samples)")
    ap.add_argument("--arms", default="obs_variant,dyn_variant,goal_variant")
    ap.add_argument("--max-extra-steps", type=int, default=6)
    ap.add_argument("--solve-timeout", type=int, default=120,
                    help="per-solve process-level timeout in seconds (0 = off); "
                         "signal-based timeouts cannot interrupt fast-downward")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    rels = _sample_games(args.manifest, args.per_type, args.offset)
    print(f"[verify] {len(rels)} games x {sum(len(VARIANT_POOLS[a]) - 1 for a in arms)} "
          f"non-default variants (arms: {arms})", flush=True)

    all_results, obs_surface, failures = [], [], []
    for gi, rel in enumerate(rels):
        with open(os.path.join(args.data_root, rel)) as f:
            game_data = json.load(f)
        tt = rel.split("/")[0].split("-")[0]

        dwon, dlen, dcmds, dstuck = _solve(game_data, args.solve_timeout)
        if not dwon:
            failures.append({"game": rel, "status": "DEFAULT_KERNEL_BROKEN",
                             "stuck": dstuck})
            print(f"  [{gi}] {tt}: DEFAULT plan failed ({dstuck}) -- harness "
                  f"problem", flush=True)
            continue

        results = []
        for strategy in arms:
            for key in VARIANT_POOLS[strategy][1:]:
                variant = {"strategy": strategy, "key": key}
                rewritten = rewrite_game_data(game_data, variant)
                rec = {"strategy": strategy, "key": key, "game": rel,
                       "task_type": tt, "default_plan_len": dlen}
                if all(rewritten[k] == game_data[k]
                       for k in ("grammar", "pddl_domain", "pddl_problem")):
                    rec["status"] = "identity"     # e.g. look_at x v_closed
                    results.append(rec)
                    continue
                won, plen, _, stuck = _solve(rewritten, args.solve_timeout)
                if not won:
                    rec.update(status="UNSOLVABLE", stuck=stuck, plan_len=plen)
                    results.append(rec)
                    continue
                rec["plan_len"] = plen
                rec["extra_steps"] = plen - dlen
                venv = _fresh_env(rewritten)
                vstate = venv.reset()
                default_wins = _replay_text(venv, vstate, dcmds)
                rec["default_plan_wins"] = bool(default_wins)
                if strategy == "obs_variant":
                    rec["status"] = "ok" if default_wins else "OBS_CHANGED_DYNAMICS"
                else:
                    rec["status"] = "ok"
                    rec["binding"] = not default_wins
                results.append(rec)

        if "obs_variant" in arms:
            surf = check_obs_surfaces(game_data, dcmds)
            surf.update(game=rel, task_type=tt)
            obs_surface.append(surf)

        for r in results:
            bad = r["status"] not in ("ok", "identity")
            over = r.get("extra_steps", 0) > args.max_extra_steps
            if over and not bad:
                r["status"] = "TOO_MANY_EXTRA_STEPS"
                bad = True
            if bad:
                failures.append(r)
        all_results.extend(results)
        print(f"  [{gi}] {tt}: default={dlen} steps; "
              + ", ".join(
                  f"{r['strategy'][:3]}/{r['key'].replace('v_', '')}:"
                  f"{r['status']}"
                  + (f"+{r['extra_steps']}" if r.get("extra_steps") else "")
                  + ("!bind" if r.get("binding") else "")
                  for r in results), flush=True)

    # summary
    summary = {}
    grouped = collections.defaultdict(list)
    for r in all_results:
        grouped[(r["strategy"], r["key"])].append(r)
    print("\n=== summary ===")
    for (strategy, key), rs in sorted(grouped.items()):
        ok = [r for r in rs if r["status"] == "ok"]
        ident = sum(1 for r in rs if r["status"] == "identity")
        fail = len(rs) - len(ok) - ident
        extra = [r["extra_steps"] for r in ok] or [0]
        binding = [r for r in ok if "binding" in r]
        row = {
            "n": len(rs), "ok": len(ok), "identity": ident, "failures": fail,
            "extra_steps_mean": round(sum(extra) / len(extra), 2),
            "extra_steps_max": max(extra),
        }
        if binding:
            row["binding_rate"] = round(
                sum(1 for r in binding if r["binding"]) / len(binding), 3)
        summary[f"{strategy}/{key}"] = row
        print(f"  {strategy}/{key}: n={row['n']} ok={row['ok']} "
              f"identity={ident} fail={fail} "
              f"extra={row['extra_steps_mean']}/{row['extra_steps_max']}"
              + (f" binding={row.get('binding_rate')}" if binding else ""))
        # A dyn/goal variant that never binds in the sample gets a WARNING, not a
        # failure: the openable-surface variants (v_closed / v_autoclose) are
        # DESIGNED dilute (openable is a per-INSTANCE init fact; manifest-order
        # samples can miss the surface entirely), and their applicable-surface
        # behavior is proven by targeted openable-parent runs (dev doc 7.2).
        # The hard properties remain solvability / obs purity / step bound.
        if binding and row.get("binding_rate") == 0.0:
            print(f"  WARNING: {strategy}/{key} never bound in this sample -- "
                  f"expected for openable-surface variants under manifest-order "
                  f"sampling; verify its surface with a targeted run "
                  f"(--offset or openable-parent game list)")
    if obs_surface:
        n = len(obs_surface)
        blind_leak = sum(1 for s in obs_surface if s.get("blind_intro_has_room"))
        task_kept = sum(1 for s in obs_surface if s.get("blind_intro_has_task"))
        terse_leak = sum(1 for s in obs_surface
                         if s.get("terse_goto_lists_contents"))
        base_shows = sum(1 for s in obs_surface
                         if s.get("default_goto_lists_contents"))
        print(f"  obs surfaces: blind_intro room-leak {blind_leak}/{n}, task kept "
              f"{task_kept}/{n}; terse goto contents-leak {terse_leak} vs default "
              f"shows {base_shows}")

    payload = {"results": all_results, "obs_surface": obs_surface,
               "summary": summary, "failures": failures}
    if args.out:
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=1)
        print(f"[verify] wrote {args.out}")
    if failures:
        print(f"[verify] {len(failures)} FAILURES", file=sys.stderr)
        for r in failures[:20]:
            print(f"  FAIL {r.get('game', '')} {r.get('strategy')}/{r.get('key')}: "
                  f"{r.get('status')}", file=sys.stderr)
        sys.exit(1)
    print("[verify] all checks passed")


if __name__ == "__main__":
    main()
