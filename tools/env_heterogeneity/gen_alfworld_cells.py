#!/usr/bin/env python
"""Generate + validate the ALFWorld (query x env) cell library.

The cell library is the substrate for the tau-strictly-zero env-het control arm
(docs/dev_doc/alfworld_query_env_decoupling.md): a *query* is one of the 550
train specs (its type-quantified ``(:goal)`` block + canonical task wording); a
*cell* is that query transplanted into a compatible host game (a specific TRIAL
-- inventories are trial-level, not scene-level). All clients can then train on
the byte-identical query set while running in disjoint environments.

Pipeline (revised per the 2026-08-23 review):

  1. INDEX  -- per manifest game, parse the path spec and extract from the
     ``pddl_problem`` init: objectType instance counts, receptacle type set,
     and the (instance-local!) canContain pair set.
  2. SCREEN -- static compatibility per (query, host TRIAL):
       (1) >= n instances of object_target (pick_two: n=2, else 1);
       (2) parent receptacle type present (look_at: DeskLamp as object);
       (3) mrecep type present when the spec has one;
       (4) clean/heat/cool need SinkBasin/Microwave/Fridge;
       (5) (canContain <recep>Type <obj>Type) present (place-type queries) --
           canContain is instance-local (34-130 pairs/problem, kitchen-vs-
           bedroom Jaccard 0.01), NOT a global type table.
     Per (query, scene) the host trial is resolved deterministically: max
     target-object count, then lexicographic path.
  3. VALIDATE -- synthetic cells only: build the cell game (host everything;
     goal block from the query donor; task rhs REPLACED by the query's
     canonical wording -- fail-loud on anchor mismatch), then fast-downward
     plan + step-through in a forked child under a PROCESS-level deadline
     (signal-based timeouts cannot interrupt fast-downward). Reject
     won@reset / no-plan / exec-failure / timeout.
  4. WRITE  -- ``alfworld_cells_v1.json``: meta (counts, rejection stats,
     content sha256) + one row per usable cell
     ``{query, task_type, scene, host, plan_len, natural}``. Natural cells
     (the query's own games) are included with plan_len = len(walkthrough).

Run inside ``verl-agent-alfworld`` (textworld + fast-downward + $ALFWORLD_DATA):

    conda run -n verl-agent-alfworld python \
        tools/env_heterogeneity/gen_alfworld_cells.py \
        --workers 8 --solve-timeout 45 \
        --out data/env_heterogeneity/alfworld_cells_v1.json

``--check`` re-derives the static screen + rebuilds cell games for a sample of
stored cells and verifies the stored content hash, gen_holdout style.
"""
import argparse
import collections
import hashlib
import json
import os
import re
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, os.path.join(_REPO, "fedagent", "envs", "alfworld", "engine"))
sys.path.insert(0, _HERE)

from agent_system.environments.alfworld_kernel_variants import (  # noqa: E402
    _extract_goal,
    canonicalize_task_text,
)

TASK_TYPES = (
    "pick_and_place_simple",
    "look_at_obj_in_light",
    "pick_clean_then_place_in_recep",
    "pick_heat_then_place_in_recep",
    "pick_cool_then_place_in_recep",
    "pick_two_obj_and_place",
)
_NEEDS_STATION = {
    "pick_clean_then_place_in_recep": "SinkBasinType",
    "pick_heat_then_place_in_recep": "MicrowaveType",
    "pick_cool_then_place_in_recep": "FridgeType",
}
_OBJTYPE_RE = re.compile(r"\(objectType (\w+?)_bar_[^\s\)]* (\w+)\)")
_RECEPTYPE_RE = re.compile(r"\(receptacleType (\w+?)_bar_[^\s\)]* (\w+)\)")
_CANCONTAIN_RE = re.compile(r"\(canContain (\w+) (\w+)\)")
_TASK_RHS_RE = re.compile(r'("rhs": "Your task is to: )([^"]*?)(")')


def parse_spec(task_dir):
    """<task_type>-<obj>-<mrecep>-<recep>-<scene> -> fields."""
    body, scene = task_dir.rsplit("-", 1)
    task_type = next(t for t in TASK_TYPES if body.startswith(t + "-"))
    obj, mrecep, recep = body[len(task_type) + 1:].split("-")
    return {"spec": body, "task_type": task_type, "obj": obj,
            "mrecep": mrecep, "recep": recep, "scene": scene}


def index_game(problem):
    """Static inventory of one host trial's init section."""
    init = problem[: problem.find("(:goal")]
    obj_counts = collections.Counter(t for _, t in _OBJTYPE_RE.findall(init))
    recep_types = {t for _, t in _RECEPTYPE_RE.findall(init)}
    can_contain = set(_CANCONTAIN_RE.findall(init))
    return {"obj": obj_counts, "recep": recep_types, "cc": can_contain}


def query_requirements(q):
    n_obj = 2 if q["task_type"] == "pick_two_obj_and_place" else 1
    return {
        "obj_type": q["obj"] + "Type",
        "n_obj": n_obj,
        "recep_type": (None if q["task_type"] == "look_at_obj_in_light"
                       else q["recep"] + "Type"),
        "lamp_type": (q["recep"] + "Type"
                      if q["task_type"] == "look_at_obj_in_light" else None),
        "mrecep_type": (q["mrecep"] + "Type" if q["mrecep"] != "None" else None),
        "station": _NEEDS_STATION.get(q["task_type"]),
    }


def host_compatible(req, inv):
    if inv["obj"].get(req["obj_type"], 0) < req["n_obj"]:
        return False
    if req["recep_type"]:
        if req["recep_type"] not in inv["recep"]:
            return False
        # canContain is instance-local: require the exact (recep, obj) pair
        if (req["recep_type"], req["obj_type"]) not in inv["cc"]:
            return False
    if req["lamp_type"] and inv["obj"].get(req["lamp_type"], 0) < 1:
        return False
    if req["mrecep_type"] and (req["mrecep_type"] not in inv["recep"]
                               and inv["obj"].get(req["mrecep_type"], 0) < 1):
        return False
    if req["station"] and req["station"] not in inv["recep"]:
        return False
    return True


def build_cell_game(host_data, donor_goal, canon_desc):
    """Host everything; query goal block; query canonical task wording."""
    problem = host_data["pddl_problem"]
    i, j = _extract_goal(problem)
    out = dict(host_data)
    out["pddl_problem"] = problem[:i] + donor_goal + problem[j:]
    grammar = host_data["grammar"]
    if len(_TASK_RHS_RE.findall(grammar)) != 1:
        raise ValueError("host grammar does not carry exactly one task rhs")
    out["grammar"] = _TASK_RHS_RE.sub(
        lambda m: m.group(1) + canon_desc + m.group(3), grammar, count=1)
    out["walkthrough"] = []      # host walkthrough is invalid for the new goal
    out["solvable"] = False      # set by validation
    return out


# --------------------------------------------------------------------------- #
# validation workers (fork + process-level deadline; parent single-threaded)  #
# --------------------------------------------------------------------------- #

def _cell_child(job, data_root, conn):
    """Build + solve one synthetic cell entirely inside the child."""
    try:
        from verify_alfworld_kernel_variants import _fresh_env, _raw_plan, _execute_raw

        with open(os.path.join(data_root, job["host"])) as f:
            host = json.load(f)
        game = build_cell_game(host, job["goal"], job["desc"])
        env = _fresh_env(game)
        state = env.reset()
        if bool(state.get("won")):
            conn.send({"status": "won_at_reset"})
            return
        plan = _raw_plan(env)
        if not plan:
            conn.send({"status": "no_plan"})
            return
        won, taken, stuck = _execute_raw(env, state, plan)
        if not won:
            conn.send({"status": "exec_failed", "stuck": str(stuck)})
            return
        conn.send({"status": "usable", "plan_len": len(plan), "plan": taken})
    except Exception as e:  # noqa: BLE001
        conn.send({"status": "error", "error": repr(e)})
    finally:
        conn.close()


def validate_parallel(jobs, data_root, workers, timeout_s, progress_every=200):
    import multiprocessing as mp
    from multiprocessing.connection import wait as conn_wait

    ctx = mp.get_context("fork")
    results = [None] * len(jobs)
    todo = collections.deque(enumerate(jobs))
    active = {}   # conn -> (idx, proc, deadline)
    done = 0
    t0 = time.time()
    while todo or active:
        while todo and len(active) < workers:
            idx, job = todo.popleft()
            parent, child = ctx.Pipe(duplex=False)
            p = ctx.Process(target=_cell_child, args=(job, data_root, child))
            p.start()
            child.close()
            active[parent] = (idx, p, time.time() + timeout_s)
        ready = conn_wait(list(active), timeout=0.5)
        now = time.time()
        finished = []
        for conn in list(active):
            idx, p, deadline = active[conn]
            if conn in ready:
                try:
                    results[idx] = conn.recv()
                except EOFError:
                    results[idx] = {"status": "child_died"}
                finished.append(conn)
            elif now > deadline:
                p.terminate()
                results[idx] = {"status": "timeout"}
                finished.append(conn)
        for conn in finished:
            idx, p, _ = active.pop(conn)
            p.join(5)
            if p.is_alive():
                p.kill()
                p.join(5)
            conn.close()
            done += 1
            if done % progress_every == 0:
                rate = done / max(time.time() - t0, 1e-9)
                print(f"  [validate] {done}/{len(jobs)} "
                      f"({rate:.1f}/s, eta {int((len(jobs)-done)/max(rate,1e-9))}s)",
                      flush=True)
    return results


# --------------------------------------------------------------------------- #

def derive(manifest_path, data_root, limit=0, per_query_cap=0):
    """INDEX + SCREEN: queries, natural cells, synthetic candidate jobs."""
    with open(manifest_path) as f:
        games = json.load(f)["games"]

    print(f"[index] {len(games)} games ...", flush=True)
    trials = {}          # rel path -> {inv, spec fields}
    by_spec = collections.defaultdict(list)
    for rel in games:
        fields = parse_spec(rel.split("/")[0])
        with open(os.path.join(data_root, rel)) as f:
            data = json.load(f)
        trials[rel] = {
            "fields": fields,
            "inv": index_game(data["pddl_problem"]),
            "walk_len": len(data.get("walkthrough") or []),
        }
        by_spec[fields["spec"]].append(rel)

    print(f"[queries] {len(by_spec)} specs", flush=True)
    queries = {}
    for spec, rels in sorted(by_spec.items()):
        donor_rel = sorted(rels)[0]
        with open(os.path.join(data_root, donor_rel)) as f:
            donor = json.load(f)
        i, j = _extract_goal(donor["pddl_problem"])
        fields = trials[donor_rel]["fields"]
        canon_grammar = canonicalize_task_text(donor["grammar"], fields["task_type"])
        desc = _TASK_RHS_RE.search(canon_grammar).group(2)
        queries[spec] = {"fields": fields, "goal": donor["pddl_problem"][i:j],
                         "desc": desc, "donor": donor_rel}

    natural_cells = []
    for spec, rels in sorted(by_spec.items()):
        fields = queries[spec]["fields"]
        by_scene = collections.defaultdict(list)
        for rel in rels:
            by_scene[trials[rel]["fields"]["scene"]].append(rel)
        for scene, srels in sorted(by_scene.items()):
            host = sorted(srels)[0]
            natural_cells.append({
                "query": spec, "task_type": fields["task_type"], "scene": scene,
                "host": host, "plan_len": trials[host]["walk_len"],
                "natural": True,
            })

    print("[screen] static compatibility over (query, trial) ...", flush=True)
    native_scenes = {spec: {trials[r]["fields"]["scene"] for r in rels}
                     for spec, rels in by_spec.items()}
    jobs = []
    for spec, q in sorted(queries.items()):
        req = query_requirements(q["fields"])
        best = {}   # scene -> (score, host_rel)
        for rel, t in trials.items():
            scene = t["fields"]["scene"]
            if scene in native_scenes[spec]:
                continue                      # natural coverage handles these
            if not host_compatible(req, t["inv"]):
                continue
            cnt = t["inv"]["obj"].get(req["obj_type"], 0)
            cur = best.get(scene)
            # max target-object count; lexicographic-first path on ties
            if cur is None or cnt > cur[0] or (cnt == cur[0] and rel < cur[1]):
                best[scene] = (cnt, rel)
        chosen = sorted(best.items())
        if per_query_cap:
            chosen = chosen[:per_query_cap]
        for scene, (_, host_rel) in chosen:
            jobs.append({"query": spec, "task_type": q["fields"]["task_type"],
                         "scene": scene, "host": host_rel,
                         "goal": q["goal"], "desc": q["desc"]})
    if limit:
        jobs = jobs[:limit]
    print(f"[screen] natural cells: {len(natural_cells)}; "
          f"synthetic candidates: {len(jobs)}", flush=True)
    return queries, natural_cells, jobs


def content_sha(payload):
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=os.path.join(
        _REPO, "data", "alfworld_games", "train.json"))
    ap.add_argument("--data-root", default=os.path.join(
        os.environ.get("ALFWORLD_DATA", os.path.expanduser("~/.cache/alfworld")),
        "json_2.1.1", "train"))
    ap.add_argument("--out", default=os.path.join(
        _REPO, "data", "env_heterogeneity", "alfworld_cells_v1.json"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--solve-timeout", type=int, default=45)
    ap.add_argument("--limit", type=int, default=0,
                    help="cap synthetic candidates (testing)")
    ap.add_argument("--per-query-cap", type=int, default=0,
                    help="cap synthetic scenes per query (testing)")
    ap.add_argument("--check", action="store_true",
                    help="re-derive the screen and verify the stored library hash")
    args = ap.parse_args()

    queries, natural_cells, jobs = derive(
        args.manifest, args.data_root, args.limit, args.per_query_cap)

    if args.check:
        with open(args.out) as f:
            stored = json.load(f)
        same_jobs = stored["meta"]["candidate_sha"] == content_sha(
            [{k: j[k] for k in ("query", "scene", "host")} for j in jobs])
        same_cells = stored["meta"]["cells_sha"] == content_sha(stored["cells"])
        print(f"[check] candidate set match: {same_jobs}; "
              f"stored cells hash match: {same_cells}")
        sys.exit(0 if (same_jobs and same_cells) else 1)

    # pre-import textworld once so every forked child inherits the modules
    import textworld  # noqa: F401
    import verify_alfworld_kernel_variants  # noqa: F401

    t0 = time.time()
    results = validate_parallel(jobs, args.data_root, args.workers,
                                args.solve_timeout)
    stats = collections.Counter(r["status"] for r in results)
    print(f"[validate] {dict(stats)} in {int(time.time()-t0)}s", flush=True)

    cells = list(natural_cells)
    for job, res in zip(jobs, results):
        if res["status"] != "usable":
            continue
        cells.append({"query": job["query"], "task_type": job["task_type"],
                      "scene": job["scene"], "host": job["host"],
                      "plan_len": res["plan_len"], "natural": False})
    cells.sort(key=lambda c: (c["query"], c["scene"], c["host"]))

    per_query = collections.Counter(c["query"] for c in cells)
    depth = sorted(per_query.values())
    plan_syn = [c["plan_len"] for c in cells if not c["natural"]]
    meta = {
        "schema": 1,
        "source_manifest_sha": json.load(open(args.manifest))["sha256"],
        "n_queries": len(queries),
        "n_cells": len(cells),
        "n_natural": len(natural_cells),
        "n_synthetic": len(cells) - len(natural_cells),
        "n_candidates": len(jobs),
        "validation_stats": dict(stats),
        "solve_timeout_s": args.solve_timeout,
        "depth_min_med_max": [depth[0], depth[len(depth) // 2], depth[-1]],
        "plan_len_synthetic_mean": (round(sum(plan_syn) / len(plan_syn), 2)
                                    if plan_syn else None),
        "cell_rule": "host everything; goal block from query donor; task rhs "
                     "replaced by the query's canonical wording; walkthrough/"
                     "solvable of the host are NOT valid for the cell",
        "candidate_sha": content_sha(
            [{k: j[k] for k in ("query", "scene", "host")} for j in jobs]),
    }
    meta["cells_sha"] = content_sha(cells)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"meta": meta, "cells": cells}, f, indent=1)
    print(f"[write] {args.out}: {meta['n_cells']} cells "
          f"({meta['n_synthetic']} synthetic / {meta['n_natural']} natural), "
          f"depth min/med/max {meta['depth_min_med_max']}, "
          f"synthetic plan_len mean {meta['plan_len_synthetic_mean']}")


if __name__ == "__main__":
    main()
