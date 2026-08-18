#!/usr/bin/env python3
"""verify_train_val_disjoint.py -- confirm WebShop & ALFWorld train and val never overlap.

PORT (2026-07-28) of the original experiment repo's tool of the same name, adapted to this
overlay's split semantics. The original mirrored the partition arithmetic by hand; this port
IMPORTS the real partition function (fedagent/hetero/webshop_uniform.py), so the check can
never drift from the shipped code.

WEBSHOP (analytical, by construction in fedagent/envs/webshop/service/server.py):
    val service (WEBSHOP_SPLIT=val):   goal index = seed % VAL_SIZE          -> [0, VAL_SIZE)
    train service (unsharded):         goal index = VAL_SIZE + seed % pool   -> [VAL_SIZE, N)
    train service (partitioned):       CLIENT_GOAL_IDXS from uniform_for_client(...,
                                       val_size=VAL_SIZE) -- and preference/coverage/hardness/
                                       catalog_split all partition with start_idx=VAL_SIZE.
The unsharded modulo path is disjoint by arithmetic; this script verifies the PARTITIONED
path: for the given (val_size, client_num, min_goals_per_client, total_goals) every client's
served goal indices stay >= val_size, so no client ever trains on a val goal. VAL_SIZE
defaults to 500 (the service's WEBSHOP_VAL_SIZE default; the in-loop val spec scores the
first 64 of those 500).

ALFWORLD (empirical, file system walk):
    train    : $ALFWORLD_DATA/json_2.1.1/train/
    val (id) : $ALFWORLD_DATA/json_2.1.1/valid_seen/
    val (ood): $ALFWORLD_DATA/json_2.1.1/valid_unseen/
Walks all three directories, collects every game.tw-pddl, and verifies trial-id sets are
disjoint between train and valid_{seen,unseen}. With --check-content, also SHA1-hashes file
contents to catch the "same file copied into two dirs" failure mode. (Which games a run
actually LOADS is guarded separately by tools/gen_alfworld_manifest.py --check.)

Exit: 0 = all disjoint, 1 = overlap found, 2 = setup error.
"""

import argparse
import hashlib
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_ALFWORLD_DATA = os.path.expanduser("~/.cache/alfworld")

# -----------------------------------------------------------------------------
# WebShop
# -----------------------------------------------------------------------------

def check_webshop(val_size: int, clients: int, min_per_client: int,
                  total_goals: int) -> bool:
    from fedagent.hetero.webshop_uniform import uniform_for_client  # the REAL shard fn

    print("== WebShop ==")
    print(f"  val_size             = {val_size}  (service WEBSHOP_VAL_SIZE; val = goals[0:{val_size}))")
    print(f"  client_num           = {clients}")
    print(f"  min_goals_per_client = {min_per_client}")
    print(f"  total_goals          = {total_goals}")

    if total_goals <= val_size:
        print(f"  FAIL: total_goals={total_goals} <= val_size={val_size}; train pool empty.")
        return False

    val_idxs = set(range(val_size))
    goals = list(range(total_goals))          # uniform_for_client only uses len(env_goals)
    train_per_client = {}
    for cid in range(clients):
        idxs = set(uniform_for_client(cid, clients, min_per_client, goals, val_size=val_size))
        train_per_client[cid] = idxs
        rng = f"[{min(idxs)}, {max(idxs) + 1})" if idxs else "EMPTY"
        if clients <= 10 or cid < 3 or cid == clients - 1:
            print(f"  client {cid}: train {rng}  size={len(idxs)}")
        elif cid == 3:
            print(f"  ... ({clients - 4} clients elided) ...")

    train_all = set().union(*train_per_client.values()) if train_per_client else set()
    overlap = val_idxs & train_all
    below_val = {i for i in train_all if i < val_size}

    print(f"  |val|                = {len(val_idxs)}")
    print(f"  |train (all clients)|= {len(train_all)}")
    print(f"  |val & train|        = {len(overlap)}")
    print(f"  |train idx < {val_size}|   = {len(below_val)}")

    # Inter-client overlap (informational -- uniform with min_samples allows it).
    cids = sorted(train_per_client)
    inter = sum(1 for i, a in enumerate(cids) for b in cids[i + 1:]
                if train_per_client[a] & train_per_client[b])
    if inter:
        print(f"  inter-client overlapping pairs (expected if min > base slice): {inter}")

    ok = not overlap and not below_val
    if not ok:
        if overlap:
            print(f"  FAIL: val & train = {sorted(overlap)[:10]}...")
        if below_val:
            print(f"  FAIL: train indices below {val_size}: {sorted(below_val)[:10]}...")
    else:
        print(f"  PASS: val & train = empty; all train indices >= {val_size}")
    print()
    return ok


# -----------------------------------------------------------------------------
# ALFWorld
# -----------------------------------------------------------------------------

def collect_games(root: Path):
    if not root.exists():
        return []
    return sorted(root.rglob("game.tw-pddl"))


def sha1_of(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def check_alfworld(data_root: Path, check_content: bool) -> bool:
    print("== ALFWorld ==")
    splits = {
        "train":        data_root / "json_2.1.1" / "train",
        "valid_seen":   data_root / "json_2.1.1" / "valid_seen",
        "valid_unseen": data_root / "json_2.1.1" / "valid_unseen",
    }
    games = {name: collect_games(p) for name, p in splits.items()}
    for name, path in splits.items():
        print(f"  {name:<13s}: {path} -> {len(games[name])} game.tw-pddl")

    if not games["train"]:
        print(f"  FAIL: train dir missing or empty: {splits['train']}")
        return False
    if not games["valid_seen"]:
        print(f"  FAIL: valid_seen missing or empty: {splits['valid_seen']}")
        return False

    # game.tw-pddl layout: .../<task_id>/<trial_id>/game.tw-pddl
    #   task_id  = pick_and_place_simple-Knife-None-SideTable-3
    #   trial_id = trial_T20190918_184236_557252
    def trial_id(p: Path) -> str:
        return p.parent.name

    def task_id(p: Path) -> str:
        return p.parent.parent.name

    trials = {name: {trial_id(p) for p in g} for name, g in games.items()}
    tasks = {name: {task_id(p) for p in g} for name, g in games.items()}

    fail = False
    print()
    print("  -- trial-id disjointness (DEMAND: 0) --")
    for v in ("valid_seen", "valid_unseen"):
        n = len(trials["train"] & trials[v])
        flag = "FAIL" if n else "PASS"
        print(f"    {flag}: |train & {v}| (trial_id) = {n}")
        if n:
            fail = True
            print(f"       sample: {list(trials['train'] & trials[v])[:5]}")

    print()
    print("  -- task-id intersection (BY DESIGN: nonzero for valid_seen) --")
    for v in ("valid_seen", "valid_unseen"):
        n = len(tasks["train"] & tasks[v])
        print(f"    |train & {v}| (task_id)  = {n}   "
              f"(valid_seen: SAME task types, different trials; "
              f"valid_unseen: DIFFERENT layouts)")

    if check_content:
        print()
        print("  -- content disjointness (SHA1 of game.tw-pddl) --")
        train_hashes = {sha1_of(p): str(p) for p in games["train"]}
        for v in ("valid_seen", "valid_unseen"):
            collisions = [(p, train_hashes[h]) for p in games[v]
                          if (h := sha1_of(p)) in train_hashes]
            if collisions:
                fail = True
                print(f"    FAIL: {v}: {len(collisions)} game files share SHA1 with train")
                for v_path, t_path in collisions[:3]:
                    print(f"       {v_path}\n         == {t_path}")
            else:
                print(f"    PASS: {v}: 0 SHA1 collisions with train ({len(games[v])} files compared)")

    print()
    print("  " + ("FAIL: ALFWorld" if fail else
                  "PASS: ALFWorld train disjoint from valid_seen/valid_unseen at trial-id"
                  + (" and content" if check_content else "")))
    print()
    return not fail


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--skip-webshop", action="store_true")
    ap.add_argument("--skip-alfworld", action="store_true")
    ap.add_argument("--webshop-val-size", type=int, default=500,
                    help="service WEBSHOP_VAL_SIZE (default: 500, the shipped default; the "
                         "in-loop val spec scores the first 64 of these)")
    ap.add_argument("--webshop-clients", type=int, default=100,
                    help="total_clients of the run to check (default: 100, the paper's "
                         "WebShop federated pool)")
    ap.add_argument("--webshop-min-per-client", type=int, default=100,
                    help="min_goals_per_client (default: 100 == the paper configs)")
    ap.add_argument("--webshop-total-goals", type=int, default=6910,
                    help="len(goals) served by the engine (default 6910 == the service's "
                         "WEBSHOP_NUM_GOALS default for items_shuffle_1000 + items_ins_v2_1000; "
                         "the service hard-fails at startup if the real pool differs)")
    ap.add_argument("--alfworld-data",
                    default=os.environ.get("ALFWORLD_DATA", DEFAULT_ALFWORLD_DATA),
                    help=f"ALFWORLD_DATA root (default: $ALFWORLD_DATA or {DEFAULT_ALFWORLD_DATA})")
    ap.add_argument("--check-content", action="store_true",
                    help="also SHA1-hash ALFWorld game.tw-pddl files to catch content-identical "
                         "files placed across splits")
    args = ap.parse_args()

    passes = []
    if not args.skip_webshop:
        passes.append(check_webshop(args.webshop_val_size, args.webshop_clients,
                                    args.webshop_min_per_client,
                                    args.webshop_total_goals))
    if not args.skip_alfworld:
        passes.append(check_alfworld(Path(args.alfworld_data), args.check_content))

    if not passes:
        print("Nothing to check (both envs skipped).", file=sys.stderr)
        return 2
    if all(passes):
        print("=== PASS: all disjointness checks passed ===")
        return 0
    print("=== FAIL: at least one disjointness check FAILED ===")
    return 1


if __name__ == "__main__":
    sys.exit(main())
