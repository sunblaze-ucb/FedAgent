#!/usr/bin/env python
"""Generate the shipped ALFWorld game manifests — the authoritative per-split game lists.

Why they exist: `AlfredTWEnv.collect_game_files` walks `$ALFWORLD_DATA` and keeps the solvable
trials, but `game.tw-pddl` and its `solvable` flag come from ALFWorld's preprocessing step, so
the walk's output depends on how completely that step ran on a given machine. Every index
downstream (client shards, `games[seed]` under `ALFWORLD_SEED_IS_INDEX`, the val set) shifts
with it. A checked-in manifest makes the task set part of the repo instead. Full rationale:
`fedagent/envs/alfworld/game_manifest.py`.

Usage (bare python is fine — stdlib only, no alfworld/textworld needed):

    # all three splits from one data root, into the shipped location
    python tools/gen_alfworld_manifest.py --data $ALFWORLD_DATA/json_2.1.1

    # one split, elsewhere
    python tools/gen_alfworld_manifest.py --data .../json_2.1.1 --split train --out /tmp/train.json

    # verify the shipped manifests still match this machine's data (CI / after a data refresh)
    python tools/gen_alfworld_manifest.py --data .../json_2.1.1 --check

`--check` never writes: it reports, per split, whether the game list on disk is identical to
the shipped one, and exits non-zero if not. That is the signal to either fix the data or
regenerate deliberately (regenerating CHANGES which games every run trains and evaluates on —
see docs/revision.md).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fedagent.envs.alfworld import game_manifest as gm  # noqa: E402

# split -> the directory under the data root that holds it (ALFWorld's own layout)
SPLIT_DIRS = {
    "train": "train",
    "eval_in_distribution": "valid_seen",
    "eval_out_of_distribution": "valid_unseen",
}


def repo_default_out(split):
    repo = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    return os.path.join(repo, gm.SHIPPED_DIR, f"{split}.json")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", required=True,
                    help="the split-holding root, e.g. $ALFWORLD_DATA/json_2.1.1")
    ap.add_argument("--split", choices=sorted(SPLIT_DIRS), action="append",
                    help="repeatable; default: all three")
    ap.add_argument("--out", default=None, help="single-split output path (default: shipped)")
    ap.add_argument("--check", action="store_true",
                    help="compare this machine's data against the shipped manifests; write nothing")
    args = ap.parse_args()

    splits = args.split or sorted(SPLIT_DIRS)
    if args.out and len(splits) != 1:
        raise SystemExit("--out needs exactly one --split")

    rc = 0
    for split in splits:
        root = os.path.join(args.data, SPLIT_DIRS[split])
        if not os.path.isdir(root):
            print(f"{split:26s} SKIP  (no {root})")
            continue
        m = gm.build(root, split)
        if args.check:
            shipped = gm.default_path(split)
            if not shipped:
                print(f"{split:26s} MISS  no shipped manifest; disk has {m['n']} games")
                rc = 1
                continue
            have = gm.load(shipped)
            if have["sha256"] == m["sha256"]:
                print(f"{split:26s} OK    {m['n']} games, sha256 {m['sha256'][:12]}")
            else:
                only_disk = sorted(set(m["games"]) - set(have["games"]))
                only_ship = sorted(set(have["games"]) - set(m["games"]))
                print(f"{split:26s} DIFF  shipped {have['n']} vs disk {m['n']}; "
                      f"+{len(only_disk)} on disk / -{len(only_ship)} missing")
                for p in only_disk[:3]:
                    print(f"{'':28s}+ {p}")
                for p in only_ship[:3]:
                    print(f"{'':28s}- {p}")
                rc = 1
            continue
        out = args.out or repo_default_out(split)
        gm.write(m, out)
        print(f"{split:26s} wrote {out}  ({m['n']} games, sha256 {m['sha256'][:12]}, "
              f"source {m['source']})")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
