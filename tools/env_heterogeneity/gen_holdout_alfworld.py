"""Generate the AlfWorld holdout-scenes list for env-level OOD eval.

PROVENANCE PORT (2026-07-28) of the original experiment repo's
``tools/env_heterogeneity/gen_holdout_alfworld.py`` -- the script that produced the
committed ``data/env_heterogeneity/holdout_alfworld_v1.json`` (seed 99999, 2 scenes per
room type = 8 FloorPlans, 264 trials). The scan + selection body is VERBATIM; only the
paths were adapted (``--data`` / $ALFWORLD_DATA instead of a hardcoded repo-relative
walk). ``--check`` regenerates in memory and diffs against the committed file.

NOTE the scan walks the LOCAL ALFWorld dataset with the FedAgent-effective filter
(movable/Sliced excluded, 6 task types, solvable=True): regeneration reproduces the
committed file only against the same-content dataset (guard the dataset itself with
``tools/gen_alfworld_manifest.py --check``).

Strategy (original): for each of the 4 room types (kitchen, living_room, bedroom,
bathroom), pick 2 scenes preferring small/medium trial counts, so the OOD eval set
covers diverse env types without eating much of the train pool. The holdout file is
consumed by the env_disjoint partition (``holdout_scenes``): no client trains on any
trial in these scenes.
"""
import argparse
import json
import os
import random
import sys
from pathlib import Path
from collections import defaultdict, Counter


REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_PATH = REPO_ROOT / "data" / "env_heterogeneity" / "holdout_alfworld_v1.json"
DEFAULT_ALFWORLD_DATA = "/gpfs/projects/b1222/userdata/canyu/.cache/alfworld"
TASK_TYPES_USED = {
    "pick_and_place_simple",
    "look_at_obj_in_light",
    "pick_clean_then_place_in_recep",
    "pick_heat_then_place_in_recep",
    "pick_cool_then_place_in_recep",
    "pick_two_obj_and_place",
}


def room_type(scene_id: str) -> str:
    """ALFRED FloorPlan id -> room type."""
    try:
        n = int(scene_id)
    except ValueError:
        return "unknown"
    if 1 <= n <= 30:
        return "kitchen"
    if 200 <= n <= 230:
        return "living_room"
    if 300 <= n <= 330:
        return "bedroom"
    if 400 <= n <= 430:
        return "bathroom"
    return "other"


def build(train_dir: Path, holdout_seed: int = 99999, per_room_type: int = 2) -> dict:
    if not train_dir.exists():
        raise FileNotFoundError(
            f"ALFWorld train split not found at {train_dir}. "
            "Set $ALFWORLD_DATA or pass --data."
        )

    # Scan: count trials per scene across the FedAgent-effective filter
    # (movable/Sliced excluded, only allowed task_types, solvable=True)
    scene_to_count = Counter()
    scene_to_specs = defaultdict(set)
    for d in os.listdir(train_dir):
        full = train_dir / d
        if not full.is_dir():
            continue
        if "movable" in d or "Sliced" in d:
            continue
        parts = d.rsplit("-", 1)
        if len(parts) != 2:
            continue
        spec, scene = parts
        for tr in os.listdir(full):
            tp = full / tr
            if not (tr.startswith("trial_") and tp.is_dir()):
                continue
            gp = tp / "game.tw-pddl"
            tj = tp / "traj_data.json"
            if not (gp.exists() and tj.exists()):
                continue
            try:
                gd = json.load(open(gp))
                if not gd.get("solvable", False):
                    continue
            except Exception:
                continue
            try:
                td = json.load(open(tj))
                if td.get("task_type") not in TASK_TYPES_USED:
                    continue
            except Exception:
                continue
            scene_to_count[scene] += 1
            scene_to_specs[scene].add(spec)

    # Group by room type and pick (small, medium) per type
    by_rt = defaultdict(list)
    for scene, count in scene_to_count.items():
        by_rt[room_type(scene)].append((scene, count))

    rng = random.Random(holdout_seed)
    holdout = []
    per_rt_chosen = {}
    for rt in sorted(by_rt):
        if rt in ("unknown", "other"):
            continue
        # Sort by trial count: prefer scenes with smaller / medium count to limit OOD set size
        candidates = sorted(by_rt[rt], key=lambda x: x[1])
        rng.shuffle(candidates)  # break ties / add seed-based variance
        chosen = sorted(candidates[: per_room_type], key=lambda x: x[0])
        per_rt_chosen[rt] = [c[0] for c in chosen]
        holdout.extend(c[0] for c in chosen)
    holdout = sorted(set(holdout), key=lambda s: int(s) if s.isdigit() else 0)

    total_holdout_trials = sum(scene_to_count[s] for s in holdout)
    total_holdout_specs = len(set().union(*(scene_to_specs[s] for s in holdout)))

    return {
        "version": "v1",
        "seed": holdout_seed,
        "n_holdout_scenes": len(holdout),
        "n_holdout_trials": total_holdout_trials,
        "n_holdout_specs_touched": total_holdout_specs,
        "scenes": holdout,
        "per_room_type": per_rt_chosen,
        "per_scene_trial_count": {s: scene_to_count[s] for s in holdout},
        "comment": (
            "Reserved FloorPlans for OOD env eval. No client training trial in "
            "these scenes (the env_disjoint partition function applies this filter "
            "before per-spec top-k selection). Pick covers all 4 room types so OOD "
            "eval is balanced across env semantics."
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default=os.environ.get("ALFWORLD_DATA", DEFAULT_ALFWORLD_DATA),
                    help=f"ALFWORLD_DATA root (default: $ALFWORLD_DATA or {DEFAULT_ALFWORLD_DATA})")
    ap.add_argument("--check", action="store_true",
                    help="regenerate in memory and diff against the committed file; "
                         "exit 1 on drift, write nothing")
    args = ap.parse_args()

    train_dir = Path(args.data) / "json_2.1.1" / "train"
    output = build(train_dir)
    if args.check:
        committed = json.load(open(OUT_PATH))
        if committed != output:
            print(f"DRIFT: regenerated holdout != {OUT_PATH}")
            for k in output:
                if committed.get(k) != output[k]:
                    print(f"  field {k!r}: committed={committed.get(k)!r} regen={output[k]!r}")
            return 1
        print(f"OK: regeneration reproduces {OUT_PATH} exactly "
              f"({output['n_holdout_scenes']} scenes, {output['n_holdout_trials']} trials, "
              f"seed {output['seed']})")
        return 0

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    json.dump(output, open(OUT_PATH, "w"), indent=2, sort_keys=False)
    print(f"Wrote {OUT_PATH}")
    print(f"  holdout scenes ({output['n_holdout_scenes']}): {output['scenes']}")
    print(f"  total holdout trials: {output['n_holdout_trials']}")
    print(f"  per room type: {output['per_room_type']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
