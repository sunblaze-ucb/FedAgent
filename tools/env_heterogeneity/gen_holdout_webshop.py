"""Generate the WebShop holdout-distractor list for env-level OOD eval.

PROVENANCE PORT (2026-07-28) of the original experiment repo's
``tools/env_heterogeneity/gen_holdout_webshop.py`` -- the script that produced the
committed ``data/env_heterogeneity/holdout_webshop_v1.json`` (seed 99999, 6 distractor
ASINs per category = 30 total). The selection body is VERBATIM; only the repo paths were
adapted to this overlay's layout. Deterministic: rerunning reproduces the committed file
byte-for-byte (checked in the port review; ``--check`` re-verifies without writing).

The holdout file is consumed by the catalog_split partition (run_fed ``holdout_file`` ->
service HOLDOUT_FILE -> fedagent/hetero/webshop_catalog_split.py): the listed ASINs are
excluded from every client's training catalog and reserved for OOD env eval.
"""
import argparse
import json
import random
import sys
from pathlib import Path
from collections import defaultdict


REPO_ROOT = Path(__file__).resolve().parents[2]
WEBSHOP_DATA = REPO_ROOT / "fedagent" / "envs" / "webshop" / "engine" / "webshop" / "data"
OUT_PATH = REPO_ROOT / "data" / "env_heterogeneity" / "holdout_webshop_v1.json"


def build(holdout_seed: int = 99999, per_category: int = 6) -> dict:
    products = json.load(open(WEBSHOP_DATA / "items_shuffle_1000.json"))
    ins = json.load(open(WEBSHOP_DATA / "items_ins_v2_1000.json"))

    # Targets are ASINs that have a non-empty 'instruction' (will be referenced by some goal)
    target_asins = {asin for asin, entry in ins.items() if entry.get("instruction")}
    distractor_asins = sorted({p["asin"] for p in products} - target_asins)

    # Stratified sample: per_category items per product category
    asin_to_cat = {p["asin"]: p["category"] for p in products}
    by_cat = defaultdict(list)
    for d in distractor_asins:
        by_cat[asin_to_cat[d]].append(d)

    rng = random.Random(holdout_seed)
    holdout = []
    cat_counts = {}
    for cat in sorted(by_cat):
        pool = sorted(by_cat[cat])
        rng.shuffle(pool)
        picked = pool[:per_category]
        holdout.extend(picked)
        cat_counts[cat] = len(picked)

    return {
        "version": "v1",
        "seed": holdout_seed,
        "n_holdout": len(holdout),
        "asins": sorted(holdout),
        "per_category_count": cat_counts,
        "comment": (
            "Reserved distractor ASINs for OOD env eval. "
            "Inject into eval_unseen catalog only; never include in any client training catalog."
        ),
        "stats": {
            "total_products": len(products),
            "target_asins": len(target_asins),
            "all_distractors": len(distractor_asins),
            "partition_distractors": len(distractor_asins) - len(holdout),
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true",
                    help="regenerate in memory and diff against the committed file; "
                         "exit 1 on drift, write nothing")
    args = ap.parse_args()

    output = build()
    if args.check:
        committed = json.load(open(OUT_PATH))
        if committed != output:
            print(f"DRIFT: regenerated holdout != {OUT_PATH}")
            for k in output:
                if committed.get(k) != output[k]:
                    print(f"  field {k!r} differs")
            return 1
        print(f"OK: regeneration reproduces {OUT_PATH} exactly "
              f"({output['n_holdout']} ASINs, seed {output['seed']})")
        return 0

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    json.dump(output, open(OUT_PATH, "w"), indent=2, sort_keys=False)
    print(f"Wrote {OUT_PATH}")
    print(f"  total holdout = {output['n_holdout']} distractor ASINs")
    print(f"  per category  = {output['per_category_count']}")
    print(f"  stats         = {output['stats']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
