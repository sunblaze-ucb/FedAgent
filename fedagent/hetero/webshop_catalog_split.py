"""WebShop env-level Catalog-Split heterogeneity (paper Variant 1, Stage 1 content).

`_distractor_disjoint_partition_webshop_v5` + `_generate_goal_asins_for_partition`
carry the verl-agent `partition_strategy.py` math unchanged (the catalog-assembly
steps 4-7 are factored into `_assemble_catalog`, byte-identical selection). The
public API used by the verl-0.8 WebShop remote service is `catalog_from_target_asins`
(+ `load_webshop_data`); the goal slice comes from `webshop_uniform.uniform_for_client`
(the slice arithmetic here and there is line-identical).

SERVED-GOAL ORDER (why the service calls `catalog_from_target_asins` at RUNTIME):
the target-ASIN floor must protect the products of the goals this client will
actually be served. SimServer SHUFFLES its goal list at construction (seed-42
window), so `goal_asins[i]` below (generation order) is NOT the asin of served
goal i. Deriving targets from this list — which `catalog_split_for_client` still
does, and which the service did before 2026-07 — protected the products of the
WRONG ~100 goals; a client's actual targets could be filtered out of its catalog,
silently breaking the paper's "every reward-bearing target stays reachable"
guarantee (main.tex, Variant 1). The service therefore computes the true targets
from `env.server.goals` after the pool is warmed and only then assembles the
catalog. `catalog_split_for_client` is kept as the legacy pre-shuffle reference.

The catalog itself is deterministic by (target set, client_id): shared u @
base_seed, per-client v @ base_seed+1000*client_id, both keyed by ASIN string, so
WHERE the targets came from cannot change the selection math.
"""
from typing import Any, Dict, List, Optional, Tuple
import json
import os

import numpy as np


def _generate_goal_asins_for_partition(
    raw_products: List[Dict[str, Any]],
    ins: Dict[str, Dict[str, Any]],
) -> List[str]:
    """Mimics get_synthetic_goals iteration order; returns asin list (one entry per goal).

    Used by the current Catalog-Split partition
    (_distractor_disjoint_partition_webshop_v5; '_v5' = impl revision, paper
    Variant 1 — NOT paper Variant 5) to compute per-client client_target_asins
    without running the full SimServer goal generation. Iteration order must match
    webshop/web_agent_site/engine/goal.py:get_synthetic_goals so that the i-th
    goal in this list maps to the i-th goal SimServer will see.

    Inline-replicates load_products' customization_options → options transform
    (engine.py:357-374) since raw json products have customization_options but
    not options. The transform is deterministic per ASIN.
    """
    goal_asins: List[str] = []
    for product in raw_products:
        asin = product['asin']
        # Filter: equivalent to load_products' merge `instruction_text = ins[asin]['instruction']`
        # then get_synthetic_goals' `if instruction_text is None: continue`.
        if not ins.get(asin, {}).get('instruction'):
            continue
        # Mimic load_products: customization_options -> options dict (engine.py:357-374)
        # Even if cust is None / empty, options stays as empty dict; goal still produced.
        cust = product.get('customization_options') or {}
        options: Dict[str, list] = {}
        for option_name, option_contents in cust.items():
            if option_contents is None:
                continue
            option_name_low = option_name.lower()
            values = []
            for oc in option_contents:
                v = oc['value'].strip().replace('/', ' | ').lower()
                values.append(v)
            if values:
                options[option_name_low] = values
        # One goal per combination of options (sorted by option_name).
        # Empty options -> itertools.product() returns [()] -> 1 combination -> 1 goal,
        # matching get_synthetic_goals' behavior for option-less products.
        n_combos = 1
        for k in sorted(options):
            n_combos *= len(options[k])
        goal_asins.extend([asin] * n_combos)
    return goal_asins


def _assemble_catalog(
    client_target_asins: set,
    all_asins_sorted: List[str],
    client_id: int,
    env_div: float,
    keep_ratio: float,
    holdout: set,
    base_seed: int = 42,
) -> Tuple[List[str], int, List[str]]:
    """Steps 4-7 of the v5 partition: distractor scoring + catalog assembly.

    Selection math is byte-identical to the original inline body. Deterministic by
    (client_target_asins, client_id) alone: u/v are keyed by ASIN string over the
    full sorted universe, so the same target set yields the same catalog no matter
    how the targets were derived (pre-shuffle mimic or served goals).

    Returns (catalog_asins, D, include_distractors).
    """
    # Step 4: distractor_pool — pool depends on per-client target, so it's per-client
    if holdout & client_target_asins:
        raise ValueError(
            f"[distractor_disjoint v5] client {client_id}: holdout list contains "
            f"{len(holdout & client_target_asins)} of this client's target ASINs"
        )
    unknown = client_target_asins - set(all_asins_sorted)
    if unknown:
        raise ValueError(
            f"[distractor_disjoint v5] client {client_id}: {len(unknown)} target "
            f"ASINs not in the product universe (sample: {sorted(unknown)[:3]}); "
            f"goal source and catalog data are out of sync"
        )
    distractor_pool = sorted((set(all_asins_sorted) - client_target_asins) - holdout)
    D = len(distractor_pool)

    # Step 5: ASIN-level u/v dictionaries (the key change in the current
    # Catalog-Split revision, i.e. what '_v5' refers to — NOT paper Variant 5).
    # WARNING: each client's distractor_pool has different content/length (their
    #   targets differ).
    # → We cannot simply do proto_rng.random(D) and index by distractor_pool
    #   position: the same ASIN occupies different indices in different clients'
    #   pools, which would break the "every client sees a consistent u for the same
    #   ASIN" invariant.
    # → Instead, key by ASIN string: compute one u and one (per-client) v for each
    #   of the full 1000 ASINs, then read them out in this client's distractor_pool
    #   order.
    proto_rng = np.random.RandomState(base_seed)
    asin_to_u = {a: float(proto_rng.random()) for a in all_asins_sorted}

    client_rng = np.random.RandomState(base_seed + 1000 * int(client_id))
    asin_to_v = {a: float(client_rng.random()) for a in all_asins_sorted}

    u = np.array([asin_to_u[a] for a in distractor_pool])
    v = np.array([asin_to_v[a] for a in distractor_pool])

    # Step 6: weighted blend + top-k (identical selection math to the legacy
    # distractor_disjoint / 'v4' revision; 'v4' = revision number, not paper Variant 4).
    e = (1.0 - env_div) * u + env_div * v
    n_keep = int(round(keep_ratio * D))
    chosen_idx = np.argsort(e)[:n_keep]
    include_distractors = [distractor_pool[i] for i in chosen_idx]

    # Step 7: assemble catalog (targets always included by construction)
    catalog_asins = sorted(client_target_asins | set(include_distractors))
    return catalog_asins, D, include_distractors


def _distractor_disjoint_partition_webshop_v5(
    products: List[Dict[str, Any]],
    ins: Dict[str, Dict[str, Any]],
    client_id: int,
    client_num: int,
    min_goals_per_client: int = 100,
    env_div: float = 0.7,
    keep_ratio: float = 0.7,
    holdout_distractor_asins: Optional[List[str]] = None,
    base_seed: int = 42,
    start_idx: int = 500,
    **kwargs,
) -> Tuple[List[str], List[int]]:
    """WebShop env-level partition (CURRENT impl) — per-client target floor distractor disjoint.

    PAPER ANCHOR: this is the implementation of the paper's Environment-Level
    *Variant 1 = Catalog Split* (Stage 1, content/catalog axis of the transition
    pipeline; paper Algorithm 1 'CatalogSplitPartition'). Selected via partition
    strategy key `catalog_split` (env var PARTITION_STRATEGY='catalog_split'),
    dispatched in fed_env_manager.py; knobs env_div, keep_ratio.

    NAMING CAUTION: the trailing '_v5' / the 'v4' and 'v5' tags below are
    *implementation-revision numbers of this Catalog-Split partition function*,
    NOT the paper's Variant 4 (Lookalike Injection) or Variant 5 (Rank Wrapper).
    Both revisions implement the SAME paper Variant 1 (Catalog Split):
      - 'v4' = `_distractor_disjoint_partition_webshop` (strategy key
        `distractor_disjoint`): LEGACY / superseded revision. All clients share
        goals[500:] (~6410 goals); the catalog protects *all* ~415 training
        target ASINs (full-target floor); distractor_pool = 1000 - ~415 = ~585
        shared across clients.
      - 'v5' = this function (strategy key `catalog_split`): CURRENT revision used
        for the reported Catalog-Split numbers. Each client gets a ~100-goal
        slice cut by uniform_partition; the catalog protects only *this client's*
        ~50-80 target ASINs (per-client floor);
        distractor_pool = 1000 - per_client_target ≈ 920 (per-client).
          → The full pairwise-Jaccard range widens from [1.000, 0.746] to
            [0.819, 0.464], strengthening the env_div signal by +40%, while the
            task partition stays strictly consistent with the main experiment
            (uniform 100/client).

    Args:
      products: list of raw product dicts (from items_shuffle_1000.json, before
                load_products merging).
      ins:      asin -> {instruction, attributes, ...} (from items_ins_v2_1000.json).
      client_id, client_num: federated args.
      min_goals_per_client: minimum number of goals per client; used by
                            uniform_partition for the task split.
      env_div, keep_ratio: env-heterogeneity knobs (identical meaning to the
                            legacy distractor_disjoint / 'v4' revision).
      holdout_distractor_asins: optional holdout list of distractor ASINs reserved
                                for OOD eval (not passed by the current
                                Catalog-Split sweeps; the 'v3/v4/v5' tags are
                                partition-impl revision numbers, NOT paper variants).
      base_seed: RNG seed (default 42, same as the legacy 'v4' revision).
      start_idx: start of the training pool (default 500; goals[0:500] are val).

    Returns:
      catalog_asins:   sorted List[str], this client's ASIN set (always contains
                       all client_target_asins).
      client_goal_idxs: List[int], this client's absolute goal indices (including
                        the start_idx offset).
                        e.g. for client 0 with min_goals_per_client=100, returns [500, 501, ..., 599].
                        webshop/envs.py assigns this to self.goal_idxs (replacing
                        the legacy distractor_disjoint code path's hardcoded
                        list(range(500, len(goals))) at envs.py, which all clients
                        shared).
    """
    holdout = set(holdout_distractor_asins or [])

    # Step 1: derive goal_asins list (deterministic, mimics SimServer's goal generation)
    goal_asins = _generate_goal_asins_for_partition(products, ins)
    n_goals = len(goal_asins)
    if n_goals <= start_idx:
        raise RuntimeError(
            f"[distractor_disjoint v5] only {n_goals} goals generated, "
            f"need > start_idx={start_idx}. Check products/ins data."
        )
    train_n = n_goals - start_idx

    # Step 2: uniform_partition-style split of the train pool -> this client's
    # goal slice. Equivalent to
    # partition_strategy.uniform_partition(data=goal_asins, ...) but only computes
    # the indices.
    base_slice_size = train_n // client_num
    start_slice = client_id * base_slice_size
    end_slice = start_slice + base_slice_size
    if (end_slice - start_slice) < min_goals_per_client:
        needed_extra = min_goals_per_client - (end_slice - start_slice)
        left_avail = start_slice
        right_avail = train_n - end_slice
        if left_avail + right_avail >= needed_extra:
            left_extra = min(needed_extra // 2, left_avail)
            right_extra = min(needed_extra - left_extra, right_avail)
            if right_extra < (needed_extra - left_extra):
                left_extra += min(needed_extra - left_extra - right_extra, left_avail - left_extra)
            start_slice -= left_extra
            end_slice += right_extra
        elif left_avail > 0:
            start_slice = 0
            right_needed = needed_extra - left_avail
            end_slice += min(right_needed, right_avail)
        elif right_avail > 0:
            end_slice += min(needed_extra, right_avail)
        else:
            start_slice = 0
            end_slice = train_n
    start_slice = max(0, start_slice)
    end_slice = min(train_n, end_slice)
    client_goal_idxs = list(range(start_idx + start_slice, start_idx + end_slice))

    # Step 3: derive client_target_asins from this slice
    client_target_asins = {goal_asins[i] for i in client_goal_idxs}

    # Steps 4-7: distractor scoring + catalog assembly (factored; math unchanged)
    all_asins_sorted = sorted({p['asin'] for p in products})
    catalog_asins, D, include_distractors = _assemble_catalog(
        client_target_asins=client_target_asins,
        all_asins_sorted=all_asins_sorted,
        client_id=client_id,
        env_div=env_div,
        keep_ratio=keep_ratio,
        holdout=holdout,
        base_seed=base_seed,
    )

    print(f"[ENV v5] WebShop client {client_id}/{client_num}: "
          f"|catalog|={len(catalog_asins)} (target={len(client_target_asins)}, "
          f"distractor={len(include_distractors)}/{D}), "
          f"|goal_idxs|={len(client_goal_idxs)}, "
          f"env_div={env_div}, keep_ratio={keep_ratio}")
    return catalog_asins, client_goal_idxs


# --------------------------------------------------------------------------- #
# Thin public API for the verl-0.8 WebShop remote service (the only additions).
# --------------------------------------------------------------------------- #
_HERE = os.path.dirname(os.path.abspath(__file__))
# Vendored WebShop catalog data: fedagent/envs/webshop/engine/webshop/data.
DEFAULT_DATA_DIR = os.path.abspath(os.path.join(
    _HERE, "..", "envs", "webshop", "engine", "webshop", "data",
))


def load_webshop_data(data_dir: Optional[str] = None):
    """Load (products, ins) from items_shuffle_1000.json / items_ins_v2_1000.json."""
    data_dir = data_dir or os.environ.get("WEBSHOP_DATA_DIR") or DEFAULT_DATA_DIR
    with open(os.path.join(data_dir, "items_shuffle_1000.json")) as f:
        products = json.load(f)
    with open(os.path.join(data_dir, "items_ins_v2_1000.json")) as f:
        ins = json.load(f)
    return products, ins


def catalog_from_target_asins(
    client_target_asins,
    *,
    client_id: int,
    env_div: float = 0.7,
    keep_ratio: float = 0.7,
    holdout_file: Optional[str] = None,
    base_seed: int = 42,
    data_dir: Optional[str] = None,
) -> List[str]:
    """Assemble one client's catalog from its TRUE target ASINs (paper Variant 1).

    This is the runtime entry the WebShop service calls in `_lifespan`, AFTER the env
    pool is warmed: `client_target_asins` must be derived from the served goals
    (`{env.server.goals[i]['asin'] for i in CLIENT_GOAL_IDXS}`), not from the
    pre-shuffle generation order — see the module docstring. Selection math is the
    verbatim v5 steps 4-7 (`_assemble_catalog`); raises if any target is missing
    from the product universe (goal source / catalog data drift).
    """
    products, _ = load_webshop_data(data_dir)
    holdout = None
    if holdout_file:
        with open(holdout_file) as f:
            holdout = json.load(f).get("asins", [])
    all_asins_sorted = sorted({p['asin'] for p in products})
    catalog_asins, D, include_distractors = _assemble_catalog(
        client_target_asins=set(client_target_asins),
        all_asins_sorted=all_asins_sorted,
        client_id=client_id,
        env_div=env_div,
        keep_ratio=keep_ratio,
        holdout=set(holdout or []),
        base_seed=base_seed,
    )
    print(f"[ENV v5] WebShop client {client_id}: runtime catalog "
          f"|catalog|={len(catalog_asins)} (target={len(set(client_target_asins))}, "
          f"distractor={len(include_distractors)}/{D}), "
          f"env_div={env_div}, keep_ratio={keep_ratio}")
    return catalog_asins


def catalog_split_for_client(
    client_id: int,
    client_num: int,
    *,
    env_div: float = 0.7,
    keep_ratio: float = 0.7,
    min_goals_per_client: int = 100,
    holdout_file: Optional[str] = None,
    base_seed: int = 42,
    data_dir: Optional[str] = None,
) -> Tuple[List[str], List[int]]:
    """LEGACY pre-shuffle path: (catalog_asins, client_goal_idxs) for one client.

    Derives the target floor from the GENERATION-order goal list, which does not
    match the shuffled order SimServer serves — kept only as the reference for the
    pre-2026-07 behavior (and for the slice arithmetic, which is order-independent
    and line-identical to `webshop_uniform.uniform_for_client`). The service now
    computes the slice via `uniform_for_client` and the catalog via
    `catalog_from_target_asins` at runtime; do not use this for new code.
    """
    products, ins = load_webshop_data(data_dir)
    holdout = None
    if holdout_file:
        with open(holdout_file) as f:
            holdout = json.load(f).get("asins", [])
    return _distractor_disjoint_partition_webshop_v5(
        products=products,
        ins=ins,
        client_id=client_id,
        client_num=client_num,
        min_goals_per_client=min_goals_per_client,
        env_div=env_div,
        keep_ratio=keep_ratio,
        holdout_distractor_asins=holdout,
        base_seed=base_seed,
    )
