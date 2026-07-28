"""catalog_split: target floor must protect the SERVED goals (post-shuffle order).

SimServer shuffles its goal list inside the seed-42 construction window, so the
generation-order list `_generate_goal_asins_for_partition` mimics is NOT the order
goals are served in. The legacy import-time path (`catalog_split_for_client`)
derived the target-ASIN floor from that pre-shuffle list, protecting the products
of the WRONG ~100 goals -- a client's actual targets could be filtered out of its
catalog, breaking the paper's Variant-1 guarantee "every reward-bearing target for
that client's goals stays reachable". The service now computes the slice at runtime
(uniform arithmetic on the real goal count) and assembles the catalog from
`env.server.goals` via `catalog_from_target_asins`.

These tests pin: (1) the factored assembly math is unchanged, (2) the runtime path
protects served goals where the legacy path demonstrably does not, (3) the catalog
depends only on the target SET (ASIN-keyed u/v), (4) the legacy slice arithmetic is
line-identical to webshop_uniform (what the service now uses), (5) the new universe
check is a real assertion.
"""
import json
import os
import random
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fedagent.hetero.webshop_catalog_split import (
    _assemble_catalog,
    _distractor_disjoint_partition_webshop_v5,
    _generate_goal_asins_for_partition,
    catalog_from_target_asins,
)
from fedagent.hetero.webshop_uniform import uniform_for_client

N_PRODUCTS, START_IDX, CLIENT_NUM, MIN_GOALS = 60, 10, 4, 10
ENV_DIV, KEEP_RATIO = 1.0, 0.5


def _mk_products():
    """Every product instruction-bearing; even asins carry a 2-value option (2 goals)."""
    products, ins = [], {}
    for i in range(N_PRODUCTS):
        asin = f"P{i:03d}"
        opts = {"size": [{"value": "small"}, {"value": "large"}]} if i % 2 == 0 else {}
        products.append({"asin": asin, "customization_options": opts})
        ins[asin] = {"instruction": f"buy product {i}"}
    return products, ins


@pytest.fixture(scope="module")
def data():
    products, ins = _mk_products()
    gen_order = _generate_goal_asins_for_partition(products, ins)
    served = list(gen_order)
    random.Random(7).shuffle(served)          # stands in for SimServer's seed-42 shuffle
    assert gen_order != served                # shuffle must actually reorder
    return products, ins, gen_order, served


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory):
    products, ins = _mk_products()
    d = tmp_path_factory.mktemp("webshop_data")
    (d / "items_shuffle_1000.json").write_text(json.dumps(products))
    (d / "items_ins_v2_1000.json").write_text(json.dumps(ins))
    return str(d)


def _legacy(products, ins, client_id):
    return _distractor_disjoint_partition_webshop_v5(
        products=products, ins=ins, client_id=client_id, client_num=CLIENT_NUM,
        min_goals_per_client=MIN_GOALS, env_div=ENV_DIV, keep_ratio=KEEP_RATIO,
        start_idx=START_IDX)


def test_factored_assembly_matches_legacy(data):
    """_assemble_catalog reproduces the legacy catalog for the legacy's own targets."""
    products, ins, gen_order, _ = data
    all_asins = sorted({p["asin"] for p in products})
    for cid in range(CLIENT_NUM):
        catalog, idxs = _legacy(products, ins, cid)
        legacy_targets = {gen_order[i] for i in idxs}
        again, _, _ = _assemble_catalog(
            client_target_asins=legacy_targets, all_asins_sorted=all_asins,
            client_id=cid, env_div=ENV_DIV, keep_ratio=KEEP_RATIO, holdout=set())
        assert again == catalog


def test_runtime_path_protects_served_goals_legacy_does_not(data, data_dir):
    """The bug and the fix, side by side, on a shuffled goal order."""
    products, ins, gen_order, served = data
    broken_somewhere = False
    for cid in range(CLIENT_NUM):
        legacy_catalog, idxs = _legacy(products, ins, cid)
        served_targets = {served[i] for i in idxs}
        # fix: catalog assembled from the goals the client is actually served
        runtime_catalog = catalog_from_target_asins(
            served_targets, client_id=cid, env_div=ENV_DIV, keep_ratio=KEEP_RATIO,
            data_dir=data_dir)
        assert served_targets <= set(runtime_catalog)
        broken_somewhere |= not served_targets <= set(legacy_catalog)
    # at keep_ratio=0.5 the legacy floor must lose at least one client's true target,
    # otherwise this fixture stopped exercising the failure mode the fix is for
    assert broken_somewhere, "legacy pre-shuffle floor unexpectedly covered all served goals"


def test_catalog_depends_only_on_target_set(data, data_dir):
    """ASIN-keyed u/v: same target set -> same catalog, however it was derived."""
    products, ins, gen_order, served = data
    _, idxs = _legacy(products, ins, 1)
    targets = {served[i] for i in idxs}
    a = catalog_from_target_asins(targets, client_id=1, env_div=ENV_DIV,
                                  keep_ratio=KEEP_RATIO, data_dir=data_dir)
    b = catalog_from_target_asins(sorted(targets), client_id=1, env_div=ENV_DIV,
                                  keep_ratio=KEEP_RATIO, data_dir=data_dir)
    assert a == b
    all_asins = sorted({p["asin"] for p in products})
    c, _, _ = _assemble_catalog(client_target_asins=set(targets), all_asins_sorted=all_asins,
                                client_id=1, env_div=ENV_DIV, keep_ratio=KEEP_RATIO,
                                holdout=set())
    assert c == a


def test_legacy_slice_equals_uniform_for_client(data):
    """The service's runtime slice (uniform_for_client) == the legacy slice arithmetic."""
    products, ins, gen_order, _ = data
    for cid in range(CLIENT_NUM):
        _, idxs = _legacy(products, ins, cid)
        uni = uniform_for_client(cid, CLIENT_NUM, min_goals_per_client=MIN_GOALS,
                                 env_goals=list(range(len(gen_order))), val_size=START_IDX)
        assert idxs == uni


def test_unknown_target_asin_raises(data_dir):
    with pytest.raises(ValueError, match="not in the product universe"):
        catalog_from_target_asins({"NOT_A_REAL_ASIN"}, client_id=0, data_dir=data_dir)
