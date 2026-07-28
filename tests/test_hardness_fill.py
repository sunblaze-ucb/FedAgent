"""Hardness F_i fill: per-client INDEPENDENT draws (webshop_hardness + partition_strategy).

The paper's Algorithm HardnessPartition fills each client's shard to L with
``F_i <- SampleWithoutReplacement(U, L-|Y_i|)`` inside a ``for i`` loop -- in any faithful
execution the draws advance one RNG stream, so two clients with equal easy quotas share
E[nu^2/|U|] (~0.5 at the paper scale) hard goals. The shipped body instead let every
client replay the SAME stream state and draw only its own fill, collapsing F_i into a
pure function of the fill SIZE: equal quotas -> byte-identical hard sets (73-97% of
clients across the xi' sweep; the near-uniform baseline, whose quotas concentrate, saw
the fewest distinct hard goals of the whole sweep). The fix replays the full fill loop
from the shared stream in every invocation and keeps only the own draw.

Marginals were never affected -- rho_i = |Y_i|/L is set upstream by the Beta sizing, so
Delta^2_hard / rho_bar / n_bar (D1-D3) are identical before and after; these tests pin
the joint: equal-quota clients now get distinct fills, while determinism and the easy
side stay exactly as they were.
"""
import hashlib
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fedagent.hetero import webshop_hardness
from fedagent.hetero._beta_sizing import assign_with_overlap, default_r, generate_client_sizes

sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..",
    "fedagent", "envs", "alfworld", "engine", "agent_system", "environments")))
# partition_strategy imports matplotlib/seaborn at module level for its plotting helpers,
# which the offline test env does not ship. Stub them ONLY for the duration of this
# import, then restore sys.modules — a lingering stub (with __spec__=None) breaks any
# later test that probes matplotlib availability via importlib.
import types  # noqa: E402
_stubbed = [m for m in ("matplotlib", "matplotlib.pyplot", "seaborn") if m not in sys.modules]
for _m in _stubbed:
    sys.modules[_m] = types.ModuleType(_m)
try:
    import partition_strategy  # noqa: E402  (the ALFWorld-shipped copy of the same body)
finally:
    for _m in _stubbed:
        del sys.modules[_m]

C, L, N_ITEMS, N_EASY = 10, 50, 500, 120


def _task_id(item):
    """The verbatim body's task-id derivation (asin + md5 of sorted goal_options)."""
    options_str = str(sorted(item["goal_options"].items()))
    h = int(hashlib.md5(options_str.encode()).hexdigest(), 16)
    return f"{item['asin']}_{abs(h)}"


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    items = [{"asin": f"A{i:04d}", "goal_options": {"size": str(i)}} for i in range(N_ITEMS)]
    easy_ids = {_task_id(it) for it in items[:N_EASY]}
    traj = {"trajectories": [
        {"task_info": {"task_id": _task_id(it)},
         "traj_info": {"success": _task_id(it) in easy_ids}} for it in items]}
    f = tmp_path_factory.mktemp("hardness") / "trajectories.json"
    f.write_text(json.dumps(traj))
    return items, easy_ids, str(f)


def _shards(mod, corpus, success_std=256):
    items, easy_ids, traj_file = corpus
    # success_std=256 (the paper's near-uniform end): quotas CONCENTRATE, so equal-quota
    # clients -- the collapsed case under the old body -- are guaranteed to exist.
    return [mod.hardness_partition(items, c, C, L, start_idx=0,
                                   trajectories_file=traj_file, success_std=success_std)
            for c in range(C)], easy_ids


@pytest.mark.parametrize("mod", [webshop_hardness, partition_strategy],
                         ids=["webshop", "alfworld-file"])
def test_equal_quota_clients_get_distinct_independent_fills(mod, corpus):
    shards, easy_ids = _shards(mod, corpus)
    fills = [frozenset(it["asin"] for it in s if _task_id(it) not in easy_ids) for s in shards]
    quotas = [sum(1 for it in s if _task_id(it) in easy_ids) for s in shards]

    assert all(len(s) == L for s in shards)                      # |X_i| = L holds
    eq_pairs = [(i, j) for i in range(C) for j in range(i + 1, C) if quotas[i] == quotas[j]]
    assert eq_pairs, "std=256 must yield equal-quota pairs (the collapsed case)"
    for i, j in eq_pairs:
        assert fills[i] != fills[j], f"clients {i},{j}: identical hard fills are the OLD bug"
        overlap = len(fills[i] & fills[j]) / max(len(fills[i]), 1)
        assert overlap < 0.5, f"clients {i},{j}: near-total fill overlap ({overlap:.2f})"
    # every client's fill is its own draw -> all distinct, and the union covers far more
    # of the hard pool than any single fill (the old body's union == one fill)
    assert len(set(fills)) == C
    assert len(frozenset().union(*fills)) > 1.5 * max(len(f) for f in fills)


@pytest.mark.parametrize("mod", [webshop_hardness, partition_strategy],
                         ids=["webshop", "alfworld-file"])
def test_marginals_and_determinism_unchanged(mod, corpus):
    shards, easy_ids = _shards(mod, corpus)
    items, _, traj_file = corpus

    # rho_i: the realized easy counts still come from the SAME Beta sizing + overlap
    # machinery (replicated here with identical arguments and seed)
    rng = np.random.default_rng(42)
    center, low, high = L // 2, 0, L
    r = default_r(N_ITEMS, C, low, center, high)
    counts = generate_client_sizes(C=C, low=low, center=center, high=high,
                                   dispersion_s=256, target_sum=int(round(r * N_ITEMS)), rng=rng)
    easy_sets, _ = assign_with_overlap(N_EASY, counts, int(round(r * N_ITEMS)) / N_EASY, rng)
    for c, s in enumerate(shards):
        assert sum(1 for it in s if _task_id(it) in easy_ids) == len(easy_sets[c])

    # determinism: recomputing any client reproduces its shard exactly (list equality)
    again = mod.hardness_partition(items, 3, C, L, start_idx=0,
                                   trajectories_file=traj_file, success_std=256)
    assert again == shards[3]
