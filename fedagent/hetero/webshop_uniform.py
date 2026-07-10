"""WebShop UNIFORM per-client goal shard (the paper's homogeneous task partition).

The original FedAgent gave every federated WebShop client a CONTIGUOUS slice of the
seed-42-shuffled train pool (goals[500:]), extended to at least `min_goals_per_client`
goals — on the uniform/decentralized families AND on the transition-level env variants
2–5 (bm25/lookalike/rank), whose task axis stays uniform (verl-agent envs.py:262-276 and
:520-535). "Uniform" therefore means *uniformly sharded*, not *unsharded*: with N=100
clients over the 6410-goal train pool each client trains on its own ~100-goal shard.

`_uniform_partition` is copied VERBATIM from verl-agent's partition_strategy.py
(uniform_partition; the [DEBUG] prints dropped, math untouched). The thin public API
`uniform_for_client(...) -> goal_idxs` mirrors the other fedagent.hetero ports: it
returns this client's absolute indices into `env_goals` (offset by the val holdout,
start_idx=500), exactly the `[500 + i for i in range(start_slice, end_slice)]` the
original computed.
"""
from typing import Any, List


def _uniform_partition(
    data: List[Any],
    client_id: int,
    client_num: int,
    min_samples_per_client: int,
    start_idx: int = 0,
) -> tuple:
    """uniform_partition — VERBATIM math from partition_strategy.py (prints removed)."""
    total_train_data = data[start_idx:]
    total_train_size = len(total_train_data)

    if total_train_size == 0:
        return [], 0, 0

    base_slice_size = total_train_size // client_num

    start_slice = client_id * base_slice_size
    end_slice = start_slice + base_slice_size

    current_slice_size = end_slice - start_slice
    if current_slice_size < min_samples_per_client:
        needed_extra = min_samples_per_client - current_slice_size

        left_available = start_slice
        right_available = total_train_size - end_slice

        if left_available + right_available >= needed_extra:
            left_extra = min(needed_extra // 2, left_available)
            right_extra = min(needed_extra - left_extra, right_available)

            if right_extra < (needed_extra - left_extra):
                left_extra += min(needed_extra - left_extra - right_extra, left_available - left_extra)

            start_slice -= left_extra
            end_slice += right_extra
        elif left_available > 0:
            start_slice = 0
            right_needed = needed_extra - left_available
            end_slice += min(right_needed, right_available)
        elif right_available > 0:
            end_slice += min(needed_extra, right_available)
        else:
            start_slice = 0
            end_slice = total_train_size

    start_slice = max(0, start_slice)
    end_slice = min(total_train_size, end_slice)

    client_data_slice = total_train_data[start_slice:end_slice]
    return client_data_slice, start_slice, end_slice


def uniform_for_client(
    client_id: int,
    client_num: int,
    min_goals_per_client: int,
    env_goals: List[Any],
    val_size: int = 500,
) -> List[int]:
    """This client's absolute goal indices under the original uniform shard.

    Mirrors verl-agent envs.py: partition_dataset(strategy='uniform', start_idx=500)
    over the env's real (seed-42 shuffled) goals, then offset back by the val holdout.
    client_num=1 (centralized / standalone service) degrades to the full train pool,
    matching the original centralized baseline.
    """
    _, start_slice, end_slice = _uniform_partition(
        data=env_goals,
        client_id=client_id,
        client_num=client_num,
        min_samples_per_client=min_goals_per_client,
        start_idx=val_size,
    )
    return [val_size + i for i in range(start_slice, end_slice)]
