"""WebShop TASK-LEVEL heterogeneity (paper task arm): Coverage (xi).

The Coverage variant draws each client's goal count from a Beta distribution and
hands out the goals with controlled cross-client overlap, so the union of all
clients covers the goal pool while individual clients see overlapping but unequal
slices -- the FULL catalog (env unperturbed), mirroring the Preference arm.

`coverage_partition` is copied VERBATIM from verl-agent's `partition_strategy.py`
(the science red line -- exact copy, no paraphrasing/improvements); it calls the
verbatim `default_r` / `generate_client_sizes` / `assign_with_overlap` helpers
(also copied verbatim, in `_beta_sizing.py`). The thin public API
`coverage_for_client(...) -> goal_idxs` mirrors `preference_for_client`: it builds
the WebShop goal list (goal i -> asin i via the catalog-split goal generator),
partitions the train pool (goals[500:]) by coverage, and returns this client's
absolute goal indices (full catalog -- task-only).
"""
from typing import Any, List, Optional

import numpy as np

from fedagent.hetero._beta_sizing import (
    assign_with_overlap,
    default_r,
    generate_client_sizes,
)
from fedagent.hetero.webshop_catalog_split import (
    _generate_goal_asins_for_partition,
    load_webshop_data,
)


def coverage_partition(
    data: List[Any],
    client_id: int,
    client_num: int,
    min_samples_per_client: int,
    start_idx: int = 0,
    overlap_ratio: float = 1.3,
    max_samples_per_client: Optional[int] = None,
    show_progress: bool = False,
    **kwargs
) -> List[Any]:
    """
    Coverage partition strategy: generate per-client sizes via a Beta distribution
    while trying to cover all samples across clients.

    The goals of this strategy are:
    1. Generate each client's sample count from a Beta distribution.
    2. Have the union of all clients' samples cover the whole dataset as far as
       possible.
    3. Guarantee each client receives at least min_samples_per_client samples.
    4. Allow sample overlap between clients.

    Args:
        data: list of items to partition.
        client_id: ID of the current client (0-based).
        client_num: total number of clients.
        min_samples_per_client: minimum number of samples each client must receive.
        start_idx: starting index into `data` (used to skip a validation set, etc.).
        overlap_ratio: overlap coefficient (total assignments / number of samples).
        max_samples_per_client: maximum samples per client; auto-computed if None.
        show_progress: whether to print progress messages (only for client_id=0).
        **kwargs: additional parameters, including dispersion_s (controls the
            spread of the Beta distribution).

    Returns:
        The data slice the current client should receive.
    """
    # Compute the actually usable training data (everything after start_idx).
    total_train_data = data[start_idx:]
    total_size = len(total_train_data)

    if total_size == 0:
        return []

    # Read the dispersion_s parameter from kwargs (replaces the former size_std).
    if 'dispersion_s' not in kwargs:
        # Fall back to size_std for backward compatibility if dispersion_s is absent.
        if 'size_std' in kwargs:
            dispersion_s = kwargs['size_std']
        else:
            raise ValueError("Missing required 'dispersion_s' parameter in coverage_partition kwargs.")
    else:
        dispersion_s = kwargs['dispersion_s']

    # Sizing parameters.
    center = 500  # kept consistent with the visualization functions
    low = min_samples_per_client
    high = 1000

    # Maximum sample count (auto-computation alternatives kept commented out).
    # if max_samples_per_client is None:
    #     high = int(center + 3*dispersion_s)  # auto-compute
    # else:
    #     high = max_samples_per_client  # use the user-specified value


    print(f"Coverage partition using Beta distribution method...")
    print(f"Parameters: center={center:.1f}, dispersion_s={dispersion_s}, low={low}, high={high}")

    # A fixed seed ensures every client computes the same allocation.
    rng = np.random.default_rng(42)

    # Compute the default r value.
    r = default_r(total_size, client_num, low, center, high)
    target_sum = int(round(r * total_size))

    if show_progress and client_id == 0:
        print(f"Default r = {r:.6f}, target_sum = {target_sum}")

    # Generate the per-client sample sizes.
    client_sizes = generate_client_sizes(
        C=client_num,
        low=low,
        center=center,
        high=high,
        dispersion_s=dispersion_s,
        target_sum=target_sum,
        rng=rng
    )

    # Assign samples using the overlap strategy.
    client_sets, k = assign_with_overlap(total_size, client_sizes, r, rng)

    # Indices of the data assigned to the current client.
    client_indices = list(client_sets[client_id])

    # Resolve indices to the actual data items.
    client_data = []
    for idx in client_indices:
        if idx < len(total_train_data):
            client_data.append(total_train_data[idx])


    print(f"Generated client sizes: {client_sizes}")
    print(f"Current client {client_id} got {len(client_data)} samples")
    print(f"Sample replication counts (min, mean, max): {k.min()}, {k.mean():.3f}, {k.max()}")

    return client_data


# --------------------------------------------------------------------------- #
# Thin public API for the verl-0.8 WebShop service (task-level; full catalog).
# --------------------------------------------------------------------------- #
def coverage_for_client(
    client_id: int,
    client_num: int,
    *,
    size_std: float,
    min_goals_per_client: int = 100,
    base_seed: int = 42,  # noqa: ARG001 (verbatim fn hardcodes 42; kept for API symmetry)
    start_idx: int = 500,
    env_goals: Optional[List[Any]] = None,
    data_dir: Optional[str] = None,
) -> List[int]:
    """This client's WebShop goal indices under Coverage(xi) -- full catalog (task-only).

    Beta-sizes + overlap-assigns the train pool (goals[start_idx:]) and returns the selected
    ABSOLUTE goal indices. `size_std` is the Beta dispersion knob (forwarded as dispersion_s).

    `env_goals` (REQUIRED for science): the env's ACTUAL `server.goals` (seed-42 shuffled).
    coverage_partition selects by POSITION within the train pool, so the indices must address
    the env's real goal order (the original partitions server.goals and maps back via
    goals.index()). The env shuffle is reproducible + identical across clients (GPU-verified),
    so this reproduces the original selection. The `data_dir` fallback (reconstructed order) is
    for offline tests ONLY and is NOT order-faithful.
    """
    if env_goals is not None:
        goals = [dict(g, _idx=i) for i, g in enumerate(env_goals)]
    else:
        products, ins = load_webshop_data(data_dir)
        goal_asins = _generate_goal_asins_for_partition(products, ins)
        # tag every goal with its absolute index; partition the train pool only
        goals = [{"_idx": i} for i in range(len(goal_asins))]
    selected = coverage_partition(
        data=goals[start_idx:],
        client_id=client_id,
        client_num=client_num,
        min_samples_per_client=min_goals_per_client,
        dispersion_s=size_std,
    )
    idxs = sorted(g["_idx"] for g in selected)
    print(f"[task coverage] WebShop client {client_id}/{client_num}: |goal_idxs|={len(idxs)} "
          f"(size_std={size_std}, full catalog, src={'env.server.goals' if env_goals is not None else 'reconstructed'})",
          flush=True)
    return idxs
