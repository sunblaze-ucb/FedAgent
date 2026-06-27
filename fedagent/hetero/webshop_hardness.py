"""WebShop TASK-LEVEL heterogeneity (paper task arm): Hardness (xi').

The Hardness variant skews each client toward easy-vs-hard goals: per-task success
labels (from a precomputed trajectories file, task_id -> success) bucket the goals
into high/low success, then a Beta distribution sets each client's count of
"success" (easy) goals, with the rest filled randomly -- the FULL catalog (env
unperturbed), mirroring the Preference/Coverage arms.

`hardness_partition` is copied VERBATIM from verl-agent's `partition_strategy.py`
(the science red line -- exact copy, no paraphrasing/improvements); it calls the
verbatim `default_r` / `generate_client_sizes` helpers (also copied verbatim, in
`_beta_sizing.py`). The thin public API `hardness_for_client(...) -> goal_idxs`
mirrors `preference_for_client`: it builds the WebShop goal list (goal i -> asin i
via the catalog-split goal generator, carrying each goal's task_id so the verbatim
body's success lookup matches), partitions the train pool (goals[start_idx:]) by
hardness, and returns this client's absolute goal indices.

A `trajectories_file` (task_id -> success labels) is REQUIRED: there is no usable
default in this package, so `hardness_for_client` raises if it is not supplied (and
the verbatim partition itself raises FileNotFoundError for a missing path).

NOTE on `path_cfg`: the verbatim `hardness_partition` body references
`path_cfg.project_root` ONLY inside its `if trajectories_file is None:` default-path
branch. Because `hardness_for_client` always supplies an explicit `trajectories_file`,
that branch is never taken. The module-level `path_cfg` below exists solely so the
verbatim body (kept character-for-character) resolves at import time; the original
source loads it from `config/paths.yaml`, which is absent in this package.
"""
from typing import Any, List, Optional
import hashlib
import os

import numpy as np
from omegaconf import OmegaConf

from fedagent.hetero._beta_sizing import default_r, generate_client_sizes
from fedagent.hetero.webshop_catalog_split import (
    _generate_goal_asins_for_partition,
    load_webshop_data,
)

# Repo root (this file lives at <root>/fedagent/hetero/webshop_hardness.py).
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
# Stand-in for the source module's `path_cfg = OmegaConf.load("config/paths.yaml")`.
# Only `.project_root` is read, and only on the unused default-path branch. We build
# it directly (config/paths.yaml is absent here) so the verbatim body imports cleanly.
path_cfg = OmegaConf.create({"project_root": _PROJECT_ROOT})


def hardness_partition(
    data: List[Any],
    client_id: int,
    client_num: int,
    min_samples_per_client: int,
    start_idx: int = 0,
    trajectories_file: str = None,
    show_progress: bool = False,
    **kwargs
) -> List[Any]:
    """
    Task-difficulty (hardness) based partition strategy.

    The goals of this strategy are:
    1. Read each task's success rate from a trajectories file.
    2. Allocate the training set according to those success rates.
    3. Make each client's number of "success" samples follow a normal distribution
       between 0 and min_goals_per_client.
    4. Fill the rest of each client's quota with randomly chosen samples.

    Args:
        data: list of items to partition.
        client_id: ID of the current client (0-based).
        client_num: total number of clients.
        min_samples_per_client: minimum number of samples each client must receive.
        start_idx: starting index into `data` (used to skip a validation set, etc.).
        trajectories_file: path to the trajectories file containing success info.
        success_std: standard deviation of the per-client success-sample count;
            controls the spread of the normal distribution.
        show_progress: whether to print progress messages.
        **kwargs: additional parameters.

    Returns:
        The data slice the current client should receive.
    """
    import json
    import os

    # Compute the actually usable training data (everything after start_idx).
    total_train_data = data[start_idx:]
    total_size = len(total_train_data)

    if total_size == 0:
        return []

    # Default trajectories file path.
    if trajectories_file is None:
        trajectories_file = os.path.join(path_cfg.project_root, "output/inference/all_trajectories.json")

    # Load the trajectories file.
    if not os.path.exists(trajectories_file):
        raise FileNotFoundError(f"Trajectories file not found: {trajectories_file}")

    if show_progress and client_id == 0:
        print(f"Loading trajectories from: {trajectories_file}")

    with open(trajectories_file, 'r') as f:
        trajectories_data = json.load(f)

    # Build a task_id -> success map (one success flag per task_id).
    task_success_map = {}
    for traj in trajectories_data.get('trajectories', []):
        task_info = traj.get('task_info', {})
        traj_info = traj.get('traj_info', {})

        task_id = task_info.get('task_id')
        success = traj_info.get('success', False)

        if task_id is not None:
            # Store the success flag directly; if a task_id has several
            # trajectories, the last one seen wins.
            task_success_map[task_id] = success


    print(f"Loaded {len(task_success_map)} tasks with success information")
    success_count = sum(task_success_map.values())
    print(f"Success distribution: {success_count}/{len(task_success_map)} tasks succeeded ({success_count/len(task_success_map)*100:.1f}%)")

    # Read the success_std parameter from kwargs (kept for backward compatibility).
    if 'success_std' not in kwargs:
        if 'dispersion_s' in kwargs:
            success_std = kwargs['dispersion_s']
        else:
            raise ValueError("Missing required 'success_std' or 'dispersion_s' parameter in kwargs for hardness partition strategy.")
    else:
        success_std = kwargs['success_std']

    # Maximum number of samples per client.
    max_samples_per_client = max(min_samples_per_client, total_size // client_num)

    print(f"Sample limits: min={min_samples_per_client}, max={max_samples_per_client}, total_data={total_size}")

    # Use a Beta distribution to generate the per-client success-sample counts.
    # A fixed seed ensures every client computes the same allocation.
    rng = np.random.default_rng(42)

    # Beta-distribution parameters.
    center = min_samples_per_client // 2  # center set to half of min_samples_per_client
    low = 0  # minimum number of success samples
    high = min_samples_per_client  # maximum number of success samples
    dispersion_s = success_std

    # Compute the target total (sum of success samples across all clients), using
    # the same approach as coverage_partition.
    # target_sum = int(round(center * client_num * 1))  # alternative: fixed coverage
    r = default_r(total_size, client_num, low, center, high)
    target_sum = int(round(r * total_size))

    print(f"Hardness partition using Beta distribution method...")
    print(f"Parameters: center={center}, dispersion_s={dispersion_s}, low={low}, high={high}")
    print(f"Target success samples sum: {target_sum}")

    # Generate the per-client success-sample counts.
    success_counts = generate_client_sizes(
        C=client_num,
        low=low,
        center=center,
        high=high,
        dispersion_s=dispersion_s,
        target_sum=target_sum,
        rng=rng
    )

    print(f"Generated success counts: {success_counts}")

    # Number of success samples assigned to the current client.
    current_success_count = success_counts[client_id]

    # Bucket the data by success rate.
    high_success_data = []  # success rate >= 0.5
    low_success_data = []   # success rate < 0.5
    unknown_success_data = []  # tasks with no success information

    for item in total_train_data:
        # Extract the task_id from the item (matching the logic in envs.py).
        task_id = None
        if isinstance(item, dict):
            if 'asin' in item:
                asin = item['asin']
                if 'goal_options' in item and item['goal_options']:
                    # For synthetic goals: use asin + goal_options hash
                    import hashlib
                    options_str = str(sorted(item['goal_options'].items()))
                    options_hash = int(hashlib.md5(options_str.encode()).hexdigest(), 16)
                    task_id = f"{asin}_{abs(options_hash)}"
                else:
                    # Fallback to asin + instruction_text hash for human goals
                    if 'instruction_text' in item:
                        import hashlib
                        instruction_hash = int(hashlib.md5(item['instruction_text'].encode()).hexdigest(), 16)
                        task_id = f"{asin}_{abs(instruction_hash)}"
                    else:
                        task_id = asin

        if task_id and task_id in task_success_map:
            success = task_success_map[task_id]
            if success:
                high_success_data.append(item)
            else:
                low_success_data.append(item)
        else:
            # Tasks with no matching success info default to "not successful".
            low_success_data.append(item)


    print(f"Data distribution: high_success={len(high_success_data)}, low_success={len(low_success_data)}")

    # Assemble the data for the current client.
    current_client_data = []

    # 1. Assign success samples (drawn preferentially from high_success_data).
    if current_success_count > 0 and high_success_data:
        # Randomly pick success samples from high_success_data.
        success_samples = rng.choice(
            high_success_data,
            size=min(current_success_count, len(high_success_data)),
            replace=False
        ).tolist()
        current_client_data.extend(success_samples)

        # Remove the chosen samples from high_success_data.
        for sample in success_samples:
            if sample in high_success_data:
                high_success_data.remove(sample)

    # 2. If more success samples are still needed, draw from low_success_data.
    remaining_success_needed = current_success_count - len([s for s in current_client_data if s in high_success_data])
    if remaining_success_needed > 0 and low_success_data:
        additional_success = rng.choice(
            low_success_data,
            size=min(remaining_success_needed, len(low_success_data)),
            replace=False
        ).tolist()
        current_client_data.extend(additional_success)

        # Remove the chosen samples from low_success_data.
        for sample in additional_success:
            if sample in low_success_data:
                low_success_data.remove(sample)

    # 3. Fill the remaining quota with randomly chosen samples, ensuring the
    #    total does not exceed max_samples_per_client.
    remaining_needed = min(max_samples_per_client - len(current_client_data), min_samples_per_client - len(current_client_data))
    if remaining_needed > 0:
        # Collect all still-unused samples.
        all_remaining_data = high_success_data + low_success_data

        if all_remaining_data:
            additional_samples = rng.choice(
                all_remaining_data,
                size=min(remaining_needed, len(all_remaining_data)),
                replace=False
            ).tolist()
            current_client_data.extend(additional_samples)

    # Final check: make sure we do not exceed the maximum.
    if len(current_client_data) > max_samples_per_client:
        current_client_data = current_client_data[:max_samples_per_client]

    print(f"Hardness partition completed for client {client_id + 1}")
    print(f"  Success samples: {current_success_count}")
    print(f"  Total samples: {len(current_client_data)} (max: {max_samples_per_client})")

    return current_client_data


# --------------------------------------------------------------------------- #
# Thin public API for the verl-0.8 WebShop service (task-level; full catalog).
# --------------------------------------------------------------------------- #
def hardness_for_client(
    client_id: int,
    client_num: int,
    *,
    success_std: float,
    trajectories_file: str,
    min_goals_per_client: int = 100,
    base_seed: int = 42,  # noqa: ARG001 (verbatim fn hardcodes 42; kept for API symmetry)
    start_idx: int = 500,
    env_goals: Optional[List[Any]] = None,
    data_dir: Optional[str] = None,
) -> List[int]:
    """This client's WebShop goal indices under Hardness(xi') -- full catalog (task-only).

    Beta-allocates easy/hard goals over the train pool (goals[start_idx:]) and returns the
    selected ABSOLUTE goal indices. `success_std` is the Beta dispersion knob.

    `env_goals` (REQUIRED for science): the env's ACTUAL `server.goals` (seed-42 shuffled
    dicts carrying asin + goal_options + instruction_text). The verbatim hardness body derives
    task_id = f"{asin}_{md5(goal_options)}" (or instruction_text fallback) -- the SAME formula
    the labelling pass records -- so the success lookup resolves ONLY when given the real goal
    dicts. The original partitions server.goals and maps back via goals.index(); the env
    shuffle is reproducible + identical across clients (GPU-verified), so this reproduces the
    original selection. The `data_dir` fallback yields asin-only dicts (task_id == asin) and is
    for offline tests ONLY -- its task_ids will NOT match an options-hash labelled file.

    `trajectories_file` (task_id -> success labels) is REQUIRED; there is no usable default.
    """
    if not trajectories_file:
        raise ValueError(
            "hardness_for_client requires `trajectories_file` (task_id -> success "
            "labels); there is no default in the fedagent package."
        )
    if not os.path.exists(trajectories_file):
        raise FileNotFoundError(f"Trajectories file not found: {trajectories_file}")

    if env_goals is not None:
        # FAITHFUL path: real goal dicts (asin/goal_options/instruction_text) -> correct task_id.
        goals = [dict(g, _idx=i) for i, g in enumerate(env_goals)]
    else:
        products, ins = load_webshop_data(data_dir)
        goal_asins = _generate_goal_asins_for_partition(products, ins)
        # Offline fallback: asin-only (task_id == asin); NOT options-hash faithful.
        goals = [{"asin": a, "_idx": i} for i, a in enumerate(goal_asins)]
    selected = hardness_partition(
        data=goals[start_idx:],
        client_id=client_id,
        client_num=client_num,
        min_samples_per_client=min_goals_per_client,
        trajectories_file=trajectories_file,
        success_std=success_std,
    )
    idxs = sorted(g["_idx"] for g in selected)
    print(f"[task hardness] WebShop client {client_id}/{client_num}: |goal_idxs|={len(idxs)} "
          f"(success_std={success_std}, full catalog, src={'env.server.goals' if env_goals is not None else 'reconstructed'})",
          flush=True)
    return idxs
