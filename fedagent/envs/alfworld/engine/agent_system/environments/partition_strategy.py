# Dataset partition strategies for federated learning.
from typing import List, Dict, Any, Optional, Union, Tuple
import json
import random
import hashlib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import math

from omegaconf import OmegaConf
import os


def _spec_hash(spec: str) -> int:
    """Process-stable hash for env_disjoint spec seeding.

    Built-in hash() is randomized per Python interpreter (PYTHONHASHSEED=random
    by default since Py3.3), so different client subprocesses see different
    hashes for the same spec → spec_seed inconsistent across clients → the
    "shared u" invariant in env_disjoint partition breaks at env_div<1.0.

    sha256-based hash gives the same 31-bit int across all processes / runs.
    See docs/heterogeneity.md
    """
    return int.from_bytes(hashlib.sha256(spec.encode()).digest()[:4], 'big') & 0x7fffffff

# ------------ Beta-distribution based sizing methods ------------
def default_r(N, C, low, center, high):
    """Compute the default overlap coefficient r.

    r is the ratio (total number of sample assignments) / (number of samples N).
    The raw value C*center/N (clients times the central per-client size, over the
    sample count) is clipped to the feasible band [C*low/N, C*high/N] so that the
    resulting target assignment count stays achievable given the per-client size
    bounds.
    """
    r_raw = (C * center) / N
    r_min = (C * low) / N
    r_max = (C * high) / N
    return float(np.clip(r_raw, r_min, r_max))

def generate_client_sizes(C, low, center, high, dispersion_s, target_sum, rng):
    """Generate per-client sizes using a Beta distribution.

    Draws C continuous sizes from a Beta distribution reparameterized by a mean
    `mu` (derived from low/center/high) and a dispersion `dispersion_s`, rescales
    them so their sum matches `target_sum`, clips each to [low, high], then rounds
    to integers while preserving the integer sum exactly (largest-remainder
    rounding plus a corrective pass). Returns an integer array of length C.
    """
    assert low <= center <= high, "Require low <= center <= high"
    assert C > 0

    # Adjust target_sum so the request is feasible given per-client bounds:
    # the achievable total lies in [C*low, C*high].
    if target_sum < C * low:
        print(f"Warning: target_sum {target_sum} < C*low {C*low}, adjusting target_sum")
        target_sum = C * low
    elif target_sum > C * high:
        print(f"Warning: target_sum {target_sum} > C*high {C*high}, adjusting target_sum")
        target_sum = C * high
    
    mu = (center - low) / (high - low) if high > low else 0.5
    mu = min(max(mu, 1e-6), 1 - 1e-6)
    s = max(dispersion_s, 2e-3)
    alpha = mu * s
    beta = (1 - mu) * s
    x = rng.beta(alpha, beta, size=C) if high > low else np.full(C, 0.5)
    cont_sizes = low + x * (high - low)
    scale = target_sum / cont_sizes.sum()
    cont_sizes = cont_sizes * scale
    cont_sizes = np.clip(cont_sizes, low, high)
    floors = np.floor(cont_sizes).astype(int)
    remainders = cont_sizes - floors
    need = int(round(target_sum)) - floors.sum()
    order = np.argsort(-remainders)
    sizes = floors.copy()
    if need > 0:
        sizes[order[:need]] += 1
    elif need < 0:
        sizes[order[::-1][:(-need)]] -= 1
    sizes = np.clip(sizes, low, high)
    # Clipping may have perturbed the sum; nudge individual sizes (respecting the
    # [low, high] bounds) until the integer total matches target_sum again.
    diff = int(round(target_sum)) - sizes.sum()
    if diff != 0:
        if diff > 0:
            room = (high - sizes)
            idxs = np.where(room > 0)[0]
            for i in idxs[:diff]:
                sizes[i] += 1
        else:
            room = (sizes - low)
            idxs = np.where(room > 0)[0]
            for i in idxs[:(-diff)]:
                sizes[i] -= 1
    
    # Final reconciliation: if the sum still does not match (e.g. the bounds left
    # no room in the loop above), force it into agreement.
    final_sum = sizes.sum()
    target_sum_int = int(round(target_sum))
    if final_sum != target_sum_int:
        print(f"Warning: Final sum {final_sum} != target {target_sum_int}, adjusting...")
        if final_sum < target_sum_int:
            # Need to add samples: increment clients that still have headroom
            # below `high`, cycling through them round-robin.
            diff = target_sum_int - final_sum
            for i in range(diff):
                # Find positions with remaining capacity to increase.
                room = (high - sizes)
                idxs = np.where(room > 0)[0]
                if len(idxs) > 0:
                    sizes[idxs[i % len(idxs)]] += 1
                else:
                    break
        else:
            # Need to remove samples: decrement clients that are still above
            # `low`, cycling through them round-robin.
            diff = final_sum - target_sum_int
            for i in range(diff):
                # Find positions that can be decreased.
                room = (sizes - low)
                idxs = np.where(room > 0)[0]
                if len(idxs) > 0:
                    sizes[idxs[i % len(idxs)]] -= 1
                else:
                    break

    # Final validation: guarantee every size lies within [low, high].
    if not np.all((sizes >= low) & (sizes <= high)):
        print(f"Warning: Some sizes are out of bounds, clipping...")
        sizes = np.clip(sizes, low, high)

    return sizes

def assign_with_overlap(N, sizes, r, rng):
    """Assign samples to clients allowing cross-client overlap.

    Each of the N samples is replicated k times (k near the overlap coefficient r:
    most samples get floor(r) copies, a random subset gets ceil(r) so the total
    equals sum(sizes)). For every sample, its copies are handed out to distinct
    clients chosen with probability proportional to each client's remaining
    capacity. A top-up pass then fills any shortfall left by capacity exhaustion.

    Returns (client_sets, k) where client_sets[c] is the set of sample ids held by
    client c, and k[i] is the replication count assigned to sample i.
    """
    total_assign = int(sizes.sum())
    r_floor = int(math.floor(r))
    r_ceil = int(math.ceil(r))
    num_high = total_assign - r_floor * N
    if not (0 <= num_high <= N):
        avg = total_assign / N
        r_floor = int(math.floor(avg))
        r_ceil = int(math.ceil(avg))
        num_high = total_assign - r_floor * N
    k = np.full(N, r_floor, dtype=int)
    if num_high > 0:
        chosen = rng.choice(N, size=num_high, replace=False)
        k[chosen] = r_ceil
    remaining = sizes.astype(int).copy()
    client_sets = [set() for _ in range(len(sizes))]

    # Distribute each sample's k copies across distinct clients.
    for sample_id in range(N):
        need = k[sample_id]
        avail = np.where(remaining > 0)[0]
        if need > len(avail):
            need = len(avail)
        if need == 0:
            continue
        if len(avail) == 0:
            break  # No clients with remaining capacity left.

        probs = remaining[avail].astype(float)
        if probs.sum() > 0:
            probs /= probs.sum()
            choices = rng.choice(avail, size=need, replace=False, p=probs)
            for c in choices:
                client_sets[c].add(sample_id)
                remaining[c] -= 1
    
    # Inspect the result; if fewer assignments were made than required, top up.
    actual_assign = sum(len(s) for s in client_sets)
    if actual_assign < total_assign:
        # Try to hand the missing assignments to clients that still have capacity.
        remaining_samples = total_assign - actual_assign
        avail_clients = np.where(remaining > 0)[0]

        if len(avail_clients) > 0:
            # Randomly pick samples to fill the remaining capacity.
            for _ in range(remaining_samples):
                if len(avail_clients) == 0:
                    break
                # Pick a random sample.
                sample_id = rng.integers(0, N)
                # Pick a random client that still has capacity.
                client_id = rng.choice(avail_clients)
                client_sets[client_id].add(sample_id)
                remaining[client_id] -= 1

                # Refresh the list of clients with remaining capacity.
                avail_clients = np.where(remaining > 0)[0]

    # Final sanity check.
    final_assign = sum(len(s) for s in client_sets)
    if final_assign != total_assign:
        print(f"Warning: Assignment mismatch - expected {total_assign}, got {final_assign}")
    
    return client_sets, k

def summarize_overlap(client_sets):
    """Summarize the result of an overlapping assignment.

    Computes per-client size statistics (min/max/mean/std) and the mean pairwise
    Jaccard similarity across all client pairs, returned as a dict.
    """
    C = len(client_sets)
    sizes = np.array([len(s) for s in client_sets])
    from itertools import combinations
    jaccs = []
    for i in range(C):
        for j in range(i+1, C):
            a, b = client_sets[i], client_sets[j]
            inter = len(a & b)
            uni = len(a | b)
            jacc = inter / uni if uni > 0 else 0.0
            jaccs.append(jacc)
    jacc_mean = float(np.mean(jaccs)) if jaccs else 0.0
    return {
        "client_sizes": sizes,
        "sizes_min": int(sizes.min()),
        "sizes_max": int(sizes.max()),
        "sizes_mean": float(sizes.mean()),
        "sizes_std": float(sizes.std(ddof=0)),
        "avg_pairwise_jaccard": jacc_mean,
    }

# Resolve the optional paths.yaml relative to this module. It is only used for the DEFAULT
# `trajectories_file` fallbacks below (lines ~828/1054/3422), which the FedAgent service
# always overrides with an explicit file. Vendored out of the verl-agent tree, the original
# repo-relative config/paths.yaml is absent -- tolerate that instead of failing at import.
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../../.."))
config_path = os.path.join(project_root, "config/paths.yaml")
if os.path.exists(config_path):
    path_cfg = OmegaConf.load(config_path)
else:
    path_cfg = OmegaConf.create({"project_root": project_root})




def uniform_partition(
    data: List[Any],
    client_id: int,
    client_num: int,
    min_samples_per_client: int,
    start_idx: int = 0
) -> tuple[List[Any], int, int]:
    """
    Uniform partition strategy: split the dataset into contiguous, equally sized
    slices, one per client.

    Args:
        data: list of items to partition.
        client_id: ID of the current client (0-based).
        client_num: total number of clients.
        min_samples_per_client: minimum number of samples each client must receive.
        start_idx: starting index into `data` (used to skip a held-out validation
            set, etc.).

    Returns:
        A tuple (client_data_slice, start_slice, end_slice): the data slice the
        current client should receive, plus its absolute start/end indices within
        the training portion (data[start_idx:]).
    """
    # Compute the actually usable training data (everything after start_idx).
    total_train_data = data[start_idx:]
    total_train_size = len(total_train_data)

    print(f"[DEBUG] uniform_partition: data_length={len(data)}, start_idx={start_idx}, total_train_size={total_train_size}")
    print(f"[DEBUG] uniform_partition: client_id={client_id}, client_num={client_num}, min_samples_per_client={min_samples_per_client}")

    # Edge case: if there is no training data, return an empty slice.
    if total_train_size == 0:
        print(f"[DEBUG] uniform_partition: total_train_size=0, returning empty data")
        return [], 0, 0

    # Base (equal) slice size per client.
    base_slice_size = total_train_size // client_num
    goals_per_client = max(base_slice_size, min_samples_per_client)

    print(f"[DEBUG] uniform_partition: base_slice_size={base_slice_size}, goals_per_client={goals_per_client}")

    # Base slice boundaries for the current client.
    start_slice = client_id * base_slice_size
    end_slice = start_slice + base_slice_size

    # If the base slice is smaller than the required minimum, extend it on both
    # sides to reach min_samples_per_client.
    current_slice_size = end_slice - start_slice
    if current_slice_size < min_samples_per_client:
        needed_extra = min_samples_per_client - current_slice_size

        # Room available to grow on the left and on the right.
        left_available = start_slice
        right_available = total_train_size - end_slice

        # Prefer growing symmetrically on both sides.
        if left_available + right_available >= needed_extra:
            # Enough room on both sides; split the extension roughly evenly.
            left_extra = min(needed_extra // 2, left_available)
            right_extra = min(needed_extra - left_extra, right_available)

            # If the right side could not absorb its full share, give the
            # leftover back to the left side.
            if right_extra < (needed_extra - left_extra):
                left_extra += min(needed_extra - left_extra - right_extra, left_available - left_extra)

            start_slice -= left_extra
            end_slice += right_extra
        elif left_available > 0:
            # Not enough room overall; consume the entire left side first, then
            # take whatever remains from the right.
            start_slice = 0
            right_needed = needed_extra - left_available
            end_slice += min(right_needed, right_available)
        elif right_available > 0:
            # No room on the left; extend to the right only.
            end_slice += min(needed_extra, right_available)
        else:
            # No room on either side; fall back to using all available data.
            start_slice = 0
            end_slice = total_train_size

    # Clamp indices into the valid range.
    start_slice = max(0, start_slice)
    end_slice = min(total_train_size, end_slice)

    print(f"[DEBUG] uniform_partition: final start_slice={start_slice}, end_slice={end_slice}")

    # Extract this client's data slice.
    client_data_slice = total_train_data[start_slice:end_slice]

    print(f"[DEBUG] uniform_partition: client_data_slice_length={len(client_data_slice)}")

    return client_data_slice, start_slice, end_slice


def preference_partition(
    data: List[Any],
    client_id: int,
    client_num: int,
    min_samples_per_client: int,
    category_key: str = 'category',
    start_idx: int = 0,
    tau: float = 0.3,
    data_type: str = 'generic',
    **kwargs
) -> List[Any]:
    """
    Preference partition strategy (non-IID over the product-category marginal;
    Dirichlet -> Multinomial, see the per-backend helpers).

    Perturbs the global category marginal per client and draws per-category counts
    via Multinomial sampling, so that each category's per-client share is spread
    around the global proportion while the global category ratios are preserved in
    expectation.

    Args:
        data: list of items to partition.
        client_id: ID of the current client (0-based).
        client_num: total number of clients.
        min_samples_per_client: minimum number of samples each client must receive.
        category_key: name of the field holding the category label.
        start_idx: starting index into `data` (used to skip a validation set, etc.).
        tau: heterogeneity parameter controlling the variance of the per-client
            category distribution. LEGACY ALIAS of the canonical preference
            knob omega (paper symbol omega); forwarded as `tau=` and resolved
            downstream via `if omega is None: omega = tau`. NOTE: this 'tau' is
            the preference knob and is UNRELATED to the paper's task-descriptor
            tau. (Renaming the kwarg itself is a separate risky change — see the
            risky-rename note.)
        data_type: data type ('generic', 'webshop', 'alfworld').
        **kwargs: additional strategy-specific parameters.

    Returns:
        The data slice the current client should receive.
    """
    # Compute the actually usable training data (everything after start_idx).
    total_train_data = data[start_idx:]

    if data_type == 'alfworld':
        # ALFWorld-specific handling: derive the task type from the file path.
        return _preference_partition_alfworld(
            data=total_train_data,
            client_id=client_id,
            client_num=client_num,
            min_samples_per_client=min_samples_per_client,
            tau=tau,
            **kwargs
        )
    else:
        # Generic handling: read the category label from each item's dict.
        # Pop fashion_sample_ratio out of kwargs to avoid passing it twice.
        fashion_sample_ratio = kwargs.pop('fashion_sample_ratio', 0.2)
        return _preference_partition_generic(
            data=total_train_data,
            client_id=client_id,
            client_num=client_num,
            min_samples_per_client=min_samples_per_client,
            category_key=category_key,
            tau=tau,
            fashion_sample_ratio=fashion_sample_ratio,
            **kwargs
        )


def _preference_partition_generic(
    data: List[Any],
    client_id: int,
    client_num: int,
    min_samples_per_client: int,
    category_key: str = 'category',
    tau: float = 0.3,
    fashion_sample_ratio: float = 0.2,
    omega: Optional[float] = None,
    **kwargs
) -> List[Any]:
    """
    PreferencePartition (Dirichlet, ω) — see docs/heterogeneity.md (Preference construction).

    q_i ~ Dir(π · (1-ω)/ω)   then   counts ~ Multinomial(L; q_i)
    E[q_i] = π exactly ∀ ω;  Δ²_pref(ω) = ω · (1 - ‖π‖²) (linear, monotone).

    `omega` is the canonical hyperparameter; legacy yamls passing only `tau`
    will have it aliased to omega (both lie in (0,1) for the values we sweep
    {0.1, 0.9}; semantic mapping is "spread fraction").

    Args:
        fashion_sample_ratio: subsampling ratio applied to the 'fashion' category
            (the default constant in the signature is 0.2, i.e. keep 20%).
        omega: spread fraction in (0, 1); large ω → high heterogeneity.
        tau: legacy kwarg; aliased to omega when omega is None.
    """
    import math

    if omega is None:
        omega = tau
    omega = float(np.clip(omega, 1e-3, 1 - 1e-3))

    total_size = len(data)

    # Group items by category, recording the sample indices for each category.
    category_to_indices = {}
    for idx, item in enumerate(data):
        category = item.get(category_key, 'unknown')
        if category not in category_to_indices:
            category_to_indices[category] = []
        category_to_indices[category].append(idx)

    # Pre-subsample the 'fashion' category to keep it from dominating.
    if 'fashion' in category_to_indices:
        fashion_indices = category_to_indices['fashion']
        fashion_count = len(fashion_indices)

        if fashion_count > 0:
            target_fashion_count = max(1, int(fashion_count * fashion_sample_ratio))

            rng = np.random.RandomState(42 + client_id)
            sampled_fashion_indices = rng.choice(
                fashion_indices,
                size=target_fashion_count,
                replace=False
            ).tolist()

            category_to_indices['fashion'] = sampled_fashion_indices

            print(f"Fashion category subsampling: {fashion_count} -> {target_fashion_count} samples (keeping {fashion_sample_ratio*100:.1f}%)")

    total_size = sum(len(indices) for indices in category_to_indices.values())

    # Global category marginal π (Laplace-smoothed for any zero-class).
    categories = list(category_to_indices.keys())
    C = len(categories)
    eps_smooth = 0.01
    raw_p = np.array([len(category_to_indices[c]) for c in categories], dtype=float)
    raw_p = raw_p / raw_p.sum() if raw_p.sum() > 0 else np.full(C, 1.0 / C)
    pi = (raw_p + eps_smooth) / (1.0 + C * eps_smooth)

    # Per-client RNG: deterministic across rounds for given client_id.
    rng = np.random.RandomState(42 + client_id)

    # PreferencePartition (Dirichlet, ω):
    #   α := π · (1-ω)/ω           (Dirichlet concentration vector)
    #   q_i ~ Dir(α);  E[q_i] = π exact;  tr Cov(q_i) = ω(1-‖π‖²)
    alpha_vec = pi * ((1.0 - omega) / omega)
    q = rng.dirichlet(alpha_vec)
    q = q / q.sum()  # numerical safety

    # Multinomial sampling of L per-class counts (line 4 of paper Algorithm 1).
    counts = rng.multinomial(min_samples_per_client, q)
    category_counts = {categories[i]: int(counts[i]) for i in range(C)}

    # Capacity fix (line 5 of Algorithm 1): if a_c > n_c, clip and redistribute
    # leftover by q to classes with spare capacity, preserving total = L.
    while True:
        overflow = 0
        donor_cats = []
        donor_q = []
        for i, c in enumerate(categories):
            n_c = len(category_to_indices[c])
            if category_counts[c] > n_c:
                overflow += category_counts[c] - n_c
                category_counts[c] = n_c
            elif category_counts[c] < n_c:
                donor_cats.append(c)
                donor_q.append(q[i])
        if overflow == 0 or not donor_cats:
            break
        donor_q = np.array(donor_q, dtype=float)
        donor_q = donor_q / donor_q.sum()
        # Redistribute overflow by q. Cap each category by its remaining capacity.
        extra = rng.multinomial(overflow, donor_q)
        for j, c in enumerate(donor_cats):
            cap_left = len(category_to_indices[c]) - category_counts[c]
            add = min(int(extra[j]), cap_left)
            category_counts[c] += add
        # Loop again in case some donor classes also overflowed; converges
        # because total assigned ≤ Σ n_c (= total_size ≥ L by precondition).
        if sum(category_counts.values()) >= min_samples_per_client:
            break
    
    # 5) Draw the concrete samples from each category without replacement.
    current_client_data = []
    for c, count in category_counts.items():
        if count > 0:
            # Randomly select from this category's index pool.
            available_indices = category_to_indices[c]
            if len(available_indices) >= count:
                selected_indices = rng.choice(available_indices, size=count, replace=False)
            else:
                # Not enough samples available; use all of them.
                selected_indices = available_indices

            # Resolve indices to the actual data items.
            for idx in selected_indices:
                current_client_data.append(data[idx])

    # 6) If the resulting set is still too small, top up from other categories.
    if len(current_client_data) < min_samples_per_client:
        needed_extra = min_samples_per_client - len(current_client_data)

        # Collect the indices already used.
        used_indices = set()
        for item in current_client_data:
            item_idx = data.index(item)
            used_indices.add(item_idx)

        available_indices = [i for i in range(len(data)) if i not in used_indices]

        if available_indices and needed_extra > 0:
            extra_count = min(needed_extra, len(available_indices))
            extra_indices = rng.choice(available_indices, size=extra_count, replace=False)
            for idx in extra_indices:
                current_client_data.append(data[idx])

    return current_client_data


def _preference_partition_alfworld(
    data: List[str],
    client_id: int,
    client_num: int,
    min_samples_per_client: int,
    tau: float = 0.3,
    omega: Optional[float] = None,
    **kwargs
) -> List[str]:
    """
    ALFWorld task-type PreferencePartition (Dirichlet, ω). See docs/heterogeneity.md (Task-level Constructions).
    """
    import json
    import os

    if omega is None:
        omega = tau
    omega = float(np.clip(omega, 1e-3, 1 - 1e-3))

    # ALFWorld task-type code -> name mapping.
    TASK_TYPES = {
        1: "pick_and_place_simple",
        2: "look_at_obj_in_light",
        3: "pick_clean_then_place_in_recep",
        4: "pick_heat_then_place_in_recep",
        5: "pick_cool_then_place_in_recep",
        6: "pick_two_obj_and_place"
    }

    total_size = len(data)

    # Group items by task type.
    category_to_indices = {}
    for idx, file_path in enumerate(data):
        json_path = None
        try:
            json_path = file_path.replace("game.tw-pddl", "traj_data.json")
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    traj_data = json.load(f)
                task_type = traj_data.get('task_type', 'unknown')
                if task_type in TASK_TYPES:
                    category = TASK_TYPES[task_type]
                else:
                    category = f"task_type_{task_type}"
            else:
                category = 'unknown'
        except Exception as e:
            error_path = json_path if json_path else file_path
            print(f"[ALFWorld Preference Partition] Error reading {error_path}: {e}")
            category = 'unknown'

        if category not in category_to_indices:
            category_to_indices[category] = []
        category_to_indices[category].append(idx)

    categories = list(category_to_indices.keys())
    C = len(categories)
    eps_smooth = 0.01
    raw_p = np.array([len(category_to_indices[c]) for c in categories], dtype=float)
    raw_p = raw_p / raw_p.sum() if raw_p.sum() > 0 else np.full(C, 1.0 / C)
    pi = (raw_p + eps_smooth) / (1.0 + C * eps_smooth)

    print(f"[ALFWorld Preference Partition] Found categories: {categories}")
    print(f"[ALFWorld Preference Partition] Smoothed π: {dict(zip(categories, pi.round(4).tolist()))}")
    print(f"[ALFWorld Preference Partition] omega={omega}; client_id={client_id}")

    rng = np.random.RandomState(42 + client_id)

    # PreferencePartition (Dirichlet, ω): q_i ~ Dir(π · (1-ω)/ω)
    alpha_vec = pi * ((1.0 - omega) / omega)
    q = rng.dirichlet(alpha_vec)
    q = q / q.sum()

    counts = rng.multinomial(min_samples_per_client, q)
    category_counts = {categories[i]: int(counts[i]) for i in range(C)}

    # Capacity fix (line 5 of paper Algorithm 1).
    while True:
        overflow = 0
        donor_cats = []
        donor_q = []
        for i, c in enumerate(categories):
            n_c = len(category_to_indices[c])
            if category_counts[c] > n_c:
                overflow += category_counts[c] - n_c
                category_counts[c] = n_c
            elif category_counts[c] < n_c:
                donor_cats.append(c)
                donor_q.append(q[i])
        if overflow == 0 or not donor_cats:
            break
        donor_q = np.array(donor_q, dtype=float)
        donor_q = donor_q / donor_q.sum()
        extra = rng.multinomial(overflow, donor_q)
        for j, c in enumerate(donor_cats):
            cap_left = len(category_to_indices[c]) - category_counts[c]
            add = min(int(extra[j]), cap_left)
            category_counts[c] += add
        if sum(category_counts.values()) >= min_samples_per_client:
            break

    # 5) Draw the concrete samples from each category without replacement.
    current_client_data = []
    for c, count in category_counts.items():
        if count > 0:
            # Randomly select from this category's index pool.
            available_indices = category_to_indices[c]
            if len(available_indices) >= count:
                selected_indices = rng.choice(available_indices, size=count, replace=False)
            else:
                # Not enough samples available; use all of them.
                selected_indices = available_indices

            # Resolve indices to the actual data items.
            for idx in selected_indices:
                current_client_data.append(data[idx])

    # 6) If the resulting set is still too small, top up from other categories.
    if len(current_client_data) < min_samples_per_client:
        needed_extra = min_samples_per_client - len(current_client_data)

        # Collect the indices already used.
        used_indices = set()
        for item in current_client_data:
            item_idx = data.index(item)
            used_indices.add(item_idx)

        available_indices = [i for i in range(len(data)) if i not in used_indices]

        if available_indices and needed_extra > 0:
            extra_count = min(needed_extra, len(available_indices))
            extra_indices = rng.choice(available_indices, size=extra_count, replace=False)
            for idx in extra_indices:
                current_client_data.append(data[idx])

    return current_client_data


def split_with_overlap(n_samples: int, n_sets: int, r: float = 1.3, 
                      size_std: float = 10, low: Optional[int] = None, 
                      high: Optional[int] = None, seed: int = 42, 
                      show_progress: bool = True) -> Tuple[List[List[int]], np.ndarray]:
    """
    Split a dataset into overlapping sets - simplified version.

    Args:
        n_samples: total number of samples.
        n_sets: number of sets to produce.
        r: overlap coefficient (total number of assignments / number of samples).
        size_std: standard deviation of the set sizes.
        low: minimum set size.
        high: maximum set size.
        seed: random seed.
        show_progress: whether to print progress messages.

    Returns:
        (list of sets, array of target sizes).
    """
    import random

    random.seed(seed)
    np.random.seed(seed)

    # Average number of samples per set.
    avg = n_samples * r / n_sets
    if low is None:
        low = max(1, int(avg - 5*size_std))
    if high is None:
        # high = int(avg + 5*size_std)  # auto-compute the maximum set size
        high = 1000

    # 1) Draw normally distributed set sizes.
    if show_progress:
        print("Generating normally distributed set sizes...")

    # Draw a normal distribution clipped to [low, high]. The center is fixed at
    # 100 here (alternatives such as (low+high)/2 or avg+150 are kept commented
    # out); size_std controls how peaked the distribution is.
    # center = (low + high) / 2
    # center = avg+150
    center =100

    # Draw the normally distributed sizes.
    sizes = []
    for _ in range(n_sets):
        size = int(np.random.normal(center, size_std))
        # Clamp into the valid range.
        size = max(low, min(high, size))
        sizes.append(size)

    sizes = np.array(sizes)

    # 2) Randomly sample the members of each set.
    if show_progress:
        print("Randomly assigning samples...")

    sets_list = []
    for k in range(n_sets):
        # Sample directly: no duplicates within one client, but duplicates across
        # clients are allowed.
        sample_indices = np.random.choice(n_samples, size=sizes[k], replace=False)
        sets_list.append(sorted(sample_indices.tolist()))

        # Verify there are no duplicates within a single client.
        assert len(set(sample_indices)) == len(sample_indices), f"Client {k} has duplicate samples!"

    if show_progress:
        print("Data split complete!")

    return sets_list, sizes


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


def hardness_partition_alfworld(
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
    Task-difficulty (hardness) based partition strategy for Alfworld.

    The goals of this strategy are:
    1. Read each task's success rate from a trajectories file.
    2. Allocate the training set according to those success rates.
    3. Make each client's number of "success" samples follow a normal distribution
       between 0 and min_goals_per_client.
    4. Fill the rest of each client's quota with randomly chosen samples.

    Args:
        data: list of items to partition (Alfworld game file paths).
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
        trajectories_file = os.path.join(path_cfg.project_root, "output/inference/all_trajectories_alfworld.json")

    # Load the trajectories file.
    if not os.path.exists(trajectories_file):
        raise FileNotFoundError(f"Trajectories file not found: {trajectories_file}")

    if show_progress and client_id == 0:
        print(f"Loading Alfworld trajectories from: {trajectories_file}")

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


    print(f"Loaded {len(task_success_map)} Alfworld tasks with success information")
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

    print(f"Alfworld sample limits: min={min_samples_per_client}, max={max_samples_per_client}, total_data={total_size}")

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

    print(f"Alfworld hardness partition using Beta distribution method...")
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
        # Extract the task_id from the Alfworld game file path.
        task_id = None
        if isinstance(item, str) and item.endswith('.tw-pddl'):
            # Derive the task_id from the game file path.
            # Path format: /path/to/pick_clean_then_place_in_recep-Plate-None-DiningTable-19/trial_T20190909_045437_991233/game.tw-pddl
            try:
                # Get the file name (with extension).
                filename = os.path.basename(item)
                if filename == 'game.tw-pddl':
                    # Parent directory name (trial_xxx).
                    parent_dir = os.path.basename(os.path.dirname(item))
                    # Grandparent directory name (task_type-xxx).
                    grandparent_dir = os.path.basename(os.path.dirname(os.path.dirname(item)))
                    # Build the task_id.
                    task_id = f"alfworld_{grandparent_dir}_{parent_dir}_game"
            except Exception as e:
                if show_progress and client_id == 0:
                    print(f"Warning: Could not extract task_id from {item}: {e}")

        if task_id and task_id in task_success_map:
            success = task_success_map[task_id]
            if success:
                high_success_data.append(item)
            else:
                low_success_data.append(item)
        else:
            # Tasks with no matching success info default to "not successful".
            low_success_data.append(item)


    print(f"Alfworld data distribution: high_success={len(high_success_data)}, low_success={len(low_success_data)}")

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

    print(f"Alfworld hardness partition completed for client {client_id + 1}")
    print(f"  Success samples: {current_success_count}")
    print(f"  Total samples: {len(current_client_data)} (max: {max_samples_per_client})")
    
    return current_client_data


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


def partition_dataset(
    data: List[Any],
    strategy: str,
    client_id: int,
    client_num: int,
    min_samples_per_client: int,
    start_idx: int = 0,
    data_type: str = 'generic',  # 'generic', 'webshop', 'alfworld'
    **kwargs
) -> Union[List[Any], Tuple[List[Any], int, int]]:
    """
    Unified dataset partition interface.

    Args:
        data: list of items to partition.
        strategy: partition strategy ('uniform', 'preference', 'coverage',
            'hardness').
        client_id: ID of the current client (0-based).
        client_num: total number of clients.
        min_samples_per_client: minimum number of samples each client must receive.
        start_idx: starting index into `data` (used to skip a validation set, etc.).
        data_type: data type ('generic', 'webshop', 'alfworld').
        **kwargs: additional strategy-specific parameters.

    Returns:
        The data slice the current client should receive.
    """
    if strategy == 'uniform':
        client_data_slice, start_slice, end_slice = uniform_partition(data, client_id, client_num, min_samples_per_client, start_idx)
        return client_data_slice, start_slice, end_slice
    elif strategy == 'preference':
        return preference_partition(data, client_id, client_num, min_samples_per_client, start_idx=start_idx, data_type=data_type, **kwargs)
    elif strategy == 'coverage':
        return coverage_partition(data, client_id, client_num, min_samples_per_client, start_idx=start_idx, **kwargs)
    elif strategy == 'hardness':
        if data_type == 'alfworld':
            return hardness_partition_alfworld(data, client_id, client_num, min_samples_per_client, start_idx=start_idx, **kwargs)
        else:
            return hardness_partition(data, client_id, client_num, min_samples_per_client, start_idx=start_idx, data_type=data_type, **kwargs)
    elif strategy == 'env_disjoint':
        # Env-level heterogeneity for AlfWorld (scene-disjoint partition).
        # See docs/heterogeneity.md
        if data_type != 'alfworld':
            raise ValueError(
                f"strategy 'env_disjoint' only supports data_type='alfworld' "
                f"(WebShop env-level uses 'distractor_disjoint' instead, called "
                f"directly from fed_env_manager.py). got data_type={data_type}"
            )
        return _env_disjoint_partition_alfworld(
            data, client_id, client_num, min_samples_per_client, **kwargs
        )
    elif strategy == 'catalog_split':
        # Catalog-Split: per-client target floor distractor disjoint
        # See docs/heterogeneity.md
        # Called directly from fed_env_manager.py (needs products / ins / goals);
        # routing through partition_dataset is not supported.
        raise NotImplementedError(
            "catalog_split: call _distractor_disjoint_partition_webshop_v5 "
            "directly from fed_env_manager.py, not via partition_dataset()"
        )
    else:
        raise ValueError(f"Unknown partition strategy: {strategy}. Supported strategies: uniform, preference, coverage, hardness, env_disjoint, catalog_split")


def get_partition_info(
    data: List[Any],
    strategy: str,
    client_id: int,
    client_num: int,
    min_samples_per_client: int,
    start_idx: int = 0,
    data_type: str = 'generic',
    **kwargs
) -> Dict[str, Any]:
    """
    Return detailed information about a partition.

    Args:
        data: list of items to partition.
        strategy: partition strategy.
        client_id: ID of the current client.
        client_num: total number of clients.
        min_samples_per_client: minimum number of samples each client must receive.
        start_idx: starting index into `data`.
        **kwargs: additional strategy-specific parameters.

    Returns:
        A dict describing the partition.
    """
    if strategy == 'uniform':
        client_data, start_slice, end_slice = uniform_partition(data, client_id, client_num, min_samples_per_client, start_idx)
        return {
            'strategy': strategy,
            'client_id': client_id,
            'client_num': client_num,
            'data_size': len(client_data),
            'start_idx': start_slice,
            'end_idx': end_slice,
            'min_samples_per_client': min_samples_per_client,
            'actual_samples': len(client_data)
        }
    elif strategy == 'preference':
        client_data = preference_partition(data, client_id, client_num, min_samples_per_client, start_idx=start_idx, data_type=data_type, **kwargs)
        return {
            'strategy': strategy,
            'client_id': client_id,
            'client_num': client_num,
            'data_size': len(client_data),
            'min_samples_per_client': min_samples_per_client,
            'actual_samples': len(client_data)
        }
    elif strategy == 'coverage':
        client_data = coverage_partition(data, client_id, client_num, min_samples_per_client, start_idx=start_idx, **kwargs)
        return {
            'strategy': strategy,
            'client_id': client_id,
            'client_num': client_num,
            'data_size': len(client_data),
            'min_samples_per_client': min_samples_per_client,
            'actual_samples': len(client_data)
        }
    elif strategy == 'hardness':
        if data_type == 'alfworld':
            client_data = hardness_partition_alfworld(data, client_id, client_num, min_samples_per_client, start_idx=start_idx, **kwargs)
        else:
            client_data = hardness_partition(data, client_id, client_num, min_samples_per_client, start_idx=start_idx, data_type=data_type, **kwargs)
        return {
            'strategy': strategy,
            'client_id': client_id,
            'client_num': client_num,
            'data_size': len(client_data),
            'min_samples_per_client': min_samples_per_client,
            'actual_samples': len(client_data)
        }
    elif strategy == 'env_disjoint':
        client_data = _env_disjoint_partition_alfworld(
            data, client_id, client_num, min_samples_per_client, **kwargs
        )
        return {
            'strategy': strategy,
            'client_id': client_id,
            'client_num': client_num,
            'data_size': len(client_data),
            'min_samples_per_client': min_samples_per_client,
            'actual_samples': len(client_data)
        }
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


# =============================================================================
# Usage examples and notes
# =============================================================================

"""
Partition strategy usage examples:

1. Uniform Partition:
   client_data, start_idx, end_idx = partition_dataset(
       data=goals,
       strategy='uniform',
       client_id=0,
       client_num=100,
       min_samples_per_client=500,
       start_idx=500
   )

2. Preference Partition:
   # Basic usage (with default tau=0.3):
   client_data = partition_dataset(
       data=goals,
       strategy='preference',
       client_id=0,
       client_num=100,
       min_samples_per_client=500,
       start_idx=500,
       category_key='category'
   )

   # Tune the tau parameter to control how uneven the distribution is:
   # Smoother distribution (closer to the global proportions)
   client_data = partition_dataset(
       data=goals,
       strategy='preference',
       client_id=0,
       client_num=100,
       min_samples_per_client=500,
       start_idx=500,
       category_key='category',
       tau=0.15  # smaller tau -> smoother distribution
   )

   # More uneven distribution (larger differences)
   client_data = partition_dataset(
       data=goals,
       strategy='preference',
       client_id=0,
       client_num=100,
       min_samples_per_client=500,
       start_idx=500,
       category_key='category',
       tau=0.5   # larger tau -> more uneven distribution
   )

3. Coverage Partition:
   # Basic usage (with defaults overlap_ratio=1.3, size_std=10):
   client_data = partition_dataset(
       data=goals,
       strategy='coverage',
       client_id=0,
       client_num=100,
       min_samples_per_client=500,
       start_idx=500
   )

   # Tune the overlap_ratio parameter to control the degree of overlap:
   # Less overlapping distribution
   client_data = partition_dataset(
       data=goals,
       strategy='coverage',
       client_id=0,
       client_num=100,
       min_samples_per_client=500,
       start_idx=500,
       overlap_ratio=1.1  # smaller overlap_ratio -> less overlap
   )

   # More overlapping distribution
   client_data = partition_dataset(
       data=goals,
       strategy='coverage',
       client_id=0,
       client_num=100,
       min_samples_per_client=500,
       start_idx=500,
       overlap_ratio=1.5   # larger overlap_ratio -> more overlap
   )

   # Tune the size_std parameter to control the variability of sample counts:
   # More uniform sample-count distribution
   client_data = partition_dataset(
       data=goals,
       strategy='coverage',
       client_id=0,
       client_num=100,
       min_samples_per_client=500,
       start_idx=500,
       size_std=5  # smaller size_std -> more uniform sample counts
   )

   # More uneven sample-count distribution
   client_data = partition_dataset(
       data=goals,
       strategy='coverage',
       client_id=0,
       client_num=100,
       min_samples_per_client=500,
       start_idx=500,
       size_std=20   # larger size_std -> larger differences in sample counts
   )

4. Parameter notes:
   - Preference strategy:
     * tau: fluctuation-strength parameter (0.1-0.6)
       - 0.1-0.25: distribution closer to the global proportions, smoother
       - 0.3-0.4: moderate unevenness (recommended default)
       - 0.4-0.6: more uneven distribution, larger differences
     * Small-category protection: for categories with very few samples, a smaller
       tau is used automatically to avoid instability from excessive fluctuation.

   - Coverage strategy:
     * overlap_ratio: overlap coefficient = total assignments / number of samples
       (1.0-2.0)
       - 1.0: no overlap; on average size/n_sets samples per set
       - 1.2-1.4: moderate overlap (recommended default 1.3)
       - 1.5-2.0: high overlap; larger average set size
     * size_std: standard deviation of the set sizes; controls the spread of the
       normal distribution (5-20)
       - 5-10: fairly uniform sample-count distribution (recommended default 10)
       - 10-15: moderate variability
       - 15-20: larger differences in sample counts
     * Coverage guarantee: every sample appears at least once.
     * Overlap allowed: samples may be shared between clients.
     * Normal distribution: per-client sample counts are approximately normal.

5. Algorithm principles:
   - Uniform: split the dataset evenly by the number of clients.
   - Preference: perturb each client's category marginal (Dirichlet) and
     determine per-category sample counts via Multinomial sampling.
   - Coverage: draw per-client sample counts from a truncated normal distribution,
     ensure coverage via round-robin assignment, then fill the remaining slots via
     weighted sampling.
"""


# ============================================================
# Environment-Level Heterogeneity (WebShop): distractor-disjoint catalog partition
#   This is paper Variant 1 = "Catalog Split" (Transition-pipeline Stage 1),
#   built by paper Algorithm 1 (CatalogSplitPartition).
#   strategy key: 'distractor_disjoint'  ->  SimServer override: catalog_filter_asins.
#
#   NAMING WARNING: in this codebase this function is the "v4 algo" / "v4"
#   catalog impl. "v4"/"v5" here are IMPLEMENTATION-REVISION numbers of the
#   Catalog-Split partitioner, NOT the paper's Variant 4 / Variant 5. This v4
#   impl is LEGACY/superseded: all clients share goals[500:], the catalog
#   protects ALL training target ASINs, and the distractor pool is the single
#   shared ~585-item set. The CURRENT impl the paper uses for Catalog Split is
#   _distractor_disjoint_partition_webshop_v5 (strategy key 'catalog_split').
#   Both implement the SAME paper Variant 1; they are UNRELATED to paper
#   Variant 4 (Lookalike Injection, lookalike_injection) and Variant 5
#   (Rank Wrapper, rank_wrapper).
#   See docs/heterogeneity.md for design.
# ============================================================
def _distractor_disjoint_partition_webshop(
    products: List[Dict[str, Any]],
    ins: Dict[str, Dict[str, Any]],
    client_id: int,
    client_num: int,
    env_div: float = 0.7,
    keep_ratio: float = 0.7,
    holdout_distractor_asins: Optional[List[str]] = None,
    base_seed: int = 42,
    **kwargs,
) -> List[str]:
    """Env-level partition for WebShop.

    Algorithm (Shared+Independent Uniform interpolation + Top-k selection):
      For each distractor d in the partition pool:
        u[d]   = global shared random scalar  (same across all clients)
        v_k[d] = per-client private random scalar  (independent per client)
        e_k[d] = (1 - env_div) * u[d] + env_div * v_k[d]
      Then client_k takes the n_keep = round(keep_ratio * D) distractors with
      lowest e_k (np.argsort(e)[:n_keep]).

    Why top-k instead of `e < keep_ratio` threshold:
      Threshold method makes P(include) = P(e < keep_ratio), which depends on
      env_div (e is Triangular at env_div=0.5 -> P=0.82 instead of 0.7),
      coupling env_div with catalog size. Top-k decouples the two: env_div
      only controls *which* distractors are picked, not *how many*.

    Properties (verified empirically):
      env_div = 0.0 -> all clients identical                  Jaccard = 1.0
      env_div = 1.0 -> independent random size-n_keep subsets,
                       per-d marginal = keep_ratio (NOT joint
                       independent Bernoulli; size constraint
                       induces per-client correlation)         Jaccard ≈ r²/(2r-r²)
      catalog size always = |target| + n_keep, regardless of env_div.

    Args:
      products: list of product dicts (from items_shuffle_*.json)
      ins: dict asin -> {attributes, instruction, ...}
           (from items_ins_v2_*.json). Only ASINs with non-empty 'instruction'
           are 'targets' (must be in every client's catalog).
      client_id, client_num: standard federated args
      env_div: heterogeneity strength in [0, 1]
      keep_ratio: per-client distractor density, in (0, 1]
      holdout_distractor_asins: distractors reserved for OOD eval
                                (excluded from any client's training catalog)
      base_seed: RNG seed (same for proto_rng across clients)

    Returns:
      catalog_asins: sorted list of ASINs for this client's SimServer.
                     Always contains all target ASINs.
    """
    holdout = set(holdout_distractor_asins or [])

    # Step 1: separate target / distractor
    target_asins = {asin for asin, entry in ins.items()
                    if entry.get('instruction')}
    # Validate holdout: must contain only distractors. If it accidentally
    # contains a target ASIN, that ASIN would silently end up back in every
    # client's catalog (via the safety check at end), defeating the OOD invariant.
    holdout_target_overlap = holdout & target_asins
    if holdout_target_overlap:
        raise ValueError(
            f"[distractor_disjoint] holdout list contains "
            f"{len(holdout_target_overlap)} target ASINs (must be distractors only): "
            f"sample = {sorted(holdout_target_overlap)[:5]}. "
            f"Regenerate with `python tools/env_heterogeneity/gen_holdout_webshop.py`."
        )
    all_asins = {p['asin'] for p in products}
    distractor_pool = sorted((all_asins - target_asins) - holdout)
    D = len(distractor_pool)

    # Step 2: shared u[d] (same for all clients)
    proto_rng = np.random.RandomState(base_seed)
    u = proto_rng.random(D)

    # Step 3: per-client v[d] + linear interpolation
    client_rng = np.random.RandomState(base_seed + 1000 * int(client_id))
    v = client_rng.random(D)
    e = (1.0 - env_div) * u + env_div * v

    # Step 4: assemble catalog with FIXED size = round(keep_ratio * D)
    # Rationale: a naive `e < keep_ratio` threshold makes
    #   P(include) = P(e < keep_ratio) which depends on env_div
    #   (e is triangular when env_div=0.5, so P shifts to ~0.82 instead of 0.7).
    # That confounds env_div with catalog size (i.e., "coverage" axis).
    # Top-k selection by e-rank decouples the two: env_div only controls
    # *which* distractors get picked, not *how many*.
    n_keep = int(round(keep_ratio * D))
    chosen_idx = np.argsort(e)[:n_keep]
    include_distractors = [distractor_pool[i] for i in chosen_idx]
    catalog_asins = sorted(target_asins | set(include_distractors))

    # Step 5: safety check (every target ASIN must be in catalog)
    missing = target_asins - set(catalog_asins)
    if missing:
        raise RuntimeError(
            f"[distractor_disjoint] client {client_id}: "
            f"{len(missing)} target ASINs missing from catalog!"
        )

    print(f"[ENV] WebShop client {client_id}/{client_num}: "
          f"|catalog|={len(catalog_asins)} (target={len(target_asins)}, "
          f"distractor={len(include_distractors)}/{D}), "
          f"env_div={env_div}, keep_ratio={keep_ratio}")
    return catalog_asins


# ============================================================
# Catalog-Split (CURRENT impl): per-client target-floor distractor disjoint.
#   This is the live realization of paper Variant 1 = "Catalog Split"
#   (Transition-pipeline Stage 1), built by paper Algorithm 1.
#   strategy key: 'catalog_split'  ->  SimServer overrides:
#       catalog_filter_asins AND client_goal_idxs.
#
#   NAMING WARNING: this is the "v5 algo" / "v5" catalog impl. "v5" is an
#   IMPLEMENTATION-REVISION number of the Catalog-Split partitioner; it is NOT
#   the paper's Variant 5 (Rank Wrapper). It supersedes the v4 impl
#   (_distractor_disjoint_partition_webshop, key 'distractor_disjoint'); both
#   implement the SAME paper Variant 1.
# See docs/heterogeneity.md
# ============================================================
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

    # Step 4: distractor_pool — pool depends on per-client target, so it's per-client
    if holdout & client_target_asins:
        raise ValueError(
            f"[distractor_disjoint v5] client {client_id}: holdout list contains "
            f"{len(holdout & client_target_asins)} of this client's target ASINs"
        )
    all_asins_sorted = sorted({p['asin'] for p in products})
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

    # Step 7: assemble catalog
    catalog_asins = sorted(client_target_asins | set(include_distractors))

    # safety check
    missing = client_target_asins - set(catalog_asins)
    if missing:
        raise RuntimeError(
            f"[distractor_disjoint v5] client {client_id}: "
            f"{len(missing)} target ASINs missing from catalog!"
        )

    print(f"[ENV v5] WebShop client {client_id}/{client_num}: "
          f"|catalog|={len(catalog_asins)} (target={len(client_target_asins)}, "
          f"distractor={len(include_distractors)}/{D}), "
          f"|goal_idxs|={len(client_goal_idxs)}, "
          f"env_div={env_div}, keep_ratio={keep_ratio}")
    return catalog_asins, client_goal_idxs


# ============================================================
# Environment-Level Heterogeneity (WebShop): BM25-variant search index/score
# strategy key: 'bm25_variant'  ->  SimServer override: bm25_in_memory_config.
# Dispatched by paper Algorithm 2 (EnvVariantPartition). This ONE function
# serves TWO paper variants, selected by the BM25_VARIANT_POOL env var:
#   * default pool (BM25_VARIANTS_DEFAULT)  -> paper Variant 3 "BM25 Reweighting"
#                                              (Stage 3 matching/score; sweeps k1/b).
#   * BM25_VARIANT_POOL=fields_only         -> paper Variant 2 "Field-Subset Index"
#                                              (Stage 2 encoding/index; varies the
#                                               field subset, fixed k1/b).
# (config keys: bm25_reweighting = V3 default pool; field_subset_index = V2 +
#  variant_pool=fields_only.)
# See docs/heterogeneity.md (BM25 Reweighting / Field-Subset Index).
#
# Each client is deterministically assigned (by client_id) to one of N
# (fields, k1, b) BM25 configs. SimServer in that client's worker swaps
# its search backend to InMemoryBM25Searcher with that config. Catalog,
# goals, reward, val env all UNCHANGED — only the search transition T(s'|s,a)
# differs across clients.
#
# The 4 default variants are picked from probe data
# (tools/env_heterogeneity/probe_bm25_real_queries.py) as the most-divergent
# pairwise combination on real agent queries (mean Jaccard@10 ~ 0.65,
# top-1 disagreement ~ 70%).
# ============================================================
BM25_VARIANTS_DEFAULT = [
    {'name': 'full',           'fields': ['name', 'Title', 'description', 'features', 'BulletPoints'], 'k1': 1.2, 'b': 0.75},
    {'name': 'full_b=0.0',     'fields': ['name', 'Title', 'description', 'features', 'BulletPoints'], 'k1': 1.2, 'b': 0.00},
    {'name': 'full_k1=0.3',    'fields': ['name', 'Title', 'description', 'features', 'BulletPoints'], 'k1': 0.3, 'b': 0.75},
    {'name': 'full_k1=5.0',    'fields': ['name', 'Title', 'description', 'features', 'BulletPoints'], 'k1': 5.0, 'b': 0.75},
    # N>=5 extension (deterministic ordering; existing N=4 yamls keep first 4 unchanged)
    {'name': 'full_k1=0.1',    'fields': ['name', 'Title', 'description', 'features', 'BulletPoints'], 'k1': 0.1, 'b': 0.75},
    {'name': 'full_b=1.0',     'fields': ['name', 'Title', 'description', 'features', 'BulletPoints'], 'k1': 1.2, 'b': 1.00},
    {'name': 'full_k1=2.0_b=0.5', 'fields': ['name', 'Title', 'description', 'features', 'BulletPoints'], 'k1': 2.0, 'b': 0.50},
    {'name': 'full_k1=0.3_b=0.0', 'fields': ['name', 'Title', 'description', 'features', 'BulletPoints'], 'k1': 0.3, 'b': 0.00},
]

# Field-Subset Index "field-subset" variant pool (mirrors doc §1's Lucene multi-index design
# but built on top of InMemoryBM25Searcher to skip the JDK/offline-indexing
# step). Same k1/b across variants; only the field subset that goes into the
# BM25 doc text differs. Selectable via env var BM25_VARIANT_POOL=fields_only.
BM25_VARIANTS_FIELDS_ONLY = [
    {'name': 'full',          'fields': ['name', 'Title', 'description', 'features', 'BulletPoints'], 'k1': 1.2, 'b': 0.75},
    {'name': 'name',          'fields': ['name', 'Title'],                                              'k1': 1.2, 'b': 0.75},
    {'name': 'desc',          'fields': ['description'],                                                'k1': 1.2, 'b': 0.75},
    {'name': 'bullets',       'fields': ['BulletPoints'],                                               'k1': 1.2, 'b': 0.75},
    # N>=5 extension
    {'name': 'features',      'fields': ['features'],                                                   'k1': 1.2, 'b': 0.75},
    {'name': 'name_bullets',  'fields': ['name', 'Title', 'BulletPoints'],                              'k1': 1.2, 'b': 0.75},
    {'name': 'desc_features', 'fields': ['description', 'features'],                                    'k1': 1.2, 'b': 0.75},
    {'name': 'no_name',       'fields': ['description', 'features', 'BulletPoints'],                    'k1': 1.2, 'b': 0.75},
]


def _bm25_variant_partition_webshop(
    client_id: int,
    client_num: int,
    N: int = 4,
    base_seed: int = 42,
    variants: Optional[List[Dict[str, Any]]] = None,
):
    """Return this client's (fields, k1, b) BM25 config dict.

    The dict is suitable as `env_kwargs['bm25_in_memory_config']` — SimServer
    will route through InMemoryBM25Searcher with these settings.

    Variant pool selection (lowest precedence first):
      1. BM25_VARIANTS_DEFAULT  -- paper Variant 3 "BM25 Reweighting"
         (Stage 3; extreme k1/b on full fields; config key bm25_reweighting)
      2. env BM25_VARIANT_POOL=fields_only -- paper Variant 2 "Field-Subset Index"
         (Stage 2; field-subset on default k1/b; config key field_subset_index)
      3. explicit `variants=` kwarg

    Assignment is deterministic by client_id so repeated launches converge
    on the same per-client variant (important for FedAvg cross-round consistency).
    """
    if variants is None:
        pool_name = os.environ.get('BM25_VARIANT_POOL', 'default').strip().lower()
        if pool_name == 'fields_only':
            variants = BM25_VARIANTS_FIELDS_ONLY
        else:
            variants = BM25_VARIANTS_DEFAULT
    pool = list(variants)
    if N > len(pool):
        raise ValueError(
            f"_bm25_variant_partition_webshop: requested N={N} but only "
            f"{len(pool)} variants defined (extend BM25_VARIANTS_DEFAULT or pass `variants=`)"
        )
    pool = pool[:N]
    rng = np.random.RandomState(base_seed + client_id)
    chosen = pool[rng.randint(N)]
    print(f"[BM25-VARIANT] client {client_id}/{client_num}: variant={chosen['name']} "
          f"fields={chosen['fields']} k1={chosen['k1']} b={chosen['b']}")
    return {
        'fields': list(chosen['fields']),
        'k1': float(chosen['k1']),
        'b': float(chosen['b']),
        '_variant_name': chosen['name'],  # bookkeeping; SimServer ignores keys it doesn't know
    }


# ============================================================
# Environment-Level Heterogeneity (WebShop): Lookalike Injection (adversarial)
#   This is paper Variant 4 = "Lookalike Injection", which spans
#   Transition-pipeline Stages 1+3 (catalog content injection + matching/score),
#   NOT Stage 4 — variant-number does NOT equal stage-number here.
#   Dispatched by paper Algorithm 2 (EnvVariantPartition).
#   strategy key: 'lookalike_injection'  ->  SimServer override: extra_products.
#   (Unrelated to the "v4 algo" Catalog-Split impl above; that v4 is an
#    impl-revision tag for paper Variant 1, not this paper Variant 4.)
# See docs/heterogeneity.md
#
# Each client deterministically assigned (by client_id) to one of N attribute-
# attack lookalike sets (price / color / ...). SimServer in that client's worker
# injects the lookalike products via env_kwargs['extra_products'] so the agent
# is forced to specifically check that attribute to filter out fakes — different
# variants force structurally different attribute-checking policies → π* divergence.
#
# Default N=2 covers the two reward-validated attacks (audit confirmed price and
# option/color attacks both flip reward components; material is not directly
# attackable since r_attribute fuzzy-matches text fields we keep identical).
# ============================================================
LOOKALIKE_VARIANTS_DEFAULT = [
    {'name': 'v_price',       'lookalike_file': 'data/env_heterogeneity/lookalike_data/lookalike_v_price.json'},
    {'name': 'v_color',       'lookalike_file': 'data/env_heterogeneity/lookalike_data/lookalike_v_color.json'},
    # N>=3 extension
    {'name': 'v_size',        'lookalike_file': 'data/env_heterogeneity/lookalike_data/lookalike_v_size.json'},
    {'name': 'v_price_color', 'lookalike_file': 'data/env_heterogeneity/lookalike_data/lookalike_v_price_color.json'},
]

_LOOKALIKE_CACHE = {}


def _load_lookalikes(file_path):
    if file_path not in _LOOKALIKE_CACHE:
        with open(file_path) as f:
            _LOOKALIKE_CACHE[file_path] = json.load(f)
    return _LOOKALIKE_CACHE[file_path]


def _lookalike_injection_partition_webshop(
    client_id: int,
    client_num: int,
    N: int = 2,
    base_seed: int = 42,
    project_root: Optional[str] = None,
    variants: Optional[List[Dict[str, Any]]] = None,
):
    """Return this client's adversarial lookalike list (raw products).

    Implements paper Variant 4 "Lookalike Injection" (Transition-pipeline
    Stages 1+3, NOT Stage 4); strategy key 'lookalike_injection'.

    The list is suitable as `env_kwargs['extra_products']` — SimServer will
    append it to the base 1000-product catalog before BM25 indexing.

    Assignment is deterministic by client_id so repeated launches converge on
    the same per-client variant.
    """
    pool = list(variants) if variants is not None else list(LOOKALIKE_VARIANTS_DEFAULT)
    if N > len(pool):
        raise ValueError(
            f"_lookalike_injection_partition_webshop: requested N={N} "
            f"but only {len(pool)} variants defined (extend LOOKALIKE_VARIANTS_DEFAULT)"
        )
    pool = pool[:N]
    rng = np.random.RandomState(base_seed + client_id)
    chosen = pool[rng.randint(N)]
    file_path = chosen['lookalike_file']
    if not os.path.isabs(file_path):
        if project_root is None:
            project_root = os.environ.get('PROJECT_ROOT', os.getcwd())
        file_path = os.path.join(project_root, file_path)
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"[LOOKALIKE-VARIANT] {file_path} does not exist.\n"
            f"  The lookalike data ships under data/env_heterogeneity/lookalike_data/; ensure it is present."
        )
    lookalikes = _load_lookalikes(file_path)
    print(f"[LOOKALIKE-VARIANT] client {client_id}/{client_num}: variant={chosen['name']} "
          f"|lookalikes|={len(lookalikes)} file={os.path.basename(file_path)}")
    return lookalikes


# ============================================================
# Environment-Level Heterogeneity (WebShop): search-engine TYPE swap
#   This is paper Variant 5 = "Rank Wrapper" (Transition-pipeline Stage 4,
#   rendering/ranking); dispatched by paper Algorithm 2 (EnvVariantPartition).
#   strategy key: 'rank_wrapper'  ->  SimServer override: search_engine_variant.
#   (Unrelated to the "v5 algo" Catalog-Split impl above; that v5 is an
#    impl-revision tag for paper Variant 1, not this paper Variant 5.)
# See docs/heterogeneity.md (search backend axis).
#
# Each variant breaks a different baseline-policy assumption:
#   v_bm25_default   -- control (BM25 ranking trustable)
#   v_shuffled_topk  -- BM25 ranks top-50 then shuffles → "click position 1" fails
#   v_inverted_topk  -- BM25 returns top-K reversed → forces "skip front, click later"
#   v_partial_random -- 50% queries return random → forces "verify each result"
#
# All 4 preserve reward gradient (target reachable in candidate set), avoiding
# the v_random pitfall where 25% of clients can never get reward signal.
# ============================================================
SEARCH_ENGINE_VARIANTS_DEFAULT = [
    {'name': 'v_bm25_default',  'type': 'bm25_default'},
    {'name': 'v_shuffled_topk', 'type': 'bm25_shuffle', 'shuffle_k': 50},
    {'name': 'v_inverted_topk', 'type': 'bm25_invert'},
    {'name': 'v_partial_random','type': 'bm25_partial', 'random_prob': 0.5},
]


def _rank_wrapper_partition_webshop(
    client_id: int,
    client_num: int,
    N: int = 4,
    base_seed: int = 42,
    variants: Optional[List[Dict[str, Any]]] = None,
):
    """Return this client's search-engine variant config.

    Implements paper Variant 5 "Rank Wrapper" (Transition-pipeline Stage 4);
    strategy key 'rank_wrapper'.

    Result is a dict suitable as `env_kwargs['search_engine_variant']` —
    SimServer routes through `init_search_engine(search_engine_variant=...)`
    which builds an InMemoryBM25 base and wraps it per the type field.
    """
    pool = list(variants) if variants is not None else list(SEARCH_ENGINE_VARIANTS_DEFAULT)
    if N > len(pool):
        raise ValueError(
            f"_rank_wrapper_partition_webshop: requested N={N} "
            f"but only {len(pool)} variants defined"
        )
    pool = pool[:N]
    rng = np.random.RandomState(base_seed + client_id)
    chosen = pool[rng.randint(N)]
    print(f"[RANK-WRAPPER] client {client_id}/{client_num}: variant={chosen['name']} "
          f"type={chosen['type']}")
    out = {k: v for k, v in chosen.items() if k != 'name'}
    # per-client unique seed so shuffle/random differ across clients of the same variant
    out['seed'] = base_seed + client_id
    return out


def visualize_webshop_env_partition(
    products: List[Dict[str, Any]],
    ins: Dict[str, Dict[str, Any]],
    client_num: int,
    env_div: float,
    keep_ratio: float,
    holdout_distractor_asins: Optional[List[str]] = None,
    save_path: str = "webshop_env_partition.png",
    base_seed: int = 42,
):
    """Visualize per-client catalog assignment with env-level heterogeneity.

    Generates a 4-panel figure:
      (a) Pairwise Jaccard matrix (client x client)
      (b) Per-client catalog size histogram
      (c) Per-distractor inclusion frequency (how many clients each d goes to)
      (d) Catalogs as a binary matrix (clients x distractors), sorted for clarity
    """
    import matplotlib.pyplot as plt

    holdout = set(holdout_distractor_asins or [])
    target_asins = {asin for asin, entry in ins.items() if entry.get('instruction')}
    all_asins = {p['asin'] for p in products}
    distractor_pool = sorted((all_asins - target_asins) - holdout)
    D = len(distractor_pool)
    distractor_idx = {a: i for i, a in enumerate(distractor_pool)}

    # Compute all client catalogs
    catalogs = []
    binary_matrix = np.zeros((client_num, D), dtype=np.uint8)
    for k in range(client_num):
        cat = _distractor_disjoint_partition_webshop(
            products=products, ins=ins,
            client_id=k, client_num=client_num,
            env_div=env_div, keep_ratio=keep_ratio,
            holdout_distractor_asins=list(holdout),
            base_seed=base_seed,
        )
        catalogs.append(set(cat))
        for asin in cat:
            if asin in distractor_idx:
                binary_matrix[k, distractor_idx[asin]] = 1

    # Pairwise Jaccard
    M = np.eye(client_num)
    for i in range(client_num):
        for j in range(i + 1, client_num):
            inter = len(catalogs[i] & catalogs[j])
            union = len(catalogs[i] | catalogs[j])
            jac = inter / max(union, 1)
            M[i, j] = M[j, i] = jac
    off_diag = M[~np.eye(client_num, dtype=bool)]

    # Per-distractor inclusion frequency
    distractor_pop = binary_matrix.sum(axis=0)  # shape (D,)
    sort_idx = np.argsort(-distractor_pop)
    binary_sorted = binary_matrix[:, sort_idx]
    pop_sorted = distractor_pop[sort_idx]

    # 4 panels
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # (a) Jaccard
    im0 = axes[0, 0].imshow(M, cmap='viridis', vmin=0, vmax=1, aspect='equal')
    axes[0, 0].set_title(
        f'(a) Pairwise Jaccard  (env_div={env_div}, keep_ratio={keep_ratio})\n'
        f'mean off-diag = {off_diag.mean():.3f} ± {off_diag.std():.3f}'
    )
    axes[0, 0].set_xlabel('client j'); axes[0, 0].set_ylabel('client i')
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046)

    # (b) Catalog sizes
    sizes = np.array([len(c) for c in catalogs])
    axes[0, 1].hist(sizes, bins=25, edgecolor='k', color='steelblue')
    axes[0, 1].axvline(sizes.mean(), color='red', linestyle='--',
                       label=f'mean = {sizes.mean():.0f}')
    axes[0, 1].set_title(
        f'(b) Per-client catalog size  '
        f'(target={len(target_asins)}, distractor_pool={D})\n'
        f'size: {sizes.mean():.0f} ± {sizes.std():.1f}'
    )
    axes[0, 1].set_xlabel('|catalog_k|')
    axes[0, 1].set_ylabel('count')
    axes[0, 1].legend()

    # (c) Distractor popularity
    axes[1, 0].plot(pop_sorted, linewidth=1, color='darkorange')
    axes[1, 0].axhline(client_num * keep_ratio, color='gray', linestyle=':',
                       label=f'expected (= keep_ratio*N = {client_num*keep_ratio:.0f})')
    axes[1, 0].set_title(
        f'(c) Per-distractor inclusion frequency\n'
        f'(distractor d ranked by # clients including it)'
    )
    axes[1, 0].set_xlabel('distractor rank')
    axes[1, 0].set_ylabel(f'# clients including (max = {client_num})')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    # (d) Binary catalog matrix (clients x distractors, sorted)
    im3 = axes[1, 1].imshow(binary_sorted, cmap='Greys', aspect='auto',
                            interpolation='nearest')
    axes[1, 1].set_title(
        f'(d) Binary catalog matrix  ({client_num} clients x {D} distractors)\n'
        f'distractors sorted by global popularity'
    )
    axes[1, 1].set_xlabel('distractor (sorted by popularity)')
    axes[1, 1].set_ylabel('client')
    plt.colorbar(im3, ax=axes[1, 1], fraction=0.046, ticks=[0, 1])

    plt.suptitle(
        f'WebShop env-level partition  (N={client_num}, env_div={env_div}, '
        f'keep_ratio={keep_ratio}, base_seed={base_seed})',
        fontsize=14, y=1.00,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"[VIZ] saved {save_path}  "
          f"(Jaccard={off_diag.mean():.3f}, size={sizes.mean():.0f}±{sizes.std():.1f})")
    return {
        'jaccard_mean': float(off_diag.mean()),
        'jaccard_std': float(off_diag.std()),
        'catalog_size_mean': float(sizes.mean()),
        'catalog_size_std': float(sizes.std()),
    }


# ============================================================
# Env-level heterogeneity: AlfWorld scene-disjoint partition
#   See docs/heterogeneity.md for design
# ============================================================
def _parse_alfworld_path(path: str):
    """Extract (spec, scene, trial_id) from a game.tw-pddl absolute path.

    Path format: .../<task_type>-<obj>-<recep1>-<recep2>-<scene>/<trial_T...>/game.tw-pddl
    Returns (spec, scene, trial) or (None, None, None) on parse failure.
    """
    parts = path.rstrip('/').split('/')
    if len(parts) < 3:
        return None, None, None
    trial = parts[-2]
    task_dir = parts[-3]
    sp = task_dir.rsplit('-', 1)
    if len(sp) != 2:
        return None, None, None
    return sp[0], sp[1], trial


def _env_disjoint_partition_alfworld(
    data: List[str],            # game_files (abs paths)
    client_id: int,
    client_num: int,
    min_samples_per_client: int = 100,    # Interface alignment; not used here
    env_div: float = 0.7,
    fallback: str = 'skip',     # 'skip' | 'shared' | 'trial-only'
    holdout_scenes: Optional[List[str]] = None,
    base_seed: int = 42,
    **kwargs,
) -> List[str]:
    """Env-level partition for AlfWorld.

    For each spec (= <task_type>-<obj>-<recep1>-<recep2>):
      * unique_scenes == 1  → handled by `fallback`:
          'skip'        → drop this spec entirely (per-user request: nothing
                          to do scene-wise; pure env-only heterogeneity)
          'shared'      → all clients get all this spec's trials (no env
                          heterogeneity contribution)
          'trial-only'  → trial-axis split via top-k (D2 weak heterogeneity)
      * unique_scenes >= 2 → top-k (scene, trial) selection per client:
          n_per_client = min(|instances|, max(1, ceil(|instances|/N · (1+env_div))))
          e_k[i] = (1-env_div) · u[i] + env_div · v_k[i]
          chosen = argsort(e_k)[:n_per_client]
        u shared across all clients (per-spec seed); v_k per (client, spec).

    Args:
      data: list of absolute paths to game.tw-pddl files (already filtered upstream)
      client_id, client_num: federated args
      env_div: heterogeneity strength in [0, 1]
      fallback: how to treat single-scene specs
      holdout_scenes: scene IDs reserved for OOD eval (excluded entirely)
      base_seed: RNG seed (shared `proto_rng`s; per-client offset for `client_rng`s)

    Returns:
      list of game.tw-pddl paths assigned to this client.
    """
    holdout = set(holdout_scenes or [])
    valid_fallbacks = {'skip', 'shared', 'trial-only'}
    if fallback not in valid_fallbacks:
        raise ValueError(f"fallback must be one of {valid_fallbacks}, got '{fallback}'")

    # Step 1: parse & bucket by spec; drop trials in holdout scenes
    spec_to_instances = defaultdict(list)
    n_dropped_holdout = 0
    n_unparsed = 0
    for path in data:
        spec, scene, trial = _parse_alfworld_path(path)
        if spec is None:
            n_unparsed += 1
            continue
        if scene in holdout:
            n_dropped_holdout += 1
            continue
        spec_to_instances[spec].append((path, scene, trial))

    if n_unparsed > 0:
        print(f"[ENV-AlfWorld] WARNING: {n_unparsed} paths failed to parse")
    if n_dropped_holdout > 0:
        print(f"[ENV-AlfWorld] dropped {n_dropped_holdout} trials in holdout scenes "
              f"({len(holdout)} scenes: {sorted(holdout)})")

    # Step 2: per-spec selection
    client_paths: List[str] = []
    n_specs_total = len(spec_to_instances)
    n_specs_skipped = 0
    n_specs_fallback_shared = 0
    n_specs_fallback_trial = 0
    n_specs_partitioned = 0

    for spec in sorted(spec_to_instances.keys()):
        instances = sorted(spec_to_instances[spec], key=lambda x: (x[1], x[2]))
        n_avail = len(instances)
        unique_scenes = set(x[1] for x in instances)

        # Single-scene spec → fallback path
        if len(unique_scenes) <= 1:
            if fallback == 'skip':
                n_specs_skipped += 1
                continue
            if fallback == 'shared':
                client_paths.extend(x[0] for x in instances)
                n_specs_fallback_shared += 1
                continue
            if fallback == 'trial-only':
                # Same shared+indep top-k mechanism but on trial axis only
                if n_avail == 1:
                    client_paths.append(instances[0][0])
                else:
                    n_per_client = max(1, int(np.ceil(n_avail / client_num * (1 + env_div))))
                    n_per_client = min(n_per_client, n_avail)
                    spec_seed = (base_seed + (_spec_hash(spec) & 0x7fffffff)) & 0x7fffffff
                    proto_rng_spec = np.random.RandomState(spec_seed)
                    u = proto_rng_spec.random(n_avail)
                    client_seed = (base_seed + 1000 * int(client_id) + (_spec_hash(spec) & 0xffff)) & 0x7fffffff
                    client_rng_spec = np.random.RandomState(client_seed)
                    v = client_rng_spec.random(n_avail)
                    e = (1.0 - env_div) * u + env_div * v
                    chosen_idx = np.argsort(e)[:n_per_client]
                    client_paths.extend(instances[i][0] for i in chosen_idx)
                n_specs_fallback_trial += 1
                continue

        # Multi-scene spec → top-k per-spec
        n_per_client = max(1, int(np.ceil(n_avail / client_num * (1 + env_div))))
        n_per_client = min(n_per_client, n_avail)
        spec_seed = (base_seed + (_spec_hash(spec) & 0x7fffffff)) & 0x7fffffff
        proto_rng_spec = np.random.RandomState(spec_seed)
        u = proto_rng_spec.random(n_avail)
        client_seed = (base_seed + 1000 * int(client_id) + (_spec_hash(spec) & 0xffff)) & 0x7fffffff
        client_rng_spec = np.random.RandomState(client_seed)
        v = client_rng_spec.random(n_avail)
        e = (1.0 - env_div) * u + env_div * v
        chosen_idx = np.argsort(e)[:n_per_client]
        client_paths.extend(instances[i][0] for i in chosen_idx)
        n_specs_partitioned += 1

    print(f"[ENV-AlfWorld] client {client_id}/{client_num}: "
          f"|game_files|={len(client_paths)} from {n_specs_partitioned} multi-scene specs "
          f"(out of {n_specs_total}); single-scene specs: skipped={n_specs_skipped}, "
          f"shared={n_specs_fallback_shared}, trial-only={n_specs_fallback_trial}; "
          f"env_div={env_div}, fallback={fallback}")
    return client_paths


def visualize_alfworld_env_partition(
    game_files: List[str],
    client_num: int,
    env_div: float,
    fallback: str = 'skip',
    holdout_scenes: Optional[List[str]] = None,
    save_path: str = "alfworld_env_partition.png",
    base_seed: int = 42,
):
    """Generate a 4-panel visualization of AlfWorld env-level partition.

    Panels:
      (a) Pairwise Jaccard matrix (client x client) on game_file sets
      (b) Per-client |game_files| histogram
      (c) Spec coverage matrix (clients x specs, # trials per cell)
      (d) Scene coverage matrix (clients x scenes, # trials per cell)
    """
    import matplotlib.pyplot as plt

    holdout = set(holdout_scenes or [])

    # Build per-client game_files via the partition function (deterministic)
    client_paths_list = []
    for k in range(client_num):
        paths = _env_disjoint_partition_alfworld(
            data=game_files, client_id=k, client_num=client_num,
            env_div=env_div, fallback=fallback,
            holdout_scenes=list(holdout), base_seed=base_seed,
        )
        client_paths_list.append(set(paths))

    # Collect all specs / scenes that show up in any client's slice
    all_specs = set()
    all_scenes = set()
    for paths in client_paths_list:
        for p in paths:
            spec, scene, _ = _parse_alfworld_path(p)
            if spec is not None:
                all_specs.add(spec)
                all_scenes.add(scene)
    spec_list = sorted(all_specs)
    scene_list = sorted(all_scenes, key=lambda s: int(s) if s.isdigit() else 0)
    spec_idx = {s: i for i, s in enumerate(spec_list)}
    scene_idx = {s: i for i, s in enumerate(scene_list)}

    # Spec coverage and scene coverage matrices
    cov_spec = np.zeros((client_num, len(spec_list)), dtype=np.uint16)
    cov_scene = np.zeros((client_num, len(scene_list)), dtype=np.uint16)
    for k, paths in enumerate(client_paths_list):
        for p in paths:
            spec, scene, _ = _parse_alfworld_path(p)
            if spec in spec_idx:
                cov_spec[k, spec_idx[spec]] += 1
            if scene in scene_idx:
                cov_scene[k, scene_idx[scene]] += 1

    # Pairwise Jaccard
    M = np.eye(client_num)
    for i in range(client_num):
        for j in range(i + 1, client_num):
            inter = len(client_paths_list[i] & client_paths_list[j])
            union = len(client_paths_list[i] | client_paths_list[j])
            M[i, j] = M[j, i] = inter / max(union, 1)
    off_diag = M[~np.eye(client_num, dtype=bool)]

    sizes = np.array([len(s) for s in client_paths_list])

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # (a) Jaccard
    im0 = axes[0, 0].imshow(M, cmap='viridis', vmin=0, vmax=1, aspect='equal')
    axes[0, 0].set_title(
        f'(a) Pairwise Jaccard  (env_div={env_div}, fallback={fallback})\n'
        f'mean off-diag = {off_diag.mean():.3f} ± {off_diag.std():.3f}'
    )
    axes[0, 0].set_xlabel('client j'); axes[0, 0].set_ylabel('client i')
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046)

    # (b) Sizes
    axes[0, 1].hist(sizes, bins=25, edgecolor='k', color='steelblue')
    axes[0, 1].axvline(sizes.mean(), color='red', linestyle='--',
                       label=f'mean = {sizes.mean():.0f}')
    axes[0, 1].set_title(
        f'(b) Per-client |game_files|\n'
        f'size: {sizes.mean():.0f} ± {sizes.std():.1f}, '
        f'min={sizes.min()}, max={sizes.max()}'
    )
    axes[0, 1].set_xlabel('|client_games|')
    axes[0, 1].set_ylabel('count')
    axes[0, 1].legend()

    # (c) Spec coverage (sorted by total popularity)
    spec_pop = cov_spec.sum(axis=0)
    spec_sort = np.argsort(-spec_pop)
    im2 = axes[1, 0].imshow(cov_spec[:, spec_sort], cmap='Blues', aspect='auto',
                            interpolation='nearest')
    axes[1, 0].set_title(
        f'(c) Spec coverage  ({client_num} clients × {len(spec_list)} specs, '
        f'sorted by total trials per spec)'
    )
    axes[1, 0].set_xlabel('spec rank (most-covered → least)')
    axes[1, 0].set_ylabel('client')
    plt.colorbar(im2, ax=axes[1, 0], fraction=0.046, label='# trials')

    # (d) Scene coverage (sorted by FloorPlan id)
    im3 = axes[1, 1].imshow(cov_scene, cmap='Greens', aspect='auto',
                            interpolation='nearest')
    axes[1, 1].set_title(
        f'(d) Scene coverage  ({client_num} clients × {len(scene_list)} FloorPlans)'
    )
    axes[1, 1].set_xlabel('FloorPlan idx (sorted)')
    axes[1, 1].set_ylabel('client')
    plt.colorbar(im3, ax=axes[1, 1], fraction=0.046, label='# trials')

    plt.suptitle(
        f'AlfWorld env-level partition  (N={client_num}, env_div={env_div}, '
        f'fallback={fallback}, holdout_scenes={sorted(holdout)}, '
        f'base_seed={base_seed})',
        fontsize=13, y=0.995,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"[VIZ] saved {save_path}  "
          f"(Jaccard={off_diag.mean():.3f}, |game_files|={sizes.mean():.0f}±{sizes.std():.1f}, "
          f"specs={len(spec_list)}, scenes={len(scene_list)})")
    return {
        'jaccard_mean': float(off_diag.mean()),
        'jaccard_std': float(off_diag.std()),
        'size_mean': float(sizes.mean()),
        'size_std': float(sizes.std()),
        'n_specs_used': len(spec_list),
        'n_scenes_used': len(scene_list),
    }


def visualize_alfworld_client_category_distribution(
    client_games_slice: List[str],
    client_id: int,
    client_num: int,
    tau: float,
    save_path: str
) -> None:
    """
    Generate a category-distribution pie chart for a single ALFWorld client.

    Args:
        client_games_slice: list of game files assigned to the current client.
        client_id: ID of the current client.
        client_num: total number of clients.
        tau: heterogeneity parameter.
        save_path: output path.
    """
    import json
    import os
    import matplotlib.pyplot as plt

    # Tally the current client's category distribution.
    category_counts = {}
    for game_file in client_games_slice:
        try:
            json_path = game_file.replace("game.tw-pddl", "traj_data.json")
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    traj_data = json.load(f)
                task_type = traj_data.get('task_type', 'unknown')
                # Map the numeric task type to its string name.
                if task_type == 1:
                    category = "pick_and_place_simple"
                elif task_type == 2:
                    category = "look_at_obj_in_light"
                elif task_type == 3:
                    category = "pick_clean_then_place_in_recep"
                elif task_type == 4:
                    category = "pick_heat_then_place_in_recep"
                elif task_type == 5:
                    category = "pick_cool_then_place_in_recep"
                elif task_type == 6:
                    category = "pick_two_obj_and_place"
                else:
                    category = f"task_type_{task_type}"
            else:
                category = 'unknown'
        except Exception:
            category = 'unknown'
        
        category_counts[category] = category_counts.get(category, 0) + 1

    # Render the pie chart.
    plt.figure(figsize=(10, 8))
    categories = list(category_counts.keys())
    counts = list(category_counts.values())

    plt.pie(counts, labels=categories, autopct='%1.1f%%', startangle=90)
    plt.title(f'ALFWorld Category Distribution - Client {client_id}/{client_num} (tau={tau})')
    plt.axis('equal')

    # Save the figure.
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Print the category distribution.
    total_games = len(client_games_slice)
    print(f"[ALFWorld Preference Partition] Client {client_id}/{client_num} (tau={tau}):")
    print(f"  Category distribution ({total_games} total games):")
    for category, count in sorted(category_counts.items()):
        percentage = (count / total_games) * 100
        print(f"    {category}: {count} ({percentage:.1f}%)")


def visualize_webshop_client_category_distribution(
    client_goals_slice: List[Dict[str, Any]],
    client_id: int,
    client_num: int,
    tau: float = None,
    save_path: str = None,
    **kwargs
) -> None:
    """
    Generate a category-distribution pie chart for a single WebShop client.

    Args:
        client_goals_slice: list of goal items assigned to the current client.
        client_id: ID of the current client.
        client_num: total number of clients.
        tau: heterogeneity parameter (if None, read from kwargs).
        save_path: output path.
        **kwargs: additional parameters, including tau.
    """
    import matplotlib.pyplot as plt

    # Read the tau parameter from kwargs if it was not passed directly.
    if tau is None:
        tau = kwargs.get('tau', 0.3)

    # Tally the current client's category distribution.
    category_counts = {}
    for goal in client_goals_slice:
        category = goal.get('category', 'unknown')
        category_counts[category] = category_counts.get(category, 0) + 1

    # Render the pie chart.
    plt.figure(figsize=(10, 8))
    categories = list(category_counts.keys())
    counts = list(category_counts.values())

    plt.pie(counts, labels=categories, autopct='%1.1f%%', startangle=90)
    plt.title(f'WebShop Category Distribution - Client {client_id}/{client_num} (tau={tau})')
    plt.axis('equal')

    # Save the figure.
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Print the category distribution.
    total_goals = len(client_goals_slice)
    print(f"[WebShop Preference Partition] Client {client_id}/{client_num} (tau={tau}):")
    print(f"  Category distribution ({total_goals} total goals):")
    for category, count in sorted(category_counts.items()):
        percentage = (count / total_goals) * 100
        print(f"    {category}: {count} ({percentage:.1f}%)")


def visualize_client_category_distribution(
    data: List[Any],
    client_num: int,
    min_samples_per_client: int,
    strategy: str = 'preference',
    category_key: str = 'category',
    start_idx: int = 0,
    tau: float = 0.3,
    omega: Optional[float] = None,
    save_path: Optional[str] = None,
    **kwargs
) -> None:
    """
    Visualize the per-client category distribution.

    For preference strategy: Dirichlet PreferencePartition (see docs/heterogeneity.md).
    Pass `omega` (preferred); `tau` accepted as legacy alias.
    """
    if strategy == 'preference' and omega is None:
        omega = tau
    # Compute the actually usable training data (everything after start_idx).
    total_train_data = data[start_idx:]

    # Collect all categories.
    all_categories = set()
    for item in total_train_data:
        category = item.get(category_key, 'unknown')
        all_categories.add(category)
    all_categories = sorted(list(all_categories))

    # Partition data for each client and tally its category distribution.
    client_category_counts = {}
    for client_id in range(client_num):
        # Get the current client's data.
        if strategy == 'uniform':
            result = uniform_partition(data, client_id, client_num, min_samples_per_client, start_idx)
            client_data, _, _ = result
        elif strategy == 'preference':
            client_data = preference_partition(data, client_id, client_num, min_samples_per_client,
                                           category_key=category_key, start_idx=start_idx, omega=omega, **kwargs)
        elif strategy == 'coverage':
            client_data = coverage_partition(data, client_id, client_num, min_samples_per_client, 
                                           start_idx=start_idx, **kwargs)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # Tally the current client's category distribution.
        category_counts = defaultdict(int)
        for item in client_data:
            category = item.get(category_key, 'unknown')
            category_counts[category] += 1

        client_category_counts[client_id] = category_counts

    # Build the visualization.
    plt.figure(figsize=(15, 8))

    # Prepare the data.
    clients = list(range(client_num))
    # Use a matplotlib colormap.
    import matplotlib.cm as cm
    category_colors = cm.get_cmap('tab20')(np.linspace(0, 1, len(all_categories)))

    # Build a stacked bar chart.
    bottom = np.zeros(client_num)

    for i, category in enumerate(all_categories):
        counts = [client_category_counts[client_id].get(category, 0) for client_id in clients]
        plt.bar(clients, counts, bottom=bottom, label=category, color=category_colors[i], alpha=0.8)
        bottom += counts

    # Configure the figure.
    plt.xlabel('Client ID', fontsize=24, fontweight='bold')
    plt.ylabel('Number of Samples', fontsize=24, fontweight='bold')
    # Remove title per user request
    plt.title("")
    plt.xlim(0, 100)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=16)
    plt.grid(True, alpha=0.3)
    # Configure x-axis ticks: label every 10th client ID, and make sure the last
    # value is included.
    x_ticks = list(range(0, client_num, 10))
    if client_num - 1 not in x_ticks:  # add the last client ID if it is missing
        x_ticks.append(client_num - 1)
    # x_ticks.append(100)
    plt.xticks(x_ticks, fontsize=16)  # further enlarge the x-axis tick labels
    plt.yticks(fontsize=16)  # enlarge the y-axis tick labels

    # Total-count annotations removed to avoid overlap.
    # for i, client_id in enumerate(clients):
    #     total = sum(client_category_counts[client_id].values())
    #     plt.text(i, total + max(bottom) * 0.01, str(total), ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)  # add headroom so the title sits farther from the plot

    # Save or display the figure.
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        # Also save as PDF
        pdf_path = save_path.replace('.png', '.pdf')
        plt.savefig(pdf_path, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
        print(f"PDF saved to: {pdf_path}")
    else:
        plt.show()

    # Print summary statistics.
    print(f"\n{'='*80}")
    print(f"Category distribution statistics (Strategy: {strategy}, Tau: {tau})")
    print(f"{'='*80}")

    # Global category statistics.
    global_category_counts = defaultdict(int)
    for item in total_train_data:
        category = item.get(category_key, 'unknown')
        global_category_counts[category] += 1

    print(f"\nGlobal category distribution:")
    for category in all_categories:
        count = global_category_counts[category]
        percentage = count / len(total_train_data) * 100
        print(f"  {category}: {count} ({percentage:.1f}%)")

    # Per-client statistics.
    print(f"\nPer-client category distribution:")
    for client_id in range(min(5, client_num)):  # only show the first 5 clients
        print(f"\nClient {client_id}:")
        total = sum(client_category_counts[client_id].values())
        for category in all_categories:
            count = client_category_counts[client_id].get(category, 0)
            if count > 0:
                percentage = count / total * 100
                print(f"  {category}: {count} ({percentage:.1f}%)")

    if client_num > 5:
        print(f"\n... (showing the first 5 clients out of {client_num} total)")


def _extract_category_from_item(item, data_type: str, category_key: str = 'category') -> str:
    """
    Extract the category label from a data item (internal helper).

    Args:
        item: a data item (either a dict or a file path).
        data_type: data type ('generic', 'webshop', 'alfworld').
        category_key: name of the category field in the dict.

    Returns:
        The category as a string.
    """
    if data_type == 'alfworld':
        # ALFWorld-specific handling: derive the task type from the file path.
        # Path format: .../train/look_at_obj_in_light-AlarmClock-None-DeskLamp-301/trial_.../game.tw-pddl
        try:
            import os
            # Derive the task type from the file path.
            # e.g. /path/to/look_at_obj_in_light-AlarmClock-None-DeskLamp-301/trial_.../game.tw-pddl
            path_parts = item.split(os.sep)

            # Find the directory name that encodes the task type.
            for part in path_parts:
                if part.startswith(('pick_and_place_simple', 'look_at_obj_in_light',
                                  'pick_clean_then_place_in_recep', 'pick_heat_then_place_in_recep',
                                  'pick_cool_then_place_in_recep', 'pick_two_obj_and_place')):
                    # Extract the task type (the part before the first '-').
                    task_type = part.split('-')[0]
                    return task_type

            # If not found, fall back to reading the JSON file.
            json_path = item.replace("game.tw-pddl", "traj_data.json")
            if os.path.exists(json_path):
                import json
                with open(json_path, 'r') as f:
                    traj_data = json.load(f)
                task_type = traj_data.get('task_type', 'unknown')
                # Map the numeric task type to its string name.
                task_type_map = {
                    1: "pick_and_place_simple",
                    2: "look_at_obj_in_light",
                    3: "pick_clean_then_place_in_recep",
                    4: "pick_heat_then_place_in_recep",
                    5: "pick_cool_then_place_in_recep",
                    6: "pick_two_obj_and_place"
                }
                return task_type_map.get(task_type, f"task_type_{task_type}")
            else:
                return 'unknown'
        except Exception:
            return 'unknown'
    else:
        # Generic handling: read the category from the item's dict.
        return item.get(category_key, 'unknown')


def visualize_all_clients_category_distribution(
    data: List[Any],
    client_num: int,
    min_samples_per_client: int,
    strategy: str = 'preference',
    category_key: str = 'category',
    start_idx: int = 0,
    tau: float = 0.3,
    save_path: Optional[str] = None,
    data_type: str = 'generic',
    **kwargs
) -> None:
    """
    Visualize the category distribution across all clients (stacked bar chart).

    Args:
        data: list of items to partition.
        client_num: total number of clients.
        min_samples_per_client: minimum number of samples each client must receive.
        strategy: partition strategy ('uniform', 'preference', 'coverage').
        category_key: name of the field holding the category label.
        start_idx: starting index into `data`.
        tau: fluctuation-strength parameter (only relevant for the category
            strategy).
        save_path: output path for the figure; if None, the figure is displayed.
        data_type: data type ('generic', 'webshop', 'alfworld').
        **kwargs: additional parameters.
    """
    # Compute the actually usable training data (everything after start_idx).
    total_train_data = data[start_idx:]

    # Collect all categories.
    all_categories = set()
    for item in total_train_data:
        category = _extract_category_from_item(item, data_type, category_key)
        all_categories.add(category)

    all_categories = sorted(list(all_categories))

    # Partition data for each client and tally its category distribution.
    client_category_counts = {}
    for client_id in range(client_num):
        # Get the current client's data.
        if strategy == 'uniform':
            result = uniform_partition(data, client_id, client_num, min_samples_per_client, start_idx)
            client_data, _, _ = result
        elif strategy == 'preference':
            client_data = preference_partition(data, client_id, client_num, min_samples_per_client, 
                                           category_key=category_key, start_idx=start_idx, tau=tau, data_type=data_type, **kwargs)
        elif strategy == 'coverage':
            client_data = coverage_partition(data, client_id, client_num, min_samples_per_client, 
                                           start_idx=start_idx, **kwargs)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # Tally the current client's category distribution.
        category_counts = {}
        for item in client_data:
            category = _extract_category_from_item(item, data_type, category_key)
            category_counts[category] = category_counts.get(category, 0) + 1

        client_category_counts[client_id] = category_counts

    # Build the visualization.
    import matplotlib.pyplot as plt
    import numpy as np

    plt.figure(figsize=(20, 10))

    # Prepare the data.
    clients = list(range(client_num))
    # Use a matplotlib colormap.
    import matplotlib.cm as cm
    category_colors = cm.get_cmap('tab20')(np.linspace(0, 1, len(all_categories)))

    # Build a stacked bar chart.
    bottom = np.zeros(client_num)

    for i, category in enumerate(all_categories):
        counts = [client_category_counts[client_id].get(category, 0) for client_id in clients]
        plt.bar(clients, counts, bottom=bottom, label=category, color=category_colors[i], alpha=0.8)
        bottom += counts

    # Configure the figure.
    plt.xlabel('Client ID', fontsize=14)
    plt.ylabel('Number of Samples', fontsize=14)
    plt.title(f'Category Distribution Across All Clients\nStrategy: {strategy}, Tau: {tau}, Min Samples: {min_samples_per_client}, Data Type: {data_type}',
              fontsize=16, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(clients[::max(1, client_num//20)])  # show only a subset of client IDs to avoid crowding

    # Add total-count annotations.
    for i, client_id in enumerate(clients):
        total = sum(client_category_counts[client_id].values())
        plt.text(i, total + max(bottom) * 0.01, str(total), ha='center', va='bottom', fontweight='bold', fontsize=8)

    plt.tight_layout()

    # Save or display the figure.
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"All clients category distribution saved to: {save_path}")
    else:
        plt.show()

    plt.close()

    # Print summary statistics.
    print(f"\n{'='*80}")
    print(f"All-client category distribution statistics (Strategy: {strategy}, Tau: {tau}, Data Type: {data_type})")
    print(f"{'='*80}")

    # Global category statistics.
    global_category_counts = {}
    for item in total_train_data:
        category = _extract_category_from_item(item, data_type, category_key)
        global_category_counts[category] = global_category_counts.get(category, 0) + 1

    print(f"\nGlobal category distribution:")
    for category in all_categories:
        count = global_category_counts.get(category, 0)
        percentage = count / len(total_train_data) * 100
        print(f"  {category}: {count} ({percentage:.1f}%)")

    # Per-client statistics (only the first 5).
    print(f"\nPer-client category distribution (first 5):")
    for client_id in range(min(5, client_num)):
        print(f"\nClient {client_id}:")
        total = sum(client_category_counts[client_id].values())
        for category in all_categories:
            count = client_category_counts[client_id].get(category, 0)
            if count > 0:
                percentage = count / total * 100
                print(f"  {category}: {count} ({percentage:.1f}%)")

    if client_num > 5:
        print(f"\n... (showing the first 5 clients out of {client_num} total)")


def visualize_coverage_normal_distribution(
    data: List[Any],
    client_num: int,
    min_samples_per_client: int,
    start_idx: int = 0,
    overlap_ratio: float = 1.3,
    dispersion_s: float = 15.0,
    max_samples_per_client: Optional[int] = None,
    save_path: Optional[str] = None,
    **kwargs
) -> None:
    """
    Dedicated visualization of the Beta-distribution behavior of the coverage
    strategy.

    Args:
        data: list of items to partition.
        client_num: total number of clients.
        min_samples_per_client: minimum number of samples each client must receive.
        start_idx: starting index into `data`.
        overlap_ratio: overlap coefficient (total assignments / number of samples).
        dispersion_s: spread parameter of the Beta distribution.
        max_samples_per_client: maximum samples per client; auto-computed if None.
        save_path: output path for the figure; if None, the figure is displayed.
        **kwargs: additional parameters.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Compute the actually usable training data (everything after start_idx).
    total_train_data = data[start_idx:]
    total_size = len(total_train_data)

    # Use the same parameter settings as coverage_partition.
    center = 500
    low = min_samples_per_client
    high = 1000


    if 'dispersion_s' not in kwargs:
        # Fall back to size_std for backward compatibility if dispersion_s is absent.
        if 'size_std' in kwargs:
            dispersion_s = kwargs['size_std']
        else:
            raise ValueError("Missing required 'dispersion_s' parameter in coverage_partition kwargs.")
    else:
        dispersion_s = kwargs['dispersion_s']
    print(f"Visualizing coverage partition with Beta distribution method...")
    print(f"Parameters: center={center}, dispersion_s={dispersion_s}, low={low}, high={high}")

    # A fixed seed ensures every client computes the same allocation.
    rng = np.random.default_rng(42)

    # Compute the default r value.
    r = default_r(total_size, client_num, low, center, high)
    target_sum = int(round(r * total_size))

    print(f"Default r = {r:.6f} (feasible range [{client_num*low/total_size:.3f}, {client_num*high/total_size:.3f}])")
    print(f"Target sum = {target_sum}")

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

    # Tally the actual sample counts.
    client_sample_counts = [len(s) for s in client_sets]

    # Compute summary statistics.
    stats = summarize_overlap(client_sets)
    
    print("Summary:")
    print({k_: (round(v, 4) if isinstance(v, float) else v) for k_,v in stats.items() if k_ != "client_sizes"})
    print(f"Total assignments = {sum(len(s) for s in client_sets)}, expected = {target_sum}")
    print(f"Each sample replication counts (min, mean, max): {k.min()}, {k.mean():.3f}, {k.max()}")

    # Build the visualization - following the example format.
    plt.figure()
    plt.hist(stats["client_sizes"], bins=12)
    # Remove title per user request
    plt.title("")
    plt.xlabel("samples per client")
    plt.ylabel("frequency")
    if save_path:
        hist_png = save_path.replace('.png', '_hist.png')
        hist_pdf = save_path.replace('.png', '_hist.pdf')
        plt.savefig(hist_png, dpi=300, bbox_inches='tight')
        plt.savefig(hist_pdf, bbox_inches='tight')
        print(f"Histogram saved to: {hist_png}")
        print(f"Histogram PDF saved to: {hist_pdf}")
    else:
        plt.show()
    plt.close()

    plt.figure()
    plt.bar(np.arange(client_num), stats["client_sizes"])
    # Remove title per user request
    plt.title("")
    plt.xlabel("client id")
    plt.ylabel("samples")
    if save_path:
        bar_png = save_path.replace('.png', '_bar.png')
        bar_pdf = save_path.replace('.png', '_bar.pdf')
        plt.savefig(bar_png, dpi=300, bbox_inches='tight')
        plt.savefig(bar_pdf, bbox_inches='tight')
        print(f"Bar chart saved to: {bar_png}")
        print(f"Bar chart PDF saved to: {bar_pdf}")
    else:
        plt.show()
    plt.close()

    # Print summary statistics.
    print(f"\n{'='*80}")
    print(f"Coverage strategy Beta-distribution statistics (overlap_ratio={overlap_ratio}, dispersion_s={dispersion_s})")
    print(f"{'='*80}")
    print(f"Total samples: {total_size}")
    print(f"Number of clients: {client_num}")
    print(f"Mean samples per client: {mean_samples:.1f}")
    print(f"Std dev: {std_samples:.1f}")
    print(f"Min samples: {min_samples}")
    print(f"Max samples: {max_samples}")
    print(f"Normality test: {normality}")


def visualize_coverage_sample_distribution(
    data: List[Any],
    client_num: int,
    min_samples_per_client: int,
    start_idx: int = 0,
    overlap_ratio: float = 1.3,
    dispersion_s: float = 15.0,
    save_path: Optional[str] = None,
    **kwargs
) -> None:
    """
    Visualize the coverage strategy's per-client sample-count distribution using
    the Beta-distribution method.

    Args:
        data: list of items to partition.
        client_num: total number of clients.
        min_samples_per_client: minimum number of samples each client must receive.
        start_idx: starting index into `data`.
        overlap_ratio: overlap coefficient (total assignments / number of samples).
        dispersion_s: spread parameter of the Beta distribution.
        save_path: output path for the figure; if None, the figure is displayed.
        **kwargs: additional parameters.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Compute the actually usable training data (everything after start_idx).
    total_train_data = data[start_idx:]
    total_size = len(total_train_data)

    # Use the same parameter settings as coverage_partition.
    center = 500
    low = min_samples_per_client
    high = 1000

    print(f"Visualizing coverage sample distribution with Beta distribution method...")
    print(f"Parameters: center={center}, dispersion_s={dispersion_s}, low={low}, high={high}")

    # A fixed seed ensures every client computes the same allocation.
    rng = np.random.default_rng(42)

    # Compute the default r value.
    r = default_r(total_size, client_num, low, center, high)
    target_sum = int(round(r * total_size))

    print(f"Default r = {r:.6f} (feasible range [{client_num*low/total_size:.3f}, {client_num*high/total_size:.3f}])")
    print(f"Target sum = {target_sum}")

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

    # Tally the actual sample counts.
    client_sample_counts = [len(s) for s in client_sets]

    # Compute summary statistics.
    stats = summarize_overlap(client_sets)
    
    print("Summary:")
    print({k_: (round(v, 4) if isinstance(v, float) else v) for k_,v in stats.items() if k_ != "client_sizes"})
    print(f"Total assignments = {sum(len(s) for s in client_sets)}, expected = {target_sum}")
    print(f"Each sample replication counts (min, mean, max): {k.min()}, {k.mean():.3f}, {k.max()}")

    # Count how many times each sample appears across clients.
    appear_counts = np.zeros(total_size, dtype=int)
    for client_indices in client_sets:
        for idx in client_indices:
            appear_counts[idx] += 1

    # Build the visualization - following the example format.
    plt.figure(figsize=(12, 8))
    plt.hist(stats["client_sizes"], bins=12)
    # Remove title per user request
    plt.title("")
    plt.xlabel("samples per client", fontsize=20, fontweight='bold')
    plt.ylabel("frequency", fontsize=20, fontweight='bold')
    plt.ylim(0, 50)  # set the y-axis range
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(True, alpha=0.3)
    if save_path:
        hist_png = save_path.replace('.png', '_hist.png')
        hist_pdf = save_path.replace('.png', '_hist.pdf')
        plt.savefig(hist_png, dpi=300, bbox_inches='tight')
        plt.savefig(hist_pdf, bbox_inches='tight')
        print(f"Histogram saved to: {hist_png}")
        print(f"Histogram PDF saved to: {hist_pdf}")
    else:
        plt.show()
    plt.close()
    
    plt.figure(figsize=(15, 8))
    plt.bar(np.arange(client_num), stats["client_sizes"])
    # Remove title per user request
    plt.title("")
    plt.xlabel("client id", fontsize=20, fontweight='bold')
    plt.ylabel("samples", fontsize=20, fontweight='bold')
    plt.ylim(0, 1000)  # set the y-axis range to 0-1000
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(True, alpha=0.3)
    if save_path:
        bar_png = save_path.replace('.png', '_bar.png')
        bar_pdf = save_path.replace('.png', '_bar.pdf')
        plt.savefig(bar_png, dpi=300, bbox_inches='tight')
        plt.savefig(bar_pdf, bbox_inches='tight')
        print(f"Bar chart saved to: {bar_png}")
        print(f"Bar chart PDF saved to: {bar_pdf}")
    else:
        plt.show()
    plt.close()
    
    # Print summary statistics.
    print(f"\n{'='*80}")
    print(f"Coverage strategy Beta-distribution statistics (overlap_ratio={overlap_ratio}, dispersion_s={dispersion_s})")
    print(f"{'='*80}")
    print(f"Total samples: {total_size}")
    print(f"Number of clients: {client_num}")
    print(f"Min samples per client: {min_samples_per_client}")
    print(f"Mean samples per client: {stats['sizes_mean']:.1f}")
    print(f"Std dev: {stats['sizes_std']:.1f}")
    print(f"Min samples: {stats['sizes_min']}")
    print(f"Max samples: {stats['sizes_max']}")
    print(f"Total samples used: {sum(len(s) for s in client_sets)}")
    print(f"Mean pairwise Jaccard similarity: {stats['avg_pairwise_jaccard']:.4f}")
    print(f"Min sample appearance count: {k.min()}")
    print(f"Max sample appearance count: {k.max()}")
    print(f"Mean sample appearance count: {k.mean():.2f}")


def visualize_hardness_distribution(
    data: List[Any],
    client_num: int,
    min_samples_per_client: int,
    start_idx: int = 0,
    success_std: float = 0.1,
    trajectories_file: str = None,
    save_path: Optional[str] = None,
    **kwargs
) -> None:
    """
    Visualize the hardness strategy's per-client success-sample distribution using
    the Beta-distribution method.

    Args:
        data: list of items to partition.
        client_num: total number of clients.
        min_samples_per_client: minimum number of samples each client must receive.
        start_idx: starting index into `data`.
        success_std: standard deviation of the success-sample count (kept for
            backward compatibility).
        trajectories_file: path to the trajectories file.
        save_path: output path for the figure; if None, the figure is displayed.
        **kwargs: additional parameters, including dispersion_s (controls the
            spread of the Beta distribution).
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import json
    import os

    # Compute the actually usable training data (everything after start_idx).
    total_train_data = data[start_idx:]
    total_size = len(total_train_data)

    if total_size == 0:
        print("No training data available for visualization")
        return

    # Default trajectories file path.
    if trajectories_file is None:
        trajectories_file = os.path.join(path_cfg.project_root, "output/inference/all_trajectories.json")

    # Load the trajectories file.
    if not os.path.exists(trajectories_file):
        print(f"Trajectories file not found: {trajectories_file}")
        return

    with open(trajectories_file, 'r') as f:
        trajectories_data = json.load(f)

    # Build a task_id -> success map.
    task_success_map = {}
    for traj in trajectories_data.get('trajectories', []):
        task_info = traj.get('task_info', {})
        traj_info = traj.get('traj_info', {})

        task_id = task_info.get('task_id')
        success = traj_info.get('success', False)

        if task_id is not None:
            task_success_map[task_id] = success

    # Read the dispersion_s parameter from kwargs (kept for backward compatibility).
    if 'dispersion_s' not in kwargs:
        dispersion_s = success_std
    else:
        dispersion_s = kwargs['dispersion_s']

    # Use a Beta distribution to generate the per-client success-sample counts.
    rng = np.random.default_rng(42)

    # Beta-distribution parameters (kept consistent with hardness_partition).
    center = min_samples_per_client // 2  # center set to half of min_samples_per_client
    low = 0  # minimum number of success samples
    high = min_samples_per_client  # maximum number of success samples

    # Compute the target total (sum of success samples across all clients).
    # target_sum = int(round(center * client_num * 0.8))  # alternative: 80% coverage
    r = default_r(total_size, client_num, low, center, high)
    target_sum = int(round(r * total_size))

    print(f"Visualizing hardness partition with Beta distribution method...")
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

    # Compute summary statistics.
    stats = {
        "client_sizes": success_counts,
        "sizes_min": int(success_counts.min()),
        "sizes_max": int(success_counts.max()),
        "sizes_mean": float(success_counts.mean()),
        "sizes_std": float(success_counts.std(ddof=0)),
    }
    
    print("Summary:")
    print({k_: (round(v, 4) if isinstance(v, float) else v) for k_,v in stats.items() if k_ != "client_sizes"})
    print(f"Total success samples = {success_counts.sum()}, expected = {target_sum}")

    # Build the visualization - following the example format.
    plt.figure()
    plt.hist(stats["client_sizes"], bins=12)
    # Remove title per user request
    plt.title("")
    plt.xlabel("success samples per client", fontsize=20, fontweight='bold')
    plt.ylabel("frequency", fontsize=20, fontweight='bold')
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(True, alpha=0.3)
    if save_path:
        hist_png = save_path.replace('.png', '_hist.png')
        hist_pdf = save_path.replace('.png', '_hist.pdf')
        plt.savefig(hist_png, dpi=300, bbox_inches='tight')
        plt.savefig(hist_pdf, bbox_inches='tight')
        print(f"Histogram saved to: {hist_png}")
        print(f"Histogram PDF saved to: {hist_pdf}")
    else:
        plt.show()
    plt.close()

    plt.figure()
    # Convert success samples to success rate (divide by 100)
    success_rates = stats["client_sizes"] / 100.0
    plt.bar(np.arange(client_num), success_rates)
    # Remove title per user request
    plt.title("")
    plt.xlabel("client id", fontsize=20, fontweight='bold')
    plt.ylabel("Success Rate", fontsize=20, fontweight='bold')
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(True, alpha=0.3)
    if save_path:
        bar_png = save_path.replace('.png', '_bar.png')
        bar_pdf = save_path.replace('.png', '_bar.pdf')
        plt.savefig(bar_png, dpi=300, bbox_inches='tight')
        plt.savefig(bar_pdf, bbox_inches='tight')
        print(f"Bar chart saved to: {bar_png}")
        print(f"Bar chart PDF saved to: {bar_pdf}")
    else:
        plt.show()
    plt.close()

    # Print summary statistics.
    print(f"\n{'='*80}")
    print(f"Hardness strategy Beta-distribution statistics (dispersion_s={dispersion_s})")
    print(f"{'='*80}")
    print(f"Total clients: {client_num}")
    print(f"Min samples per client: {min_samples_per_client}")
    print(f"Mean success samples per client: {stats['sizes_mean']:.1f}")
    print(f"Std dev: {stats['sizes_std']:.1f}")
    print(f"Min success samples: {stats['sizes_min']}")
    print(f"Max success samples: {stats['sizes_max']}")
    print(f"Total tasks: {len(task_success_map)}")
    print(f"Successful tasks: {sum(task_success_map.values())}")
    print(f"Success rate: {sum(task_success_map.values())/len(task_success_map)*100:.1f}%")
    print(f"Total success samples: {success_counts.sum()}")
    print(f"Target success samples: {target_sum}")


def visualize_hardness_distribution_alfworld(
    data: List[Any],
    client_num: int,
    min_samples_per_client: int,
    start_idx: int = 0,
    success_std: float = 0.1,
    trajectories_file: str = None,
    save_path: Optional[str] = None,
    **kwargs
) -> None:
    """
    Visualize the Alfworld hardness strategy's per-client success-sample
    distribution using the Beta-distribution method.

    Args:
        data: list of items to partition (Alfworld game file paths).
        client_num: total number of clients.
        min_samples_per_client: minimum number of samples each client must receive.
        start_idx: starting index into `data`.
        success_std: standard deviation of the success-sample count (kept for
            backward compatibility).
        trajectories_file: path to the trajectories file.
        save_path: output path for the figure; if None, the figure is displayed.
        **kwargs: additional parameters, including dispersion_s (controls the
            spread of the Beta distribution).
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import json
    import os

    # Compute the actually usable training data (everything after start_idx).
    total_train_data = data[start_idx:]
    total_size = len(total_train_data)

    if total_size == 0:
        print("No training data available for visualization")
        return

    # Default trajectories file path.
    if trajectories_file is None:
        trajectories_file = "output/inference/all_trajectories_alfworld.json"

    # Load the trajectories file.
    if not os.path.exists(trajectories_file):
        print(f"Alfworld trajectories file not found: {trajectories_file}")
        return

    with open(trajectories_file, 'r') as f:
        trajectories_data = json.load(f)

    # Build a task_id -> success map.
    task_success_map = {}
    for traj in trajectories_data.get('trajectories', []):
        task_info = traj.get('task_info', {})
        traj_info = traj.get('traj_info', {})

        task_id = task_info.get('task_id')
        success = traj_info.get('success', False)

        if task_id is not None:
            task_success_map[task_id] = success

    # Read the dispersion_s parameter from kwargs (kept for backward compatibility).
    if 'dispersion_s' not in kwargs:
        dispersion_s = success_std
    else:
        dispersion_s = kwargs['dispersion_s']

    # Use a Beta distribution to generate the per-client success-sample counts.
    rng = np.random.default_rng(42)

    # Beta-distribution parameters (kept consistent with
    # hardness_partition_alfworld).
    center = min_samples_per_client // 2  # center set to half of min_samples_per_client
    low = 0  # minimum number of success samples
    high = min_samples_per_client  # maximum number of success samples

    # Compute the target total (sum of success samples across all clients), at 80%
    # coverage.
    target_sum = int(round(center * client_num * 0.8))  # 80% coverage

    print(f"Visualizing Alfworld hardness partition with Beta distribution method...")
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

    # Compute summary statistics.
    stats = {
        "client_sizes": success_counts,
        "sizes_min": int(success_counts.min()),
        "sizes_max": int(success_counts.max()),
        "sizes_mean": float(success_counts.mean()),
        "sizes_std": float(success_counts.std(ddof=0)),
    }
    
    print("Summary:")
    print({k_: (round(v, 4) if isinstance(v, float) else v) for k_,v in stats.items() if k_ != "client_sizes"})
    print(f"Total success samples = {success_counts.sum()}, expected = {target_sum}")

    # Build the visualization - following the example format.
    plt.figure()
    plt.hist(stats["client_sizes"], bins=12)
    # Remove title per user request
    plt.title("")
    plt.xlabel("success samples per client", fontsize=20, fontweight='bold')
    plt.ylabel("frequency", fontsize=20, fontweight='bold')
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(True, alpha=0.3)
    if save_path:
        hist_png = save_path.replace('.png', '_hist.png')
        hist_pdf = save_path.replace('.png', '_hist.pdf')
        plt.savefig(hist_png, dpi=300, bbox_inches='tight')
        plt.savefig(hist_pdf, bbox_inches='tight')
        print(f"Histogram saved to: {hist_png}")
        print(f"Histogram PDF saved to: {hist_pdf}")
    else:
        plt.show()
    plt.close()
    
    plt.figure()
    # Convert success samples to success rate (divide by 100)
    success_rates = stats["client_sizes"] / 100.0
    plt.bar(np.arange(client_num), success_rates)
    # Remove title per user request
    plt.title("")
    plt.xlabel("client id", fontsize=15, fontweight='bold')
    plt.ylabel("Success Rate", fontsize=15, fontweight='bold')
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(True, alpha=0.3)
    if save_path:
        bar_png = save_path.replace('.png', '_bar.png')
        bar_pdf = save_path.replace('.png', '_bar.pdf')
        plt.savefig(bar_png, dpi=300, bbox_inches='tight')
        plt.savefig(bar_pdf, bbox_inches='tight')
        print(f"Bar chart saved to: {bar_png}")
        print(f"Bar chart PDF saved to: {bar_pdf}")
    else:
        plt.show()
    plt.close()
    
    # Print summary statistics.
    print(f"\n{'='*80}")
    print(f"Alfworld hardness strategy Beta-distribution statistics (dispersion_s={dispersion_s})")
    print(f"{'='*80}")
    print(f"Total clients: {client_num}")
    print(f"Min samples per client: {min_samples_per_client}")
    print(f"Mean success samples per client: {stats['sizes_mean']:.1f}")
    print(f"Std dev: {stats['sizes_std']:.1f}")
    print(f"Min success samples: {stats['sizes_min']}")
    print(f"Max success samples: {stats['sizes_max']}")
    print(f"Total tasks: {len(task_success_map)}")
    print(f"Successful tasks: {sum(task_success_map.values())}")
    print(f"Success rate: {sum(task_success_map.values())/len(task_success_map)*100:.1f}%")
    print(f"Total success samples: {success_counts.sum()}")
    print(f"Target success samples: {target_sum}")


def compare_partition_strategies(
    data: List[Any],
    client_num: int,
    min_samples_per_client: int,
    category_key: str = 'category',
    start_idx: int = 0,
    tau_values: List[float] = [0.1, 0.3, 0.5],
    save_dir: Optional[str] = None
) -> None:
    """
    Compare the distribution behavior across different partition strategies and
    tau values.

    Args:
        data: list of items to partition.
        client_num: total number of clients.
        min_samples_per_client: minimum number of samples each client must receive.
        category_key: name of the field holding the category label.
        start_idx: starting index into `data`.
        tau_values: list of tau values to compare.
        save_dir: directory in which to save the figures.
    """
    strategies = ['uniform', 'preference', 'coverage']
    
    for strategy in strategies:
        if strategy == 'preference':
            for tau in tau_values:
                save_path = f"{save_dir}/distribution_{strategy}_tau_{tau}.png" if save_dir else None
                print(f"\n{'='*60}")
                print(f"Visualizing {strategy} strategy (tau={tau})")
                print(f"{'='*60}")
                visualize_client_category_distribution(
                    data=data,
                    client_num=client_num,
                    min_samples_per_client=min_samples_per_client,
                    strategy=strategy,
                    category_key=category_key,
                    start_idx=start_idx,
                    tau=tau,
                    save_path=save_path
                )
        elif strategy == 'coverage':
            save_path = f"{save_dir}/distribution_{strategy}.png" if save_dir else None
            print(f"\n{'='*60}")
            print(f"Visualizing {strategy} strategy")
            print(f"{'='*60}")
            visualize_coverage_sample_distribution(
                data=data,
                client_num=client_num,
                min_samples_per_client=min_samples_per_client,
                start_idx=start_idx,
                overlap_ratio=1.3,
                size_std=10,
                save_path=save_path
            )
        else:
            save_path = f"{save_dir}/distribution_{strategy}.png" if save_dir else None
            print(f"\n{'='*60}")
            print(f"Visualizing {strategy} strategy")
            print(f"{'='*60}")
            visualize_client_category_distribution(
                data=data,
                client_num=client_num,
                min_samples_per_client=min_samples_per_client,
                strategy=strategy,
                category_key=category_key,
                start_idx=start_idx,
                save_path=save_path
            )