"""Beta-distribution sizing helpers for the Coverage/Hardness task-level variants.

`default_r`, `generate_client_sizes`, and `assign_with_overlap` are copied VERBATIM
from verl-agent's `partition_strategy.py` (the science red line -- exact copy, no
paraphrasing/improvements). They are the shared sizing primitives that
`coverage_partition` and `hardness_partition` call to draw per-client sizes from a
Beta distribution and (for Coverage) hand out samples with cross-client overlap.

Nothing else in this module is original logic beyond this docstring and the `math`/
`numpy` imports the verbatim bodies require.
"""
import math

import numpy as np


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
