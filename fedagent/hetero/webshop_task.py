"""WebShop TASK-LEVEL heterogeneity (paper task arm): Preference (omega).

The FAITHFUL task-heterogeneity partition (vs the `task_disjoint` stand-in): each client
gets a category-skewed goal distribution (Dirichlet over the goal `category`), observable
in the prompt, with the FULL catalog (env unperturbed) -- the FedAvg-robust arm of the
Input-Dynamics Asymmetry.

`_preference_partition_generic` is copied VERBATIM from verl-agent's partition_strategy.py
(base_seed=42 hardcoded -- the science red line). The thin public API
`preference_for_client(...) -> goal_idxs` builds the WebShop goal->category list (goal i ->
asin i via the catalog-split goal generator -> product category) and returns this client's
absolute goal indices. Coverage(xi)/Hardness(xi') to be added next (Coverage needs the
Beta-sizing helpers; Hardness needs a precomputed success-labels file).
"""
from typing import Any, List, Optional

import numpy as np

from fedagent.hetero.webshop_catalog_split import (
    _generate_goal_asins_for_partition,
    load_webshop_data,
)


def _preference_partition_generic(
    data: List[Any],
    client_id: int,
    client_num: int,
    min_samples_per_client: int,
    category_key: str = "category",
    tau: float = 0.3,
    fashion_sample_ratio: float = 0.2,
    omega: Optional[float] = None,
    **kwargs,
) -> List[Any]:
    """PreferencePartition (Dirichlet, omega) -- VERBATIM from partition_strategy.py.

    q_i ~ Dir(pi * (1-omega)/omega) then counts ~ Multinomial(L; q_i). E[q_i]=pi exact;
    spread grows with omega. base_seed=42 (per-client RandomState(42+client_id)).
    """
    import math  # noqa: F401  (kept verbatim)

    if omega is None:
        omega = tau
    omega = float(np.clip(omega, 1e-3, 1 - 1e-3))

    total_size = len(data)

    category_to_indices = {}
    for idx, item in enumerate(data):
        category = item.get(category_key, "unknown")
        if category not in category_to_indices:
            category_to_indices[category] = []
        category_to_indices[category].append(idx)

    if "fashion" in category_to_indices:
        fashion_indices = category_to_indices["fashion"]
        fashion_count = len(fashion_indices)
        if fashion_count > 0:
            target_fashion_count = max(1, int(fashion_count * fashion_sample_ratio))
            rng = np.random.RandomState(42 + client_id)
            sampled_fashion_indices = rng.choice(
                fashion_indices, size=target_fashion_count, replace=False
            ).tolist()
            category_to_indices["fashion"] = sampled_fashion_indices
            print(f"Fashion category subsampling: {fashion_count} -> {target_fashion_count} "
                  f"samples (keeping {fashion_sample_ratio*100:.1f}%)")

    total_size = sum(len(indices) for indices in category_to_indices.values())

    categories = list(category_to_indices.keys())
    C = len(categories)
    eps_smooth = 0.01
    raw_p = np.array([len(category_to_indices[c]) for c in categories], dtype=float)
    raw_p = raw_p / raw_p.sum() if raw_p.sum() > 0 else np.full(C, 1.0 / C)
    pi = (raw_p + eps_smooth) / (1.0 + C * eps_smooth)

    rng = np.random.RandomState(42 + client_id)

    alpha_vec = pi * ((1.0 - omega) / omega)
    q = rng.dirichlet(alpha_vec)
    q = q / q.sum()

    counts = rng.multinomial(min_samples_per_client, q)
    category_counts = {categories[i]: int(counts[i]) for i in range(C)}

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

    current_client_data = []
    for c, count in category_counts.items():
        if count > 0:
            available_indices = category_to_indices[c]
            if len(available_indices) >= count:
                selected_indices = rng.choice(available_indices, size=count, replace=False)
            else:
                selected_indices = available_indices
            for idx in selected_indices:
                current_client_data.append(data[idx])

    if len(current_client_data) < min_samples_per_client:
        needed_extra = min_samples_per_client - len(current_client_data)
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


# --------------------------------------------------------------------------- #
# Thin public API for the verl-0.8 WebShop service (task-level; full catalog).
# --------------------------------------------------------------------------- #
def preference_for_client(
    client_id: int,
    client_num: int,
    *,
    omega: float = 0.5,
    min_goals_per_client: int = 100,
    base_seed: int = 42,  # noqa: ARG001 (verbatim fn hardcodes 42; kept for API symmetry)
    start_idx: int = 500,
    data_dir: Optional[str] = None,
) -> List[int]:
    """This client's WebShop goal indices under Preference(omega) -- full catalog (task-only).

    Builds goal->category (goal i -> asin i via the catalog-split goal generator -> product
    `category`), Dirichlet-partitions the train pool (goals[start_idx:]) by category, and
    returns the selected ABSOLUTE goal indices.
    """
    products, ins = load_webshop_data(data_dir)
    goal_asins = _generate_goal_asins_for_partition(products, ins)
    asin_to_cat = {p["asin"]: p.get("category", "unknown") for p in products}
    # tag every goal with its absolute index + category; partition the train pool only
    goals = [{"category": asin_to_cat.get(a, "unknown"), "_idx": i} for i, a in enumerate(goal_asins)]
    selected = _preference_partition_generic(
        data=goals[start_idx:],
        client_id=client_id,
        client_num=client_num,
        min_samples_per_client=min_goals_per_client,
        category_key="category",
        omega=omega,
    )
    idxs = sorted(g["_idx"] for g in selected)
    print(f"[task pref] WebShop client {client_id}/{client_num}: |goal_idxs|={len(idxs)} "
          f"(omega={omega}, full catalog)", flush=True)
    return idxs
