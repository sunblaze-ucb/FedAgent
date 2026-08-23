#!/usr/bin/env python
"""E2 stage 3 -- render the delta_eff results as the tables the rebuttal needs.

Emits the working sheet (todo.md 5.4), the per-arm search/click decomposition, and the
Table R4 delta column. Pure stdlib + numpy; run in either conda env.

Usage:
    python tools/env_heterogeneity/report_delta_occ.py e2_delta_occ/delta_occ_val64.json
"""
import json
import sys
from pathlib import Path

import numpy as np

# Table R1 / R4 final success (GRPO, standard env) per paper variant row.
FINAL_SUCCESS = {
    "catalog_split": "43.8%", "bm25_field_subset": "15.6%", "bm25_reweight": "15.6%",
    "lookalike": "12.5%", "rank_wrapper": "10.9%",
}
PAPER_NAME = {
    "catalog_split": "Catalog Split", "bm25_field_subset": "Field-Subset Index",
    "bm25_reweight": "BM25 Reweighting", "lookalike": "Lookalike Injection",
    "rank_wrapper": "Rank Wrapper", "uniform": "Uniform (control)",
}
STAGE = {
    "catalog_split": "1 (content)", "bm25_field_subset": "2 (encoding)",
    "bm25_reweight": "3 (matching)", "lookalike": "1+3 (joint)",
    "rank_wrapper": "4 (rendering)", "uniform": "-",
}


def family(arm):
    for k in ("catalog_split", "bm25_field_subset", "bm25_reweight", "lookalike",
              "rank_wrapper", "uniform"):
        if arm.startswith(k):
            return k
    return arm


def pct(x):
    return f"{100 * x:.1f}%"


def main():
    path = Path(sys.argv[1])
    d = json.loads(path.read_text())
    meta = d["meta"]
    faithful = any(s.get("occupancy") == "faithful" for s in d["summaries"])

    print(f"# delta_eff results -- {path.name}\n")
    print(f"* trace: `{meta['trace']}`  ({meta['episodes_used']} episodes, "
          f"{meta['trace_meta']['num_goals']} val goals rolled out)")
    print(f"* reference policy: `{meta['trace_meta']['model']}`, "
          f"{meta['trace_meta']['rollout_mode']}, temperature {meta['trace_meta']['temperature']}, "
          f"max_turns {meta['trace_meta']['max_turns']}")
    print(f"* occupancy: {'FAITHFUL (row i sampled in M_i)' if faithful else 'SIMPLIFIED (one rollout on the unperturbed env)'}")
    print(f"* WEBSHOP_SEARCH_RETURN_N = {meta['search_return_n']}, "
          f"clients per arm = 100\n")

    if faithful:
        print("## Faithful occupancy (delta_eff(pi; i, j), occupancy from M_i)\n")
        print("| Arm | kernels rolled out | N (s,a) | delta max-pairwise | delta mean-pairwise | replay fidelity |")
        print("|---|---|---|---|---|---|")
        for s in d["summaries"]:
            print(f"| {s['arm']} | {len(s['rows_measured'])} | {s['n_sa_total']} | "
                  f"{pct(s['delta_max_pairwise'])} | {pct(s['delta_mean_pairwise'])} | {s['fidelity']} |")
        return

    print("## 5.4 working sheet\n")
    print("| Variant (config) | N (s,a) | delta_eff max-pairwise | mean-pairwise | any-pair | "
          "empirical sup | replay fidelity |")
    print("|---|---|---|---|---|---|---|")
    for s in d["summaries"]:
        print(f"| {s['arm']} | {s['n_sa']} | **{pct(s['delta_max_pairwise'])}** | "
              f"{pct(s['delta_mean_pairwise'])} | {pct(s['any_pair_rate'])} | "
              f"{s['empirical_sup']:.0f} | {s['fidelity']} |")

    print("\n## Decomposition (which transitions carry the divergence)\n")
    print("| Variant (config) | retrieval-invoking (s,a) | delta on retrieval steps | "
          "delta on non-retrieval steps | max unavailable-product rate |")
    print("|---|---|---|---|---|")
    for s in d["summaries"]:
        share = s["n_search_sa"] / max(s["n_sa"], 1)
        print(f"| {s['arm']} | {s['n_search_sa']}/{s['n_sa']} ({pct(share)}) | "
              f"{pct(s['delta_max_search_only'])} | {pct(s['delta_max_click_only'])} | "
              f"{pct(s['unavailable_rate_max'])} |")

    rw = [s for s in d["summaries"] if "delta_max_pairwise_analytic" in s]
    if rw:
        print("\n## Rank Wrapper: empirical vs exact TV\n")
        print("The shuffle / partial-random searchers are STOCHASTIC given (s,a), so the "
              "disagreement rate only upper-bounds the TV. Exact values below.\n")
        print("| Config | max-pairwise (disagreement) | max-pairwise (exact TV) | "
              "mean-pairwise (disagreement) | mean-pairwise (exact TV) |")
        print("|---|---|---|---|---|")
        for s in rw:
            print(f"| {s['arm']} | {pct(s['delta_max_pairwise'])} | "
                  f"{pct(s['delta_max_pairwise_analytic'])} | {pct(s['delta_mean_pairwise'])} | "
                  f"{pct(s['delta_mean_pairwise_analytic'])} |")

    print("\n## Table R4 delta column (paper configs)\n")
    print("| Variant | Stage | $\\hat\\delta_{\\text{occ}}$ | Final success (GRPO) |")
    print("|---|---|---|---|")
    for s in d["summaries"]:
        f = family(s["arm"])
        if f == "uniform":
            continue
        print(f"| {PAPER_NAME[f]} | {STAGE[f]} | {pct(s['delta_max_pairwise'])} "
              f"({s['arm']}) | {FINAL_SUCCESS.get(f, '')} |")

    print("\n## Per-kernel matrices\n")
    for s in d["summaries"]:
        m = np.asarray(s["backend_matrix"])
        if m.shape[0] > 10:
            print(f"### {s['arm']}: {m.shape[0]} kernels (matrix omitted; "
                  f"off-diagonal min {pct(m[~np.eye(len(m), dtype=bool)].min())}, "
                  f"max {pct(m.max())})\n")
            continue
        print(f"### {s['arm']}\n")
        labels = s["backend_labels"]
        print("| | " + " | ".join(labels) + " |")
        print("|---" * (len(labels) + 1) + "|")
        for i, lab in enumerate(labels):
            print(f"| {lab} | " + " | ".join(pct(v) for v in m[i]) + " |")
        print()


if __name__ == "__main__":
    main()
