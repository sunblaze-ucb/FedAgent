"""summarize_val_dump_by_tasktype: the paper-table estimator (ONE pass, partitioned).

The paper's ALFWorld per-type columns are a partition of a single 64-trial val pass:
every per-type cell is a multiple of 1/n_t for that type's episode count in the pass,
and the n_t-weighted pool equals the All column exactly (verified on 40/40 table rows
by the 2026-07 audit). The previously shipped 7-pass tool measured a different
quantity (per-type full pools with game repetition, every denominator = n_envs) and
could not reproduce the tables even in principle. These tests pin the new single-pass
grouping to the table estimator's defining identities.
"""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fedagent.fed.eval_dumps import summarize_val_dump, summarize_val_dump_by_tasktype

# the paper runs' val-64 composition (Pick/Look/Clean/Heat/Cool/Pick2)
COMPOSITION = {
    "pick_and_place_simple": 14,
    "look_at_obj_in_light": 8,
    "pick_clean_then_place_in_recep": 12,
    "pick_heat_then_place_in_recep": 9,
    "pick_cool_then_place_in_recep": 11,
    "pick_two_obj_and_place": 10,
}
SUCCESSES = {  # one concrete table row: 2/14, 2/8, 2/12, 2/9, 3/11, 2/10 -> All 13/64
    "pick_and_place_simple": 2, "look_at_obj_in_light": 2,
    "pick_clean_then_place_in_recep": 2, "pick_heat_then_place_in_recep": 2,
    "pick_cool_then_place_in_recep": 3, "pick_two_obj_and_place": 2,
}


def _write_dump(d, rows, step=4):
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{step}.jsonl").write_text("\n".join(json.dumps(r) for r in rows))


def _rows(tag=True):
    rows = []
    for t, n in COMPOSITION.items():
        for j in range(n):
            r = {"traj_success": 1.0 if j < SUCCESSES[t] else 0.0, "score": 10.0 if j < SUCCESSES[t] else 0.0}
            if tag:
                r["task_type"] = t
            rows.append(r)
    return rows


def test_partition_identities(tmp_path):
    """Denominators = the pass composition; per-type rates on the 1/n_t grid; pool == All."""
    _write_dump(tmp_path, _rows())
    g = summarize_val_dump_by_tasktype(tmp_path)
    assert g["All"]["n"] == 64 and g["All"]["success_rate"] == round(13 / 64, 4)
    assert {t: s["n"] for t, s in g["by_type"].items()} == COMPOSITION
    for t, s in g["by_type"].items():
        assert s["success_rate"] == round(SUCCESSES[t] / COMPOSITION[t], 4)
    # the pooling identity that defines the table estimator (and that 7 independent
    # passes cannot satisfy): sum of per-type successes / 64 == All, up to the 4-decimal
    # rounding stats carry (each rate rounds within 5e-5 -> pooled within ~1e-4)
    pooled = sum(s["success_rate"] * s["n"] for s in g["by_type"].values()) / 64
    assert abs(pooled - g["All"]["success_rate"]) < 2e-4


def test_untagged_rows_grouped_not_dropped(tmp_path):
    rows = _rows(tag=False)
    _write_dump(tmp_path, rows)
    g = summarize_val_dump_by_tasktype(tmp_path)
    assert set(g["by_type"]) == {"untagged"} and g["by_type"]["untagged"]["n"] == 64
    assert g["All"]["n"] == 64  # All still meaningful even when the tag is missing


def test_latest_step_numeric_and_plain_summary_unchanged(tmp_path):
    _write_dump(tmp_path, _rows(), step=4)
    stale = [{"traj_success": 1.0, "score": 10.0, "task_type": "pick_and_place_simple"}]
    _write_dump(tmp_path, _rows(), step=10)   # latest numerically, same content
    (tmp_path / "4.jsonl").write_text("\n".join(json.dumps(r) for r in stale))
    g = summarize_val_dump_by_tasktype(tmp_path)
    assert g["All"]["n"] == 64                # read 10.jsonl, not the 1-row 4.jsonl
    flat = summarize_val_dump(tmp_path)       # refactor kept the flat summary identical
    assert flat == g["All"]


def test_empty_dir_returns_none(tmp_path):
    assert summarize_val_dump_by_tasktype(tmp_path) is None
