"""The shared validation-dump reader (fedagent/fed/eval_dumps.py).

Every number on the paper's figures passes through this: the federated loop folds it into
``federated_summary.json`` during the run, and ``tools/rebuild_summary.py`` folds it again when
a run dies before its teardown. Those were two copies of the same 20 lines until 2026-07-26 --
now one module, imported by both, so the "which file / which keys" decisions cannot drift.

Pure stdlib: no verl, no omegaconf, no GPU (the whole point of the module living apart from
``run_fed``, which cannot be imported without omegaconf).
"""
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fedagent.fed.eval_dumps import dumps_by_step, summarize_val_dump  # noqa: E402


def _dump(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def test_latest_dump_is_chosen_numerically(tmp_path):
    """verl names each dump after its global_step. Lexicographically '10' < '4', so a plain
    sorted()[-1] would summarize the OLDER dump once a dir ever holds two."""
    d = tmp_path / "val_samples"
    _dump(str(d / "0.jsonl"), [{"traj_success": 0.0, "score": 0.0}])
    _dump(str(d / "4.jsonl"), [{"traj_success": 0.0, "score": 0.0}])
    _dump(str(d / "10.jsonl"), [{"traj_success": 1.0, "score": 10.0}])
    assert [p.stem for p in dumps_by_step(d)] == ["0", "4", "10"]
    assert summarize_val_dump(d)["success_rate"] == 1.0


def test_non_numeric_names_sort_last(tmp_path):
    d = tmp_path / "val_samples"
    _dump(str(d / "3.jsonl"), [{"traj_success": 0.0, "score": 0.0}])
    _dump(str(d / "final.jsonl"), [{"traj_success": 1.0, "score": 10.0}])
    assert [p.stem for p in dumps_by_step(d)] == ["3", "final"]


def test_means_and_missing_task_score(tmp_path):
    """task_score is WebShop-only; ALFWorld dumps (and pre-plumbing dumps) must yield None
    rather than 0.0, which would silently plot as a real score."""
    d = tmp_path / "val_samples"
    _dump(str(d / "0.jsonl"), [{"traj_success": 1.0, "score": 10.0},
                               {"traj_success": 0.0, "score": 0.0},
                               {"traj_success": 0.0, "score": 0.0},
                               {"traj_success": 0.0, "score": 0.0}])
    assert summarize_val_dump(d) == {"n": 4, "success_rate": 0.25, "reward_mean": 2.5,
                                     "task_score_mean": None}
    _dump(str(d / "1.jsonl"), [{"traj_success": 1.0, "score": 1.0, "task_score": 0.5},
                               {"traj_success": 0.0, "score": 0.0, "task_score": 0.25}])
    assert summarize_val_dump(d)["task_score_mean"] == 0.375


def test_partial_rows_survive_a_truncated_tail(tmp_path):
    """A run killed mid-dump leaves a half-written last line; losing the whole round's point
    over it would be worse than averaging what did land."""
    d = tmp_path / "val_samples"
    d.mkdir(parents=True)
    (d / "0.jsonl").write_text(
        json.dumps({"traj_success": 1.0, "score": 10.0}) + "\n"
        + json.dumps({"traj_success": 1.0, "score": 10.0}) + "\n"
        + '{"traj_success": 1.0, "sco')
    m = summarize_val_dump(d)
    assert m["n"] == 2 and m["success_rate"] == 1.0


def test_empty_and_missing_dirs_return_none(tmp_path):
    assert summarize_val_dump(tmp_path / "nope") is None
    (tmp_path / "empty").mkdir()
    assert summarize_val_dump(tmp_path / "empty") is None
    d = tmp_path / "blank"
    d.mkdir()
    (d / "0.jsonl").write_text("")
    assert summarize_val_dump(d) is None
