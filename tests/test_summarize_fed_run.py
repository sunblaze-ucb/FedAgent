"""Per-round metric reading across the layouts run_fed actually writes.

``tools/summarize_fed_run.py`` originally read ONLY ``round_*/client_*/training.log``. That
file exists only on the subprocess-per-client path; the persistent path (fed/run_fed.py:1758)
writes one shared ``round_<k>/json_logs/metrics.json`` per round, and the lane path writes
``json_logs_lane*/``. So every persistent-mode run -- which is what the ALFWorld/WebShop paper
runs use -- summarized as "(no data)" while its metrics sat on disk. These tests pin the
reading of each layout, including the two ways the shared stream has to be cut back into
clients (per-round: one segment per client; cross-round: the log is cumulative over rounds, so
the round's clients are the LAST segments).

Also covers tools/rebuild_summary.py's dump fold-in, which must match run_fed's own
summarize_val_dump (4-dp rounding, last dump file wins).

Pure stdlib + tmp dirs: no verl, no GPU, no env services.
"""
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "tools")))

import rebuild_summary as rs  # noqa: E402
import summarize_fed_run as sfr  # noqa: E402


def _entries(steps, reward):
    return [{"step": s, "metrics": {"critic/rewards/mean": reward + s}} for s in steps]


def _write(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f)


def test_reads_subprocess_per_client_json_logs(tmp_path):
    r = tmp_path / "round_1"
    _write(str(r / "client_7" / "json_logs" / "metrics.json"), _entries([1, 2], 0.0))
    _write(str(r / "client_9" / "json_logs" / "metrics.json"), _entries([1, 2], 10.0))
    got = sfr.run_rounds(str(tmp_path), sfr.KEY)
    assert set(got[1]) == {7, 9}
    assert got[1][7] == (1.5, 2.0)          # mean over steps, max step
    assert got[1][9] == (11.5, 12.0)


def test_reads_legacy_training_log(tmp_path):
    """The original source must keep working (runs predating json_logs)."""
    log = tmp_path / "round_2" / "client_3" / "training.log"
    os.makedirs(os.path.dirname(str(log)))
    log.write_text(
        "step:1 - actor/entropy:1.0 - critic/rewards/mean:0.5 - critic/score/mean:0.5 - "
        "actor/lr:1e-6 - training/global_step:1\n"
        "step:2 - actor/entropy:1.0 - critic/rewards/mean:1.5 - critic/score/mean:1.5 - "
        "actor/lr:1e-6 - training/global_step:2\n"
    )
    got = sfr.run_rounds(str(tmp_path), sfr.KEY)
    assert got[2][3] == (1.0, 1.5)


def test_persistent_shared_stream_splits_on_step_reset(tmp_path):
    """One file per ROUND, clients concatenated in plan order, step counters restarting."""
    r = tmp_path / "round_5"
    _write(str(r / "persistent_plan.json"), [{"client": 51}, {"client": 9}])
    _write(str(r / "json_logs" / "metrics.json"), _entries([1, 2, 3], 0.0) + _entries([1, 2, 3], 100.0))
    got = sfr.run_rounds(str(tmp_path), sfr.KEY)
    assert set(got[5]) == {51, 9}
    assert got[5][51] == (2.0, 3.0)         # 1,2,3
    assert got[5][9] == (102.0, 103.0)      # 101,102,103 -- plan ORDER decides the mapping


def test_cross_round_cumulative_log_takes_this_rounds_clients(tmp_path):
    """Cross-round shares ONE launch log, so round k's file holds rounds 1..k: take the last
    len(clients) segments, not the first."""
    r = tmp_path / "round_3"
    _write(str(r / "persistent_plan.json"), [{"client": 4}, {"client": 8}])
    stream = (_entries([1, 2], 0.0) + _entries([1, 2], 10.0)      # round 1 (stale)
              + _entries([1, 2], 20.0) + _entries([1, 2], 30.0)   # round 2 (stale)
              + _entries([1, 2], 40.0) + _entries([1, 2], 50.0))  # round 3 (this one)
    _write(str(r / "json_logs" / "metrics.json"), stream)
    got = sfr.run_rounds(str(tmp_path), sfr.KEY)
    assert got[3][4] == (41.5, 42.0)
    assert got[3][8] == (51.5, 52.0)


def test_lane_layout_maps_lanes_to_plan_order(tmp_path):
    r = tmp_path / "round_2"
    _write(str(r / "persistent_plan.json"), [{"client": 11}, {"client": 22}])
    _write(str(r / "json_logs_lane0" / "metrics.json"), _entries([1], 0.0))
    _write(str(r / "json_logs_lane1" / "metrics.json"), _entries([1], 5.0))
    got = sfr.run_rounds(str(tmp_path), sfr.KEY)
    assert got[2][11] == (1.0, 1.0)
    assert got[2][22] == (6.0, 6.0)


def test_stale_rounds_never_leak_into_the_table(tmp_path):
    """quarantine_stale_rounds parks crashed attempts in _stale_rounds/round_K.N; a round_*
    glob at the run root must not pick them up."""
    _write(str(tmp_path / "round_1" / "client_1" / "json_logs" / "metrics.json"), _entries([1], 0.0))
    _write(str(tmp_path / "_stale_rounds" / "round_1.0" / "client_1" / "json_logs" / "metrics.json"),
           _entries([1], 99.0))
    got = sfr.run_rounds(str(tmp_path), sfr.KEY)
    assert list(got) == [1] and got[1][1] == (1.0, 1.0)


def _dump(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def test_rebuild_summary_folds_val_and_client_dumps(tmp_path):
    _dump(str(tmp_path / "round_0" / "eval" / "val_samples" / "0.jsonl"),
          [{"traj_success": 0.0, "score": 0.0, "task_type": "pick"}] * 3
          + [{"traj_success": 1.0, "score": 10.0, "task_type": "pick"}])
    _dump(str(tmp_path / "round_1" / "eval" / "val_samples" / "0.jsonl"),
          [{"traj_success": 1.0, "score": 10.0, "task_type": "pick"}] * 4)
    _dump(str(tmp_path / "round_1" / "client_2" / "eval" / "val_samples" / "4.jsonl"),
          [{"traj_success": 0.0, "score": 0.0, "task_type": "pick"}] * 4)
    _write(str(tmp_path / "round_1" / "persistent_plan.json"), [{"client": 2}])
    _write(str(tmp_path / "round_1" / "json_logs" / "metrics.json"), _entries([1, 2], 0.0))

    s = rs.rebuild(tmp_path, epochs_per_round=3)
    assert [v["round"] for v in s["val_curve"]] == [0, 1]
    assert s["val_curve"][0]["success_rate"] == 0.25 and s["val_curve"][0]["reward_mean"] == 2.5
    assert s["val_curve"][0]["model"] == "base" and s["val_curve"][1]["model"] == "aggregated"
    assert s["client_curve"] == [{"round": 1, "client": 2, "n": 4, "success_rate": 0.0,
                                  "reward_mean": 0.0, "task_score_mean": None}]
    assert s["env_kind"] == "alfworld"          # task_type present, task_score absent
    assert s["rounds"] == [{"round": 1, "clients": [2], "train_steps": 2,
                            "train_reward_mean": 1.5, "train_reward_max": 2.0}]
    assert s["_reconstructed"] is True


def test_rebuild_summary_prefers_the_highest_step_dump(tmp_path):
    """verl dumps one file per global_step; summarize_val_dump reads the LATEST -- by NUMBER.
    Lexicographically '10' < '4', so a plain sorted()[-1] would read the older dump."""
    d = tmp_path / "round_0" / "eval" / "val_samples"
    _dump(str(d / "0.jsonl"), [{"traj_success": 0.0, "score": 0.0}] * 2)
    _dump(str(d / "4.jsonl"), [{"traj_success": 0.0, "score": 0.0}] * 2)
    _dump(str(d / "10.jsonl"), [{"traj_success": 1.0, "score": 10.0}] * 2)
    s = rs.rebuild(tmp_path)
    assert s["val_curve"][0]["success_rate"] == 1.0
