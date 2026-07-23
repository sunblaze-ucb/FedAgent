"""Plot-tool metric semantics (tools/plot_training_dynamics.py, data path only).

Context: WebShop's training reward is the binarized {0,10} success signal, so a
val/reward_mean curve is success_rate x10 — the same information twice (observed as two
"identical" live figures in the field). The paper pair for WebShop is task_score
(partial-credit [0,1]) + success_rate; the tool's default metric set is exactly that
pair, with a metric that has NO recorded points skipped instead of failing the call
(task_score is WebShop-only and absent from dumps predating its plumbing).

Offline: matplotlib is imported lazily inside the renderer, so the data helpers and the
metric-set loop (renderer stubbed) are testable without it.
"""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "tools")))

import plot_training_dynamics as ptd  # noqa: E402


def _write_summary(tmp_path):
    summary = {
        "epochs_per_round": 3,
        "val_curve": [
            {"round": 0, "success_rate": 0.03, "reward_mean": 0.3, "task_score_mean": None},
            {"round": 1, "success_rate": 0.10, "reward_mean": 1.0, "task_score_mean": 0.25},
            {"round": 2, "success_rate": 0.50, "reward_mean": 5.0, "task_score_mean": 0.62},
        ],
        "client_curve": [
            {"round": 1, "client": 0, "success_rate": 0.08, "task_score_mean": 0.2},
        ],
    }
    (tmp_path / "federated_summary.json").write_text(json.dumps(summary))
    return tmp_path


def test_load_summary_curves_task_score_skips_null_rounds(tmp_path):
    _write_summary(tmp_path)
    agg, circles, stride = ptd.load_summary_curves(tmp_path, "val/task_score")
    # round-0 entry has task_score_mean=None (dump predating the plumbing) -> dropped;
    # the round-k aggregate is the model ENTERING round k+1 (x placement contract)
    assert agg == [(2, 0.25), (3, 0.62)]
    assert circles == {1: [(0, 0.2)]}
    assert stride == 3


def test_load_summary_curves_success_rate_and_unknown_metric(tmp_path):
    _write_summary(tmp_path)
    agg, _, _ = ptd.load_summary_curves(tmp_path, "val/success_rate")
    assert agg == [(1, 0.03), (2, 0.10), (3, 0.50)]
    with pytest.raises(ValueError, match="not in federated_summary.json"):
        ptd.load_summary_curves(tmp_path, "val/nonsense")


def test_plot_metric_set_default_renders_four_and_skips_missing_data(tmp_path, monkeypatch):
    _write_summary(tmp_path)                     # has client_curve -> both variants kept
    calls = []

    def fake_render(folder, metric, *, out_path=None, title=None, with_clients=False, **kw):
        calls.append({"metric": metric, "out": out_path, "wc": with_clients, "title": title})
        if metric == "val/task_score":
            raise ValueError(f"no '{metric}' points in {folder}/federated_summary.json")
        return f"{out_path}"

    monkeypatch.setattr(ptd, "plot_training_dynamics", fake_render)
    outs = ptd.plot_metric_set(str(tmp_path), ["val/task_score", "val/success_rate"],
                               title="std-1")
    # the default set = 2 metrics x {plain, with_clients} = 4 attempted figures;
    # task_score data absent -> its two variants skipped, success_rate's two render
    assert [(c["metric"], c["wc"]) for c in calls] == [
        ("val/task_score", False), ("val/task_score", True),
        ("val/success_rate", False), ("val/success_rate", True)]
    assert len(outs) == 2
    # default out names get per-metric/per-variant suffixes (a shared name would overwrite)
    assert calls[0]["out"].endswith("training_dynamics_task_score.pdf")
    assert calls[1]["out"].endswith("training_dynamics_task_score_with_clients.pdf")
    assert calls[2]["out"].endswith("training_dynamics_success_rate.pdf")
    assert calls[3]["out"].endswith("training_dynamics_success_rate_with_clients.pdf")
    # titles disambiguated per metric
    assert "success rate" in calls[2]["title"]


def test_plot_metric_set_drops_client_variant_without_client_curve(tmp_path, monkeypatch):
    summary = json.loads((_write_summary(tmp_path) / "federated_summary.json").read_text())
    summary.pop("client_curve")
    (tmp_path / "federated_summary.json").write_text(json.dumps(summary))
    calls = []

    def fake_render(folder, metric, *, out_path=None, with_clients=False, **kw):
        calls.append((metric, with_clients))
        return f"{out_path}"

    monkeypatch.setattr(ptd, "plot_training_dynamics", fake_render)
    outs = ptd.plot_metric_set(str(tmp_path), ["val/task_score", "val/success_rate"])
    # no client_curve (client_end_eval off) -> the with-clients variants would just
    # duplicate the plain figures; auto mode drops them
    assert calls == [("val/task_score", False), ("val/success_rate", False)]
    assert len(outs) == 2
    # explicit --with-clients still forces the overlay variant
    calls.clear()
    ptd.plot_metric_set(str(tmp_path), ["val/success_rate"], with_clients=True)
    assert calls == [("val/success_rate", True)]


def test_plot_metric_set_raises_when_nothing_rendered_or_name_bad(tmp_path, monkeypatch):
    def none_have_data(folder, metric, **kw):
        raise ValueError(f"no '{metric}' points in the summary")

    monkeypatch.setattr(ptd, "plot_training_dynamics", none_have_data)
    with pytest.raises(ValueError, match="none of"):
        ptd.plot_metric_set(str(tmp_path), ["val/task_score", "val/success_rate"])

    def bad_name(folder, metric, **kw):
        raise ValueError("metric 'oops' is not in federated_summary.json ...")

    monkeypatch.setattr(ptd, "plot_training_dynamics", bad_name)
    with pytest.raises(ValueError, match="is not in federated_summary"):
        ptd.plot_metric_set(str(tmp_path), ["oops", "val/success_rate"])


def test_default_metrics_is_the_paper_pair():
    assert ptd.DEFAULT_METRICS == "val/task_score,val/success_rate"
    assert "reward_mean" not in ptd.DEFAULT_METRICS      # redundant: success_rate x10
    assert "val/reward_mean" in ptd._SUMMARY_METRIC      # explicit opt-in stays possible
