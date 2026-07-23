"""Regression test for the PPO critic value-loss parity overlay (fedagent/ppo_critic_loss.py).

Stock verl 0.8: per-micro loss = 0.5 * local-token-mean, backwarded with NO 1/M division ->
critic gradient = 0.5*M x the paper fork's Sigma micro_mean / M objective (2x at the paper
recipe, M=4). The wrapper rescales the stock per-micro output so the engine's Sigma-backward
+ FSDP DP-mean reproduces:

- legacy_exact:            (1/(dp*M)) * Sigma_all micro_token_means, coefficient 1.0 (the fork)
- global_token_paper_coef: global token-mean, coefficient 1.0, micro/DP-invariant

The wrapper only touches `data` via __getitem__ ("dp_size", "global_batch_size",
"batch_num_tokens", "response_mask"), so a plain dict duck-types the runtime TensorDict here.
Pure offline (torch only; no verl import).
"""
import os
import sys
import types

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fedagent.ppo_critic_loss import _wrap_value_loss  # noqa: E402

CFG = types.SimpleNamespace(loss_agg_mode="token-mean")


def _stock_value_loss(config, model_output, data, dp_group=None):
    """Verbatim stock semantics: 0.5 * LOCAL token-mean of the (unclipped-here) sq error."""
    rm = data["response_mask"].to(torch.bool)
    err2 = (data["values"] - data["returns"]) ** 2
    local = (err2 * rm).sum() / rm.sum()
    vf = 0.5 * local
    return vf, {"critic/vf_loss": float(vf.item()), "critic/vf_clipfrac": 0.0,
                "critic/vpred_mean": 0.0}


def _micro(rows, toks, err):
    """One micro-batch: `rows` rows, `toks` valid tokens/row, constant per-token error `err`
    -> local token-mean = err^2, masked sum = rows*toks*err^2."""
    T = 8
    rm = torch.zeros(rows, T)
    rm[:, :toks] = 1
    return {"values": torch.zeros(rows, T), "returns": torch.full((rows, T), float(err)),
            "response_mask": rm}


def _run(mode, micros_by_rank, dp, global_rows, global_tokens):
    wrapped = _wrap_value_loss(mode, _stock_value_loss)
    total = 0.0
    for rank_micros in micros_by_rank:
        for m in rank_micros:
            data = dict(m)
            data["dp_size"] = dp
            data["global_batch_size"] = global_rows
            data["batch_num_tokens"] = global_tokens
            loss, metrics = wrapped(CFG, {}, data)
            assert metrics["critic/vf_loss"] == float(loss.item())
            assert "critic/vf_loss_scale" in metrics
            total += float(loss.item())
    return total / dp   # FSDP averages gradients (hence effective loss) across DP ranks


def test_legacy_exact_matches_fork_objective():
    # dp=2 ranks x M=2 equal-row micros; per-micro token-means: err^2 = 1, 4, 9, 16
    ranks = [[_micro(4, 3, 1.0), _micro(4, 3, 2.0)],
             [_micro(4, 3, 3.0), _micro(4, 3, 4.0)]]
    got = _run("legacy_exact", ranks, dp=2, global_rows=16, global_tokens=48)
    fork = (1.0 + 4.0 + 9.0 + 16.0) / 4          # (1/(dp*M)) * Sigma micro means, NO 0.5
    assert abs(got - fork) < 1e-6


def test_legacy_exact_micro_split_invariant_for_equal_rows():
    # same 8 rows on one rank, split M=2 vs M=4 (equal rows): identical effective loss
    a = _run("legacy_exact", [[_micro(4, 3, 1.0), _micro(4, 3, 3.0)]],
             dp=1, global_rows=8, global_tokens=24)
    b = _run("legacy_exact", [[_micro(2, 3, 1.0), _micro(2, 3, 1.0),
                               _micro(2, 3, 3.0), _micro(2, 3, 3.0)]],
             dp=1, global_rows=8, global_tokens=24)
    assert abs(a - b) < 1e-6


def test_global_token_is_global_token_mean_even_for_unequal_micros():
    # unequal token counts: micro A 4x2 tokens err=1, micro B 4x6 tokens err=3
    # global token mean = (8*1 + 24*9) / 32 = 7.0
    got = _run("global_token_paper_coef", [[_micro(4, 2, 1.0), _micro(4, 6, 3.0)]],
               dp=1, global_rows=8, global_tokens=32)
    assert abs(got - 7.0) < 1e-6
    # micro/DP-invariance: same tokens split across 2 "ranks"
    got2 = _run("global_token_paper_coef", [[_micro(4, 2, 1.0)], [_micro(4, 6, 3.0)]],
                dp=2, global_rows=8, global_tokens=32)
    assert abs(got2 - 7.0) < 1e-6


def test_stock_vs_legacy_ratio_is_half_M_at_paper_recipe():
    # paper recipe shape: dp=4, M=4 equal micros -> stock Sigma-backward = 0.5*M*mean = 2x fork
    micros = [[_micro(4, 3, 2.0) for _ in range(4)] for _ in range(4)]
    legacy = _run("legacy_exact", micros, dp=4, global_rows=64, global_tokens=192)
    stock = sum(float(_stock_value_loss(CFG, {}, dict(m))[0].item())
                for rank in micros for m in rank) / 4
    assert abs(stock / legacy - 2.0) < 1e-6      # 0.5 * M = 2 at M=4


def test_non_token_mean_mode_fails_closed():
    wrapped = _wrap_value_loss("legacy_exact", _stock_value_loss)
    data = dict(_micro(2, 2, 1.0)); data.update(dp_size=1, global_batch_size=2, batch_num_tokens=4)
    bad = types.SimpleNamespace(loss_agg_mode="seq-mean-token-sum")
    try:
        wrapped(bad, {}, data)
    except RuntimeError as e:
        assert "token-mean" in str(e)
    else:
        raise AssertionError("expected RuntimeError for non-token-mean mode")


def test_missing_global_info_fails_closed():
    wrapped = _wrap_value_loss("legacy_exact", _stock_value_loss)
    data = dict(_micro(2, 2, 1.0))               # no dp_size/global_batch_size injected
    try:
        wrapped(CFG, {}, data)
    except RuntimeError as e:
        assert "global batch info" in str(e)
    else:
        raise AssertionError("expected RuntimeError when engine info is absent")


def test_source_guard_rejects_post_6957_value_loss():
    # upstream 2eb020a (#6957) makes value_loss consume the global batch info itself;
    # rescaling that output would double-correct -> the guard must refuse loudly.
    from fedagent.ppo_critic_loss import _assert_stock_value_loss

    def post_fix_value_loss(config, model_output, data, dp_group=None):
        batch_num_tokens = data["batch_num_tokens"]  # the post-#6957 signature marker
        return batch_num_tokens, {}

    try:
        _assert_stock_value_loss(post_fix_value_loss)
    except RuntimeError as e:
        assert "double-correct" in str(e)
    else:
        raise AssertionError("expected RuntimeError for a post-#6957 value_loss")
    _assert_stock_value_loss(_stock_value_loss)   # pre-fix shape passes
