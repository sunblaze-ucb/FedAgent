"""Regression test: the FedProx proximal anchor is ACTOR-ONLY (fedagent/fedprox.py).

The fork (8c6a000) patched dp_actor.update_policy and left dp_critic untouched. verl 0.8
builds the actor AND the PPO critic on the same FSDPEngine class, so a class-level
optimizer_step patch reaches both -- the wrapper must pass value-model engines
(engine.model_config.model_type == "value_model") straight through, or PPO+FedProx would
gain a critic regularizer the paper recipe never had.

Pure offline: exercises _make_optimizer_step on fake engines (torch only; no verl import).
"""
import os
import sys
import types

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fedagent.fedprox import _make_optimizer_step  # noqa: E402

MU = 0.01


def _engine(model_type):
    """Fake FSDPEngine: one 2x2 linear (bias off), grads set to zero so the proximal term
    is the ONLY gradient contribution after the wrapper runs."""
    m = torch.nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        m.weight.fill_(1.0)
    m.weight.grad = torch.zeros_like(m.weight)
    eng = types.SimpleNamespace(module=m)
    if model_type is not None:
        eng.model_config = types.SimpleNamespace(model_type=model_type)
    return eng


def _run_two_steps(eng):
    """Step once (snapshots w_t), drift the weights, step again; return the final grad."""
    calls = []
    step = _make_optimizer_step(lambda self: calls.append(1), MU)
    step(eng)                                    # first call: snapshot only (w == w_t -> +0)
    with torch.no_grad():
        eng.module.weight.add_(2.0)              # drift: w - w_t == 2
    eng.module.weight.grad.zero_()
    step(eng)
    assert len(calls) == 2                       # original optimizer_step always runs
    return eng.module.weight.grad


def test_actor_engine_is_anchored():
    g = _run_two_steps(_engine("language_model"))
    assert torch.allclose(g, torch.full((2, 2), MU * 2.0))


def test_value_model_engine_passes_through_unanchored():
    eng = _engine("value_model")
    g = _run_two_steps(eng)
    assert torch.equal(g, torch.zeros(2, 2))     # no proximal grad ...
    assert not hasattr(eng, "_fedprox_w_t")      # ... and no snapshot taken


def test_engine_without_model_config_is_anchored_back_compat():
    # single-engine GRPO paths predate the discriminator; absence => anchor (old behavior)
    g = _run_two_steps(_engine(None))
    assert torch.allclose(g, torch.full((2, 2), MU * 2.0))
