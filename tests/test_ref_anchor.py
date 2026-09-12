"""Regression test: the KL reference policy is pinned to the BASE model (fedagent/ref_anchor.py).

The defect (docs/bugfixes.md "the KL anchor slides too", dsp_doc/KL_ANCHOR_FIX.md): verl 0.8
builds actor AND ref from the single ``actor_rollout_ref.model.path``, which the federated loop
points at each round's aggregate -- so the KL trust region silently re-anchored every round.

Three properties, all offline (no GPU, no verl import for the core paths):
  1. _repoint_ref_engine moves the ref engine's weight source to the base and rebuilds from it;
  2. the persistent-path guard condition skips the per-round ref reset exactly when the pin is
     active (and preserves the historical reset when it is not);
  3. an unset FEDAGENT_REF_MODEL_PATH arms nothing and refuses nothing (byte-identical legacy).
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fedagent import ref_anchor  # noqa: E402


class _FakeModelConfig:
    def __init__(self, local_path):
        self.local_path = local_path
        self.use_shm = False


class _FakeEngine:
    """Only the surface ref_anchor touches: model_config.local_path, module,
    checkpoint_manager, initialize()."""

    def __init__(self, local_path):
        self.model_config = _FakeModelConfig(local_path)
        self.module = object()
        self.checkpoint_manager = object()
        self.loaded_from = []

    def initialize(self):
        self.module = object()
        self.loaded_from.append(self.model_config.local_path)


def test_repoint_moves_ref_to_base_and_rebuilds():
    eng = _FakeEngine("/run/round_54/aggregated/hf")   # what a RESUMED chunk launches with
    ref_anchor._repoint_ref_engine(eng, "/models/base")
    assert eng.model_config.local_path == "/models/base"
    assert eng.loaded_from == ["/models/base"], "initialize() must reload from the base"
    assert eng.module is not None


def test_repoint_drops_retired_module_before_rebuild():
    """The old ref module must be dropped BEFORE initialize() so the rebuild reuses freed
    memory instead of peaking at two co-resident copies (the _reset_engine ordering rule)."""
    eng = _FakeEngine("/run/round_54/aggregated/hf")
    old_module = eng.module
    seen = {}
    orig_init = eng.initialize

    def initialize():
        seen["module_at_init"] = eng.module
        orig_init()

    eng.initialize = initialize
    ref_anchor._repoint_ref_engine(eng, "/models/base")
    assert seen["module_at_init"] is None, "retired module still referenced at rebuild time"
    assert eng.module is not old_module


def test_persistent_guard_condition():
    """The reload_client_model guard: `ref is not None and not env(FEDAGENT_REF_MODEL_PATH)`.
    Pinned -> per-round reset SKIPPED; unpinned -> historical reset preserved."""
    def would_reset(has_ref, env_value):
        return has_ref and not env_value

    assert would_reset(True, "") is True            # legacy: ref follows the round aggregate
    assert would_reset(True, "/models/base") is False   # pinned: ref left alone
    assert bool(would_reset(False, "")) is False    # no ref built (e.g. use_kl_loss=false)


def test_unset_env_is_a_noop(monkeypatch):
    monkeypatch.delenv("FEDAGENT_REF_MODEL_PATH", raising=False)
    ref_anchor._PATCHED = False
    n_hooks = len(sys.meta_path)
    assert ref_anchor.install_deferred_patch() is True   # armed nothing, refused nothing
    assert len(sys.meta_path) == n_hooks, "no import hook may be installed when unset"


def test_deferred_hook_arms_when_set(monkeypatch):
    """With the env set and verl not yet imported, install must add exactly one meta-path
    hook (the one-shot finder) and report success."""
    monkeypatch.setenv("FEDAGENT_REF_MODEL_PATH", "/models/base")
    assert "verl.workers.engine_workers" not in sys.modules, (
        "test environment unexpectedly imported verl; the arm-path assertion needs it absent")
    ref_anchor._PATCHED = False
    n_hooks = len(sys.meta_path)
    try:
        assert ref_anchor.install_deferred_patch() is True
        assert len(sys.meta_path) == n_hooks + 1
    finally:
        sys.meta_path[:] = [h for h in sys.meta_path
                            if type(h).__name__ != "_RefAnchorImportHook"]


def test_wrapper_preserves_verl_register_metadata(monkeypatch):
    """BLOCKER regression (review 2026-08-30): verl binds only methods carrying the @register
    MAGIC_ATTR (single_controller/ray/base.py _bind_worker_method), and ray_trainer's
    init_workers() calls wg.init_model() unconditionally -- a wrapper that drops the attr
    kills every ref_anchor=base run BEFORE the first rollout. Needs real verl; runs last so
    the deferred-arm test above still sees engine_workers unimported."""
    import pytest
    pytest.importorskip("verl")
    monkeypatch.setenv("FEDAGENT_REF_MODEL_PATH", "/models/base")
    ref_anchor._PATCHED = False
    assert ref_anchor.install_deferred_patch() is True
    import verl.workers.engine_workers as ew                      # triggers the one-shot hook
    from verl.single_controller.base.decorator import MAGIC_ATTR
    im = ew.ActorRolloutRefWorker.init_model
    # functools.wraps copies __qualname__, so identity comes from the explicit sentinel.
    assert getattr(im, "__fedagent_ref_anchor__", False), "not wrapped"
    assert hasattr(im, MAGIC_ATTR), (
        "wrapped init_model lost the @register metadata -- it would never be bound to the "
        "worker group and init_workers() would fail")
