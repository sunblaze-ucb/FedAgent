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


# --- the 2026-09-16 default flip (round -> base) and its resume guard --------------------------

def _run_fed_cfg(out, **kw):
    """A run_fed config over DEFAULTS (offline: run_fed is imported for its pure helpers)."""
    from omegaconf import OmegaConf
    from fedagent.fed import run_fed
    cfg = OmegaConf.create(dict(run_fed.DEFAULTS))
    cfg.output_dir = str(out)
    cfg.model_path = "/models/base"
    for k, v in kw.items():
        cfg[k] = v
    return run_fed, cfg


def test_default_anchor_is_base_and_resolves_to_the_actor_base(tmp_path):
    import pytest
    run_fed, cfg = _run_fed_cfg(tmp_path)
    assert run_fed.DEFAULTS["ref_anchor"] == "base"
    assert run_fed.resolve_ref_model_path(cfg) == "/models/base"          # "" => the actor base
    assert run_fed.resolve_ref_model_path(cfg, "/snap/base") == "/snap/base"
    cfg.ref_model_path = "/models/explicit"
    assert run_fed.resolve_ref_model_path(cfg) == "/models/explicit"
    cfg.ref_model_path = ""
    cfg.ref_anchor = "round"
    assert run_fed.resolve_ref_model_path(cfg) == ""                       # rolling: no fixed path
    cfg.ref_model_path = "/models/explicit"
    with pytest.raises(ValueError, match="conflicts"):
        run_fed.resolve_ref_model_path(cfg)
    cfg.ref_model_path, cfg.ref_anchor = "", "sideways"
    with pytest.raises(ValueError, match="base|round"):
        run_fed.resolve_ref_model_path(cfg)


def test_persistent_env_pins_the_reference_by_default(tmp_path):
    run_fed, cfg = _run_fed_cfg(tmp_path, env_kind="tinyguess")
    plan = [{"out_dir": str(tmp_path / "c0")}]
    args = (plan, tmp_path / "plan.json", "/run/round_3/aggregated/hf", None, 4)
    _cmd, env = run_fed._persistent_cmd_env(cfg, *args, {}, n_gpus=1, worker_eval=False)
    assert env["FEDAGENT_REF_MODEL_PATH"] == "/models/base", "default launch must pin the ref to the base"
    cfg.ref_anchor = "round"
    _cmd, env = run_fed._persistent_cmd_env(cfg, *args, {"FEDAGENT_REF_MODEL_PATH": "/leaked"},
                                            n_gpus=1, worker_eval=False)
    assert "FEDAGENT_REF_MODEL_PATH" not in env, "round must strip an inherited pin"


def test_resume_objective_guard(tmp_path):
    import json
    import pytest
    out = tmp_path / "run"
    (out / "round_3" / "aggregated" / "hf").mkdir(parents=True)
    run_fed, cfg = _run_fed_cfg(out)                     # ref_anchor: base (the new default)
    # a directory that predates the record trained with the rolling reference => refused
    with pytest.raises(ValueError, match="RESUME REFUSED.*ref_anchor"):
        run_fed.check_resume_objective(cfg, start_round=4, base_model="/models/base")
    assert not (out / run_fed.OBJECTIVE_RECORD).exists(), "a refusal must not write a record"
    # continuing it unchanged works and writes the record
    cfg.ref_anchor = "round"
    rec = run_fed.check_resume_objective(cfg, 4, "/models/base")
    assert rec == {"ref_anchor": "round", "ref_model_path": None, "adv_estimator": "grpo"}
    on_disk = json.loads((out / run_fed.OBJECTIVE_RECORD).read_text())
    assert on_disk["ref_anchor"] == "round" and on_disk["written_at_round"] == 4
    # flipping the anchor on the next resume is refused; allow_objective_change lets it through, recorded
    cfg.ref_anchor = "base"
    with pytest.raises(ValueError, match="'round' \\(directory\\) vs 'base'"):
        run_fed.check_resume_objective(cfg, 5, "/models/base")
    cfg.allow_objective_change = True
    rec = run_fed.check_resume_objective(cfg, 5, "/models/base")
    assert rec["ref_anchor"] == "base" and rec["objective_changed_at_round"] == 5
    assert rec["previous"]["ref_anchor"] == "round"
    # a later resume under the same (new) objective keeps the switch on the record
    cfg.allow_objective_change = False
    rec = run_fed.check_resume_objective(cfg, 6, "/models/base")
    assert rec["ref_anchor"] == "base" and rec["previous"]["ref_anchor"] == "round"
    # a fresh launch never refuses: it (re)writes the record for whatever it runs
    cfg.ref_anchor = "round"
    assert run_fed.check_resume_objective(cfg, 1, "/models/base")["ref_anchor"] == "round"
    # a FINISHED pre-knob run (summary without the key) => rolling reference => refused under base
    old = tmp_path / "old"
    old.mkdir()
    (old / "federated_summary.json").write_text(json.dumps({"adv_estimator": "grpo"}))
    run_fed, cfg2 = _run_fed_cfg(old)
    with pytest.raises(ValueError, match="RESUME REFUSED"):
        run_fed.check_resume_objective(cfg2, 71, "/models/base")
    # a finished run that recorded ref_anchor=base in its summary (2026-09-12..16 window) matches
    (old / "federated_summary.json").write_text(json.dumps({"adv_estimator": "grpo", "ref_anchor": "base"}))
    assert run_fed.check_resume_objective(cfg2, 71, "/models/base")["ref_anchor"] == "base"
    # switching the algorithm on resume is refused the same way
    cfg2.adv_estimator = "gae"
    with pytest.raises(ValueError, match="adv_estimator"):
        run_fed.check_resume_objective(cfg2, 71, "/models/base")


def test_inferred_prior_is_marked_and_kept_on_the_record(tmp_path):
    import pytest
    run_fed, cfg = _run_fed_cfg(tmp_path / "old")          # base by default
    (tmp_path / "old" / "round_2" / "aggregated" / "hf").mkdir(parents=True)
    prior, src = run_fed.prior_objective(tmp_path / "old")
    assert prior["ref_anchor"] == "round" and prior["inferred"] is True and "no run_objective.json" in src
    with pytest.raises(ValueError, match="INFERRED"):
        run_fed.check_resume_objective(cfg, 3, "/models/base")
    cfg.allow_objective_change = True
    rec = run_fed.check_resume_objective(cfg, 3, "/models/base")
    assert rec["previous"]["inferred"] is True and rec["objective_changed_at_round"] == 3
    # a recorded prior never carries the mark
    prior, src = run_fed.prior_objective(tmp_path / "old")
    assert src == run_fed.OBJECTIVE_RECORD and "inferred" not in prior


def test_ref_model_path_trailing_slash_is_normalized(tmp_path):
    run_fed, cfg = _run_fed_cfg(tmp_path)
    cfg.ref_model_path = "/models/explicit/"
    assert run_fed.resolve_ref_model_path(cfg) == "/models/explicit"
    cfg.ref_model_path = ""
    cfg.model_path = "/models/base/"
    assert run_fed.resolve_ref_model_path(cfg) == "/models/base"
