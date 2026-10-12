"""Regression test: the KL reference policy is pinned to the BASE model (fedagent/ref_anchor.py).

The defect (docs/bugfixes.md "the KL anchor slides too", dsp_doc/KL_ANCHOR_FIX.md): verl 0.8
builds actor AND ref from the single ``actor_rollout_ref.model.path``, which the federated loop
points at each round's aggregate -- so the KL trust region silently re-anchored every round.

Four properties, all offline (no GPU, no verl import for the core paths):
  1. the ref is BUILT at the base (_ref_pinning_factory), with _repoint_ref_engine (move the ref
     engine's weight source to the base and rebuild from it) as the fallback;
  2. the persistent-path guard condition skips the per-round ref reset exactly when the pin is
     active (and preserves the historical reset when it is not);
  3. an unset FEDAGENT_REF_MODEL_PATH arms nothing and refuses nothing (byte-identical legacy);
  4. through verl's REAL init_model (the client config composed from fedagent_ppo.yaml; only Ray,
     the process group and the engines faked) the ref loads once, from the base, and the actor
     from the round's aggregate.
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


class _FakeTWConfig:
    def __init__(self, local_path, forward_only):
        self.model_config = _FakeModelConfig(local_path)
        self.engine_config = type("E", (), {"forward_only": forward_only})()


def test_pinning_factory_builds_ref_at_base_and_leaves_actor_alone():
    """Pin-before-build (2026-09-24): the ref (verl builds it first, forward_only) must be
    CONSTRUCTED at the base, so no actor-path copy is ever built and stranded; the actor's config
    passes through untouched."""
    built = []

    def orig_tw(config):
        built.append(config.model_config.local_path)
        return config

    pinned = {}
    make = ref_anchor._ref_pinning_factory(orig_tw, "/models/base", pinned)
    make(_FakeTWConfig("/run/round_1/aggregated/hf", forward_only=True))    # ref
    make(_FakeTWConfig("/run/round_1/aggregated/hf", forward_only=False))   # actor
    assert built == ["/models/base", "/run/round_1/aggregated/hf"]
    assert pinned == {"was": "/run/round_1/aggregated/hf"}


def test_pinning_factory_pins_only_the_first_forward_only_worker():
    built = []
    pinned = {}
    make = ref_anchor._ref_pinning_factory(
        lambda c: built.append(c.model_config.local_path), "/models/base", pinned)
    make(_FakeTWConfig("/a", forward_only=True))
    make(_FakeTWConfig("/b", forward_only=True))
    assert built == ["/models/base", "/b"] and pinned == {"was": "/a"}


def test_pinning_factory_without_a_ref_records_nothing():
    """No forward_only worker (no ref role) -> nothing pinned, so init_model can tell
    'pinned' from 'never saw the ref' and fall back to the rebuild."""
    pinned = {}
    make = ref_anchor._ref_pinning_factory(lambda c: c, "/models/base", pinned)
    cfg = make(_FakeTWConfig("/run/hf", forward_only=False))
    assert cfg.model_config.local_path == "/run/hf" and pinned == {}


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


def test_wrapper_pins_the_ref_and_restores_training_worker(monkeypatch):
    """The init_model WRAPPER alone, over a fake init_model that builds ref then actor through
    ``engine_workers.TrainingWorker``: the ref is built at the base, the actor is untouched, and
    the module global is put back on success and when init_model raises. It does not run verl's
    init_model, so it cannot tell how verl reaches the ref's TrainingWorker -- that is
    test_real_init_model_builds_the_ref_once_at_the_base below."""
    import types

    import pytest
    pytest.importorskip("verl")
    import verl.utils.fs as vfs
    import verl.workers.engine_workers as ew
    from verl.single_controller.base.decorator import MAGIC_ATTR

    registered = ew.ActorRolloutRefWorker.init_model      # what the wrapper wraps (MAGIC_ATTR source)
    agg, base = "/run/round_1/aggregated/hf", "/models/base"
    built = []

    def recorder(config):
        built.append((config.engine_config.forward_only, config.model_config.local_path))
        return types.SimpleNamespace(engine=types.SimpleNamespace(model_config=config.model_config))

    def fake_init_model(self):
        self.ref = ew.TrainingWorker(_FakeTWConfig(agg, forward_only=True))     # verl: ref first
        self.actor = ew.TrainingWorker(_FakeTWConfig(agg, forward_only=False))
        if self.boom:
            raise RuntimeError("init failed")

    setattr(fake_init_model, MAGIC_ATTR, getattr(registered, MAGIC_ATTR))
    monkeypatch.setenv("FEDAGENT_REF_MODEL_PATH", base)
    monkeypatch.setattr(vfs, "copy_to_local", lambda path, use_shm=False: path)
    monkeypatch.setattr(ew, "TrainingWorker", recorder)
    monkeypatch.setattr(ew.ActorRolloutRefWorker, "init_model", fake_init_model)
    monkeypatch.setattr(ref_anchor, "_PATCHED", False)
    assert ref_anchor._apply_ref_anchor_patch()
    patched = ew.ActorRolloutRefWorker.init_model

    def worker(boom):
        return types.SimpleNamespace(config=types.SimpleNamespace(model={}), boom=boom)

    w = worker(boom=False)
    patched(w)
    assert built == [(True, base), (False, agg)]        # ref built ONCE, at the base; actor untouched
    assert w.ref.engine.model_config.local_path == base
    assert ew.TrainingWorker is recorder                 # restored

    with pytest.raises(RuntimeError, match="init failed"):
        patched(worker(boom=True))
    assert ew.TrainingWorker is recorder                 # restored on the error path too


# --- the pin through verl's REAL init_model (review 2026-09-25) -------------------------------

def _tiny_hf_model(path):
    """A config.json and nothing else: all HFModelConfig reads when load_tokenizer=false."""
    from transformers import Qwen2Config

    Qwen2Config(vocab_size=128, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                num_attention_heads=2, num_key_value_heads=1,
                architectures=["Qwen2ForCausalLM"]).save_pretrained(path)
    return str(path)


def _real_init_model_worker(monkeypatch, tmp_path, pin=True, actor_build_fails=False):
    """An ActorRolloutRefWorker whose init_model is verl's OWN (the registered method, not a
    stand-in), on the actor_rollout_ref config a round-2 client runs: fedagent_ppo.yaml over verl's
    ppo_trainer, with model.path = the round's aggregate.

    Real: the config composition and dataclass conversion (HFModelConfig; the ref's forward_only
    comes from verl's ref/dp_ref.yaml), Worker and TrainingWorker construction, reset(). Faked: only
    what needs Ray, a process group or a GPU -- process-group init, NUMA pinning, the rollout
    server, the checkpoint engine, and the FSDP engine, whose initialize() records the path it would
    load weights from. Recording at the engine is what makes this test see a ref that reaches its
    TrainingWorker by a route the pin misses (a captured alias, a new constructor): that ref loads
    from the aggregate, then again from the base via the rebuild fallback.

    Returns (worker, loads, base, agg); ``loads`` gets (forward_only, local_path, model_config)
    per engine initialize()."""
    import types

    import pytest
    pytest.importorskip("verl")
    import torch
    import verl.workers.engine as verl_engine
    import verl.workers.engine_workers as ew
    from hydra import compose, initialize_config_dir

    from fedagent.fed import run_fed

    base = _tiny_hf_model(tmp_path / "base")
    agg = _tiny_hf_model(tmp_path / "round_1" / "aggregated" / "hf")
    monkeypatch.setenv("VERL_CFG", run_fed.verl_cfg_dir())      # fedagent_ppo.yaml's searchpath
    cfg_dir = os.path.join(os.path.dirname(os.path.abspath(ref_anchor.__file__)), "config")
    with initialize_config_dir(config_dir=cfg_dir, version_base=None):
        cfg = compose(config_name="fedagent_ppo",
                      overrides=[f"actor_rollout_ref.model.path={agg}",
                                 "+actor_rollout_ref.model.load_tokenizer=false"])
    # Worker.__init__ reads the rank env and writes it back (plus REDIS_STORE_SERVER_HOST); set every
    # key here so monkeypatch restores this process's environment afterwards.
    for key, val in {"WORLD_SIZE": "1", "RANK": "0", "LOCAL_WORLD_SIZE": "1", "LOCAL_RANK": "0",
                     "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": "29500",
                     "REDIS_STORE_SERVER_HOST": ""}.items():
        monkeypatch.setenv(key, val)

    loads = []

    class _Engine:
        """FSDPEngine stand-in: initialize() is where the real one loads model_config.local_path."""

        def __init__(self, model_config, engine_config, **_):
            self.model_config, self.engine_config = model_config, engine_config

        def initialize(self):
            if actor_build_fails and not self.engine_config.forward_only:
                raise RuntimeError("actor build failed")
            loads.append((self.engine_config.forward_only, self.model_config.local_path,
                          self.model_config))

        def get_data_parallel_rank(self):
            return 0

        def is_mp_src_rank_with_outputs(self):
            return True

    monkeypatch.setattr(verl_engine.EngineRegistry, "new", lambda **kw: _Engine(**kw))
    monkeypatch.setattr(ew, "initialize_global_process_group_ray", lambda **_: None)
    monkeypatch.setattr(ew, "set_numa_affinity", lambda: None)
    monkeypatch.setattr(ew, "init_device_mesh", lambda *a, **k: None)
    monkeypatch.setattr(ew, "get_rollout_class", lambda *a: lambda **k: types.SimpleNamespace())
    monkeypatch.setattr(ew, "CheckpointEngineRegistry", types.SimpleNamespace(new=lambda *a, **k: None))
    monkeypatch.setattr(ew, "aggressive_empty_cache", lambda **_: None)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda *a, **k: 0)

    # Start from verl's registered init_model even when an earlier test left the wrapper installed
    # (the deferred-hook test above arms it for the rest of the session); monkeypatch restores it.
    registered = ew.ActorRolloutRefWorker.init_model
    while getattr(registered, "__fedagent_ref_anchor__", False):
        registered = registered.__wrapped__
    monkeypatch.setattr(ew.ActorRolloutRefWorker, "init_model", registered)
    monkeypatch.setattr(ref_anchor, "_PATCHED", False)
    if pin:
        monkeypatch.setenv("FEDAGENT_REF_MODEL_PATH", base)
    else:
        monkeypatch.delenv("FEDAGENT_REF_MODEL_PATH", raising=False)
    assert ref_anchor._apply_ref_anchor_patch()
    worker = ew.ActorRolloutRefWorker(cfg.actor_rollout_ref, role="actor_rollout_ref")
    return worker, loads, base, agg


def test_real_init_model_builds_the_ref_once_at_the_base(monkeypatch, tmp_path):
    """The gap in the wrapper test above (review 2026-09-25): a verl whose init_model built ONLY the
    ref through a captured alias of TrainingWorker passed it -- the actor still names TrainingWorker
    -- while in a run the ref would miss the pin and take the leaking rebuild fallback. Here verl's
    own body runs: one ref load, from the base; one actor load, from the aggregate; separate model
    configs; the module global restored."""
    worker, loads, base, agg = _real_init_model_worker(monkeypatch, tmp_path)
    import verl.workers.engine_workers as ew

    training_worker = ew.TrainingWorker
    worker.init_model()
    assert [(fo, path) for fo, path, _ in loads] == [(True, base), (False, agg)]
    ref_cfg, actor_cfg = worker.ref.engine.model_config, worker.actor.engine.model_config
    assert ref_cfg is loads[0][2] and actor_cfg is loads[1][2] and ref_cfg is not actor_cfg
    assert (ref_cfg.local_path, actor_cfg.local_path) == (base, agg)
    assert ew.TrainingWorker is training_worker


def test_real_init_model_restores_training_worker_when_it_raises(monkeypatch, tmp_path):
    import pytest

    worker, loads, base, _ = _real_init_model_worker(monkeypatch, tmp_path, actor_build_fails=True)
    import verl.workers.engine_workers as ew

    training_worker = ew.TrainingWorker
    with pytest.raises(RuntimeError, match="actor build failed"):
        worker.init_model()
    assert [(fo, path) for fo, path, _ in loads] == [(True, base)]    # the ref was already pinned
    assert ew.TrainingWorker is training_worker


def test_real_init_model_without_the_pin_builds_the_ref_at_the_aggregate(monkeypatch, tmp_path):
    """Control, and the ref_anchor=round opt-out: FEDAGENT_REF_MODEL_PATH unset wraps nothing, and
    verl builds the ref from the round's aggregate (the rolling reference). Shows that the loads
    above are verl's real paths, not values the harness put there."""
    worker, loads, _, agg = _real_init_model_worker(monkeypatch, tmp_path, pin=False)
    worker.init_model()
    assert [(fo, path) for fo, path, _ in loads] == [(True, agg), (False, agg)]


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
