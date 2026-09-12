"""Give verl 0.8 an explicit, fixed KL-reference model role.

THE DEFECT. verl exposes ONE model path -- ``actor_rollout_ref.model.path`` -- and
``ActorRolloutRefWorker.init_model`` (engine_workers.py:497/517/543) builds the actor AND the
reference policy from it:

    model_config = omega_conf_to_dataclass(self.config.model)        # :497
    ref_config.model_config = deepcopy(model_config)                 # :517  <-- ref
    actor_config.model_config = model_config                         # :543  <-- actor

The federated loop carries the FedAvg'd weights forward by pointing that same key at the round's
aggregated actor model, so the reference policy silently became "this round's starting
weights". The actor then optimizes

    J - kl_loss_coef * KL(pi || pi_{r-1})       # a per-round proximal term (an accidental FedProx)

instead of the old recipe's fixed-reference objective

    J - kl_loss_coef * KL(pi || pi_fixed)       # one fixed trust region across all T rounds

The verl-agent 0.3.1 reference stack never hit this: it left ``model.path`` at the base HF id and
moved the FedAvg weights via ``trainer.resume_from_path`` + ``resume_mode=resume_path``, and
``ray_trainer._load_checkpoint`` restores the ACTOR and CRITIC only -- the ref worker is built from
``model.path`` and never touched. Both behaviours are side effects of *how the weights travel*;
neither was chosen. This is a confirmed migration/objective mismatch, but it is not by itself a
proof that the reference choice explains the historical performance gap: high-performing old
fixed-reference runs and a weaker rolling-reference run both exist. This module makes the role
explicit so the two algorithms can be tested rather than conflated.

OBSERVABLE SIGNATURE. ``actor/kl_loss`` IS KL(pi||ref). A fixed reference generally accumulates
more displacement than a per-round reference, but monotonicity is not guaranteed and KL magnitude
is a diagnostic of the selected objective, not a performance-causality test.

WHY THIS CANNOT BE A CONFIG OVERRIDE. ``config/ref/ref.yaml`` has no ``model`` section, and
``init_model`` OVERWRITES whatever the ref config carried with ``deepcopy(model_config)``. There
is no hydra key that expresses "the ref uses a different path".

THE SEAM. ``HFModelConfig.path`` is frozen (BaseConfig.__setattr__ raises unless the field is in
``_mutable_fields``); ``local_path`` is NOT frozen (verl/workers/config/model.py:81) and is what
``_build_module`` actually loads from (transformer_impl.py:252) -- which is exactly why
persistent_patch._reset_engine swaps weights by assigning ``eng.model_config.local_path`` and
calling ``eng.initialize()``. We reuse that production-proven path: after verl finishes
``init_model``, re-point the ref engine at the base and rebuild it once. Cost: one extra HF load
per trainer process (~3 s at 1.5B); on a FRESH run round 1 launches with model.path == base, so
the hook detects "already at base" and rebuilds nothing. The rebuild matters on RESUMED chunks
(e.g. pwpf2 resumed at round 54: model.path is round_54/aggregated/hf, and without this hook the
ref would freeze THERE -- the wrong anchor a third way).

The user-facing config is ``ref_model_path`` + ``ref_anchor``. run_fed resolves those roles and
bridges the fixed path through ``FEDAGENT_REF_MODEL_PATH`` because upstream verl 0.8 has no
separate ref path. Unset => deliberate rolling/stock behaviour. Fail-closed via sitecustomize:
if requested and not armable, the process dies rather than train against the wrong objective.
The companion guard lives in
persistent_patch.reload_client_model, which must NOT re-point the ref each round when this is
active -- the two pieces are co-required; see the note there.
"""
import os
import sys

_PATCHED = False


def _base_model_path() -> str:
    return (os.environ.get("FEDAGENT_REF_MODEL_PATH", "") or "").strip()


def _repoint_ref_engine(eng, base_local: str) -> None:
    """Drop the ref module built from the actor path and rebuild it from ``base_local``.

    Mirrors persistent_patch._reset_engine minus everything that only applies to a TRAINING
    engine: the ref is forward_only (no optimizer, no LR scheduler, no FedProx anchor), so the
    only retired object is the module itself. Drop it FIRST, then gc + empty_cache, so the
    rebuild reuses the freed blocks instead of peaking at two co-resident copies."""
    import gc

    import torch

    eng.model_config.local_path = base_local        # mutable field; _build_module reads it
    for _attr in ("module", "checkpoint_manager"):
        if getattr(eng, _attr, None) is not None:
            setattr(eng, _attr, None)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    eng.initialize()                                # _build_model_optimizer -> new module @ base


def _apply_ref_anchor_patch() -> bool:
    """Wrap ActorRolloutRefWorker.init_model so the ref ends up anchored on the base model."""
    global _PATCHED
    if _PATCHED:
        return True
    base = _base_model_path()
    if not base:
        return True                                 # not requested -> stock verl, no-op

    import functools

    from verl.single_controller.base.decorator import MAGIC_ATTR
    from verl.utils.fs import copy_to_local
    from verl.workers.engine_workers import ActorRolloutRefWorker

    orig_init_model = ActorRolloutRefWorker.init_model

    # verl only BINDS worker methods that carry the @register metadata: _bind_worker_method
    # (single_controller/ray/base.py) scans for MAGIC_ATTR and skips everything else, and
    # ray_trainer.init_workers() then calls wg.init_model() unconditionally. A plain wrapper
    # (the 2026-08-30 first cut -- caught by review, reproduced: patched_has_magic=False,
    # bound_init_model=False) therefore fails the whole run at init_workers, before a single
    # rollout. functools.wraps copies the function __dict__, which is where @register stores
    # the attrs -- and the assert makes this load-bearing property fail CLOSED, not silently.
    @functools.wraps(orig_init_model)
    def init_model(self):
        orig_init_model(self)
        ref = getattr(self, "ref", None)
        if ref is None:      # no ref built (use_kl_loss=false / role without "ref") -> nothing to anchor
            return
        eng = ref.engine
        was = getattr(eng.model_config, "local_path", None)
        base_local = copy_to_local(base, use_shm=getattr(eng.model_config, "use_shm", False))
        if was == base_local:
            # Fresh run: round 1 launches with model.path == base. Nothing to do -- but say so,
            # because "no line in the log" must never be ambiguous with "not armed".
            print(f"[model-role] ref already at fixed path ({base_local}); no rebuild", flush=True)
            return
        _repoint_ref_engine(eng, base_local)
        # Assert the swap took. A silent no-op here is the exact failure this module exists to
        # prevent, and in the metrics it would look identical to "patch not installed".
        assert eng.model_config.local_path == base_local, (
            f"[ref-anchor] re-point failed: local_path is {eng.model_config.local_path!r}, "
            f"expected {base_local!r}"
        )
        print(f"[model-role] KL reference FIXED independently: {was} -> {base_local}", flush=True)

    assert getattr(init_model, MAGIC_ATTR, None) == getattr(orig_init_model, MAGIC_ATTR, None) \
        and getattr(init_model, MAGIC_ATTR, None) is not None, (
        "[ref-anchor] wrapper lost verl's @register metadata (MAGIC_ATTR) -- init_model would "
        "never be bound to the worker group and init_workers() would fail. Refusing to arm.")
    # functools.wraps makes the wrapper LOOK like the original (__qualname__ included), which is
    # exactly right for verl but hides the wrap from tests/forensics -- so mark it explicitly.
    init_model.__fedagent_ref_anchor__ = True
    ActorRolloutRefWorker.init_model = init_model
    _PATCHED = True
    print(f"[model-role] explicit ref adapter enabled (ref={base})", flush=True)
    return True


def install_deferred_patch() -> bool:
    """Arm a one-shot import hook that patches ActorRolloutRefWorker the moment verl first
    imports ``verl.workers.engine_workers`` -- i.e. AFTER Ray has set the worker's per-rank
    CUDA_VISIBLE_DEVICES. Patching eagerly at interpreter startup would import torch before
    device assignment and break FSDP rank isolation ("Duplicate GPU detected"), the same trap
    fedprox/persistent_patch document. Idempotent; returns True if armed or applied."""
    import importlib.abc
    import importlib.util

    if _PATCHED:
        return True
    if not _base_model_path():
        return True
    TARGET = "verl.workers.engine_workers"
    if TARGET in sys.modules:
        return _apply_ref_anchor_patch()

    class _RefAnchorImportHook(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path=None, target=None):
            if name != TARGET:
                return None
            sys.meta_path.remove(self)              # one-shot
            spec = importlib.util.find_spec(TARGET)
            if spec is None or spec.loader is None:
                return None
            orig_exec = spec.loader.exec_module

            def exec_module(module):
                orig_exec(module)
                _apply_ref_anchor_patch()

            spec.loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, _RefAnchorImportHook())
    return True
