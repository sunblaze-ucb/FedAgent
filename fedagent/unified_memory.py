"""Two verl 0.8 memory fixes for unified-memory GPUs (GB10 / DGX Spark), where "host" and "device"
are one LPDDR pool that the trainer, vLLM and everything else share.

1. KEEP THE KL-REFERENCE MODEL ON THE GPU. verl's FSDP engine forces every forward_only engine
   (the ref) onto the host regardless of ``actor_rollout_ref.ref.fsdp_config.param_offload``
   (verl/workers/engine/fsdp/transformer_impl.py, ``FSDPEngine._build_fsdp_module``):

       cpu_offload = None
       if self.engine_config.forward_only:
           cpu_offload = CPUOffload(offload_params=True)      # unconditional for the ref

   so ``ref.fsdp_config.param_offload=false`` is a silent no-op. On a discrete GPU that is the
   right trade. On unified memory offloading saves nothing and costs memory: FSDP keeps the host
   copy plus pinned staging buffers, and ``compute_ref_log_prob`` streams the params through them
   every step (the staging is never returned; malloc_trim does not help).

   Seam: ``_build_fsdp_module`` imports ``CPUOffload`` LOCALLY from ``torch.distributed.fsdp`` on
   every call, so for one call on a forward_only engine we rebind that package attribute to a
   factory yielding ``CPUOffload(offload_params=False)``. Nothing else in the method changes (it
   still clears ``_is_offload_param`` for the ref: no manual offload either). The actor/critic
   never construct ``CPUOffload`` on the FSDP1 path. FSDP2 (``CPUOffloadPolicy``) is not handled
   -- logged, stock behaviour.

2. RELEASE THE TRAINER'S ALLOCATOR CACHE BEFORE THE WEIGHT SYNC. ``ActorRolloutRefWorker.
   update_weights`` wakes vLLM's weights (``rollout.resume(tags=["weights"])``), gathers the actor's
   params and pushes them, and only THEN calls ``aggressive_empty_cache``. Right after
   ``update_actor`` + ``save_checkpoint`` the trainer's caching allocator holds everything the
   backward pass and optimizer step freed: at 1.5B, 17.3 GiB allocated vs 36.7 GiB reserved. With
   actor and rollout colocated, vLLM wakes into that same device memory on any GPU; what unified
   memory adds is that the pool is also the host's (Ray, dataloaders, page cache), so the cached
   blocks are memory the whole node runs short of -- the 3B run was killed by Ray's memory monitor
   exactly here (dgx_spark.md §2.2). We run the same ``aggressive_empty_cache`` once more at the
   START of ``update_weights``. Cost: one gc + empty_cache + sync per step.

The knob is run_fed's ``unified_memory`` -> ``FEDAGENT_UNIFIED_MEMORY``:
    auto (default)  apply both iff ``torch.cuda.get_device_properties().is_integrated``
                    (GB10 reports 1; discrete GPUs 0 -> byte-identical stock behaviour)
    on              always apply
    off             stock verl (nothing armed)
Both are memory-placement/timing changes only: weights, dtypes, forward passes and the synced
weights are identical to stock.

MEASURED (one GB10, TinyGuess windowed, 2 clients x 2 rounds, rollout util 0.3, dgx_spark.md §2).
"""
import os
import sys
import traceback

_MODES = ("auto", "on", "off")
_TRANSFORMER_IMPL = "verl.workers.engine.fsdp.transformer_impl"
_ENGINE_WORKERS = "verl.workers.engine_workers"
_attempted = set()


def mode() -> str:
    m = (os.environ.get("FEDAGENT_UNIFIED_MEMORY", "") or "off").strip().lower()
    if m not in _MODES:
        raise ValueError(f"FEDAGENT_UNIFIED_MEMORY must be one of {_MODES}, got {m!r}")
    return m


def _device_is_integrated() -> bool:
    import torch

    if not torch.cuda.is_available():
        return False
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    return bool(getattr(props, "is_integrated", False))


def active() -> bool:
    """Decided per call, inside the Ray worker (the driver may not see a GPU)."""
    m = mode()
    return m == "on" or (m == "auto" and _device_is_integrated())


# --------------------------------------------------------------------------- 1. ref placement
def _wrap_build_fsdp_module(orig):
    import functools

    @functools.wraps(orig)
    def _build_fsdp_module(self, module):
        cfg = self.engine_config
        if not getattr(cfg, "forward_only", False) or not active():
            return orig(self, module)
        if getattr(cfg, "strategy", None) != "fsdp":
            print(f"[unified-memory] strategy={cfg.strategy!r} not handled; ref stays "
                  "host-offloaded (stock verl)", flush=True)
            return orig(self, module)

        import torch.distributed.fsdp as fsdp_pkg

        cpu_offload_cls = fsdp_pkg.CPUOffload

        def _on_device(*_args, **_kwargs):
            return cpu_offload_cls(offload_params=False)

        fsdp_pkg.CPUOffload = _on_device
        try:
            out = orig(self, module)
        finally:
            fsdp_pkg.CPUOffload = cpu_offload_cls
        co = getattr(out, "cpu_offload", None)
        print(f"[unified-memory] KL reference kept on GPU "
              f"(cpu_offload.offload_params={getattr(co, 'offload_params', '?')})", flush=True)
        return out

    _build_fsdp_module.__fedagent_unified_memory__ = True
    return _build_fsdp_module


def _apply_ref_placement() -> None:
    from verl.workers.engine.fsdp.transformer_impl import FSDPEngine

    FSDPEngine._build_fsdp_module = _wrap_build_fsdp_module(FSDPEngine._build_fsdp_module)


# --------------------------------------------------------------------------- 2. pre-sync release
def _wrap_update_weights(orig, empty_cache):
    import functools

    @functools.wraps(orig)          # keeps verl's @register metadata (MAGIC_ATTR) -- see ref_anchor
    async def update_weights(self, *args, **kwargs):
        if active():
            empty_cache(force_sync=True)
        return await orig(self, *args, **kwargs)

    update_weights.__fedagent_unified_memory__ = True
    return update_weights


def _apply_presync_release() -> None:
    from verl.single_controller.base.decorator import MAGIC_ATTR
    from verl.utils.memory_utils import aggressive_empty_cache
    from verl.workers.engine_workers import ActorRolloutRefWorker

    orig = ActorRolloutRefWorker.update_weights
    wrapped = _wrap_update_weights(orig, aggressive_empty_cache)
    assert getattr(wrapped, MAGIC_ATTR, None) is not None and \
        getattr(wrapped, MAGIC_ATTR, None) == getattr(orig, MAGIC_ATTR, None), (
        "[unified-memory] update_weights wrapper lost verl's @register metadata; refusing to arm")
    ActorRolloutRefWorker.update_weights = wrapped


_PATCHES = {_TRANSFORMER_IMPL: _apply_ref_placement, _ENGINE_WORKERS: _apply_presync_release}


def _apply(name: str) -> None:
    """Patch one target, once. This runs inside verl's own import (the hook below) and, under the
    default ``auto``, on every platform -- so a failure must not propagate: it would fail
    ``import verl.workers.engine_workers`` and every run with it, discrete GPUs included, where the
    patch is inert. Not fail-closed (see sitecustomize): log it and leave that target stock."""
    if name in _attempted:
        return
    _attempted.add(name)
    try:
        _PATCHES[name]()
    except Exception:
        traceback.print_exc()
        print(f"WARNING [unified-memory] could not patch {name}; stock verl memory placement for it",
              file=sys.stderr, flush=True)


def install_deferred_unified_memory_patch() -> bool:
    """Arm a one-shot import hook per target module (same deferral rationale as
    ref_anchor/fedprox: importing torch before Ray assigns CUDA_VISIBLE_DEVICES breaks rank
    isolation). No-op under ``off``. Idempotent; returns True if armed or applied."""
    import importlib.abc
    import importlib.util

    if mode() == "off":
        return True
    pending = [n for n in _PATCHES if n not in _attempted]
    for name in [n for n in pending if n in sys.modules]:
        _apply(name)
    pending = [n for n in pending if n not in _attempted]
    if not pending:
        return True

    class _UnifiedMemoryImportHook(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path=None, target=None):
            if name not in pending:
                return None
            pending.remove(name)
            sys.meta_path.remove(self)              # step aside so find_spec below cannot recurse
            try:
                spec = importlib.util.find_spec(name)
            finally:
                if pending:                          # re-arm for the other target
                    sys.meta_path.insert(0, self)
            if spec is None or spec.loader is None:
                return None
            orig_exec = spec.loader.exec_module

            def exec_module(module, _name=name):
                orig_exec(module)
                _apply(_name)

            spec.loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, _UnifiedMemoryImportHook())
    return True
