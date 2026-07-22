"""Overlay worker-class patch for lever #4 (persistent trainer; docs/acceleration.md).

Attaches ``ActorRolloutRefWorker.reload_client_model`` -- a ONE_TO_ALL remote method that
re-points the live actor (+ref) FSDP engines at a new aggregated model dir and rebuilds them
(fresh weights + fresh optimizer + fresh LR scheduler + dropped FedProx anchor), so ONE
long-lived RayPPOTrainer can train successive federated clients WITHOUT the per-client
subprocess cold-start (the measured ~76-88% overhead, docs/acceleration.md §2.6).

Why a DEFERRED import hook (mirrors fedprox.install_deferred_patch): the method must exist on
the worker CLASS inside every Ray FSDP-worker process, but importing
``verl.workers.engine_workers`` EAGERLY at interpreter startup pulls in the FSDP engine before
Ray assigns per-rank ``CUDA_VISIBLE_DEVICES`` -> "Duplicate GPU detected: rank N and rank 0".
So we arm a one-shot MetaPathFinder that patches the class the moment verl itself imports
engine_workers (after device assignment). Enabled via env var ``FEDAGENT_PERSISTENT=1``.

Reload primitive -- verified against verl 0.8 source:
  ``TrainingWorker.reset()`` (engine_workers.py:165) -> ``engine.initialize()``
  (transformer_impl.py:183) -> ``_build_model_optimizer`` (transformer_impl.py:543):
    * ``_build_module`` reads ``model_config.local_path`` (transformer_impl.py:252) -> NEW weights
    * ``_build_optimizer`` (569) -> fresh Adam (zero m/v)
    * ``_build_lr_scheduler`` (571) -> fresh schedule at step 0
  The FedProx anchor ``engine._fedprox_w_t`` (fedprox.py:37) lives on the engine instance and
  SURVIVES initialize() -> we explicitly ``del`` it so the proximal term re-anchors per client.
"""
_PATCHED = False


def _apply_persistent_patch() -> bool:
    """Attach reload_client_model onto ActorRolloutRefWorker (idempotent). Runs in the
    process that is importing engine_workers (driver TaskRunner actor + each FSDP worker)."""
    global _PATCHED
    if _PATCHED:
        return True
    from verl.single_controller.base.decorator import Dispatch, register
    from verl.workers.engine_workers import ActorRolloutRefWorker, TrainingWorker

    def _reset_engine(eng, model_local_path):
        import gc

        import torch

        eng.model_config.local_path = model_local_path
        # Cross-round leak fix: initialize() (_build_model_optimizer, transformer_impl.py:427-429)
        # REBINDS self.module/optimizer/lr_scheduler without freeing the old ones -- and the CUDA
        # caching allocator keeps the dropped blocks RESERVED (never returned) absent empty_cache().
        # Per round that accretes ~one full model + Adam state (PPO leaks TWICE: actor here +
        # critic via reload_critic_model; the ref engine is forward_only -> no optimizer), so
        # reserved memory climbs ~0.6GB/round -> cross-round OOM. Drop the old refs FIRST (so the
        # rebuild reuses freed memory instead of peaking at 2x), then gc + empty_cache.
        for _attr in ("module", "optimizer", "lr_scheduler", "checkpoint_manager"):
            if getattr(eng, _attr, None) is not None:
                setattr(eng, _attr, None)
        gc.collect()
        torch.cuda.empty_cache()
        eng.initialize()  # _build_model_optimizer: new module(new weights)+optimizer+scheduler
        if hasattr(eng, "_fedprox_w_t"):
            del eng._fedprox_w_t  # re-anchor FedProx to this client's aggregated model

    def _load_model_shards(eng, shard_dir):
        """Tier-2d (hf_export=final): model-only load of the AGGREGATED FSDP shard checkpoint
        into the freshly initialize()d engine -- skipping the per-round model_merger HF round
        trip entirely. The aggregated dir carries ONLY weight shards (save_contents=[model]);
        the optimizer/LR scheduler keep the fresh state initialize() just built, which is
        exactly what the HF-merge path produced (fresh Adam + schedule at step 0). Rank r loads
        model_world_size_W_rank_r.pt via the engine's own FSDPCheckpointManager (the resume
        path), temporarily narrowed to load_contents=['model'] so the missing optimizer/extra
        files are never requested."""
        cm = eng.checkpoint_manager
        saved = cm.checkpoint_load_contents
        cm.checkpoint_load_contents = ["model"]
        try:
            eng.load_checkpoint(local_path=shard_dir, hdfs_path=None, del_local_after_load=False)
        finally:
            cm.checkpoint_load_contents = saved

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def reload_client_model(self, model_local_path: str, shard_dir: str = None):
        """Hot-swap the live actor (+ref) engines to a new aggregated model + rebuild them.

        Exactly what a fresh subprocess gets for free: new weights from model_local_path,
        a fresh optimizer (zero Adam moments), a fresh LR scheduler at step 0, and no stale
        FedProx anchor. Same-architecture clients -> hf_config/tokenizer stay valid; only the
        weight source (local_path) changes.

        ``shard_dir`` (Tier-2d, hf_export=final): model_local_path is the BASE model (constant
        all run -- rebuilds module/optimizer/scheduler), then the aggregated FSDP shards from
        shard_dir overwrite the weights in place (model-only). Net weights == the HF-merge path,
        without the merger."""
        _reset_engine(self.actor.engine, model_local_path)
        if shard_dir:
            _load_model_shards(self.actor.engine, shard_dir)
        if getattr(self, "ref", None) is not None:
            _reset_engine(self.ref.engine, model_local_path)  # ref forward_only: weights only
            if shard_dir:
                _load_model_shards(self.ref.engine, shard_dir)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def reload_critic_model(self, model_local_path: str, shard_dir: str = None):
        """Critic (PPO/gae) counterpart. The critic is a plain TrainingWorker with self.engine
        (no CriticWorker class in verl 0.8); rebuild the value engine = fresh value weights +
        fresh critic optimizer/scheduler. ``shard_dir``: as in reload_client_model (aggregated
        critic shards over a base-initialized value engine)."""
        _reset_engine(self.engine, model_local_path)
        if shard_dir:
            _load_model_shards(self.engine, shard_dir)

    ActorRolloutRefWorker.reload_client_model = reload_client_model
    TrainingWorker.reload_critic_model = reload_critic_model
    _PATCHED = True
    print("[persistent] reload_client_model + reload_critic_model attached", flush=True)
    return True


def install_deferred_persistent_patch() -> bool:
    """Arm a one-shot import hook that patches ActorRolloutRefWorker the moment verl first
    imports ``verl.workers.engine_workers`` (after Ray sets per-rank CUDA_VISIBLE_DEVICES).
    Mirrors fedprox.install_deferred_patch. Idempotent; returns True if armed/applied."""
    import importlib.abc
    import importlib.util
    import sys

    if _PATCHED:
        return True
    TARGET = "verl.workers.engine_workers"
    if TARGET in sys.modules:  # already imported -> patch now
        return _apply_persistent_patch()

    class _PersistentImportHook(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path=None, target=None):
            if name != TARGET:
                return None
            try:
                sys.meta_path.remove(self)  # one-shot; let the real finders resolve it
            except ValueError:
                pass
            spec = importlib.util.find_spec(TARGET)
            if spec is not None and spec.loader is not None:
                _orig_exec = spec.loader.exec_module

                def exec_module(module, _o=_orig_exec):
                    _o(module)  # run engine_workers body (class now defined, device set)
                    if not _apply_persistent_patch():
                        raise RuntimeError("[persistent] deferred patch did not apply")

                spec.loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, _PersistentImportHook())
    print("[persistent] deferred patch armed (engine_workers import)", flush=True)
    return True
