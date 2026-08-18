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

    # FEDAGENT_MEM_DEBUG=1: arm allocator-history recording in every worker process, so the
    # _mem_debug_dump snapshots carry allocation STACKS for C++-held memory (autograd graph,
    # dynamo caches, FSDP internals) -- the part a python gc-walk cannot attribute.
    import os as _os
    if _os.environ.get("FEDAGENT_MEM_DEBUG"):
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.memory._record_memory_history(max_entries=200000)
                print("[mem-debug] allocator history recording ON (this process)", flush=True)
        except Exception as _e:  # instrumentation must never kill a run
            print(f"[mem-debug] record_memory_history failed (non-fatal): {_e}", flush=True)

    def _mem_debug_dump(tag):
        """FEDAGENT_MEM_DEBUG=1: post-release forensic dump, called at the cleanest point of the
        reload cycle (old engine nulled + gc'd + empty_cache'd, new one not yet built). Whatever
        CUDA memory is still allocated HERE is the cross-fit residue. Emits (a) a gc-walk of all
        live CUDA tensors >=32MB with their referrer types (names PYTHON-reachable holders) and
        (b) a full allocator snapshot with stacks (names C++-held memory: autograd graph, dynamo
        caches, FSDP internals). Never raises: instrumentation must not kill a run."""
        import os

        try:
            import gc
            import time

            import torch

            alloc = torch.cuda.memory_allocated() / 2**30
            print(f"[mem-debug] {tag}: post-release current allocated = {alloc:.2f} GiB", flush=True)

            import types

            def _describe(r, child):
                """One line for a referrer: its type, plus the attribute/key it holds the child
                under (that's the holder's NAME, which is what this hunt is for)."""
                t = type(r).__name__
                try:
                    if isinstance(r, dict):
                        keys = [k for k, v in list(r.items())[:2000] if v is child]
                        owner = ""
                        for o2 in gc.get_referrers(r)[:8]:
                            if getattr(o2, "__dict__", None) is r:
                                owner = f" (__dict__ of {type(o2).__name__})"
                                break
                        return f"dict{owner} key={keys[:3]}"
                    if isinstance(r, (list, tuple, set)):
                        return f"{t}[len={len(r)}]"
                except Exception:
                    pass
                return t

            n, chains = 0, 0
            for o in gc.get_objects():
                try:
                    if not (isinstance(o, torch.Tensor) and o.is_cuda):
                        continue
                    if o.numel() * o.element_size() < 32 * 2**20:
                        continue
                    n += 1
                    try:  # metadata size vs ACTUAL storage: 0-byte = pinned-but-released phantom
                        sto = o.untyped_storage().size() / 2**30
                    except Exception:
                        sto = float("nan")
                    print(f"[mem-debug]   live {type(o).__name__} {tuple(o.shape)} {o.dtype} "
                          f"{o.numel() * o.element_size() / 2**30:.3f} GiB (storage {sto:.3f})",
                          flush=True)
                    if chains < 6:   # full 2-hop referrer chain for the first few only
                        chains += 1
                        for r1 in [r for r in gc.get_referrers(o)
                                   if not isinstance(r, types.FrameType)][:5]:
                            print(f"[mem-debug]     <- {_describe(r1, o)}", flush=True)
                            for r2 in [r for r in gc.get_referrers(r1)
                                       if not isinstance(r, types.FrameType) and r is not o][:5]:
                                print(f"[mem-debug]        <- {_describe(r2, r1)}", flush=True)
                except Exception:
                    continue
            print(f"[mem-debug] {tag}: {n} python-reachable CUDA tensors >=32MB", flush=True)
            d = os.environ.get("FEDAGENT_MEM_DEBUG_DIR", "/tmp")
            os.makedirs(d, exist_ok=True)
            p = os.path.join(d, f"snap_{tag}_{os.getpid()}_{int(time.time())}.pickle")
            torch.cuda.memory._dump_snapshot(p)
            print(f"[mem-debug] snapshot -> {p}", flush=True)
        except Exception as e:
            print(f"[mem-debug] dump failed (non-fatal): {e}", flush=True)

    def _hard_release_fsdp_storages(module):
        """Free the CUDA storages of a RETIRED engine's parameters/grads/FSDP shard copies
        directly, instead of trusting refcounts+gc to do it.

        Why: on world_size=1 PyTorch degrades FULL_SHARD to NO_SHARD, and there the fp32
        flat params of a released engine outlive _reset_engine's gc.collect() -- FSDP-internal
        containers (handle/wrapper param lists, flat_param._local_shard/_mp_shard views) plus,
        for tensors that were CUDA-IPC'd to vLLM (>bucket-size direct send), the exporter-side
        IPC handle, all keep references whose owners die later than the reload. Net effect
        measured on the shared 24GB 4090: ~+1.33 GiB (one fp32 non-embed model copy) surviving
        per reload, compounding into the round-4 vLLM wake_up OOM (docs/bugfixes.md 2026-08-18).
        The retired module is never touched again -- initialize() builds a brand-new tree -- so
        freeing storages out from under any zombie holder is safe, and is exactly how FSDP
        itself retires _mp_shard (torch.distributed.utils._free_storage). Returns bytes freed.
        Kill switch for A/B repro: FEDAGENT_DISABLE_HARD_RELEASE=1."""
        import torch

        freed = 0
        seen = set()

        def _free(t):
            nonlocal freed
            if not isinstance(t, torch.Tensor) or not t.is_cuda:
                return
            try:
                st = t.untyped_storage()
            except Exception:
                return
            n = st.size()
            if n == 0 or st.data_ptr() in seen:
                return
            seen.add(st.data_ptr())
            try:
                st.resize_(0)
                freed += n
            except Exception:
                pass

        for sub in module.modules():
            # FSDP1 wrappers: the flat param + its mixed-precision/shard copies + grad.
            for h in [getattr(sub, "_handle", None)] + list(getattr(sub, "_handles", None) or []):
                fp = getattr(h, "flat_param", None) if h is not None else None
                if fp is None:
                    continue
                _free(fp.grad)
                for attr in ("_mp_shard", "_local_shard", "_full_param_padded",
                             "_saved_grad_shard"):
                    _free(getattr(fp, attr, None))
                _free(fp)
            # Raw leaves (registered _flat_param aliases, unwrapped params, buffers).
            # Storage-level dedup makes revisits and shared/tied views free.
            for p in sub.parameters(recurse=False):
                _free(p.grad)
                _free(p)
            for b in sub.buffers(recurse=False):
                _free(b)
        return freed

    def _reset_engine(eng, model_local_path):
        import gc
        import os as _os

        import torch

        eng.model_config.local_path = model_local_path
        # Deterministic release of the RETIRED engine's CUDA storages (2026-08-18): dropping
        # the attrs below is NOT enough on ws=1/NO_SHARD -- see _hard_release_fsdp_storages.
        mod = getattr(eng, "module", None)
        if mod is not None and not _os.environ.get("FEDAGENT_DISABLE_HARD_RELEASE"):
            try:
                freed = _hard_release_fsdp_storages(mod)
                if freed:
                    print(f"[persistent-patch] reload hard-release: freed {freed / 2**30:.2f} GiB "
                          f"of retired {eng.__class__.__name__} param storage", flush=True)
            except Exception as _e:
                print(f"[persistent-patch] hard-release failed (non-fatal): {_e!r}", flush=True)
        # Cross-round leak fix: initialize() (_build_model_optimizer, transformer_impl.py:427-429)
        # REBINDS self.module/optimizer/lr_scheduler without freeing the old ones -- and the CUDA
        # caching allocator keeps the dropped blocks RESERVED (never returned) absent empty_cache().
        # Per round that accretes ~one full model + Adam state (PPO leaks TWICE: actor here +
        # critic via reload_critic_model; the ref engine is forward_only -> no optimizer), so
        # reserved memory climbs ~0.6GB/round -> cross-round OOM. Drop the old refs FIRST (so the
        # rebuild reuses freed memory instead of peaking at 2x), then gc + empty_cache.
        # FedProx: drop the OLD client's anchor FIRST -- a full fp32 param-shard clone
        # (~1.44 GB/GPU for 1.5B @ 4-way FULL_SHARD). Deleting it AFTER initialize() (the
        # previous ordering) kept it co-resident with the rebuild AND pinned its blocks across
        # the empty_cache() below (allocated mid-training, interleaved with activation
        # segments -> exactly the reserved-creep this function exists to prevent). Deleting it
        # here is semantics-neutral: the next client re-snapshots lazily at its first
        # optimizer_step (fedprox.py: `getattr(self, "_fedprox_w_t", None) is None`).
        if hasattr(eng, "_fedprox_w_t"):
            del eng._fedprox_w_t
        for _attr in ("module", "optimizer", "lr_scheduler", "checkpoint_manager"):
            if getattr(eng, _attr, None) is not None:
                setattr(eng, _attr, None)
        gc.collect()
        torch.cuda.empty_cache()
        if _os.environ.get("FEDAGENT_MEM_DEBUG"):
            _mem_debug_dump(f"reset_{eng.__class__.__name__}")
        eng.initialize()  # _build_model_optimizer: new module(new weights)+optimizer+scheduler

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
