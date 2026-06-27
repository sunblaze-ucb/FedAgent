"""FedProx proximal term for verl 0.8 (non-fork, one-method monkeypatch).

FedProx anchors each client's drifting local weights w to the round-start global model
w_t by adding mu*(w - w_t) to the actor gradient before every optimizer step. In
FedAgent's subprocess-per-round design each client-round is a FRESH process that loads
the aggregated model, so w_t is simply the params at the first optimizer step -- no
external per-round reset is needed.

Seam (verl 0.8): verl/workers/engine/fsdp/transformer_impl.py
  - FSDPEngine.optimizer_step() clips grads then calls optimizer.step().
We wrap it: snapshot w_t on the first call (params still == the loaded global model),
then on every call add the proximal grad per LOCAL shard (FSDP1 sharded view / FSDP2
DTensor -> elementwise on each shard is correct). GRPO has no critic and the ref
policy never steps, so patching the base engine's optimizer_step affects only the
actor. Mirrors verl-agent 0.3.1 dp_actor.update_policy (snapshot + grad.add_).

Enabled by run_fed via env var FEDPROX_MU>0 (so aggregation_method=fedprox uses it;
plain FedAvg leaves mu=0 = no-op).
"""
import os

_PATCHED = False


def enable_fedprox(mu: float) -> bool:
    """Monkeypatch FSDPEngine.optimizer_step to add the FedProx proximal gradient. Idempotent."""
    global _PATCHED
    if mu is None or mu <= 0 or _PATCHED:
        return False
    from verl.workers.engine.fsdp.transformer_impl import FSDPEngine

    _orig_optimizer_step = FSDPEngine.optimizer_step

    def optimizer_step(self):
        snap = getattr(self, "_fedprox_w_t", None)
        if snap is None:
            # first step of this round-process: params still == loaded global model w_t
            snap = {n: p.detach().clone() for n, p in self.module.named_parameters()}
            self._fedprox_w_t = snap
        for n, p in self.module.named_parameters():
            if p.grad is not None and n in snap:
                # per-local-shard: grad += mu * (w - w_t)   (FSDP/DTensor elementwise-safe)
                p.grad.add_(p.data - snap[n].to(p.grad.device), alpha=mu)
        return _orig_optimizer_step(self)

    FSDPEngine.optimizer_step = optimizer_step
    _PATCHED = True
    print(f"[fedprox] enabled: proximal mu={mu} (FSDPEngine.optimizer_step patched)", flush=True)
    return True


def maybe_enable_from_env() -> bool:
    """Enable FedProx if FEDPROX_MU>0 in the environment. Call before run_ppo()."""
    try:
        mu = float(os.environ.get("FEDPROX_MU", "0") or "0")
    except ValueError:
        mu = 0.0
    return enable_fedprox(mu)


def install_deferred_patch(mu: float) -> bool:
    """Install the FedProx patch LAZILY -- applied the moment verl first imports its FSDP-engine
    module (``verl.workers.engine.fsdp.transformer_impl``), which happens AFTER the Ray worker has
    its per-rank ``CUDA_VISIBLE_DEVICES`` set. Importing FSDPEngine EAGERLY at interpreter startup
    (from sitecustomize) instead pulls in torch/verl before device assignment and breaks per-rank
    GPU isolation at multi-GPU ("Duplicate GPU detected: rank N and rank 0 ..."). Returns True if
    the hook was armed (or the patch applied directly because the module was already loaded)."""
    import importlib
    import importlib.abc
    import importlib.util
    import sys

    if mu is None or mu <= 0 or _PATCHED:
        return False
    TARGET = "verl.workers.engine.fsdp.transformer_impl"
    if TARGET in sys.modules:          # verl already imported it -> safe to patch right now
        return enable_fedprox(mu)

    class _FedProxImportHook(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path=None, target=None):
            if name != TARGET:
                return None
            try:
                sys.meta_path.remove(self)      # one-shot; let the real finders resolve it
            except ValueError:
                pass
            spec = importlib.util.find_spec(TARGET)
            if spec is not None and spec.loader is not None:
                _orig_exec = spec.loader.exec_module

                def exec_module(module, _o=_orig_exec, _mu=mu):
                    _o(module)                  # run the module body (FSDPEngine now defined + device set)
                    if not enable_fedprox(_mu):
                        raise RuntimeError("[fedprox] deferred patch did not apply")

                spec.loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, _FedProxImportHook())
    print(f"[fedprox] deferred patch armed: FSDPEngine.optimizer_step will be patched on verl's "
          f"first FSDP import (mu={mu})", flush=True)
    return True


def worker_setup():
    """Legacy Ray worker_process_setup_hook entry. SUPERSEDED by the repo-root
    ``sitecustomize.py``: wiring this as ``ray_init.runtime_env.worker_process_setup_hook``
    clobbered verl's per-worker ``CUDA_VISIBLE_DEVICES`` (all FSDP ranks -> GPU 0,
    "Duplicate GPU detected"). FedProx is now injected via ``sitecustomize`` (interpreter
    startup, no runtime_env), so this is kept only for back-compat / manual use."""
    maybe_enable_from_env()
