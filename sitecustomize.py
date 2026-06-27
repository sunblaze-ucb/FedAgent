"""Repo-root sitecustomize: auto-imported by CPython at interpreter startup for every
process that has this repo root on PYTHONPATH (the federated driver AND its Ray workers,
since run_fed sets PYTHONPATH=REPO_ROOT and Ray workers inherit it).

Purpose: inject FedAgent's FedProx optimizer patch into every actor worker WITHOUT a Ray
`runtime_env.worker_process_setup_hook`. That hook works (the patch fires) but the
cluster-level `runtime_env` clobbers verl's per-worker `CUDA_VISIBLE_DEVICES` assignment,
so all FSDP ranks land on GPU 0 ("Duplicate GPU detected: rank N and rank 0 both on CUDA
device ..."). sitecustomize runs at plain interpreter startup, touches no runtime_env, and
so leaves verl's GPU isolation intact.

Safety / fail-closed: this runs in EVERY python process on PYTHONPATH, but the patch is only
attempted when FEDPROX_MU>0 -- i.e. in the federated client training subprocess and its Ray
workers (run_fed sets it there); env-service conda envs never set it, so they no-op before the
import. We distinguish the two cases by whether `verl` is importable:
  - verl ABSENT  -> not a trainer process (e.g. a service env that inherited a globally
    exported FEDPROX_MU). FedProx is N/A here -> silent no-op.
  - verl PRESENT -> a trainer process where the patch MUST apply. Any failure PROPAGATES
    (fail closed): silently downgrading a requested FedProx run to FedAvg would corrupt the
    experiment. fedprox prints "[fedprox] enabled ..." on success for log verification.
The patch is applied LAZILY (install_deferred_patch): FSDPEngine is imported only when verl
itself first imports its FSDP-engine module -- i.e. AFTER the Ray worker has its per-rank
CUDA_VISIBLE_DEVICES set. Importing it EAGERLY here (at interpreter startup, before device
assignment) breaks per-rank GPU isolation at multi-GPU ("Duplicate GPU detected: rank N and
rank 0 ..."); deferral avoids that while still patching before the first optimizer step.
"""
import importlib.util
import os

try:
    _mu = float(os.environ.get("FEDPROX_MU", "0") or "0")
except ValueError:
    _mu = 0.0

if _mu > 0:
    if importlib.util.find_spec("verl") is None:
        pass  # not a trainer process (verl absent) -> FedProx N/A, no-op
    else:
        # trainer process: DEFER the patch to verl's first FSDP-engine import (after the worker
        # sets CUDA_VISIBLE_DEVICES). fail CLOSED: install_deferred_patch raising / returning
        # False propagates rather than silently downgrading FedProx to FedAvg.
        from fedagent.fedprox import install_deferred_patch

        if not install_deferred_patch(_mu):
            raise RuntimeError(
                f"sitecustomize: FEDPROX_MU={_mu} and verl is present, but the FedProx deferred "
                "patch could not be armed -- refusing to run silently as FedAvg."
            )
