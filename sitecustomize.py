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

# Lever #4 (docs/acceleration.md): persistent-trainer worker patch. Same deferral rationale as
# FedProx (attach to the worker class on verl's first engine_workers import, AFTER Ray sets
# per-rank CUDA_VISIBLE_DEVICES). Gated on FEDAGENT_PERSISTENT=1 (set only by persistent_main),
# so normal subprocess/service processes never arm it. verl ABSENT -> N/A (no-op).
if os.environ.get("FEDAGENT_PERSISTENT") == "1" and importlib.util.find_spec("verl") is not None:
    from fedagent.fed.persistent_patch import install_deferred_persistent_patch

    if not install_deferred_persistent_patch():
        raise RuntimeError(
            "sitecustomize: FEDAGENT_PERSISTENT=1 and verl is present, but the persistent "
            "worker patch could not be armed -- refusing to run without reload_client_model."
        )

# PPO critic value-loss parity (docs/bugfixes.md 2026-07-22 "critic loss scale"): stock verl
# 0.8's value_loss skips the engine's global-batch normalization (the actor's ppo_loss uses
# it, losses.py:65-68) and adds a 0.5 the paper fork never had -> the critic's pre-clip
# gradient scales with 0.5*M (critic micro count; 2x at the paper recipe). run_fed sets
# FEDAGENT_CRITIC_LOSS_MODE only for gae (PPO) clients; GRPO/service processes never arm it.
# Same deferral + fail-closed rationale as FedProx above.
_critic_mode = os.environ.get("FEDAGENT_CRITIC_LOSS_MODE", "")
if _critic_mode and _critic_mode != "upstream_standard" and importlib.util.find_spec("verl") is not None:
    from fedagent.ppo_critic_loss import install_deferred_critic_loss_patch

    if not install_deferred_critic_loss_patch(_critic_mode):
        raise RuntimeError(
            f"sitecustomize: FEDAGENT_CRITIC_LOSS_MODE={_critic_mode} and verl is present, but "
            "the critic-loss patch could not be armed -- refusing to train with the "
            "unnormalized stock critic loss."
        )

# fp32 FedAvg merge (docs/bugfixes.md 2026-07-23 "bf16 merge truncation"): stock verl's
# model_merger casts the fp32-AGGREGATED shards to bf16 on the FSDP->HF hop at every round
# boundary; the fork kept fp32 end-to-end. run_fed sets FEDAGENT_MERGE_FP32=1 only in the
# aggregated-merge subprocess env (merge_to_hf(fp32=True); client-eval merges stay bf16).
# Same deferral + fail-closed rationale as FedProx above.
if os.environ.get("FEDAGENT_MERGE_FP32") == "1" and importlib.util.find_spec("verl") is not None:
    from fedagent.merge_fp32 import install_deferred_merge_fp32_patch

    if not install_deferred_merge_fp32_patch():
        raise RuntimeError(
            "sitecustomize: FEDAGENT_MERGE_FP32=1 and verl is present, but the fp32-merge "
            "patch could not be armed -- refusing to silently truncate the aggregated "
            "weights to bf16."
        )
