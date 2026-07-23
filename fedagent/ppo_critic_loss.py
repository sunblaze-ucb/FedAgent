"""PPO critic value-loss parity for verl 0.8 (non-fork, one-attribute monkeypatch).

Stock verl 0.8's critic loss deviates from the paper fork in two compounding ways
(see docs/bugfixes.md, 2026-07-22 "PPO critic loss scale"):

1. **Missing global normalization.** The FSDP engine backwards every micro-batch with NO
   1/M gradient-accumulation division (transformer_impl.forward_backward_batch) -- by design:
   loss functions are expected to normalize by the GLOBAL mini-batch denominator the engine
   injects into the batch (``batch_num_tokens``/``dp_size``, transformer_impl.py:620-627;
   ``global_batch_size`` from ray_trainer._update_critic:1346). The actor's ``ppo_loss``
   consumes them (losses.py:65-68); ``value_loss`` does NOT (losses.py:167-173 calls
   ``agg_loss`` bare), so each critic micro is normalized LOCALLY and the summed gradient
   scales with the micro count M.
2. **A 0.5 coefficient the fork never had.** verl 0.8 ``compute_value_loss`` returns
   ``0.5 * agg_loss(...)`` (core_algos.py:2121); the paper fork's returns ``agg_loss(...)``
   bare (fork core_algos.py:542) and divides each micro loss by the gradient-accumulation
   count instead (fork dp_critic.py:243).

Net: at the paper recipe (critic mini 16/gpu, micro 4/gpu -> M=4, dp=4) the migrated critic's
pre-clip gradient is 0.5*M = 2x the fork's objective, and the scale drifts with micro size,
DP world size and rollout.n. PPO-only (GRPO builds no critic); actor unaffected.

Fix: wrap ``verl.workers.utils.losses.value_loss`` and rescale its (0.5 x local-token-mean)
output per micro so the engine's Σ-backward + FSDP DP-mean reproduces a chosen contract:

- ``legacy_exact``   (paper default): scale = 2 * dp_size * micro_rows / global_batch_size.
  Equal-size micros (the paper configs divide evenly) => per-micro loss = local_token_mean/M,
  exactly the fork's Σ micro_mean / M objective with its 1.0 coefficient.
- ``global_token_paper_coef``: scale = 2 * dp_size * micro_tokens / batch_num_tokens.
  A true global-token-mean objective (micro/DP-invariant even for unequal micros), still with
  the fork's 1.0 coefficient. NOT byte-equivalent to the fork when micro token counts differ.
- ``upstream_standard``: no patch (stock 0.5 x per-micro behavior), for A/B forensics only.

Injection mirrors fedprox.py: run_fed sets FEDAGENT_CRITIC_LOSS_MODE for gae clients; the
repo-root sitecustomize arms a deferred import hook so the rebind happens in EVERY process
(driver, TaskRunner actor, Ray workers) the moment verl first imports its losses module --
BEFORE ray_trainer's lazy ``from verl.workers.utils.losses import value_loss`` (init_workers,
ray_trainer.py:879) binds the symbol into the critic worker's loss_fn partial.
"""
import os

_PATCHED = False
_VALID_MODES = ("legacy_exact", "global_token_paper_coef")


def _mask_counts(rm):
    """(rows, valid_token_count) for a dense or nested response_mask tensor."""
    if getattr(rm, "is_nested", False):
        parts = rm.unbind()
        return len(parts), float(sum(t.sum().item() for t in parts))
    return int(rm.shape[0]), float(rm.to(dtype=bool).sum().item())


def _wrap_value_loss(mode: str, orig):
    def value_loss_parity(config, model_output, data, dp_group=None):
        loss, metrics = orig(config, model_output, data, dp_group=dp_group)
        agg = str(getattr(config, "loss_agg_mode", "token-mean"))
        if agg != "token-mean":
            # fail CLOSED: silently keeping the stock scale would defeat the requested parity.
            raise RuntimeError(
                f"[critic-loss] mode={mode} implements loss_agg_mode=token-mean only "
                f"(the paper recipe); got {agg!r} -- refusing to guess a rescale."
            )
        try:
            dp_size = int(data["dp_size"])
            rows, tokens = _mask_counts(data["response_mask"])
            if mode == "legacy_exact":
                # fork objective: Sigma micro_token_mean / M, coefficient 1.0.
                # 2 cancels the stock 0.5; dp*rows/global_rows == 1/M for equal-size micros
                # (global_batch_size = global mini ROWS, injected by _update_critic).
                scale = 2.0 * dp_size * rows / float(int(data["global_batch_size"]))
            else:  # global_token_paper_coef
                # global token-mean, coefficient 1.0 (batch_num_tokens = DP-all-reduced valid
                # token count of the whole mini, injected by forward_backward_batch).
                scale = 2.0 * dp_size * tokens / float(data["batch_num_tokens"])
        except KeyError as e:
            raise RuntimeError(
                "[critic-loss] engine did not inject the global batch info this rescale needs "
                "(dp_size / global_batch_size / batch_num_tokens) -- refusing to train with the "
                "unnormalized stock critic loss."
            ) from e
        loss = loss * scale
        metrics = dict(metrics)
        metrics["critic/vf_loss"] = float(loss.detach().item())
        metrics["critic/vf_loss_scale"] = float(scale)
        return loss, metrics

    return value_loss_parity


def _assert_stock_value_loss(fn) -> None:
    """Source-version guard: this wrapper assumes the PRE-#6957 stock value_loss (returns
    0.5 x LOCAL per-micro token-mean). Upstream verl fixed the normalization itself in
    2eb020a ("[algo] fix: normalize critic value loss over the global mini-batch, not per
    micro-batch", #6957, 2026-07-07 -- after our pinned 7aed6b2): its value_loss consumes
    dp_size/batch_num_tokens/global_batch_size exactly like ppo_loss. Rescaling THAT output
    would double-correct, so refuse loudly on a post-fix verl (the modes then need
    re-deriving: post-fix stock == global-token with the 0.5 kept)."""
    import inspect

    try:
        src = inspect.getsource(fn)
    except (OSError, TypeError):
        return  # source unavailable (frozen/patched): fall through, the KeyError guard remains
    if "batch_num_tokens" in src or "global_batch_info" in src:
        raise RuntimeError(
            "[critic-loss] this verl's value_loss ALREADY consumes the engine's global batch "
            "info (upstream #6957 / 2eb020a or later) -- the legacy_exact/global_token rescale "
            "would double-correct. Re-derive fedagent/ppo_critic_loss.py for this verl "
            "(post-fix stock == global token-mean WITH the 0.5; legacy_exact then needs only "
            "the coefficient + micro-weighting delta) before arming."
        )


def enable_critic_loss_parity(mode: str) -> bool:
    """Rebind verl.workers.utils.losses.value_loss to the parity wrapper. Idempotent."""
    global _PATCHED
    if not mode or mode not in _VALID_MODES or _PATCHED:
        return False
    import verl.workers.utils.losses as _losses

    _assert_stock_value_loss(_losses.value_loss)
    _losses.value_loss = _wrap_value_loss(mode, _losses.value_loss)
    _PATCHED = True
    print(f"[critic-loss] enabled: mode={mode} (verl.workers.utils.losses.value_loss wrapped; "
          f"restores the fork's coefficient-1.0, M-normalized critic objective)", flush=True)
    return True


def install_deferred_critic_loss_patch(mode: str) -> bool:
    """Arm the patch LAZILY on verl's first import of its losses module (same rationale as
    fedprox.install_deferred_patch: importing verl eagerly at interpreter startup would pull
    torch in before Ray assigns per-rank CUDA_VISIBLE_DEVICES). Returns True if armed or
    applied directly (module already imported)."""
    import importlib
    import importlib.abc
    import importlib.util
    import sys

    if not mode or mode not in _VALID_MODES or _PATCHED:
        return False
    TARGET = "verl.workers.utils.losses"
    if TARGET in sys.modules:
        return enable_critic_loss_parity(mode)

    class _CriticLossImportHook(importlib.abc.MetaPathFinder):
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

                def exec_module(module, _o=_orig_exec, _mode=mode):
                    _o(module)
                    if not enable_critic_loss_parity(_mode):
                        raise RuntimeError("[critic-loss] deferred patch did not apply")

                spec.loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, _CriticLossImportHook())
    print(f"[critic-loss] deferred patch armed: value_loss will be wrapped on verl's first "
          f"losses import (mode={mode})", flush=True)
    return True


def maybe_enable_from_env() -> bool:
    """Enable from FEDAGENT_CRITIC_LOSS_MODE (manual/back-compat entry)."""
    return enable_critic_loss_parity(os.environ.get("FEDAGENT_CRITIC_LOSS_MODE", ""))
