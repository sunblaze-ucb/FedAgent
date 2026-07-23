"""fp32 FedAvg round-boundary merge for verl 0.8 (non-fork, source-guarded rebind).

The fork handed each round's fp32-averaged state_dict straight to the next round's load
(aggregator.py kept fp32 end-to-end). The migrated round boundary inserts an FSDP->HF
hop: client fp32 shards -> fedagent/fed/aggregate_fedavg_fsdp.py (in-place average,
dtype preserved -> still fp32) -> ``python -m verl.model_merger merge`` -> HF dir ->
next round's engines (verl's TRAINING load forces torch_dtype=fp32,
transformer_impl.py:236-238, so whatever dtype the HF file has is what the fp32 masters
start from). Stock verl's merger TRUNCATES in that hop:

  - fsdp_model_merger._load_and_merge_state_dicts casts every collected shard to bf16
    (``tensor._local_tensor.bfloat16()`` :169 for DTensor, ``tensor.bfloat16()`` :181);
  - base_model_merger.save_hf_model_and_tokenizer builds the save skeleton with
    ``torch_dtype=torch.bfloat16`` (:379), stamping config.json accordingly.

So every round boundary quantizes the aggregated weights to bf16 (~3 significant decimal
digits) before the next round reloads them. At actor lr 1e-6 the per-round weight motion
is comparable to the bf16 ULP at typical weight magnitudes -- a fraction of each round's
averaged update is rounded away, every round, in a path the fork ran losslessly
(docs/bugfixes.md 2026-07-23 "bf16 merge truncation").

The pinned verl (7aed6b2) has no dtype knob on the merger CLI, and others/verl stays a
clean upstream checkout, so the fix is an overlay: RECOMPILE the two stock methods from
their own source with the casts flipped to fp32. The rebind is guarded by exact marker
counts -- if a verl upgrade changes either method, arming fails loudly instead of
silently merging with whatever the new code does (same upgrade-trap discipline as
ppo_critic_loss._assert_stock_value_loss).

Injection mirrors ppo_critic_loss.py: run_fed sets FEDAGENT_MERGE_FP32=1 in the merger
subprocess env for AGGREGATED merges only (run_fed.merge_to_hf(fp32=True); client-eval
merges stay bf16 -- they only feed the bf16 vLLM eval rollout), and the repo-root
sitecustomize arms a deferred import hook so the rebind lands on the merger's first
``verl.model_merger.fsdp_model_merger`` import.
"""
import os

_PATCHED = False


def _rebind_with_source_patch(cls, method_name: str, replacements) -> None:
    """Recompile ``cls.<method_name>`` from its own source with textual ``(old, new,
    expected_count)`` replacements applied, then rebind it. Fails CLOSED: any marker
    whose occurrence count differs from ``expected_count`` means the vendored verl
    changed shape, and a blind rebind could silently drop (or double-apply) the fix."""
    import inspect
    import textwrap

    fn = getattr(cls, method_name)
    src = textwrap.dedent(inspect.getsource(fn))
    for old, new, expected in replacements:
        found = src.count(old)
        if found != expected:
            raise RuntimeError(
                f"[merge-fp32] {cls.__name__}.{method_name}: expected {expected} x {old!r} "
                f"in the stock source, found {found} -- this verl's merger changed; "
                f"re-derive fedagent/merge_fp32.py before arming (refusing a blind rebind)."
            )
        src = src.replace(old, new)
    mod = inspect.getmodule(fn)
    ns = dict(mod.__dict__)   # rebound fn resolves module-level names (torch, DTensor, ...) here
    exec(compile(src, f"<merge-fp32 {mod.__name__}.{method_name}>", "exec"), ns)
    setattr(cls, method_name, ns[method_name])


def enable_fp32_merge() -> bool:
    """Rebind the two truncating merger methods to fp32-preserving recompiles. Idempotent."""
    global _PATCHED
    if _PATCHED:
        return False
    from verl.model_merger.base_model_merger import BaseModelMerger
    from verl.model_merger.fsdp_model_merger import FSDPModelMerger

    _rebind_with_source_patch(
        FSDPModelMerger, "_load_and_merge_state_dicts",
        # .float() (not "drop the cast") so an unexpectedly-bf16 shard still merges in fp32
        [(".bfloat16()", ".float()", 2)],
    )
    _rebind_with_source_patch(
        BaseModelMerger, "save_hf_model_and_tokenizer",
        # skeleton dtype also stamps config.json's torch_dtype -> keep it consistent w/ weights
        [("torch_dtype=torch.bfloat16", "torch_dtype=torch.float32", 1)],
    )
    _PATCHED = True
    print("[merge-fp32] enabled: model_merger keeps the FedAvg-aggregated weights in fp32 "
          "(fork round-boundary precision restored)", flush=True)
    return True


def install_deferred_merge_fp32_patch() -> bool:
    """Arm the rebind LAZILY on the merger subprocess's first import of
    verl.model_merger.fsdp_model_merger (same rationale as fedprox/ppo_critic_loss:
    importing verl eagerly at interpreter startup would pull torch in first). By the time
    that module finishes executing it has itself imported base_model_merger, so both
    rebinds apply together. Returns True if armed or applied directly."""
    import importlib
    import importlib.abc
    import importlib.util
    import sys

    if _PATCHED:
        return False
    TARGET = "verl.model_merger.fsdp_model_merger"
    if TARGET in sys.modules:
        return enable_fp32_merge()

    class _MergeFp32ImportHook(importlib.abc.MetaPathFinder):
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

                def exec_module(module, _o=_orig_exec):
                    _o(module)
                    if not enable_fp32_merge():
                        raise RuntimeError("[merge-fp32] deferred patch did not apply")

                spec.loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, _MergeFp32ImportHook())
    print("[merge-fp32] deferred patch armed: model_merger will keep aggregated weights "
          "in fp32 on its first fsdp_model_merger import", flush=True)
    return True


def maybe_enable_from_env() -> bool:
    """Enable from FEDAGENT_MERGE_FP32=1 (manual/back-compat entry)."""
    return enable_fp32_merge() if os.environ.get("FEDAGENT_MERGE_FP32") == "1" else False
