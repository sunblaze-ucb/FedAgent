"""Regression tests for the fp32 FedAvg-merge overlay (fedagent/merge_fp32.py).

Stock verl's model_merger truncates the fp32-aggregated FSDP shards to bf16 on the
FSDP->HF hop (fsdp_model_merger casts each collected shard .bfloat16(); base merger
builds a bf16 skeleton). The overlay recompiles those two methods from source with the
casts flipped to fp32, guarded by exact marker counts.

Offline: the rebind mechanics run on fake classes (torch only); the vendored-source
contract test reads verl's files as TEXT via find_spec (no verl import).
"""
import ast
import importlib.util
import os
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fedagent.merge_fp32 import _rebind_with_source_patch  # noqa: E402


class _FakeMerger:
    """Mimics the stock shape: two .bfloat16() call sites in one method."""

    def collect(self, t):
        out = [t.bfloat16()]                  # stand-in for the DTensor local-shard cast
        out.append(t.detach().bfloat16())     # stand-in for the plain-tensor cast
        return out


def test_rebind_flips_casts_and_preserves_fp32_values():
    t = torch.tensor([1.0000001, 2.0], dtype=torch.float32)   # 1.0000001 is NOT bf16-representable
    before = _FakeMerger().collect(t)
    assert all(x.dtype == torch.bfloat16 for x in before)
    assert not torch.equal(before[0].float(), t)              # stock path really loses bits

    _rebind_with_source_patch(_FakeMerger, "collect", [(".bfloat16()", ".float()", 2)])
    after = _FakeMerger().collect(t)
    assert all(x.dtype == torch.float32 for x in after)
    assert torch.equal(after[0], t) and torch.equal(after[1], t)


def test_marker_count_mismatch_fails_closed():
    class _DriftedMerger:
        def collect(self, t):
            return t.bfloat16()               # only ONE cast: "verl changed shape"

    with pytest.raises(RuntimeError, match="re-derive"):
        _rebind_with_source_patch(_DriftedMerger, "collect", [(".bfloat16()", ".float()", 2)])


def _vendored_verl_source(*parts) -> str:
    """Read a verl source file WITHOUT importing verl (find_spec only locates the package)."""
    spec = importlib.util.find_spec("verl")
    if spec is None or not spec.submodule_search_locations:
        pytest.skip("verl not resolvable in this environment")
    p = Path(list(spec.submodule_search_locations)[0]).joinpath(*parts)
    if not p.is_file():
        pytest.skip(f"vendored verl file missing: {p}")
    return p.read_text()


def _method_source(file_src: str, cls_name: str, method_name: str) -> str:
    tree = ast.parse(file_src)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == cls_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == method_name:
                    return ast.get_source_segment(file_src, item)
    raise AssertionError(f"{cls_name}.{method_name} not found")


def test_vendored_merger_still_matches_the_patch_contract():
    """The exact marker counts enable_fp32_merge() asserts at runtime, checked against the
    vendored verl's REAL source -- if a verl bump changes the merger, this fails in CI
    instead of at round-1 merge time."""
    fsdp = _method_source(
        _vendored_verl_source("model_merger", "fsdp_model_merger.py"),
        "FSDPModelMerger", "_load_and_merge_state_dicts")
    assert fsdp.count(".bfloat16()") == 2

    base = _method_source(
        _vendored_verl_source("model_merger", "base_model_merger.py"),
        "BaseModelMerger", "save_hf_model_and_tokenizer")
    assert base.count("torch_dtype=torch.bfloat16") == 1
