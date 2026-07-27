"""The windowed union tag must live for exactly ONE union (bugfix 2026-07-26).

``windowed_manager`` marks the per-turn TRAIN expansion with
``meta_info["__windowed_expanded__"] = size_divisor``. The patched ``DataProto.union`` reads
that tag to ADOPT the (longer) per-turn batch instead of letting verl's equal-size union
truncate it, and the patched ``DataProto.slice`` reads it to neutralize fit()'s
``combined_gen_output.slice(0, num_sampled_prompts)``.

The tag used to be popped only inside the ``len(self) != len(other)`` branch. When every
episode of a batch produced exactly one turn the lengths MATCH, the stock union runs, and it
MERGES meta_info -- so the tag rode onto the merged training batch and made ``_windowed_slice``
silently no-op every later legitimate ``slice(0, k<len)`` on it (REMAX's baseline slice, any
future trainer slicing). Popping the tag whenever it is present bounds its lifetime to the one
union it exists for.

Needs verl's DataProto (fedagent-verl08 env). Importing windowed_manager applies the patch
process-wide, which is exactly the state under test; untagged batches are unaffected by
construction, so it cannot leak into other tests.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

torch = pytest.importorskip("torch")
protocol = pytest.importorskip("verl.protocol", reason="needs verl (fedagent-verl08 env)")
wm = pytest.importorskip("fedagent.agent_loops.windowed_manager", reason="needs verl")

DataProto = protocol.DataProto
TAG = wm._TAG


def _proto(n, key, meta=None):
    return DataProto.from_dict(tensors={key: torch.arange(n * 2).reshape(n, 2)},
                               meta_info=dict(meta or {}))


def test_longer_tagged_batch_is_adopted_and_padded():
    """The normal windowed case: more per-turn rows than prompts -> adopt, pad to the divisor."""
    prompts = _proto(4, "p", {"temperature": 0.4})
    turns = _proto(10, "t", {TAG: 4, "timing": 1.0})
    out = prompts.union(turns)
    assert len(out) == 12                      # 10 padded up to a multiple of 4
    assert "t" in out.batch and "p" not in out.batch   # other's ROWS adopted wholesale
    assert out.meta_info["temperature"] == 0.4        # self's meta keys survive
    assert TAG not in out.meta_info


def test_equal_length_tagged_batch_consumes_the_tag():
    """Every episode == 1 turn -> lengths match -> stock union merges meta_info. The tag must
    NOT survive into the merged training batch."""
    prompts = _proto(4, "p", {"temperature": 0.4})
    turns = _proto(4, "t", {TAG: 4})
    out = prompts.union(turns)
    assert len(out) == 4
    assert "p" in out.batch and "t" in out.batch       # stock union ran (columns merged)
    assert TAG not in out.meta_info                    # <-- the bug: used to be 4 here
    assert TAG not in turns.meta_info
    # ...and this is why: the stock union DOES carry the tag through, so popping it first is the
    # only thing standing between an equal-length round and a permanently tagged training batch.
    leaked = wm._orig_union(_proto(4, "p"), _proto(4, "t", {TAG: 4}))
    assert leaked.meta_info[TAG] == 4                  # pre-fix behavior, pinned


def test_a_batch_that_kept_the_tag_would_break_slicing():
    """Why the above matters: _windowed_slice no-ops a from-start shortening slice on ANY tagged
    batch, so a surviving tag silently disables truncation downstream."""
    tagged = _proto(8, "x", {TAG: 4})
    assert len(tagged.slice(0, 3)) == 8        # neutralized -- correct for the gen batch...
    untagged = _proto(8, "x")
    assert len(untagged.slice(0, 3)) == 3      # ...and plain slicing is untouched otherwise


def test_untagged_union_is_byte_for_byte_stock():
    a = _proto(3, "a", {"k": 1})
    b = _proto(3, "b", {"j": 2})
    out = a.union(b)
    assert len(out) == 3 and set(out.batch.keys()) == {"a", "b"}
    assert out.meta_info == {"k": 1, "j": 2}


def test_divisor_padding_duplicates_from_the_front():
    """_adjust_to_divisor pads by duplicating rows deterministically (verl-agent adjust_batch)."""
    padded = wm._adjust_to_divisor(_proto(5, "x"), 4)
    assert len(padded) == 8
    assert torch.equal(padded.batch["x"][5:], padded.batch["x"][:3])
    same = _proto(8, "x")
    assert wm._adjust_to_divisor(same, 4) is same      # already aligned -> untouched
