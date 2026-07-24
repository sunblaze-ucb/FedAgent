"""PPO critic warm start (run_fed: hf_weight_keys / has_value_head / resolve_start_critic).

Field motivation (2026-07-24): a PPO continuation seeded a NEW output_dir with
``--model-path <prev_run>/round_37/aggregated/hf``. The actor came along; the critic did
NOT -- ``model_path`` was the only warm-start entry, so the value model fell back to the
actor dir and verl built ``...ForTokenClassification(num_labels=1)`` over it with a
RANDOMLY INITIALIZED ``score`` head. PPO then opened with an uncalibrated GAE baseline
while the trained critic sat unused at ``round_37/aggregated/critic_hf``. Nothing warned.

The detector cannot use config.json: verl's model_merger writes
``architectures: [<Arch>ForCausalLM]`` for the critic AND the actor (verified against a
real merged pair). The value head shows up only in the weights (``score.weight`` [1,H] +
``score.bias`` [1]).

Offline: run_fed is torch-free by design, so the reader parses the safetensors container
by hand; these tests build real containers to prove it against the actual format.
"""
import json
import os
import sys

import pytest
from omegaconf import OmegaConf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fedagent.fed.run_fed import (  # noqa: E402
    has_value_head,
    hf_weight_keys,
    is_aggregated_actor,
    resolve_start_critic,
)

ACTOR_KEYS = ["model.embed_tokens.weight", "model.layers.0.mlp.down_proj.weight",
              "model.norm.weight"]
CRITIC_KEYS = ACTOR_KEYS + ["score.bias", "score.weight"]      # verl's num_labels=1 head


def _write_safetensors(path, names):
    """A REAL (minimal) safetensors container: u64 LE header length + JSON header + data."""
    header, off = {}, 0
    for n in names:
        header[n] = {"dtype": "F32", "shape": [1], "data_offsets": [off, off + 4]}
        off += 4
    blob = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(len(blob).to_bytes(8, "little"))
        f.write(blob)
        f.write(b"\0" * off)


def _mk_hf(d, names, sharded=False):
    """A dir that passes _valid_hf_dir, holding `names` as weights."""
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps({"architectures": ["Qwen2ForCausalLM"]}))
    if sharded:
        _write_safetensors(d / "model-00001-of-00001.safetensors", names)
        (d / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {n: "model-00001-of-00001.safetensors" for n in names}}))
    else:
        _write_safetensors(d / "model.safetensors", names)
    return d


def test_fixture_is_real_safetensors(tmp_path):
    """Guard the fixture itself: if this drifts from the real format the rest is vacuous."""
    safetensors = pytest.importorskip("safetensors")
    _write_safetensors(tmp_path / "m.safetensors", CRITIC_KEYS)
    with safetensors.safe_open(str(tmp_path / "m.safetensors"), framework="np") as f:
        assert sorted(f.keys()) == sorted(CRITIC_KEYS)


def test_has_value_head_single_and_sharded(tmp_path):
    actor = _mk_hf(tmp_path / "hf", ACTOR_KEYS)
    critic = _mk_hf(tmp_path / "critic_hf", CRITIC_KEYS)
    assert has_value_head(critic) is True
    assert has_value_head(actor) is False
    # sharded checkpoints answer from the index's weight_map
    assert has_value_head(_mk_hf(tmp_path / "sh_c", CRITIC_KEYS, sharded=True)) is True
    assert has_value_head(_mk_hf(tmp_path / "sh_a", ACTOR_KEYS, sharded=True)) is False
    assert sorted(hf_weight_keys(critic)) == sorted(CRITIC_KEYS)
    # trl's fallback head name is also recognized
    assert has_value_head(_mk_hf(tmp_path / "trl", ACTOR_KEYS + ["v_head.summary.weight"])) is True


def test_has_value_head_unknown_formats(tmp_path):
    # legacy .bin / missing / corrupt header -> None ("undeterminable"), never a guess
    d = tmp_path / "bin_model"
    d.mkdir()
    (d / "config.json").write_text("{}")
    (d / "pytorch_model.bin").write_bytes(b"\x80\x02")
    assert has_value_head(d) is None
    assert has_value_head(tmp_path / "missing") is None
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "model.safetensors").write_bytes((10**9).to_bytes(8, "little") + b"{}")
    assert has_value_head(bad) is None


def test_resolve_auto_sibling_warm_start(tmp_path):
    agg = tmp_path / "prev_run" / "round_37" / "aggregated"
    actor = _mk_hf(agg / "hf", ACTOR_KEYS)
    critic = _mk_hf(agg / "critic_hf", CRITIC_KEYS)
    path, mode = resolve_start_critic(OmegaConf.create({}), str(actor))
    assert (path, mode) == (str(critic), "auto-sibling")   # THE regression this fixes
    assert is_aggregated_actor(actor) and not is_aggregated_actor(tmp_path / "Qwen2.5-1.5B")


def test_resolve_falls_back_when_no_sibling_or_pretrained_base(tmp_path):
    # aggregated actor, critic_hf absent (e.g. hf_export=final) -> actor backbone, flagged
    actor = _mk_hf(tmp_path / "run" / "round_9" / "aggregated" / "hf", ACTOR_KEYS)
    assert resolve_start_critic(OmegaConf.create({}), str(actor)) == (str(actor),
                                                                     "fresh-value-head")
    # a pretrained base is unchanged from the pre-fix behaviour (no regression for fresh runs)
    base = _mk_hf(tmp_path / "Qwen2.5-1.5B-Instruct", ACTOR_KEYS)
    assert resolve_start_critic(OmegaConf.create({}), str(base)) == (str(base),
                                                                    "fresh-value-head")


def test_resolve_explicit_wins_and_is_validated(tmp_path):
    agg = tmp_path / "run" / "round_5" / "aggregated"
    actor = _mk_hf(agg / "hf", ACTOR_KEYS)
    _mk_hf(agg / "critic_hf", CRITIC_KEYS)
    other = _mk_hf(tmp_path / "other_critic", CRITIC_KEYS)

    cfg = OmegaConf.create({"critic_model_path": str(other)})
    assert resolve_start_critic(cfg, str(actor)) == (str(other), "explicit")  # beats sibling

    # pointing --critic-path at an ACTOR is the silent-reset mistake -> hard error, not a shrug
    with pytest.raises(ValueError, match="no value head"):
        resolve_start_critic(OmegaConf.create({"critic_model_path": str(actor)}), str(actor))
    with pytest.raises(FileNotFoundError, match="not a complete HF model dir"):
        resolve_start_critic(OmegaConf.create({"critic_model_path": str(tmp_path / "nope")}),
                             str(actor))
    # undeterminable format (.bin) is accepted rather than blocking the run
    b = tmp_path / "bin_critic"
    b.mkdir()
    (b / "config.json").write_text("{}")
    (b / "pytorch_model.bin").write_bytes(b"\x80\x02")
    assert resolve_start_critic(OmegaConf.create({"critic_model_path": str(b)}),
                                str(actor)) == (str(b), "explicit")
