"""Regression test for AgenticDataset per-epoch goal resampling (audit data-dataset-seeds-6).

The original fed sampler drew a FRESH goal batch every local epoch (up to E x n_envs distinct
tasks per client-round); a verl dataset replays identical rows every epoch. The fix: run_fed
launches clients with trainer.total_epochs=1 + FEDAGENT_DATA_EPOCHS=E, and AgenticDataset
emits each spec's rows E times with a distinct per-epoch-slot seed (e * n_envs stride,
epoch-major). This locks the seed layout, the E=1/unset no-op, the FEDAGENT_DATA_EPOCHS_FILE
guard (the persistent worker's val spec must NOT expand), and the widened cross-spec
seed-window guard.

Needs torch + omegaconf (AgenticDataset imports them); no verl, no network.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fedagent.data.agentic_dataset import AgenticDataset  # noqa: E402

SPEC = """
envs:
  - name: WebShopRemote
    n_envs: 4
    max_turns: 15
    agent_name: gym_text_windowed
"""

TWO_SPECS = """
envs:
  - name: A
    n_envs: 400
    max_turns: 5
  - name: B
    n_envs: 2
    max_turns: 5
"""


def _write(tmp_path, text, name="spec.yaml"):
    p = tmp_path / name
    p.write_text(text)
    return str(p)


def _seeds(ds):
    return [it["seed"] for it in ds.items]


def test_default_layout_unchanged(tmp_path, monkeypatch):
    monkeypatch.setenv("FEDAGENT_BASE_SEED", "7")
    monkeypatch.delenv("FEDAGENT_DATA_EPOCHS", raising=False)
    monkeypatch.delenv("FEDAGENT_DATA_EPOCHS_FILE", raising=False)
    ds = AgenticDataset(_write(tmp_path, SPEC))
    assert _seeds(ds) == [700_000 + i for i in range(4)]


def test_epoch_expansion_layout(tmp_path, monkeypatch):
    spec = _write(tmp_path, SPEC)
    monkeypatch.setenv("FEDAGENT_BASE_SEED", "7")
    monkeypatch.setenv("FEDAGENT_DATA_EPOCHS", "3")
    monkeypatch.setenv("FEDAGENT_DATA_EPOCHS_FILE", spec)
    ds = AgenticDataset(spec)
    # E x n_envs rows, epoch-major, e*n_envs + i stride: 3 disjoint fresh 4-row draws.
    assert len(ds) == 12
    assert _seeds(ds) == [700_000 + e * 4 + i for e in range(3) for i in range(4)]
    assert all(it["env_name"] == "WebShopRemote" and it["max_turns"] == 15 for it in ds.items)
    assert len(set(_seeds(ds))) == 12   # all epoch slots distinct -> distinct served goals


def test_e1_is_a_noop(tmp_path, monkeypatch):
    spec = _write(tmp_path, SPEC)
    monkeypatch.setenv("FEDAGENT_BASE_SEED", "7")
    monkeypatch.setenv("FEDAGENT_DATA_EPOCHS", "1")
    monkeypatch.setenv("FEDAGENT_DATA_EPOCHS_FILE", spec)
    assert _seeds(AgenticDataset(spec)) == [700_000 + i for i in range(4)]


def test_file_guard_protects_other_specs(tmp_path, monkeypatch):
    # the persistent worker builds the worker-eval (val spec) dataloader in the SAME process;
    # expansion must stay confined to the file named by FEDAGENT_DATA_EPOCHS_FILE.
    train = _write(tmp_path, SPEC, "train.yaml")
    val = _write(tmp_path, SPEC, "val.yaml")
    monkeypatch.setenv("FEDAGENT_BASE_SEED", "7")
    monkeypatch.setenv("FEDAGENT_DATA_EPOCHS", "3")
    monkeypatch.setenv("FEDAGENT_DATA_EPOCHS_FILE", train)
    assert len(AgenticDataset(train)) == 12
    assert len(AgenticDataset(val)) == 4          # val untouched
    assert _seeds(AgenticDataset(val)) == [700_000 + i for i in range(4)]


def test_expanded_window_guard_refuses_cross_spec_alias(tmp_path, monkeypatch):
    # 400 n_envs x E=3 = 1200 rows > the 1000-wide per-spec seed window with a later spec
    # present -> must refuse loudly (was n_envs-only before the expansion existed).
    spec = _write(tmp_path, TWO_SPECS)
    monkeypatch.setenv("FEDAGENT_BASE_SEED", "7")
    monkeypatch.setenv("FEDAGENT_DATA_EPOCHS", "3")
    monkeypatch.setenv("FEDAGENT_DATA_EPOCHS_FILE", spec)
    try:
        AgenticDataset(spec)
    except ValueError as e:
        assert "collide" in str(e)
    else:
        raise AssertionError("expected ValueError on cross-spec seed aliasing")
