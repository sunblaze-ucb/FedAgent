"""Regression tests for the 2026-09-10 infrastructure fixes (docs/bugfixes.md, 2026-09-10 §1-§3):

* the resume scan is keyed on the disk's ``round_<k>/`` dirs, not on ``range(total_rounds)``;
* a resume whose ``total_rounds`` is below the highest round on disk is REFUSED (not archived);
* the persistent worker's launch env always carries the round number.

Offline: no GPU, no verl, no Ray -- run_fed is imported for its pure helpers only.
"""
import sys
from pathlib import Path

import pytest
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fedagent.fed import run_fed  # noqa: E402


def _hf(d: Path) -> None:
    """A dir that passes run_fed._valid_hf_dir (config.json + one weight file)."""
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text("{}")
    (d / "model.safetensors").write_bytes(b"\0" * 8)


def _cfg(out: Path, **kw):
    cfg = OmegaConf.create(dict(run_fed.DEFAULTS))
    cfg.output_dir = str(out)
    for k, v in kw.items():
        cfg[k] = v
    return cfg


def test_resume_scan_is_disk_keyed_not_total_rounds_bounded(tmp_path):
    _hf(tmp_path / "round_70" / "aggregated" / "hf")
    (tmp_path / "round_0" / "eval").mkdir(parents=True)   # base-model eval dir: never an anchor
    (tmp_path / "round_71").mkdir()                       # partial attempt (no hf): skipped
    (tmp_path / "_stale_rounds").mkdir()                  # a prior --fresh's archive: ignored
    cfg = _cfg(tmp_path, total_rounds=2)                  # the 2-round smoke on a 70-round dir
    assert run_fed._disk_rounds(cfg) == [71, 70]
    k, actor_hf, critic_hf = run_fed.find_resume_round(cfg, is_ppo=False)
    assert k == 70 and critic_hf is None
    assert actor_hf == tmp_path / "round_70" / "aggregated" / "hf"


def test_ppo_round_is_complete_only_with_its_critic(tmp_path):
    _hf(tmp_path / "round_3" / "aggregated" / "hf")                 # actor merged, critic not
    _hf(tmp_path / "round_2" / "aggregated" / "hf")
    _hf(tmp_path / "round_2" / "aggregated" / "critic_hf")
    cfg = _cfg(tmp_path, total_rounds=70)
    assert run_fed.find_resume_round(cfg, is_ppo=True)[0] == 2
    assert run_fed.find_resume_round(cfg, is_ppo=False)[0] == 3


def test_empty_or_missing_dir_starts_fresh(tmp_path):
    assert run_fed.find_resume_round(_cfg(tmp_path / "nope", total_rounds=5), False) == (0, None, None)
    assert run_fed._disk_rounds(_cfg(tmp_path / "nope")) == []


def test_resume_refuses_a_schedule_below_the_disk(tmp_path):
    _hf(tmp_path / "round_70" / "aggregated" / "hf")
    with pytest.raises(ValueError, match="RESUME REFUSED"):
        run_fed.refuse_resume_below_disk(_cfg(tmp_path, total_rounds=2))
    # a partial round above the schedule counts too: it is evidence of a longer run
    (tmp_path / "round_71").mkdir()
    with pytest.raises(ValueError, match="round_71"):
        run_fed.refuse_resume_below_disk(_cfg(tmp_path, total_rounds=70))
    assert run_fed.refuse_resume_below_disk(_cfg(tmp_path, total_rounds=71)) == 71
    assert run_fed.refuse_resume_below_disk(_cfg(tmp_path / "empty", total_rounds=2)) == 0


@pytest.mark.parametrize("round_num", [1, 2, 70])
def test_persistent_cmd_env_carries_the_round_number(tmp_path, round_num):
    cfg = _cfg(tmp_path, env_kind="tinyguess", model_path="/base")   # base: the default ref pin resolves to it
    plan = [{"out_dir": str(tmp_path / "c0")}]
    _cmd, env = run_fed._persistent_cmd_env(cfg, plan, tmp_path / "plan.json", "/base", None,
                                            round_num, {}, n_gpus=1, worker_eval=False)
    assert env["FEDAGENT_XROUND_START_ROUND"] == str(round_num)
    assert env["FEDAGENT_PERSISTENT"] == "1"


def test_load_cfg_warns_on_keys_the_runner_does_not_read(tmp_path, capsys):
    """A key outside DEFAULTS (e.g. the dev doc's `holdout_file` for `alfworld_holdout_file`) is
    merged by OmegaConf but read by nothing; load_cfg names it instead of staying silent."""
    import argparse
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text("total_rounds: 3\nholdout_file: data/x.json\nref_anchor_base: true\n")
    args = argparse.Namespace(config=str(cfg_path), model_path=None, critic_path=None, output_dir=None,
                              rounds=None, clients=None, n_gpus=None, base_seed=None, port_base=None,
                              fedprox_mu=None, local_client_id=None, fresh=False)
    cfg = run_fed.load_cfg(args)
    out = capsys.readouterr().out
    assert cfg.total_rounds == 3
    assert "holdout_file" in out and "ref_anchor_base" in out and "[warn]" in out
    cfg_path.write_text("total_rounds: 4\n")
    assert run_fed.load_cfg(args).total_rounds == 4
    assert "[warn]" not in capsys.readouterr().out


def test_phys_gpu_ids_maps_through_the_driver_visible_set(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert run_fed._phys_gpu_ids(0, 2) == "0,1"                 # unset: legacy literal ids
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
    assert run_fed._phys_gpu_ids(0, 1) == "2" and run_fed._phys_gpu_ids(1, 2) == "3"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-aaaa,GPU-bbbb")
    assert run_fed._phys_gpu_ids(1, 2) == "GPU-bbbb"           # UUID tokens pass through
    with pytest.raises(ValueError, match="exposes only 2"):
        run_fed._phys_gpu_ids(0, 3)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    with pytest.raises(ValueError, match="set but empty"):     # no GPUs is not "unset"
        run_fed._phys_gpu_ids(0, 1)
