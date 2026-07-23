"""Round-launch reliability (run_fed helpers): FedAvg rendezvous banding, one-shot
collision relaunch for cross-round/lane workers, RESUME quarantine of incomplete rounds.

Field motivation (2026-07-23, ~round 43 of a 70-round run): a random ephemeral-port
collision killed a round at startup. Self-healing worked (outer loop + resume, ~10 min
lost), but three reliability gaps remained: the FedAvg torchrun drew a random ephemeral
rendezvous port (--standalone) with no retry; a launch-time collision on the
cross-round/lane BgProc paths had no retry; and a re-run round inherited the dead
attempt's partial artifacts (latest_actor_dir picks the HIGHEST global_step with shards,
so a stale higher-step checkpoint from a config-tweaked earlier attempt would get
FedAvg'd). Offline: no verl/torch import (run_fed's own imports are stdlib + omegaconf).
"""
import os
import socket
import sys

import pytest
from omegaconf import OmegaConf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fedagent.fed import run_fed  # noqa: E402
from fedagent.port_band import _CURSORS  # noqa: E402


def _band_cfg(tmp_path=None, base=20000, stride=8):
    d = {"port_band_base": base, "port_band_stride": stride}
    if tmp_path is not None:
        d["output_dir"] = str(tmp_path)
    return OmegaConf.create(d)


def test_agg_rdzv_args_banded_slot32():
    _CURSORS.clear()
    args = run_fed._agg_rdzv_args(_band_cfg())
    assert len(args) == 1 and args[0].startswith("--master_port=")
    port = int(args[0].split("=")[1])
    start = 20000 + run_fed.AGG_BAND_SLOT * 8          # slot 32 -> its own sub-band
    assert start <= port < start + 8
    # consecutive draws rotate (a collision relaunch must NOT re-race the same port)
    port2 = int(run_fed._agg_rdzv_args(_band_cfg())[0].split("=")[1])
    assert port2 != port


def test_agg_rdzv_args_standalone_when_band_off_or_exhausted():
    assert run_fed._agg_rdzv_args(OmegaConf.create({"port_band_base": 0})) == ["--standalone"]
    # band exhausted -> fail-OPEN to --standalone (aggregation must not die because the
    # band is busy; contrast the trainer-side probe, which fails closed)
    _CURSORS.clear()
    start = 20000 + run_fed.AGG_BAND_SLOT * 8
    squat = []
    try:
        for p in range(start, start + 8):
            s = socket.socket()
            try:
                s.bind(("", p))
                squat.append(s)
            except OSError:
                s.close()
                pytest.skip("band ports busy on this host")
        assert run_fed._agg_rdzv_args(_band_cfg()) == ["--standalone"]
    finally:
        for s in squat:
            s.close()


def test_quarantine_stale_rounds(tmp_path):
    for name in ("round_0", "round_1", "round_2", "round_43", "round_x"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "marker.txt").write_text(name)
    cfg = OmegaConf.create({"output_dir": str(tmp_path)})

    run_fed.quarantine_stale_rounds(cfg, last_complete=1)
    # above last_complete -> archived (contents intact); at/below + non-numeric -> untouched
    assert not (tmp_path / "round_2").exists() and not (tmp_path / "round_43").exists()
    assert (tmp_path / "_stale_rounds" / "round_2.0" / "marker.txt").read_text() == "round_2"
    assert (tmp_path / "_stale_rounds" / "round_43.0").is_dir()
    for kept in ("round_0", "round_1", "round_x"):
        assert (tmp_path / kept).is_dir()

    # a second crashed attempt archives under the NEXT suffix, never clobbering the first
    (tmp_path / "round_2").mkdir()
    run_fed.quarantine_stale_rounds(cfg, last_complete=1)
    assert (tmp_path / "_stale_rounds" / "round_2.1").is_dir()
    assert (tmp_path / "_stale_rounds" / "round_2.0" / "marker.txt").is_file()

    # round_0 (base-eval dir) is never archived; missing output_dir is a no-op
    run_fed.quarantine_stale_rounds(cfg, last_complete=0)
    assert (tmp_path / "round_0").is_dir()
    run_fed.quarantine_stale_rounds(OmegaConf.create({"output_dir": str(tmp_path / "nope")}), 0)


def _bg(cmd_str, log_path, tag):
    return run_fed.BgProc(["bash", "-c", cmd_str], dict(os.environ), log_path, tag=tag)


def test_wait_launch_port_retry_relaunches_on_collision(tmp_path):
    xdir = tmp_path / "xdir"
    xdir.mkdir()
    done = xdir / "done_1"
    (xdir / "go_1").write_text("stale")            # stale signals must be cleared pre-relaunch
    log_path = tmp_path / "worker.log"

    dead = _bg("echo 'RuntimeError: ... EADDRINUSE'; exit 1", log_path, "t-dead")
    relaunches = []

    def mk():
        relaunches.append(1)
        return _bg(f"touch {done}; sleep 5", log_path, "t-retry")

    proc = run_fed._wait_launch_port_retry(dead, mk, done, "launch test", log_path, xdir)
    try:
        assert relaunches == [1] and proc is not dead
        assert done.exists()
        assert (tmp_path / "worker.log.portfail").is_file()      # forensics preserved
        assert not (xdir / "go_1").exists()                      # stale signal cleared
    finally:
        proc.proc.kill()
        proc.wait(timeout=10)


def test_wait_launch_port_retry_reraises_without_signature(tmp_path):
    xdir = tmp_path / "xdir"
    xdir.mkdir()
    log_path = tmp_path / "worker.log"
    dead = _bg("echo 'ValueError: unrelated crash'; exit 1", log_path, "t-dead2")
    with pytest.raises(RuntimeError, match="died while waiting"):
        run_fed._wait_launch_port_retry(dead, lambda: pytest.fail("must not relaunch"),
                                        xdir / "done_1", "launch test", log_path, xdir)
    assert not (tmp_path / "worker.log.portfail").exists()
