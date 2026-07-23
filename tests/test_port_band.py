"""Regression tests for the deterministic port bands (fedagent/port_band.py).

vLLM's get_open_port and verl's WorkerHelper._get_free_port both draw random
EPHEMERAL ports with a use-after-probe window; on shared-netns hosts the draw
occasionally collides and the round dies with "Address already in use". The overlay
confines both pickers to a private per-process band + retries the launch once on a
collision signature. Pure offline (sockets + tmp files; no verl import).
"""
import os
import socket
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fedagent.port_band import (  # noqa: E402
    _CURSORS,
    PORT_COLLISION_SIGS,
    _parse_band,
    assign_vllm_port,
    port_collision_in_log,
    probe_in_band,
)


def _free_base(span: int = 8) -> int:
    """Find a small all-free band in a quiet range for socket tests."""
    for base in range(20100, 24000, span):
        try:
            socks = []
            for p in range(base, base + span):
                s = socket.socket()
                s.bind(("", p))
                socks.append(s)
            for s in socks:
                s.close()
            return base
        except OSError:
            for s in socks:
                s.close()
            continue
    pytest.skip("no quiet band available on this host")


def test_probe_returns_salted_start_and_skips_occupied():
    base = _free_base()
    _CURSORS.clear()
    assert probe_in_band(base, 8, salt=0) == base    # empty band, salt 0 -> first port
    _CURSORS.clear()
    with socket.socket() as squat:
        squat.bind(("", base))                       # occupy the first port
        assert probe_in_band(base, 8, salt=0) == base + 1    # probing skips it
    _CURSORS.clear()
    assert probe_in_band(base, 8, salt=3) == base + 3   # pid-salt spreads process starts


def test_consecutive_draws_differ_even_before_binding():
    # the probe socket closes before the CONSUMER binds; without the rotating cursor a
    # second draw would hand out the SAME still-unbound port (the 2026-07-23 same-start
    # regression class, in-process flavor).
    base = _free_base()
    _CURSORS.clear()
    a = probe_in_band(base, 8, salt=0)
    b = probe_in_band(base, 8, salt=0)
    assert a == base and b == base + 1 and a != b


def test_probe_fails_closed_when_band_exhausted():
    base = _free_base()
    _CURSORS.clear()
    s1, s2 = socket.socket(), socket.socket()
    try:
        s1.bind(("", base))
        s2.bind(("", base + 1))
        with pytest.raises(RuntimeError, match="band exhausted"):
            probe_in_band(base, 2)
    finally:
        s1.close(), s2.close()


def test_assign_vllm_port_salted_upper_half(monkeypatch):
    monkeypatch.setenv("FEDAGENT_PORT_BAND", "26000:100")
    monkeypatch.setenv("VLLM_PORT", "26050")     # stale static value MUST be overridden
    monkeypatch.setattr(os, "getpid", lambda: 12345)
    assert assign_vllm_port() is True
    v = int(os.environ["VLLM_PORT"])
    assert 26050 <= v < 26092                    # upper half, probing headroom respected
    assert v == 26050 + (12345 * 7919) % 42
    monkeypatch.setattr(os, "getpid", lambda: 12346)
    assign_vllm_port()                           # distinct pid -> distinct start (the
    assert int(os.environ["VLLM_PORT"]) != v     # regression was ONE start shared by all)
    monkeypatch.delenv("FEDAGENT_PORT_BAND")
    assert assign_vllm_port() is False           # no band -> untouched


def test_parse_band_env(monkeypatch):
    monkeypatch.delenv("FEDAGENT_PORT_BAND", raising=False)
    assert _parse_band() is None
    monkeypatch.setenv("FEDAGENT_PORT_BAND", "26000:100")
    assert _parse_band() == (26000, 100)
    for bad in ("", "26000", "a:b", "0:100", "26000:0"):
        monkeypatch.setenv("FEDAGENT_PORT_BAND", bad)
        assert _parse_band() is None, bad


def test_collision_signature_matcher(tmp_path):
    dead = tmp_path / "dead.log"
    dead.write_text("... RuntimeError: The server socket has failed to listen on any local "
                    "network address. ... (errno: 98 - Address already in use).\n")
    assert port_collision_in_log(dead)

    # Ray/grpc surface the same errno LOWERCASE -- the matcher must be case-insensitive
    ray_dead = tmp_path / "ray_dead.log"
    ray_dead.write_text("RuntimeError: Failed to start the GCS server: "
                        "bind: address already in use\n")
    assert port_collision_in_log(ray_dead)

    # vllm's BENIGN in-band probe line must NOT trigger a retry
    benign = tmp_path / "benign.log"
    benign.write_text("INFO ... Port 26050 is already in use, trying port 26051\n"
                      "Traceback ... ValueError: something unrelated\n")
    assert not port_collision_in_log(benign)
    for sig in PORT_COLLISION_SIGS:
        assert sig not in benign.read_text()

    assert not port_collision_in_log(tmp_path / "missing.log")
