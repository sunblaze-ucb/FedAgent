"""Deterministic port bands for the random-port pickers inside a verl trainer process.

Two independent components pick listening ports by binding port 0 (kernel ephemeral
range, Linux default 32768-60999), CLOSING the probe socket, and re-using the number
later -- a TOCTOU window that occasionally loses the port on a busy shared-netns host
(devbox), killing the client/round with
``The server socket has failed to listen ... (errno: 98 - Address already in use)``:

  - **vLLM** ``get_open_port()`` (vllm/utils): feeds ``distributed_init_method``'s
    TCPStore, the DP master, the api server. When ``VLLM_PORT`` is set it instead
    probes UPWARD from that value until a bind succeeds -- collision-tolerant by
    design (vllm logs "Port X is already in use, trying port X+1").
  - **verl** ``WorkerHelper._get_free_port`` (single_controller/base/worker.py:59,
    bare ``bind(("", 0))``): feeds the FSDP process-group ``MASTER_PORT``
    (single_controller/ray/base.py:637). NO env knob upstream.

FedAgent amplifies the lottery: per-round trainer rebuilds (subprocess mode and the
per-round persistent worker) redraw ports every round x lane x eval -- ~hundreds of
draws per 70-round run, so "occasional" becomes "expected a few times per run".

Fix (docs/bugfixes.md 2026-07-23 "vLLM/verl random-port collisions"): run_fed gives
every launched trainer/eval process a private band OUTSIDE the ephemeral range via
``FEDAGENT_PORT_BAND="<start>:<span>"``; the repo-root sitecustomize arms a deferred
import hook that rebinds ``WorkerHelper._get_free_port`` to probe INSIDE
``[start, start+span)`` (first free port wins), and run_fed sets
``VLLM_PORT = start + span//2`` so vLLM's own upward probing works the upper half of
the same quiet band. Both pickers keep their occupied-port retry semantics; the band
just moves the draws out of the contended range, where the only possible squatter is
a stale listener of our own previous process -- which probing skips.

The literal TOCTOU (probe-close -> real-bind) cannot be zeroed at the overlay level
(that needs vLLM/torch to hand over BOUND sockets), so run_fed pairs the band with a
ONE-SHOT relaunch of the client/persistent step when a failed attempt's log tail
carries a collision signature (``stream``-level retry) -- that also covers any picker
we did not pin (e.g. Ray internals).
"""
import os

_PATCHED = False

# Signatures of a listen/bind port collision in a dead trainer's log tail. Deliberately
# NOT matching vllm's benign probe line "Port X is already in use, trying port X+1".
PORT_COLLISION_SIGS = (
    "Address already in use",   # torch TCPStore / OS errno 98 text
    "EADDRINUSE",
    "DistNetworkError",
)


def port_collision_in_log(log_path, tail_bytes: int = 65536) -> bool:
    """True iff the tail of ``log_path`` carries a port-collision signature."""
    try:
        with open(log_path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - tail_bytes))
            tail = f.read().decode("utf-8", "replace")
    except OSError:
        return False
    return any(sig in tail for sig in PORT_COLLISION_SIGS)


def _parse_band():
    """FEDAGENT_PORT_BAND="<start>:<span>" -> (start, span) or None if unset/invalid."""
    band = os.environ.get("FEDAGENT_PORT_BAND", "")
    if not band:
        return None
    try:
        start_s, span_s = band.split(":")
        start, span = int(start_s), int(span_s)
    except ValueError:
        return None
    if start <= 0 or span <= 0:
        return None
    return start, span


def probe_in_band(start: int, span: int) -> int:
    """First bindable port in [start, start+span) -- the banded replacement for a bare
    bind(("", 0)) draw. Raises (fail-closed) if the whole band is occupied: silently
    falling back to the ephemeral lottery would resurrect the bug unobserved."""
    import socket

    for port in range(start, start + span):
        try:
            with socket.socket() as sock:
                sock.bind(("", port))
                return sock.getsockname()[1]
        except OSError:
            continue
    raise RuntimeError(
        f"[port-band] no free port in [{start}, {start + span}) -- band exhausted; "
        f"raise port_band_stride or clear stale listeners on this host."
    )


def enable_port_band() -> bool:
    """Rebind verl's WorkerHelper._get_free_port to band probing. Idempotent; no-op
    without FEDAGENT_PORT_BAND."""
    global _PATCHED
    band = _parse_band()
    if band is None or _PATCHED:
        return False
    from verl.single_controller.base.worker import WorkerHelper

    start, span = band
    WorkerHelper._get_free_port = staticmethod(lambda: probe_in_band(start, span))
    _PATCHED = True
    print(f"[port-band] enabled: verl master-port draws confined to [{start}, {start + span}) "
          f"(vLLM half starts at VLLM_PORT={os.environ.get('VLLM_PORT', '<unset>')})", flush=True)
    return True


def install_deferred_port_band_patch() -> bool:
    """Arm the rebind LAZILY on the first import of verl.single_controller.base.worker
    (same rationale as fedprox/ppo_critic_loss/merge_fp32: importing verl eagerly at
    interpreter startup would pull torch in before Ray assigns per-rank
    CUDA_VISIBLE_DEVICES). Returns True if armed or applied directly."""
    import importlib
    import importlib.abc
    import importlib.util
    import sys

    if _parse_band() is None or _PATCHED:
        return False
    TARGET = "verl.single_controller.base.worker"
    if TARGET in sys.modules:
        return enable_port_band()

    class _PortBandImportHook(importlib.abc.MetaPathFinder):
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
                    if not enable_port_band():
                        raise RuntimeError("[port-band] deferred patch did not apply")

                spec.loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, _PortBandImportHook())
    print("[port-band] deferred patch armed: verl free-port draws will be confined to "
          f"FEDAGENT_PORT_BAND={os.environ.get('FEDAGENT_PORT_BAND')}", flush=True)
    return True
