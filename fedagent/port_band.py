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
    """True iff the tail of ``log_path`` carries a port-collision signature.
    Case-INSENSITIVE: torch prints "Address already in use" but Ray/grpc surfaces the
    same errno as lowercase "address already in use" -- a cased match would miss those
    and skip the retry. vLLM's benign probe line ("Port X is already in use, trying
    port X+1") stays excluded either way: it never contains the word "address"."""
    try:
        with open(log_path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - tail_bytes))
            tail = f.read().decode("utf-8", "replace").lower()
    except OSError:
        return False
    return any(sig.lower() in tail for sig in PORT_COLLISION_SIGS)


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


_CURSORS: dict = {}   # (start, span) -> next probe offset (per process)


def probe_in_band(start: int, span: int, salt=None) -> int:
    """A bindable port in [start, start+span) -- the banded replacement for a bare
    bind(("", 0)) draw. Two hardenings over naive first-free scanning (2026-07-23
    regression fix, docs/bugfixes.md "port-band same-start race"):

    - the scan STARTS at a per-process pid-salted offset (``salt``), so concurrent
      processes sharing one band (e.g. Ray workers of the same trainer) probe from
      DIFFERENT starting points instead of all racing the first free port;
    - a per-process rotating cursor advances past every returned port, so consecutive
      draws differ even while the earlier port is probed-but-not-yet-bound (the
      probe socket closes before the consumer binds -- first-free would hand the
      SAME port out twice).

    Raises (fail-closed) if the whole band is occupied: silently falling back to the
    ephemeral lottery would resurrect the original bug unobserved."""
    import socket

    key = (start, span)
    cur = _CURSORS.get(key)
    if cur is None:
        cur = ((os.getpid() * 7919) if salt is None else salt) % span
    for k in range(span):
        port = start + (cur + k) % span
        try:
            with socket.socket() as sock:
                sock.bind(("", port))
                _CURSORS[key] = (cur + k + 1) % span
                return sock.getsockname()[1]
        except OSError:
            continue
    raise RuntimeError(
        f"[port-band] no free port in [{start}, {start + span}) -- band exhausted; "
        f"raise port_band_stride or clear stale listeners on this host."
    )


def assign_vllm_port() -> bool:
    """Give THIS process its own pid-salted VLLM_PORT inside the band's upper half,
    OVERRIDING any inherited value. The 4a85d12 design exported one static VLLM_PORT
    per launched job -- but verl's rollout starts one vLLM replica PER GPU (TP<world)
    concurrently, all inheriting that same value, and vLLM's upward probing closes its
    probe socket before the engine binds: every replica probed the same start and won
    the same port -> deterministic EADDRINUSE at engine init (observed in the field at
    round-1 init on a 4-GPU run). sitecustomize runs in every process (driver, Ray
    workers, spawned engine cores), so assigning here spreads the replicas' starting
    points; vLLM's own probing handles the rest."""
    band = _parse_band()
    if band is None:
        return False
    start, span = band
    half = max(1, span // 2)
    headroom = 8                                  # room to probe upward past the salt
    salt = (os.getpid() * 7919) % max(1, half - headroom)
    os.environ["VLLM_PORT"] = str(start + half + salt)
    return True


def enable_port_band() -> bool:
    """Rebind verl's WorkerHelper._get_free_port to band probing. Idempotent; no-op
    without FEDAGENT_PORT_BAND."""
    global _PATCHED
    band = _parse_band()
    if band is None or _PATCHED:
        return False
    from verl.single_controller.base.worker import WorkerHelper

    start, span = band
    # STRICT half partition (hardening on the same-start regression fix): verl draws ONLY
    # from the lower half; assign_vllm_port salts vLLM into the upper half. Without the
    # cap, a verl draw could claim-but-not-yet-bind a port in vLLM's territory (the PG
    # master binds its TCPStore LATE, after handing the number to workers) while a vLLM
    # replica probes, binds it first, and the late binder dies with EADDRINUSE.
    lower = max(1, span // 2)
    WorkerHelper._get_free_port = staticmethod(lambda: probe_in_band(start, lower))
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
    assign_vllm_port()   # per-process pid-salted VLLM_PORT (see its docstring; 4a85d12 regression fix)
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
