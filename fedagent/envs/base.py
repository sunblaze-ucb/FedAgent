"""Async multi-turn text-environment contract for FedAgent on verl 0.8.

Every environment a FedAgent agent-loop drives implements this interface. It mirrors
the per-instance async contract verl 0.8's agent-loop expects (ONE env instance per
dataset row), generalised from the Phase 0(b) spike and aligned with VAGEN's
``GymBaseEnv``. WebShop (Phase 2) and ALFWorld (Phase 3) subclass this.

Observation convention: a dict with at least ``obs_str`` (the text shown to the model).
Image/multimodal envs may later add ``multi_modal_data`` without changing this contract.

The old verl-0.3.1 code drove envs in a *batched, synchronous* ``EnvironmentManager``
(one obs dict for the whole batch). verl 0.8's agent-loop is per-row async, so the
env becomes a single-instance object with ``await``-able reset/step.
"""
import itertools
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

Obs = Dict[str, Any]

# Replica round-robin cursor, PER AGENT-LOOP WORKER PROCESS. PID-offset the start so workers
# spawned together don't all send their first episode to replica 0. Round-robin (not hash) keeps
# the per-replica session count balanced within +/-1 per worker, which lets the per-replica env
# pool be sized ~pool_size/K without /create starvation.
_REPLICA_RR = itertools.count(os.getpid())


def _pick_replica(url: str) -> str:
    """A resolved service URL may be a COMMA-SEPARATED REPLICA LIST (env-service sharding: K
    identical service processes over the same client shard -- kills per-process env serialization,
    e.g. ALFWorld's process-global ``_TW_LOCK``; see docs/acceleration.md). Each env instance picks
    ONE replica here, round-robin per worker process, and stays STICKY to it for its whole episode
    (``base_url`` is captured once at construction). Single-URL strings pass through unchanged, so
    every existing routing source (file / process-env / spec / default) keeps working."""
    if "," not in url:
        return url.rstrip("/")
    urls = [u.strip().rstrip("/") for u in url.split(",") if u.strip()]
    if not urls:
        return url.rstrip("/")
    return urls[next(_REPLICA_RR) % len(urls)]


def resolve_service_url(env_var: str, env_config: Dict[str, Any], default: str) -> str:
    """Resolve an env-service base URL, in priority order:

    1. ``$FEDAGENT_SERVICE_URL_FILE`` -- persistent/cross-round PER-CLIENT routing (lever #4). When
       the federated runner trains many clients in ONE process, it rewrites this file with the
       CURRENT client's service URL before each client's fit(); the shared agent-loop workers (which
       build the env per episode) read it here, so each client hits its OWN service. Process-env
       routing can't do this -- one process has one os.environ for all its clients.
    2. ``$<env_var>`` (e.g. WEBSHOP_SERVICE_URL) -- the subprocess-per-client path sets this.
    3. ``env_config['service_url']`` -- ad-hoc single-service fallback.
    4. ``default``.

    Any source may carry a comma-separated REPLICA LIST -- this instance then binds to one
    replica, round-robin (see ``_pick_replica``).
    """
    f = os.environ.get("FEDAGENT_SERVICE_URL_FILE")
    if f:
        try:
            url = Path(f).read_text().strip()
            if url:
                return _pick_replica(url)
        except FileNotFoundError:
            pass   # driver hasn't written it yet -> fall through to the static sources
    return _pick_replica(os.environ.get(env_var) or env_config.get("service_url") or default)


class BaseTextEnv(ABC):
    """One episode of one environment instance, driven by an AgentLoop."""

    def __init__(self, env_config: Optional[Dict[str, Any]] = None):
        self.env_config: Dict[str, Any] = dict(env_config or {})

    @abstractmethod
    async def system_prompt(self) -> Obs:
        """Return the system message (``{"obs_str": ...}``) shown once at episode start."""
        raise NotImplementedError

    @abstractmethod
    async def reset(self, seed: int = 0) -> Tuple[Obs, Dict[str, Any]]:
        """Reset to a fresh episode, deterministically in ``seed``. Returns ``(obs, info)``."""
        raise NotImplementedError

    @abstractmethod
    async def step(self, action_str: str) -> Tuple[Obs, float, bool, Dict[str, Any]]:
        """Apply the model's decoded text action. Returns ``(obs, reward, done, info)``.

        ``info`` should carry ``success`` (bool) so the agent-loop can record the
        episode outcome (FedAgent's headline metric is ``val/success_rate``).
        """
        raise NotImplementedError

    async def close(self) -> None:
        """Release any resources held by this instance (override if needed)."""
        return None
