"""ALFWorld env — thin async HTTP client to the ALFWorld remote service.

Runs in the trainer env (fedagent-verl08). The real ALFWorld/textworld env lives in the
verl-agent-alfworld env behind the ``service/`` backend (``fedagent.envs.alfworld.service``,
HTTP), because ALFWorld's deps (alfworld + textworld + gymnasium + torchvision pins)
conflict with verl 0.8.

Action parsing (``alfworld_projection``) happens server-side; this client ferries the
model's text in and formats observations out using verl-agent's ALFWorld prompt content
(``ALFWORLD_TEMPLATE_NO_HIS``) so the information the policy sees matches the 0.3.1
baseline (scientific-equivalence bar). The concat-chat ``GymTextAgentLoop`` supplies
multi-turn history as the literal chat, so per-turn observations carry only the current
observation + admissible actions (the no-history template is the per-turn body; verl-agent
reuses it every turn when history_length<=0).

Mirrors ``fedagent.envs.webshop.WebShopEnv`` exactly in structure.
"""
import asyncio
import os
import random
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import httpx

from fedagent.envs.base import BaseTextEnv, Obs, resolve_service_url
from fedagent.envs.legacy_prompts import build_alfworld_obs


def _extract_alf_task(obs: str) -> str:
    # legacy env_manager.extract_task: substring after "Your task is to: " in the init obs
    i = (obs or "").find("Your task is to: ")
    return obs[i + len("Your task is to: "):].strip() if i != -1 else ""


def _gamefile_to_task_id(gamefile: str) -> Optional[str]:
    """Derive the hardness task_id from a game.tw-pddl path, VERBATIM with
    partition_strategy.py::hardness_partition (the ALFWorld branch, line ~1177):
      .../{task_type-obj-...-scene}/{trial_xxx}/game.tw-pddl
      -> f"alfworld_{grandparent}_{parent}_game"
    So the hardness-labelling dump keys match what the partition looks up. Returns None
    if the path is not a recognizable game file (leaves normal runs untouched)."""
    if not gamefile:
        return None
    fn = os.path.basename(gamefile)
    if fn != "game.tw-pddl":
        return None
    parent = os.path.basename(os.path.dirname(gamefile))
    grandparent = os.path.basename(os.path.dirname(os.path.dirname(gamefile)))
    if not parent or not grandparent:
        return None
    return f"alfworld_{grandparent}_{parent}_game"


def _gamefile_to_task_type(gamefile: str) -> Optional[str]:
    """ALFWorld task family (the six pick_*/look_* types), for the eval breakdown tag —
    the leading segment of the grandparent dir before the first hyphen (e.g.
    'pick_clean_then_place_in_recep-Plate-None-DiningTable-19' -> 'pick_clean_then_place_in_recep')."""
    if not gamefile or os.path.basename(gamefile) != "game.tw-pddl":
        return None
    grandparent = os.path.basename(os.path.dirname(os.path.dirname(gamefile)))
    return grandparent.split("-", 1)[0] if grandparent else None

# Format/reasoning instructions (env-level, no per-episode task) -> system message.
# This is the instruction tail of verl-agent's ALFWORLD_TEMPLATE_NO_HIS, lifted to the
# system turn (the per-turn body below carries the observation + admissible actions).
ALFWORLD_SYSTEM = (
    "You are an expert agent operating in the ALFRED Embodied Environment.\n"
    "Now it's your turn to take an action.\n"
    "You should first reason step-by-step about the current situation. This reasoning "
    "process MUST be enclosed within <think> </think> tags.\n"
    "Once you've finished your reasoning, you should choose an admissible action for "
    "current step and present it within <action> </action> tags."
)
# Per-turn body — the observation lines of ALFWORLD_TEMPLATE_NO_HIS, verbatim.
_OBS = (
    "Your current observation is: {obs}\n"
    "Your admissible actions of the current situation are: [{actions}]."
)


def _fmt_actions(cmds: List[str]) -> str:
    # mirrors verl-agent build_text_obs: "\n ".join(f"'{s}'" ...), 'help' already excluded
    # server-side. Quote each command and newline-join.
    return "\n ".join(f"'{s}'" for s in (cmds or []))


class AlfworldEnv(BaseTextEnv):
    def __init__(self, env_config: Optional[Dict[str, Any]] = None):
        super().__init__(env_config)
        # Per-client routing: FEDAGENT_SERVICE_URL_FILE (persistent/cross-round, lever #4) wins;
        # else ALFWORLD_SERVICE_URL (subprocess path sets it per client); else spec service_url;
        # else default. See resolve_service_url for why the file beats process-env in persistent mode.
        self.base_url = resolve_service_url("ALFWORLD_SERVICE_URL", self.env_config,
                                            "http://localhost:8081")
        self.timeout = float(self.env_config.get("timeout", 120.0))
        self.session_id = uuid4().hex
        self._step_id = 0      # idempotency key for /step (incremented only after a success)
        # WINDOWED (faithful) mode: history_length>0 reproduces the paper's per-turn prompt
        # (task + last-N (obs, action) pairs + current obs). 0 (default) = concat mode.
        # FEDAGENT_HISTORY_LENGTH (set by run_fed per rollout_mode: windowed=2, concat=0) is
        # AUTHORITATIVE so ONE shared env spec drives both modes; the spec's history_length is the
        # fallback for direct (non-run_fed) runs.
        self._history_length = int(os.environ.get("FEDAGENT_HISTORY_LENGTH")
                                   or self.env_config.get("history_length", 0))
        self._memory: list = []     # [{"text_obs": <raw obs before action>, "action": <projected action>}]
        self._pre_obs = ""          # raw obs that led to the pending action (legacy pre_text_obs)
        self._task = ""             # extracted from the init obs ("Your task is to: ...")
        self._goal_id = None        # hardness task_id from the episode's gamefile (see step())
        self._task_type = None      # ALFWorld task family (eval breakdown tag)
        self._client: Optional[httpx.AsyncClient] = None

    def _c(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout)
        return self._client

    async def _post(self, path: str, payload: dict, *, retry: bool = False,
                    block: bool = False, retries: int = 8, base: float = 0.3):
        """POST to the env service; raise on HTTP errors, and (only when ``retry``) retry transport errors.

        ``retry=True`` is used for ALL stateful endpoints (/create, /reset, /step): at the full PPO
        batch the rollout fires train_batch_size x rollout.n episodes at once, so they hit this
        client's pooled per-client service near-simultaneously and the HTTP boundary is overwhelmed
        (sockets reset mid-response -> httpx.ReadError). Bounded backoff + jitter spreads the retried
        requests across the pool. /step mutates env state, so a naive replay would corrupt the
        trajectory -- it is made retry-SAFE by an idempotency key (``step_id``): the server applies
        each id exactly once and replays the cached response for a re-sent id (see service/server.py).
        We therefore increment ``self._step_id`` only AFTER a success, so the in-flight id is the only
        one ever re-sent. raise_for_status() ensures a 4xx/5xx body (e.g. {"detail":"unknown session"}
        or a 409 step-ordering error) is never silently parsed as an empty observation, and -- being an
        HTTPStatusError, not a TransportError -- is NOT retried (a real desync surfaces loudly).

        ``block=True`` (used for /create) disables the per-request read timeout: borrowing a pooled
        env legitimately blocks until one frees, and that wait scales with batch/pool, NOT with the
        180s timeout. ALFWorld is the stress case -- 512 episodes of mean ~30 (max 50) turns share a
        small env pool, so a waiting /create routinely exceeds 180s; a hard timeout there crashes the
        whole rollout. A blocking /create also removes the duplicate-create race: no timeout -> no
        retry-resend -> exactly one borrow per session, so the idempotency check can't be bypassed.
        """
        c = self._c()
        for attempt in range(retries + 1):
            try:
                # block: only the READ wait is unbounded (server-side _pool.get() can exceed the
                # 180s default); connect/write/pool stay bounded so a dead service still fails fast
                # instead of hanging the rollout forever.
                resp = (await c.post(path, json=payload, timeout=httpx.Timeout(self.timeout, read=None)) if block
                        else await c.post(path, json=payload))
                resp.raise_for_status()
                return resp
            except httpx.TransportError:
                if not retry or attempt >= retries:
                    raise
                await asyncio.sleep(min(base * (2 ** attempt), 4.0) + random.uniform(0.0, base))

    async def system_prompt(self) -> Obs:
        return {"obs_str": ALFWORLD_SYSTEM}

    async def reset(self, seed: int = 0) -> Tuple[Obs, Dict[str, Any]]:
        await self._post("/create", {"session_id": self.session_id}, retry=True, block=True)
        r = await self._post("/reset", {"session_id": self.session_id, "seed": int(seed)}, retry=True)
        self._step_id = 0   # fresh episode -> restart the /step idempotency counter (server does too)
        d = r.json()
        # Episode identity: /reset always returns this episode's gamefile (the game the seed
        # selected on the borrowed pool env). Derive the hardness task_id + task family so the
        # windowed/concat loop can tag every sample -- REQUIRED by the hardness-labelling
        # aggregation (mirrors WebShopEnv's goal_id surfacing). None on non-game paths.
        gamefile = d.get("gamefile")
        self._goal_id = _gamefile_to_task_id(gamefile)
        self._task_type = _gamefile_to_task_type(gamefile)
        raw = d.get("obs", "") or ""
        avail_str = _fmt_actions(d.get("admissible_commands", []))
        if self._history_length > 0:        # WINDOWED (faithful) mode: full legacy template
            self._task = _extract_alf_task(raw)
            self._memory = []
            self._pre_obs = raw
            obs_str = build_alfworld_obs(task=self._task, memory=self._memory, current_obs=raw,
                                         admissible_str=avail_str, history_length=self._history_length,
                                         init=True)
        else:                               # concat mode (unchanged)
            obs_str = _OBS.format(obs=raw, actions=avail_str)
        return {"obs_str": obs_str}, {}

    async def step(self, action_str: str) -> Tuple[Obs, float, bool, Dict[str, Any]]:
        # retry=True is SAFE here: step_id makes the server apply/replay exactly once. Increment
        # only after the await returns (success) so a retried request always carries this same id.
        r = await self._post(
            "/step",
            {"session_id": self.session_id, "text": action_str, "step_id": self._step_id},
            retry=True,
        )
        self._step_id += 1
        d = r.json()
        raw = d.get("obs", "") or ""
        avail_str = _fmt_actions(d.get("admissible_commands", []))
        if self._history_length > 0:        # WINDOWED (faithful) mode
            # store (raw obs that led to this action, PROJECTED action) -- matches legacy memory
            self._memory.append({"text_obs": self._pre_obs, "action": d.get("action", action_str)})
            self._pre_obs = raw
            obs_str = build_alfworld_obs(task=self._task, memory=self._memory, current_obs=raw,
                                         admissible_str=avail_str, history_length=self._history_length,
                                         init=False)
        else:                               # concat mode (unchanged)
            obs_str = _OBS.format(obs=raw, actions=avail_str)
        info = {
            "success": bool(d.get("success", False)),
            "is_action_valid": bool(d.get("is_action_valid", True)),
        }
        # String tags kept in verl's validation dump (skipped by metric aggregation): goal_id is
        # REQUIRED by gen_hardness_trajectories' per-task aggregation; task_type feeds the
        # ALFWorld eval breakdown. Parity with WebShopEnv (which surfaces goal_id the same way).
        if self._goal_id is not None:
            info["goal_id"] = self._goal_id
        if self._task_type is not None:
            info["task_type"] = self._task_type
        return {"obs_str": obs_str}, float(d.get("reward", 0.0)), bool(d.get("done", False)), info

    async def close(self) -> None:
        try:
            if self._client is not None:
                await self._client.post("/close", json={"session_id": self.session_id})
                await self._client.aclose()
        except Exception:
            pass
        self._client = None
