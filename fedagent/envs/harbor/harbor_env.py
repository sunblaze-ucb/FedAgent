"""Harbor env -- any Harbor task (instruction + sandbox + verifier) as one FedAgent episode.

Runs IN-PROCESS in the trainer env (Harbor needs Python >= 3.12, which the trainer env already
is; ``pip install harbor``). No service: an episode is one real Harbor ``Trial``, so the sandbox
lifecycle, the task's timeouts and its verifier all go through Harbor's own trial path, on
whichever sandbox backend ``environment_type`` names (docker by default).

Division of labor (the "external agent" pattern -- the trainer owns the turn loop):
  * FedAgent's agent-loop generates every turn on the policy being trained, exactly as for
    WebShop/ALFWorld, so token ids / log-probs / the response mask are exact by construction and
    no model-API capture proxy is involved.
  * ``reset(seed)`` picks a task deterministically in ``seed``, starts a Harbor trial whose agent
    is ``bridge.BridgeAgent``, and returns the task instruction once the sandbox is up.
  * ``step(text)`` parses ONE action out of the model's text -- a shell command, or ``submit`` --
    and has the bridge run the command in the sandbox; the command's output is the observation.
  * On ``submit`` or on the last turn the bridge returns, Harbor runs the task's verifier, and its
    reward is the episode reward (0 on every earlier turn); ``success`` = reward >= threshold.

The harness is deliberately minimal (one bash tool, a fresh shell per command): what is being
integrated is Harbor's task format, not a particular agent scaffold. Its two shaping rules exist
because training without them went wrong in a specific way (``refuse_repeats`` and ``turn_cost``
below; the runs are in the README's Verification section).

Both rollout modes are supported from ONE env spec, like the other envs: ``history_length`` 0 =
concat (per-turn body only; the loop keeps the chat), > 0 = windowed (this env renders task +
the last N (action, output) pairs each turn). ``FEDAGENT_HISTORY_LENGTH`` (set by run_fed per
``rollout_mode``) is authoritative; the spec value is the fallback.
"""
import asyncio
import logging
import os
import random
import re
import shlex
import shutil
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from fedagent.envs.base import BaseTextEnv, Obs

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
BRIDGE_AGENT = "fedagent.envs.harbor.bridge:BridgeAgent"
FAST_STOP_COMPOSE = Path(__file__).with_name("compose_fast_stop.yaml")
SHARED_NETWORK_COMPOSE = Path(__file__).with_name("compose_shared_network.yaml")
SHARED_NETWORK = "fedagent-harbor"      # the name compose_shared_network.yaml attaches to
_COMPOSE_BACKENDS = ("docker", "podman")

# Format instructions (env-level, no per-episode task) -> system message in concat mode, the head
# of the per-turn template in windowed mode.
HARBOR_SYSTEM = (
    "You are an expert agent operating in a Linux sandbox through a bash shell.\n"
    "Each turn, first reason step-by-step about the current situation inside <think> </think> "
    "tags, then give exactly one action inside <action> </action> tags. The action is either one "
    "shell command to run, for example <action>ls -la</action>, or the single word submit, "
    "<action>submit</action>, once the task is complete.\n"
    "Every command runs in a new non-interactive bash shell that starts in the working directory, "
    "so `cd` and shell variables do not carry over to the next turn: chain steps with && or use "
    "absolute paths.\n"
    "Submit as soon as the task is done: a further command cannot improve a finished task, it "
    "can only break it. A command identical to your previous one is not run again."
)
_TASK = (
    "Your task:\n{task}\n\n"
    "{workdir}"
    "You have {max_turns} turns; your work is graded when you submit or when the turns run out."
)
_WORKDIR = "Working directory: {workdir}\n"
_YOUR_TURN = ("Now it's your turn: reason inside <think> </think> tags, then give one action "
              "inside <action> </action> tags.")
_INVALID_OBS = ("Invalid response: no action found. Reply with your reasoning inside "
                "<think> </think> tags followed by one action inside <action> </action> tags.")
_REPEAT_OBS = ("Not run: this is the same command as your previous one, and its result has not "
               "changed. Give a different command, or submit if the task is done.")

_ACTION = re.compile(r"<action>(.*?)</action>", re.DOTALL | re.IGNORECASE)
_FENCE = re.compile(r"^```[\w+-]*\n(.*?)\n?```$", re.DOTALL)

_RAW_OUTPUT_LIMIT = 65536     # bytes of command output kept at the source (`head -c`)
_EXEC_BACKSTOP_SEC = 15       # host-side grace on top of the in-sandbox `timeout`
_TIMEOUT_EXIT_CODE = 124      # coreutils `timeout`

# One gate per event loop (= per agent-loop worker process): a generation batch fires
# train_batch_size x rollout.n resets at once, and that many simultaneous sandbox starts
# stall the container runtime.
_START_GATES: Dict[int, asyncio.Semaphore] = {}
_shared_network_ready = False


def parse_action(text: str) -> Tuple[str, str]:
    """``(kind, command)`` for one model response; kind is ``command`` | ``submit`` | ``invalid``.

    The FIRST ``<action>`` block counts (the same rule as the WebShop/ALFWorld projections).
    Unlike those, the body is NOT lowercased: shell is case-sensitive.
    """
    m = _ACTION.search(text or "")
    if not m:
        return "invalid", ""
    body = m.group(1).strip()
    fenced = _FENCE.match(body)
    if fenced:                       # small models often wrap the command in a ``` fence
        body = fenced.group(1).strip()
    if not body:
        return "invalid", ""
    if body.lower() == "submit":
        return "submit", ""
    return "command", body


def wrap_command(command: str, timeout_sec: int) -> str:
    """The shell line actually exec'd for one agent command.

    Caps the output at the SOURCE (a runaway ``yes`` would otherwise be buffered whole by the
    trainer process) and bounds the runtime INSIDE the sandbox, so a timed-out command is really
    killed there rather than just abandoned by the client. ``timeout`` is probed because a
    minimal image may not ship it; the host-side backstop still applies then.

    ``timeout`` wraps the command only, NOT the pipeline: it signals its own process group, and
    a ``head`` killed alongside would take the partial output down with it (unflushed stdio).
    Outside that group, ``head`` just sees EOF and delivers what the command printed before it
    was killed.
    """
    quoted = shlex.quote(command)
    return (f"{{ if command -v timeout >/dev/null 2>&1; "
            f"then timeout -k 2 {int(timeout_sec)} bash -c {quoted}; else bash -c {quoted}; fi; }} "
            f"2>&1 | head -c {_RAW_OUTPUT_LIMIT}; exit ${{PIPESTATUS[0]}}")


def clip_middle(text: str, limit: int) -> str:
    """Keep the head and the tail of an over-long output (both ends matter in a terminal)."""
    if len(text) <= limit:
        return text
    head = limit * 2 // 3
    tail = limit - head
    return f"{text[:head]}\n... [{len(text) - limit} characters omitted] ...\n{text[-tail:]}"


def render_output(return_code: int, output: str, timeout_sec: int, limit: int) -> str:
    body = clip_middle((output or "").rstrip("\n"), limit) or "(no output)"
    if return_code == _TIMEOUT_EXIT_CODE:
        return f"[timed out after {int(timeout_sec)}s]\n{body}"
    return f"[exit code {return_code}]\n{body}"


def task_block(task: str, workdir: str, max_turns: int) -> str:
    """The per-episode part of the prompt. The working-directory line is dropped when the probe
    for it failed (a task whose workdir does not exist in its image)."""
    return _TASK.format(task=task, max_turns=max_turns,
                        workdir=_WORKDIR.format(workdir=workdir) if workdir else "")


def build_windowed_obs(*, task: str, workdir: str, max_turns: int, memory: List[Dict[str, str]],
                       history_length: int) -> str:
    """The full per-turn prompt of windowed mode: format rules + task + the last
    ``history_length`` (action, output) pairs. In a terminal the current observation IS the
    output of the latest action, so it rides as the last pair instead of a separate field.

    Past actions are shown in the SAME ``<action>`` markup the model must produce. Labelling
    them ``[Action N]`` instead primed Qwen2.5-1.5B to answer ``[action] ...`` in brackets: on the
    example suite's val tasks 30% of its zero-shot turns had no parseable action under that
    labelling, 15% under this one."""
    head = f"{HARBOR_SYSTEM}\n\n" + task_block(task, workdir, max_turns)
    if not memory:
        return f"{head}\n\nYou have not run anything yet.\n\n{_YOUR_TURN}"
    recent = memory[-history_length:]
    start = len(memory) - len(recent)
    parts = "".join(
        f"\n\nTurn {start + j + 1} action: {rec['action']}\nTurn {start + j + 1} output:\n{rec['output']}"
        for j, rec in enumerate(recent))
    return (f"{head}\n\nYou have used {len(memory)} of {max_turns} turns. Your most recent "
            f"{len(recent)} turn(s):{parts}\n\n{_YOUR_TURN}")


@lru_cache(maxsize=None)
def list_tasks(tasks_dir: str) -> Tuple[str, ...]:
    """Task directories (those holding a ``task.toml``) directly under ``tasks_dir``, sorted by
    name -- the fixed order that ``task_slice`` and the seed -> task map index into."""
    root = Path(os.path.expandvars(tasks_dir)).expanduser()
    if not root.is_absolute():
        root = REPO_ROOT / root
    tasks = tuple(sorted(str(p.parent) for p in root.glob("*/task.toml")))
    if not tasks:
        raise FileNotFoundError(f"No Harbor tasks (<task>/task.toml) found under {str(root)!r}")
    return tasks


def pick_task(tasks: Tuple[str, ...], seed: int, sampling: str) -> str:
    """Deterministic seed -> task.

    ``sequential``: ``tasks[seed % N]`` -- a fixed sweep; the val spec uses it so its n_envs rows
    (seeds 0..n_envs-1) are the first n_envs tasks, each exactly once, every round.

    ``shuffle``: AgenticDataset lays row seeds out as ``base*100_000 + spec*1_000 + row``, with
    ``base`` advancing per (round, client). A plain modulo over that is badly behaved -- whenever
    N divides 100_000 every client-round would start at task 0 and train on the same few tasks
    forever. So the 1000-block picks a permutation and the row indexes into it: rows of one
    client-round get DISTINCT tasks (while rows <= N), and each client-round a fresh draw.
    """
    n = len(tasks)
    if sampling == "sequential":
        return tasks[seed % n]
    if sampling != "shuffle":
        raise ValueError(f"Harbor env: unknown sampling {sampling!r} (shuffle | sequential)")
    block, row = divmod(int(seed), 1000)
    return tasks[random.Random(block).sample(range(n), n)[row % n]]


def _quiet_harbor_logging() -> None:
    """Harbor narrates every trial phase at DEBUG on a DEBUG-level logger, and the trainer's root
    handler would print all of it for every episode of every batch. Cap what reaches the console
    (``FEDAGENT_HARBOR_LOG_LEVEL``, default WARNING); each trial's own ``trial.log`` is written by
    a handler on the trial's logger and still gets everything (see ``keep_trials``)."""
    harbor_logger = logging.getLogger("harbor.utils.logger")
    if not harbor_logger.propagate:
        return
    handler = logging.StreamHandler()
    handler.setLevel(os.environ.get("FEDAGENT_HARBOR_LOG_LEVEL", "WARNING").upper())
    harbor_logger.addHandler(handler)
    harbor_logger.propagate = False


async def _ensure_shared_network() -> None:
    """Create the shared sandbox network once (idempotent across the agent-loop worker processes
    racing to do it). Inter-container traffic is disabled on it: sharing the bridge must not let
    one episode's sandbox reach another's."""
    global _shared_network_ready
    if _shared_network_ready:
        return
    proc = await asyncio.create_subprocess_exec(
        "docker", "network", "create", "-o", "com.docker.network.bridge.enable_icc=false",
        SHARED_NETWORK, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
    out, _ = await proc.communicate()
    if proc.returncode != 0 and b"already exists" not in out:
        raise RuntimeError(f"could not create docker network {SHARED_NETWORK!r}: "
                           f"{out.decode(errors='replace').strip()}")
    _shared_network_ready = True


def _can_share_network(trial) -> bool:
    """Only a plain task may be moved onto the shared network: one whose environment is a single
    container (no docker-compose.yaml of its own -- sidecars need their project network) and whose
    network policy is public throughout (Harbor enforces no-network / allowlist policies through
    an egress sidecar, and explicit networking on ``main`` would bypass it). Unknown => False:
    falling back to Harbor's per-trial network is always correct, just bounded in concurrency."""
    if (trial.task.paths.environment_dir / "docker-compose.yaml").exists():
        return False
    return getattr(trial.agent_environment, "_enable_egress_control", True) is False


def _start_gate(limit: int) -> asyncio.Semaphore:
    loop_id = id(asyncio.get_running_loop())
    if loop_id not in _START_GATES:
        _START_GATES[loop_id] = asyncio.Semaphore(limit)
    return _START_GATES[loop_id]


class HarborEnv(BaseTextEnv):
    def __init__(self, env_config: Optional[Dict[str, Any]] = None):
        super().__init__(env_config)
        cfg = self.env_config
        if not cfg.get("tasks_dir"):
            raise ValueError("Harbor env: config.tasks_dir is required (a directory of Harbor tasks)")
        tasks = list_tasks(str(cfg["tasks_dir"]))
        if cfg.get("task_slice"):            # [start, end) over the sorted task list (train/val split)
            start, end = cfg["task_slice"]
            tasks = tasks[int(start):int(end)]
            if not tasks:
                raise ValueError(f"Harbor env: task_slice {cfg['task_slice']} selects no task")
        self._tasks = tasks
        self._sampling = str(cfg.get("sampling", "shuffle"))
        # MUST equal the spec-level max_turns the agent-loop enforces: the verifier runs on THIS
        # env's last turn, so a larger value here would end every episode ungraded (reward 0).
        # The shipped specs tie the two with `max_turns: ${..max_turns}`.
        self.max_turns = int(cfg.get("max_turns", 8))
        self._history_length = int(os.environ.get("FEDAGENT_HISTORY_LENGTH")
                                   or cfg.get("history_length", 0))
        self._environment_type = str(cfg.get("environment_type", "docker"))
        self._environment_kwargs = dict(cfg.get("environment_kwargs") or {})
        self._fast_stop = bool(cfg.get("fast_stop", True))
        self._shared_network = bool(cfg.get("shared_network", True))
        self._exec_timeout = int(cfg.get("exec_timeout_sec", 30))
        self._max_obs_chars = int(cfg.get("max_obs_chars", 2000))
        self._agent_timeout = cfg.get("agent_timeout_sec")      # None => the task's own [agent] timeout
        self._timeout_multiplier = float(cfg.get("timeout_multiplier", 1.0))
        self._reward_key = str(cfg.get("reward_key", "reward"))
        self._success_threshold = float(cfg.get("success_threshold", 1.0))
        # Multiplies the verifier's reward into the episode reward (success / task_score stay on
        # the verifier's own scale). WebShop and ALFWorld pay {0, 10}; 10.0 here puts a pass/fail
        # Harbor task on that scale, where the 0.1 invalid-action penalty and PPO's value targets
        # were set. 1.0 = the verifier's reward as is.
        self._reward_scale = float(cfg.get("reward_scale", 1.0))
        # Per-turn discount on a GRADED reward: reward = score * (reward_scale - turn_cost * turns).
        # Without it a solved task pays the same whether the model submits or keeps going, and the
        # only signal `submit` ever gets is the penalty for a premature one; Qwen2.5-1.5B then
        # unlearned submitting within two rounds and started overwriting its own correct results
        # in the turns it no longer needed. Scaled by the score on purpose, so a failed episode
        # pays 0 at any length: a flat per-turn cost would, under GRPO's per-group normalization,
        # make "give up early" the dominant signal on every task whose whole group fails.
        self._turn_cost = float(cfg.get("turn_cost", 0.0))
        # An exact repeat of the previous command is not executed and counts as an invalid action.
        # In a fresh shell per command a repeat does nothing new, yet it is the cheapest way to
        # spend a turn: once a one-command task is solved, repeating that command is harmless,
        # and windowed credit pays every turn of a successful episode. Qwen2.5-1.5B learned
        # exactly that within six rounds ("mkdir -p out" six times on a two-step task; its
        # multi-step families fell to 0 while the one-command ones rose). The invalid-action
        # penalty is per TURN, so this is the one signal that lands on the repeat itself.
        self._refuse_repeats = bool(cfg.get("refuse_repeats", True))
        self._start_retries = int(cfg.get("start_retries", 2))
        self._start_concurrency = int(cfg.get("start_concurrency", 8))
        self._keep_trials = bool(cfg.get("keep_trials", False))
        self._trials_dir = Path(cfg.get("trials_dir") or os.environ.get("FEDAGENT_HARBOR_TRIALS_DIR")
                                or Path(tempfile.gettempdir()) / "fedagent_harbor_trials")
        self._task_dir = ""
        self._task_name = ""
        self._instruction = ""
        self._workdir = ""
        self._turn = 0
        self._last_command = ""
        # [{"action": "<action>cmd</action>" | "(none)", "output": <rendered output>}], oldest first
        self._memory: List[Dict[str, str]] = []
        self._agent = None                         # the trial's BridgeAgent
        self._trial_task: Optional[asyncio.Task] = None
        self._trial_dir: Optional[Path] = None

    async def system_prompt(self) -> Obs:
        return {"obs_str": HARBOR_SYSTEM}

    # ------------------------------------------------------------------ harbor trial plumbing
    async def _from_agent(self):
        """Next bridge message, or None when the trial ended instead (sandbox start failed, or
        Harbor's own agent timeout fired and it moved on to the verifier)."""
        getter = asyncio.ensure_future(self._agent.to_env.get())
        done, _ = await asyncio.wait({getter, self._trial_task}, return_when=asyncio.FIRST_COMPLETED)
        if getter in done:
            return getter.result()
        getter.cancel()
        return None

    async def _start_trial(self) -> str:
        """Start one Harbor trial on the picked task; return its instruction once the sandbox is up."""
        from harbor.models.trial.config import AgentConfig, EnvironmentConfig, TaskConfig, TrialConfig
        from harbor.trial.trial import Trial

        _quiet_harbor_logging()
        compose = ([FAST_STOP_COMPOSE]
                   if self._fast_stop and self._environment_type in _COMPOSE_BACKENDS else [])
        config = TrialConfig(
            task=TaskConfig(path=Path(self._task_dir)),
            trial_name=f"fedagent-{uuid4().hex[:12]}",
            trials_dir=self._trials_dir,
            timeout_multiplier=self._timeout_multiplier,
            agent=AgentConfig(import_path=BRIDGE_AGENT, override_timeout_sec=self._agent_timeout),
            environment=EnvironmentConfig(type=self._environment_type, extra_docker_compose=compose,
                                          kwargs=self._environment_kwargs),
        )
        trial = await Trial.create(config)
        self._trial_dir = trial.paths.trial_dir
        if trial.task.has_steps:
            raise NotImplementedError(
                f"Harbor task {self._task_name!r} is multi-step ([[steps]]); HarborEnv drives "
                f"single-step tasks only")
        if self._shared_network and self._environment_type == "docker" and _can_share_network(trial):
            await _ensure_shared_network()
            trial.agent_environment.extra_docker_compose_paths.append(SHARED_NETWORK_COMPOSE)
        self._agent = trial.agent
        self._trial_task = asyncio.ensure_future(trial.run())
        message = await self._from_agent()
        if message is None:
            result = self._trial_task.result()
            err = result.exception_info
            raise RuntimeError(
                f"Harbor sandbox for task {self._task_name!r} did not start: "
                f"{err.exception_type}: {err.exception_message}" if err else
                f"Harbor trial for task {self._task_name!r} ended before the agent phase")
        return message[1]

    async def _exec(self, command: str) -> Optional[str]:
        """Run one command in the sandbox; the rendered observation, or None if the trial ended."""
        await self._agent.to_agent.put(
            (wrap_command(command, self._exec_timeout), self._exec_timeout + _EXEC_BACKSTOP_SEC))
        message = await self._from_agent()
        if message is None:
            return None
        kind, payload = message
        if kind == "error":
            return f"[command failed: {payload}]"
        return render_output(payload.return_code, payload.stdout or "", self._exec_timeout,
                             self._max_obs_chars)

    async def _finish(self) -> float:
        """End the agent phase, let Harbor verify + tear down, and return the episode reward."""
        if not self._trial_task.done():
            await self._agent.to_agent.put(None)
        result = await self._trial_task
        rewards = (result.verifier_result.rewards if result.verifier_result else None) or {}
        if self._reward_key in rewards:
            return float(rewards[self._reward_key])
        if len(rewards) == 1:
            return float(next(iter(rewards.values())))
        # No usable reward: the verifier crashed / timed out / wrote several keys. Scored 0, but
        # LOUDLY -- an infra fault that reads as a model failure is the classic silent trap here.
        err = result.exception_info
        logger.warning("Harbor task %s gave no '%s' reward (rewards=%s, exception=%s); scoring 0.0",
                       self._task_name, self._reward_key, rewards or None,
                       f"{err.exception_type}: {err.exception_message}" if err else None)
        return 0.0

    async def _abort_trial(self) -> None:
        """Cancel a live trial (Harbor tears the sandbox down on cancel) and drop its directory."""
        task, self._trial_task, self._agent = self._trial_task, None, None
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except BaseException:   # CancelledError (expected) or a teardown error: nothing to salvage
                pass
        if self._trial_dir is not None and not self._keep_trials:
            shutil.rmtree(self._trial_dir, ignore_errors=True)
        self._trial_dir = None

    # ------------------------------------------------------------------ BaseTextEnv
    def _obs(self) -> str:
        return build_windowed_obs(task=self._instruction, workdir=self._workdir,
                                  max_turns=self.max_turns, memory=self._memory,
                                  history_length=self._history_length)

    async def reset(self, seed: int = 0) -> Tuple[Obs, Dict[str, Any]]:
        self._task_dir = pick_task(self._tasks, int(seed), self._sampling)
        self._task_name = Path(self._task_dir).name
        self._turn, self._memory, self._last_command = 0, [], ""
        for attempt in range(self._start_retries + 1):
            try:
                async with _start_gate(self._start_concurrency):
                    self._instruction = (await self._start_trial()).strip()
                break
            except NotImplementedError:
                await self._abort_trial()
                raise
            except Exception as exc:
                await self._abort_trial()
                if attempt >= self._start_retries:
                    raise
                logger.warning("Harbor sandbox start failed (attempt %d/%d): %s", attempt + 1,
                               self._start_retries + 1, exc)
                await asyncio.sleep(1.0 + attempt)
        await self._agent.to_agent.put(("pwd", self._exec_timeout))
        message = await self._from_agent()
        probe = message[1] if message and message[0] == "result" else None
        probed = (probe.stdout or "").strip() if probe is not None else ""
        self._workdir = probed if probe is not None and probe.return_code == 0 else ""
        if not self._workdir:
            logger.warning("Harbor task %s: `pwd` failed in the sandbox (%s); its commands will "
                           "likely fail too -- does the task's workdir exist in its image?",
                           self._task_name, probed[:200] or message)
        if self._history_length > 0:        # WINDOWED mode: full per-turn template
            obs_str = self._obs()
        else:                               # concat mode: the task once; the loop keeps the chat
            obs_str = task_block(self._instruction, self._workdir, self.max_turns)
        return {"obs_str": obs_str}, {}

    async def step(self, action_str: str) -> Tuple[Obs, float, bool, Dict[str, Any]]:
        self._turn += 1
        kind, command = parse_action(action_str)
        finished = kind == "submit"
        output = ""
        if kind == "command" and self._refuse_repeats and command == self._last_command:
            kind, output = "invalid", _REPEAT_OBS
        elif kind == "command":
            self._last_command = command
            output = await self._exec(command)
            finished = output is None       # Harbor's agent timeout ended the trial under us
        elif kind == "invalid":
            output = _INVALID_OBS
        # The SAME keys on every step, graded or not: the concat loop can abandon an episode
        # between steps (context / response budget spent) and reports the last info it saw, and
        # verl's batch postprocess does info[key] over all rows with row 0's key set -- one row
        # without `task_score` is a KeyError that takes the whole batch down.
        info: Dict[str, Any] = {"success": False, "is_action_valid": kind != "invalid",
                                "goal_id": self._task_name, "task_score": 0.0}
        if finished or self._turn >= self.max_turns:
            score = await self._finish()
            info["success"] = score >= self._success_threshold
            info["task_score"] = score
            reward = score * max(self._reward_scale - self._turn_cost * self._turn, 0.0)
            return {"obs_str": "The task has been graded."}, reward, True, info
        self._memory.append({"action": f"<action>{command}</action>" if command else "(none)",
                             "output": output})
        if self._history_length > 0:
            obs_str = self._obs()
        else:
            obs_str = f"{output}\n({self.max_turns - self._turn} turn(s) left)"
        return {"obs_str": obs_str}, 0.0, False, info

    async def close(self) -> None:
        await self._abort_trial()
