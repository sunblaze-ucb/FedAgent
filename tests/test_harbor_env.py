"""Harbor env: the pieces that decide what an episode IS, tested without a sandbox.

``HarborEnv`` turns one Harbor trial into one FedAgent episode. Four contracts carry the
correctness of everything trained on it:

1. Action parsing -- the first ``<action>`` block, case PRESERVED (shell is case-sensitive),
   ``submit`` ends the episode, anything else unparseable is an invalid action.
2. seed -> task -- deterministic, and immune to the row-seed layout: AgenticDataset's seeds are
   ``base*100_000 + row``, so a plain ``seed % N`` would start EVERY client-round at task 0
   whenever N divides 100_000 and train on the same few tasks forever.
3. The exec wrapper -- output capped at the source, runtime bounded in the sandbox, the command's
   own exit code preserved through the pipe. Run here against the host's real bash.
4. Episode flow -- the verifier's reward arrives on ``submit`` OR on the env's last turn (0 before),
   and the shipped specs keep the env's turn budget equal to the loop's: if the env's were larger,
   every episode would end ungraded with reward 0.

The flow tests drive the REAL ``HarborEnv`` against a stand-in for the Harbor trial (the bridge's
two queues + a scripted verifier): no docker, no harbor install, no model. The opt-in test at the
bottom runs one real trial (``FEDAGENT_HARBOR_DOCKER_TEST=1``; needs harbor + a Docker daemon).
"""
import asyncio
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from fedagent.envs.harbor import harbor_env as he  # noqa: E402
from fedagent.envs.harbor.harbor_env import HarborEnv  # noqa: E402


def say(action: str) -> str:
    return f"<think>reasoning</think><action>{action}</action>"


# --------------------------------------------------------------------------- 1. action parsing
@pytest.mark.parametrize("text, expected", [
    (say("ls -la"), ("command", "ls -la")),
    (say("grep -c ERROR Logs/App.log"), ("command", "grep -c ERROR Logs/App.log")),   # case kept
    (say("  submit \n"), ("submit", "")),
    (say("SUBMIT"), ("submit", "")),
    (say("```bash\nsort -u a.txt > b.txt\n```"), ("command", "sort -u a.txt > b.txt")),
    (say("cat > f.py <<'EOF'\nprint(1)\nEOF"), ("command", "cat > f.py <<'EOF'\nprint(1)\nEOF")),
    (say("first") + say("second"), ("command", "first")),                            # first block wins
    ("I would run ls here", ("invalid", "")),
    (say("   "), ("invalid", "")),
    ("<action>ls", ("invalid", "")),
    ("", ("invalid", "")),
])
def test_parse_action(text, expected):
    assert he.parse_action(text) == expected


# --------------------------------------------------------------------------- 2. seed -> task
TASKS = tuple(f"task-{i:03d}" for i in range(64))     # 64 divides nothing useful -- but see below


def row_seeds(base_seed: int, n_rows: int):
    """AgenticDataset's layout for the first spec: base*100_000 + row."""
    return [base_seed * 100_000 + i for i in range(n_rows)]


def test_sequential_is_a_fixed_sweep():
    picked = [he.pick_task(TASKS, s, "sequential") for s in range(len(TASKS))]
    assert picked == list(TASKS)                       # val rows 0..N-1 = each task exactly once


def test_shuffle_rows_of_one_client_round_are_distinct_tasks():
    for base in (42, 142, 143, 7042):
        picked = [he.pick_task(TASKS, s, "shuffle") for s in row_seeds(base, 24)]
        assert len(set(picked)) == 24


def test_shuffle_redraws_per_client_round_even_when_n_divides_the_seed_stride():
    tasks = tuple(f"task-{i:03d}" for i in range(100))     # 100 | 100_000: `seed % N` collapses here
    firsts = {he.pick_task(tasks, row_seeds(42 + r * 100 + c, 1)[0], "shuffle")
              for r in range(1, 9) for c in range(2)}
    assert {s % 100 for r in range(1, 9) for c in range(2)
            for s in row_seeds(42 + r * 100 + c, 1)} == {0}  # the trap a plain modulo walks into
    assert len(firsts) > 8                                    # 16 client-rounds, (nearly) all different


def test_shuffle_is_deterministic():
    seeds = row_seeds(142, 16)
    assert [he.pick_task(TASKS, s, "shuffle") for s in seeds] == \
           [he.pick_task(TASKS, s, "shuffle") for s in seeds]


def test_unknown_sampling_is_refused():
    with pytest.raises(ValueError):
        he.pick_task(TASKS, 0, "random")


def test_list_tasks_sorted_and_slice(tmp_path):
    for name in ("b-task", "a-task", "c-task"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "task.toml").write_text("")
    (tmp_path / "not-a-task").mkdir()
    env = HarborEnv({"tasks_dir": str(tmp_path), "task_slice": [1, 3]})
    assert [Path(t).name for t in env._tasks] == ["b-task", "c-task"]
    with pytest.raises(FileNotFoundError):
        HarborEnv({"tasks_dir": str(tmp_path / "not-a-task")})
    with pytest.raises(ValueError):
        HarborEnv({})


# --------------------------------------------------------------------------- 3. exec wrapper
def run_wrapped(command: str, timeout_sec: int = 5):
    proc = subprocess.run(["bash", "-c", he.wrap_command(command, timeout_sec)],
                          capture_output=True, text=True, timeout=60)
    return proc.returncode, proc.stdout


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_wrap_command_keeps_output_and_exit_code():
    assert run_wrapped("echo out; echo err >&2") == (0, "out\nerr\n")      # stderr folded in, in order
    assert run_wrapped("echo partial; exit 3") == (3, "partial\n")         # the command's own code
    assert run_wrapped("cat <<'EOF'\nline one\nEOF") == (0, "line one\n")  # heredoc survives the wrap
    assert run_wrapped("echo 'unclosed")[0] != 0                            # syntax error -> an error, not a hang


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_wrap_command_caps_output_at_the_source():
    code, out = run_wrapped("yes")
    assert len(out) == he._RAW_OUTPUT_LIMIT          # an endless producer is cut off, not buffered


@pytest.mark.skipif(shutil.which("timeout") is None, reason="needs coreutils timeout")
def test_wrap_command_times_out_inside_the_sandbox():
    code, out = run_wrapped("echo started; sleep 30", timeout_sec=1)
    assert code == he._TIMEOUT_EXIT_CODE and out == "started\n"
    assert he.render_output(code, out, 1, 100) == "[timed out after 1s]\nstarted"


def test_render_and_clip():
    assert he.render_output(0, "", 30, 100) == "[exit code 0]\n(no output)"
    assert he.render_output(2, "boom\n", 30, 100) == "[exit code 2]\nboom"
    clipped = he.clip_middle("a" * 300 + "z" * 300, 90)
    assert clipped.startswith("a" * 60) and clipped.endswith("z" * 30)
    assert "[510 characters omitted]" in clipped


def test_windowed_obs_numbers_the_last_pairs_absolutely():
    memory = [{"action": f"<action>cmd{i}</action>", "output": f"out{i}"} for i in range(1, 5)]
    first = he.build_windowed_obs(task="do it", workdir="/app", max_turns=6, memory=[], history_length=2)
    assert "You have not run anything yet." in first and "do it" in first
    obs = he.build_windowed_obs(task="do it", workdir="/app", max_turns=6, memory=memory, history_length=2)
    assert "You have used 4 of 6 turns" in obs
    assert ("Turn 3 action: <action>cmd3</action>\nTurn 3 output:\nout3\n\n"
            "Turn 4 action: <action>cmd4</action>\nTurn 4 output:\nout4") in obs
    assert "[Action" not in obs and "[action" not in obs     # no bracket labels to imitate
    assert "cmd2" not in obs                         # outside the window


# --------------------------------------------------------------------------- 4. episode flow
class FakeTrial:
    """Stands in for a Harbor trial + its BridgeAgent: same two queues, same message shapes.
    The 'sandbox' echoes commands; the 'verifier' pays 1.0 iff ``touch answer`` was run."""

    def __init__(self, instruction="solve it", rewards=None, die_after=None):
        self.to_agent, self.to_env = asyncio.Queue(), asyncio.Queue()
        self.instruction, self.rewards, self.die_after = instruction, rewards, die_after
        self.commands = []
        self.task = asyncio.ensure_future(self._run())

    async def _run(self):
        await self.to_env.put(("ready", self.instruction))
        while True:
            request = await self.to_agent.get()
            if request is None:
                break
            self.commands.append(request[0])
            if self.die_after is not None and len(self.commands) > self.die_after:
                break                                # Harbor's agent timeout: the trial moves on by itself
            out = "/app\n" if request[0] == "pwd" else f"ran: {request[0][:20]}\n"
            await self.to_env.put(("result", SimpleNamespace(return_code=0, stdout=out)))
        solved = any("touch answer" in c for c in self.commands)
        rewards = self.rewards if self.rewards is not None else {"reward": 1.0 if solved else 0.0}
        return SimpleNamespace(verifier_result=SimpleNamespace(rewards=rewards), exception_info=None)


def make_env(tmp_path, monkeypatch, trial: FakeTrial, **cfg) -> HarborEnv:
    (tmp_path / "only-task").mkdir(exist_ok=True)
    (tmp_path / "only-task" / "task.toml").write_text("")
    monkeypatch.delenv("FEDAGENT_HISTORY_LENGTH", raising=False)
    env = HarborEnv({"tasks_dir": str(tmp_path), "max_turns": 4, **cfg})

    async def start_trial():
        env._agent, env._trial_task = trial, trial.task
        return (await env._from_agent())[1]

    monkeypatch.setattr(env, "_start_trial", start_trial)
    return env


def run_flow(tmp_path, monkeypatch, responses, **kwargs):
    """Drive one episode the way the agent loops do; returns (reset obs, steps, the fake trial)."""
    async def go():
        fake = FakeTrial(**{k: kwargs.pop(k) for k in ("rewards", "die_after") if k in kwargs})
        env = make_env(tmp_path, monkeypatch, fake, **kwargs)
        steps = []
        try:
            obs, _ = await env.reset(seed=0)
            for text in responses:
                steps.append(await env.step(text))
                if steps[-1][2]:
                    break
        finally:
            await env.close()
        return obs, steps, fake
    return asyncio.run(go())


def test_submit_ends_the_episode_with_the_verifier_reward(tmp_path, monkeypatch):
    obs, (s1, s2), fake = run_flow(tmp_path, monkeypatch, [say("touch answer"), say("submit")])
    assert "solve it" in obs["obs_str"] and "Working directory: /app" in obs["obs_str"]
    assert s1[1:3] == (0.0, False) and s1[3]["task_score"] == 0.0      # nothing is paid before grading
    assert s2[1:3] == (1.0, True)
    assert s2[3] == {"success": True, "is_action_valid": True, "goal_id": "only-task", "task_score": 1.0}
    assert fake.commands[0] == "pwd" and "touch answer" in fake.commands[1]


def test_info_keys_are_identical_on_every_step(tmp_path, monkeypatch):
    """verl's agent-loop postprocess takes the reward_extra_info key set from row 0 and indexes
    every row with it. The concat loop reports the LAST info it saw, which is an ungraded one when
    it abandons an episode (context spent), so a key that appears only on the graded step
    (`task_score` did) is a KeyError for the whole batch. GPU-hit 2026-10-06, concat val pass."""
    _, steps, _ = run_flow(tmp_path, monkeypatch, ["no tags", say("ls"), say("touch answer"), say("submit")])
    assert len({frozenset(step[3]) for step in steps}) == 1
    assert set(steps[0][3]) == {"success", "is_action_valid", "goal_id", "task_score"}


def test_last_turn_is_graded_without_a_submit(tmp_path, monkeypatch):
    _, steps, _ = run_flow(tmp_path, monkeypatch, [say("touch answer")] + [say(f"true {i}") for i in range(5)])
    assert len(steps) == 4                                   # the env's max_turns, not the 6 offered
    assert [s[2] for s in steps] == [False, False, False, True]
    assert steps[-1][1] == 1.0 and steps[-1][3]["success"] is True


def test_unsolved_episode_scores_zero(tmp_path, monkeypatch):
    _, steps, _ = run_flow(tmp_path, monkeypatch, [say("ls"), say("submit")])
    assert steps[-1][1] == 0.0 and steps[-1][3]["success"] is False


def test_invalid_action_is_flagged_and_costs_a_turn(tmp_path, monkeypatch):
    _, steps, fake = run_flow(tmp_path, monkeypatch, ["no tags here", say("touch answer"), say("submit")])
    assert steps[0][3]["is_action_valid"] is False and steps[0][2] is False
    assert "Invalid response" in steps[0][0]["obs_str"]
    _, steps_w, _ = run_flow(tmp_path, monkeypatch, ["no tags here", say("submit")], history_length=2)
    assert "Turn 1 action: (none)\nTurn 1 output:\nInvalid response" in steps_w[0][0]["obs_str"]
    assert len(fake.commands) == 2                           # pwd + the one real command: nothing ran for the invalid turn
    assert steps[-1][1] == 1.0


def test_exact_repeat_is_not_run_and_is_an_invalid_action(tmp_path, monkeypatch):
    _, steps, fake = run_flow(tmp_path, monkeypatch,
                              [say("touch answer"), say("touch answer"), say("ls"), say("touch answer")],
                              history_length=3)
    assert [s[3]["is_action_valid"] for s in steps] == [True, False, True, True]
    assert "Not run: this is the same command" in steps[1][0]["obs_str"]
    assert "Turn 2 action: <action>touch answer</action>" in steps[1][0]["obs_str"]   # the model sees what it sent
    assert len(fake.commands) == 4            # pwd + 3 executed: the repeat never reached the sandbox;
    assert steps[-1][1] == 1.0                # the same command AFTER a different one is a new command
    _, steps, fake = run_flow(tmp_path, monkeypatch, [say("ls"), say("ls"), say("submit")],
                              refuse_repeats=False)
    assert steps[1][3]["is_action_valid"] is True and len(fake.commands) == 3


def test_windowed_mode_returns_the_full_template_each_turn(tmp_path, monkeypatch):
    _, steps, _ = run_flow(tmp_path, monkeypatch, [say("ls"), say("touch answer"), say("submit")],
                        history_length=1)
    assert he.HARBOR_SYSTEM in steps[0][0]["obs_str"]
    assert "Turn 1 action: <action>ls</action>" in steps[0][0]["obs_str"]
    assert "Turn 2 action: <action>touch answer</action>" in steps[1][0]["obs_str"]
    assert "Turn 1 action" not in steps[1][0]["obs_str"]


def test_history_length_env_var_overrides_the_spec(tmp_path, monkeypatch):
    (tmp_path / "t").mkdir()
    (tmp_path / "t" / "task.toml").write_text("")
    monkeypatch.setenv("FEDAGENT_HISTORY_LENGTH", "3")
    assert HarborEnv({"tasks_dir": str(tmp_path), "history_length": 0})._history_length == 3


def test_partial_credit_and_threshold(tmp_path, monkeypatch):
    _, steps, _ = run_flow(tmp_path, monkeypatch, [say("submit")], rewards={"reward": 0.5})
    assert steps[-1][1] == 0.5 and steps[-1][3]["success"] is False and steps[-1][3]["task_score"] == 0.5
    _, steps, _ = run_flow(tmp_path, monkeypatch, [say("submit")], rewards={"reward": 0.5}, success_threshold=0.5)
    assert steps[-1][3]["success"] is True


def test_reward_scale_leaves_success_and_task_score_on_the_verifier_scale(tmp_path, monkeypatch):
    _, steps, _ = run_flow(tmp_path, monkeypatch, [say("touch answer"), say("submit")], reward_scale=10.0)
    assert steps[-1][1] == 10.0                              # the WebShop/ALFWorld {0, 10} convention
    assert steps[-1][3]["success"] is True and steps[-1][3]["task_score"] == 1.0


def test_turn_cost_discounts_a_pass_and_never_charges_a_fail(tmp_path, monkeypatch):
    """reward = score * (reward_scale - turn_cost * turns): among solved episodes the one that
    submits sooner pays more; an unsolved one pays 0 at ANY length (a flat per-turn cost would
    reward giving up early wherever a whole GRPO group fails)."""
    cfg = dict(reward_scale=10.0, turn_cost=0.2)
    _, quick, _ = run_flow(tmp_path, monkeypatch, [say("touch answer"), say("submit")], **cfg)
    _, slow, _ = run_flow(tmp_path, monkeypatch, [say("touch answer")] + [say(f"true {i}") for i in range(3)], **cfg)
    assert quick[-1][1] == pytest.approx(9.6) and slow[-1][1] == pytest.approx(9.2)   # 2 turns vs 4
    assert quick[-1][3]["task_score"] == slow[-1][3]["task_score"] == 1.0             # success itself is unchanged
    _, gave_up, _ = run_flow(tmp_path, monkeypatch, [say("submit")], **cfg)
    _, tried, _ = run_flow(tmp_path, monkeypatch, [say(f"ls {i}") for i in range(4)], **cfg)
    assert gave_up[-1][1] == tried[-1][1] == 0.0
    _, steps, _ = run_flow(tmp_path, monkeypatch, [say("touch answer"), say("submit")],
                           reward_scale=1.0, turn_cost=5.0)
    assert steps[-1][1] == 0.0                                                          # never negative


def test_reward_key_fallbacks(tmp_path, monkeypatch):
    _, steps, _ = run_flow(tmp_path, monkeypatch, [say("submit")], rewards={"score": 1.0})   # a single other key
    assert steps[-1][1] == 1.0
    _, steps, _ = run_flow(tmp_path, monkeypatch, [say("submit")], rewards={"a": 1.0, "b": 1.0})  # ambiguous -> 0
    assert steps[-1][1] == 0.0
    _, steps, _ = run_flow(tmp_path, monkeypatch, [say("submit")], rewards={})                # verifier wrote nothing
    assert steps[-1][1] == 0.0


def test_trial_ending_under_the_agent_still_gets_graded(tmp_path, monkeypatch):
    # die_after=1: the fake trial serves `pwd`, then ends at the next command (Harbor's agent timeout)
    _, steps, _ = run_flow(tmp_path, monkeypatch, [say("touch answer"), say("ls")], die_after=1)
    assert len(steps) == 1 and steps[0][2] is True and steps[0][1] == 1.0


def test_close_cancels_a_live_trial(tmp_path, monkeypatch):
    async def go():
        fake = FakeTrial()
        env = make_env(tmp_path, monkeypatch, fake)
        await env.reset(seed=0)
        await env.step(say("ls"))
        await env.close()                                   # the loop broke early (e.g. context overflow)
        return fake
    fake = asyncio.run(go())
    assert fake.task.cancelled()


# --------------------------------------------------------------------------- shared sandbox network
def test_only_plain_public_tasks_share_the_network(tmp_path):
    """The shared network replaces Harbor's per-trial one ONLY where that is safe: explicit
    networking on `main` would bypass Harbor's egress sidecar (no-network / allowlist tasks), and
    a task with its own compose file needs its project network for its sidecars."""
    env_dir = tmp_path / "environment"
    env_dir.mkdir()
    trial = SimpleNamespace(task=SimpleNamespace(paths=SimpleNamespace(environment_dir=env_dir)),
                            agent_environment=SimpleNamespace(_enable_egress_control=False))
    assert he._can_share_network(trial)
    trial.agent_environment = SimpleNamespace(_enable_egress_control=True)
    assert not he._can_share_network(trial)
    trial.agent_environment = SimpleNamespace()              # attribute gone in another Harbor: fail safe
    assert not he._can_share_network(trial)
    trial.agent_environment = SimpleNamespace(_enable_egress_control=False)
    (env_dir / "docker-compose.yaml").write_text("services: {}\n")
    assert not he._can_share_network(trial)


def test_shared_network_compose_names_the_network_the_env_creates():
    yaml = pytest.importorskip("yaml")
    doc = yaml.safe_load(he.SHARED_NETWORK_COMPOSE.read_text())
    assert doc == {"networks": {"default": {"name": he.SHARED_NETWORK, "external": True}}}
    assert yaml.safe_load(he.FAST_STOP_COMPOSE.read_text()) == {"services": {"main": {"stop_grace_period": "0s"}}}


# --------------------------------------------------------------------------- shipped specs
@pytest.mark.parametrize("spec", ["harbor_shell.yaml", "harbor_shell_val.yaml"])
def test_shipped_specs_tie_the_env_turn_budget_to_the_loop(spec):
    OmegaConf = pytest.importorskip("omegaconf").OmegaConf
    cfg = OmegaConf.to_container(OmegaConf.load(REPO / "fedagent" / "config" / "envs" / spec), resolve=True)
    for env_spec in cfg["envs"]:
        assert env_spec["name"] == "Harbor"
        assert env_spec["config"]["max_turns"] == env_spec["max_turns"]
    if spec.endswith("_val.yaml"):
        assert cfg["envs"][0]["validate"] is True and cfg["envs"][0]["config"]["sampling"] == "sequential"


def test_registered():
    from fedagent.envs.registry import ENV_REGISTRY
    assert ENV_REGISTRY["Harbor"] is HarborEnv


# --------------------------------------------------------------------------- opt-in: one real trial
@pytest.mark.skipif(os.environ.get("FEDAGENT_HARBOR_DOCKER_TEST") != "1",
                    reason="set FEDAGENT_HARBOR_DOCKER_TEST=1 (needs harbor + a Docker daemon)")
def test_real_trial_roundtrip(tmp_path, monkeypatch):
    pytest.importorskip("harbor")
    task = tmp_path / "tasks" / "write-file"
    (task / "tests").mkdir(parents=True)
    (task / "environment").mkdir()                           # uploaded into the workdir -> /app exists
    (task / "environment" / "notes.txt").write_text("hello\n")
    (task / "instruction.md").write_text("Write the word done to the file answer.txt.\n")
    (task / "task.toml").write_text(
        'schema_version = "1.4"\n[verifier]\ntimeout_sec = 60.0\n[agent]\ntimeout_sec = 300.0\n'
        '[environment]\ndocker_image = "python:3.12-slim"\nworkdir = "/app"\n')
    (task / "tests" / "test.sh").write_text(
        '#!/bin/bash\n[ "$(cat /app/answer.txt 2>/dev/null)" = "done" ] && r=1 || r=0\n'
        'echo $r > /logs/verifier/reward.txt\n')
    monkeypatch.delenv("FEDAGENT_HISTORY_LENGTH", raising=False)

    async def go(responses):
        env = HarborEnv({"tasks_dir": str(tmp_path / "tasks"), "max_turns": 3,
                         "trials_dir": str(tmp_path / "trials")})
        try:
            obs, _ = await env.reset(seed=0)
            for text in responses:
                step = await env.step(text)
        finally:
            await env.close()
        return obs, step

    obs, step = asyncio.run(go([say("echo done > answer.txt"), say("submit")]))
    assert "Write the word done" in obs["obs_str"] and step[1:3] == (1.0, True)
    _, step = asyncio.run(go([say("submit")]))
    assert step[1:3] == (0.0, True)
    assert not any((tmp_path / "trials").iterdir())          # trial directories are cleaned up
