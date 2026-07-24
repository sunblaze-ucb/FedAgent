"""Weight-transfer IPC namespace isolation (run_fed: _unique_ipc_env / patch preflight / sweep).

verl names the FSDP->vLLM weight-transfer ZMQ socket after the Ray job id
(``ipc:///tmp/rl-colocate-zmq-<job_id>-replica-<r>-rank-<lr>.sock``). A process that dies
hard leaves the socket FILE behind, so any later process computing the SAME path fails
binding it.

Field motivation (2026-07-24): a WebShop PPO run lost a round to a "stale IPC" engine
error, and the same round kept dying until the whole run was restarted. Two overlay holes
produced that signature:

  1. the non-cross_round persistent worker relaunches a NEW process every round but its
     job id was ``<tag>-persist`` with NO round -- one crashed round poisoned every later
     round of the run (a fresh run got a new uuid, which is why a full restart "fixed" it);
  2. the one-shot port-collision relaunches re-ran with the dead attempt's identical id.

The fix makes uniqueness structural (per LAUNCH, at the two launch primitives) rather than
relying on the job-id format strings being collectively injective.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fedagent.fed import run_fed  # noqa: E402


def test_unique_ipc_env_gives_every_launch_its_own_namespace():
    env = {"VERL_RAY_JOB_ID": "abcd1234-persist-r48", "OTHER": "keep"}
    a = run_fed._unique_ipc_env(env)
    b = run_fed._unique_ipc_env(env)          # SAME input env == a relaunch of the same step
    assert a["VERL_RAY_JOB_ID"] != b["VERL_RAY_JOB_ID"], "a retry must not reuse the dead id"
    for got in (a, b):
        assert got["VERL_RAY_JOB_ID"].startswith("abcd1234-persist-r48-")
        assert got["OTHER"] == "keep"
    assert env["VERL_RAY_JOB_ID"] == "abcd1234-persist-r48"   # caller's dict untouched


def test_unique_ipc_env_noop_without_job_id():
    # aggregator / merger launches run no vLLM and set no job id -> env passed through as-is
    env = {"PATH": "/usr/bin"}
    assert run_fed._unique_ipc_env(env) is env
    assert run_fed._unique_ipc_env({"VERL_RAY_JOB_ID": ""}) == {"VERL_RAY_JOB_ID": ""}


def test_launch_primitives_apply_the_namespace(tmp_path, monkeypatch):
    """stream() and BgProc are the two places a verl process is born; both must isolate."""
    seen = []

    class _FakeProc:
        returncode = 0
        stdout = iter(())

        def wait(self, timeout=None):
            return 0

        def poll(self):
            return 0

    def fake_popen(cmd, env=None, **kw):
        seen.append(env["VERL_RAY_JOB_ID"])
        return _FakeProc()

    monkeypatch.setattr(run_fed.subprocess, "Popen", fake_popen)
    env = {"VERL_RAY_JOB_ID": "tag-train-c0-r48"}
    run_fed.stream(["true"], env, tmp_path / "a.log", tag="t")
    run_fed.stream(["true"], env, tmp_path / "b.log", tag="t")     # the retry path
    run_fed.BgProc(["true"], env, tmp_path / "c.log", tag="t")
    assert len(set(seen)) == 3, f"launches shared an IPC namespace: {seen}"
    assert all(s.startswith("tag-train-c0-r48-") for s in seen)


def test_patch_preflight_detects_override_support(tmp_path, monkeypatch):
    import importlib.util

    def fake_find_spec(name, pkg=None, _orig=importlib.util.find_spec):
        if name != "verl":
            return _orig(name, pkg)
        return importlib.util.spec_from_file_location("verl", str(tmp_path / "verl" / "__init__.py"))

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
    src = tmp_path / "verl" / "workers" / "rollout" / "vllm_rollout"
    src.mkdir(parents=True)
    f = src / "vllm_rollout.py"

    f.write_text('        job_id = ray.get_runtime_context().get_job_id()\n')   # stock
    assert run_fed.verl_honors_job_id_override() is False
    f.write_text('        job_id = os.environ.get("VERL_RAY_JOB_ID") or ray_id\n')   # patched
    assert run_fed.verl_honors_job_id_override() is True
    f.unlink()                                          # unreadable -> unknown, never a guess
    assert run_fed.verl_honors_job_id_override() is None


def test_sweep_touches_only_this_runs_sockets(tmp_path, monkeypatch):
    mine = tmp_path / f"rl-colocate-zmq-{run_fed._RUN_TAG}-train-c0-r1-a3-replica-0-rank-0.sock"
    theirs = tmp_path / "rl-colocate-zmq-deadbeef-train-c0-r1-a0-replica-0-rank-0.sock"
    for p in (mine, theirs):
        p.write_text("")

    real_path = run_fed.Path      # redirect only the literal "/tmp" lookup at the sweep site
    monkeypatch.setattr(run_fed, "Path",
                        lambda p="", _r=real_path, _t=str(tmp_path): _r(_t if p == "/tmp" else p))
    assert run_fed.sweep_own_ipc_sockets() == 1
    assert not mine.exists()
    assert theirs.exists(), "a CONCURRENT run's live socket must never be swept"
