"""The Harbor example suite: every generated task must be solvable, and must not be solved for free.

``tools/gen_harbor_shell_tasks.py`` emits the tasks the Harbor env trains and validates on, so a
family whose verifier rejects its own reference solution would train toward an unreachable
reward, and one whose verifier passes on the untouched sandbox would hand out reward for doing
nothing. For EVERY family, on several instances, this runs the real ``tests/check.py`` against a
copy of ``environment/`` twice -- untouched (must FAIL) and after ``solution/solve.sh``'s command
(must PASS) -- with the host's bash standing in for the sandbox. Plus the two properties the
train/val split relies on: generation is deterministic, and growing the val split never changes
a train task.

No docker, no harbor. (``harbor run -a oracle`` / ``-a nop`` is the same check through Harbor
itself; it scored 1.0 / 0.0 on the shipped defaults.)
"""
import filecmp
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tools import gen_harbor_shell_tasks as gen  # noqa: E402

pytestmark = pytest.mark.skipif(
    any(shutil.which(tool) is None for tool in ("bash", "awk", "find", "tar", "sha256sum", "python3")),
    reason="needs bash + GNU coreutils/findutils/awk/tar on the host")


def verdict(task_dir: Path, sandbox: Path) -> int:
    """Exit code of the task's own check.py run against ``sandbox`` instead of the image's /app."""
    check = sandbox.parent / f"{sandbox.name}-tests"
    shutil.copytree(task_dir / "tests", check, dirs_exist_ok=True)
    source = (check / "check.py").read_text()
    assert f'WORKDIR = "{gen.WORKDIR}"' in source
    (check / "check.py").write_text(source.replace(f'WORKDIR = "{gen.WORKDIR}"', f'WORKDIR = "{sandbox}"'))
    return subprocess.run([sys.executable, str(check / "check.py")], capture_output=True).returncode


@pytest.mark.parametrize("family", sorted(gen.FAMILIES))
def test_family_is_solvable_and_not_free(family, tmp_path):
    for idx in range(4):
        task = gen.FAMILIES[family](random.Random(f"0/{family}/{idx}"))
        task_dir = tmp_path / f"{family}-{idx:03d}"
        gen.write_task(task_dir, family, task)
        sandbox = tmp_path / f"sandbox-{idx}"
        shutil.copytree(task_dir / "environment", sandbox)
        assert verdict(task_dir, sandbox) != 0, f"{family}-{idx}: passes with nothing done"
        solved = subprocess.run(["bash", "-c", task.solution], cwd=sandbox, capture_output=True, text=True)
        assert solved.returncode == 0, f"{family}-{idx}: reference solution failed: {solved.stderr}"
        assert verdict(task_dir, sandbox) == 0, f"{family}-{idx}: reference solution rejected"


def test_task_directory_is_a_harbor_task(tmp_path):
    tomllib = pytest.importorskip("tomllib")
    counts = gen.generate(tmp_path / "suite", train_per_family=1, val_per_family=1, seed=0)
    assert counts == {"train": len(gen.FAMILIES), "val": len(gen.FAMILIES)}
    for task_dir in sorted((tmp_path / "suite").glob("*/*")):
        assert (task_dir / "instruction.md").read_text().strip()
        config = tomllib.loads((task_dir / "task.toml").read_text())
        assert config["environment"]["docker_image"] == gen.IMAGE
        assert config["environment"]["workdir"] == gen.WORKDIR
        # Harbor uploads environment/ into the workdir ONLY for an image-only task: a Dockerfile
        # next to the data would silently switch it to a build that never sees these files.
        assert any((task_dir / "environment").iterdir())
        assert not (task_dir / "environment" / "Dockerfile").exists()
        assert json.loads((task_dir / "tests" / "spec.json").read_text())["checks"]
        for script in ("tests/test.sh", "solution/solve.sh"):
            assert (task_dir / script).stat().st_mode & 0o111


def same_tree(a: Path, b: Path) -> bool:
    cmp = filecmp.dircmp(a, b)
    if cmp.left_only or cmp.right_only or cmp.funny_files:
        return False
    _, mismatch, errors = filecmp.cmpfiles(a, b, cmp.common_files, shallow=False)
    return not mismatch and not errors and all(same_tree(a / d, b / d) for d in cmp.common_dirs)


def test_generation_is_deterministic_and_val_size_leaves_train_alone(tmp_path):
    gen.generate(tmp_path / "a", train_per_family=2, val_per_family=1, seed=0)
    gen.generate(tmp_path / "b", train_per_family=2, val_per_family=1, seed=0)
    gen.generate(tmp_path / "c", train_per_family=2, val_per_family=3, seed=0)
    gen.generate(tmp_path / "d", train_per_family=2, val_per_family=1, seed=1)
    assert same_tree(tmp_path / "a", tmp_path / "b")
    assert same_tree(tmp_path / "a" / "train", tmp_path / "c" / "train")
    assert not same_tree(tmp_path / "a" / "train", tmp_path / "d" / "train")
    train = {p.name for p in (tmp_path / "c" / "train").iterdir()}
    val = {p.name for p in (tmp_path / "c" / "val").iterdir()}
    assert not train & val and len(val) == 3 * len(gen.FAMILIES)
