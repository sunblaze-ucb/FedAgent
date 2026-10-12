#!/usr/bin/env python
"""Generate the shell-task suite in the Harbor task format: the example data of the Harbor env.

The Harbor env (fedagent/envs/harbor/) trains on ANY directory of Harbor tasks; this script
writes a small, fully offline one so the env has something to run on out of the box -- the role
TinyGuess plays for the in-process wiring, but with real sandboxes and a real verifier. It is not
part of the research suite.

24 task families (count / extract / transform / filesystem chores that take 1-4 commands),
each instantiated with seeded names, paths and file contents. Every task is a standard Harbor
task directory:

    <family>-<idx>/
        instruction.md      what to do
        task.toml           prebuilt image (no build step) + /app workdir + timeouts
        environment/        the task's input files; Harbor uploads them into the workdir
        tests/              test.sh -> check.py + spec.json; writes /logs/verifier/reward.txt (0/1)
        solution/solve.sh   a reference solution (`harbor run -a oracle` scores 1.0 on all tasks)

All tasks share ONE stock image and carry their data in environment/, so nothing is built and a
sandbox is up in about a second. Train and val hold disjoint instances of the same families.

Usage:
    python tools/gen_harbor_shell_tasks.py                    # -> data/harbor/shell_v1/{train,val}
    python tools/gen_harbor_shell_tasks.py --out /some/dir --train-per-family 12 --val-per-family 6

Deterministic in --seed (byte-identical output for the same arguments). Check a generated suite
with Harbor itself:
    harbor run -p data/harbor/shell_v1/val -a oracle -n 8     # mean reward 1.0
    harbor run -p data/harbor/shell_v1/val -a nop -n 8        # mean reward 0.0
"""
import argparse
import gzip
import hashlib
import io
import json
import random
import shutil
import tarfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Union

REPO_ROOT = Path(__file__).resolve().parents[1]
IMAGE = "python:3.12-slim"      # bash + coreutils + python3; the verifier needs python3
WORKDIR = "/app"

WORDS = ["alpha", "bravo", "cedar", "delta", "ember", "falcon", "garnet", "harbor", "indigo",
         "juniper", "kestrel", "lumen", "maple", "nimbus", "onyx", "pepper", "quartz", "raven",
         "sable", "tundra", "umber", "violet", "willow", "xenon", "yarrow", "zephyr"]
LEVELS = ["ERROR", "WARN", "INFO", "DEBUG"]
MESSAGES = ["connection reset by peer", "cache refreshed", "request completed", "retrying upload",
            "disk usage checked", "session expired", "job scheduled", "config reloaded",
            "payload rejected", "heartbeat received"]
REGIONS = ["north", "south", "east", "west"]
COUNTRIES = ["DE", "FR", "JP", "BR", "US"]

CHECK_PY = '''#!/usr/bin/env python3
"""Verifier for one generated shell task: every check in spec.json must pass."""
import hashlib
import json
import os
import sys

WORKDIR = "__WORKDIR__"


def norm(text):
    return "\\n".join(line.rstrip() for line in text.strip().splitlines())


def read(path):
    with open(path, "rb") as f:
        return f.read()


def same_value(got, expected):
    if got == expected:
        return True
    try:                        # 17 == 17.0: the task asked for a quantity, not a spelling
        return float(got) == float(expected)
    except ValueError:
        return False


def run(check):
    kind = check["type"]
    if kind == "answer":
        got = norm(read(check["path"]).decode("utf-8", "replace"))
        if check.get("ignore_case"):
            return got.lower() == check["expected"].lower()
        return same_value(got, check["expected"])
    if kind == "text":
        return norm(read(check["path"]).decode("utf-8", "replace")) == norm(check["expected"])
    if kind == "sha256":
        return hashlib.sha256(read(check["path"])).hexdigest() == check["expected"]
    if kind == "mode":
        return oct(os.stat(check["path"]).st_mode & 0o777)[2:] == check["expected"]
    if kind == "symlink":
        return (os.path.islink(check["path"])
                and os.path.realpath(check["path"]) == os.path.realpath(check["target"]))
    if kind == "listing":
        return sorted(os.listdir(check["path"])) == sorted(check["expected"])
    raise ValueError("unknown check type: %r" % kind)


def main():
    os.chdir(WORKDIR)
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "spec.json")) as f:
        spec = json.load(f)
    ok = True
    for check in spec["checks"]:
        try:
            passed = bool(run(check))
        except Exception as exc:
            passed = False
            print("ERROR %s %s: %s" % (check["type"], check.get("path"), exc))
        print("PASS" if passed else "FAIL", check["type"], check.get("path"))
        ok = ok and passed
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
'''.replace("__WORKDIR__", WORKDIR)

TEST_SH = """#!/bin/bash
# Harbor verifier entry point: reward 1 iff every check in spec.json passes.
mkdir -p /logs/verifier
if python3 /tests/check.py; then
  echo 1 > /logs/verifier/reward.txt
else
  echo 0 > /logs/verifier/reward.txt
fi
"""

TASK_TOML = """schema_version = "1.4"

[metadata]
suite = "fedagent-shell-v1"
family = "{family}"
difficulty = {difficulty}

[verifier]
timeout_sec = 60.0

[agent]
timeout_sec = 900.0

[environment]
docker_image = "{image}"
workdir = "{workdir}"
cpus = 1
memory_mb = 512
"""

ANSWER = "Write just {what} to the file `answer.txt` in the working directory."


@dataclass
class Task:
    instruction: str
    files: Dict[str, Union[str, bytes]]     # path under the workdir -> content
    checks: List[dict]
    solution: str                           # shell, run from the workdir
    difficulty: int                         # 1 one obvious command / 2 a pipeline or a look first / 3 several steps
    modes: Dict[str, int] = field(default_factory=dict)


def answer(expected) -> List[dict]:
    return [{"type": "answer", "path": "answer.txt", "expected": str(expected)}]


def log_lines(rng: random.Random, n: int) -> List[str]:
    return [f"2026-0{rng.randint(1, 9)}-{rng.randint(10, 28)} {rng.choice(LEVELS)} "
            f"{rng.choice(MESSAGES)} id={rng.randint(1000, 9999)}" for _ in range(n)]


def text(lines: List[str]) -> str:
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------------- families
def count_lines(rng):
    level = rng.choice(LEVELS)
    path = f"{rng.choice(['logs', 'var/log', 'data'])}/{rng.choice(['app', 'server', 'worker', 'sync'])}.log"
    lines = log_lines(rng, rng.randint(40, 120))
    return Task(f"How many lines of the file `{path}` contain the word {level}? "
                + ANSWER.format(what="that number"),
                {path: text(lines)}, answer(sum(f" {level} " in line for line in lines)),
                f"grep -c ' {level} ' {path} > answer.txt", 1)


def count_files(rng):
    ext = rng.choice(["txt", "md", "csv"])
    files, count = {}, 0
    for _ in range(rng.randint(8, 16)):
        sub = "/".join(rng.sample(WORDS, rng.randint(0, 2)))
        e = rng.choice(["txt", "md", "csv", "json"])
        path = f"docs/{sub + '/' if sub else ''}{rng.choice(WORDS)}{rng.randint(1, 99)}.{e}"
        if path not in files:
            files[path] = f"{rng.choice(WORDS)}\n"
            count += e == ext
    return Task(f"How many files with the extension `.{ext}` are there under the directory "
                f"`docs/`, counting all of its subdirectories? " + ANSWER.format(what="that number"),
                files, answer(count), f"find docs -type f -name '*.{ext}' | wc -l > answer.txt", 1)


def nth_line(rng):
    path = f"data/{rng.choice(['names', 'hosts', 'items'])}.txt"
    lines = [f"{rng.choice(WORDS)}-{rng.randint(100, 999)}" for _ in range(rng.randint(15, 40))]
    k = rng.randint(3, len(lines) - 2)
    return Task(f"Write line number {k} of the file `{path}` (the first line is line 1) to the "
                f"file `answer.txt` in the working directory.",
                {path: text(lines)}, answer(lines[k - 1]), f"sed -n '{k}p' {path} > answer.txt", 1)


def copy_file(rng):
    name = f"{rng.choice(WORDS)}.csv"
    src, dst = f"reports/{name}", f"backup/{rng.choice(WORDS)}-copy.csv"
    body = text([f"{i},{rng.choice(WORDS)},{rng.randint(1, 500)}" for i in range(rng.randint(5, 20))])
    return Task(f"Copy the file `{src}` to `{dst}`. The directory `backup/` does not exist yet. "
                f"Leave the original file in place.",
                {src: body},
                [{"type": "sha256", "path": dst, "expected": hashlib.sha256(body.encode()).hexdigest()},
                 {"type": "sha256", "path": src, "expected": hashlib.sha256(body.encode()).hexdigest()}],
                f"mkdir -p backup && cp {src} {dst}", 1)


def tail_lines(rng):
    path = f"logs/{rng.choice(WORDS)}.log"
    lines = log_lines(rng, rng.randint(20, 60))
    k = rng.randint(3, 8)
    return Task(f"Write the last {k} lines of the file `{path}` to the file `out/last.txt`. "
                f"The directory `out/` does not exist yet.",
                {path: text(lines)}, [{"type": "text", "path": "out/last.txt", "expected": text(lines[-k:])}],
                f"mkdir -p out && tail -n {k} {path} > out/last.txt", 1)


def word_count(rng):
    path = f"docs/{rng.choice(WORDS)}.txt"
    lines = [" ".join(rng.choice(WORDS) for _ in range(rng.randint(3, 12))) for _ in range(rng.randint(6, 20))]
    return Task(f"How many words does the file `{path}` contain (as `wc -w` counts them)? "
                + ANSWER.format(what="that number"),
                {path: text(lines)}, answer(sum(len(line.split()) for line in lines)),
                f"wc -w < {path} > answer.txt", 1)


def chmod_mode(rng):
    path = f"bin/{rng.choice(WORDS)}.sh"
    mode = rng.choice(["700", "750", "755", "744", "500"])
    return Task(f"Set the permissions of the file `{path}` to {mode} (octal).",
                {path: "#!/bin/bash\necho ok\n"}, [{"type": "mode", "path": path, "expected": mode}],
                f"chmod {mode} {path}", 1, modes={path: 0o644})


def symlink(rng):
    versions = [f"v{i}" for i in rng.sample(range(1, 9), 3)]
    target = f"releases/{rng.choice(versions)}"
    return Task(f"Create a symbolic link named `current` in the working directory that points to "
                f"the directory `{target}`.",
                {f"releases/{v}/VERSION": f"{v}\n" for v in versions},
                [{"type": "symlink", "path": "current", "target": target}],
                f"ln -s {target} current", 1)


def _sales(rng):
    return [(i, rng.choice(REGIONS), rng.randint(5, 900)) for i in range(1, rng.randint(15, 40))]


def sum_column(rng):
    rows = _sales(rng)
    return Task("The file `data/sales.csv` is comma-separated and its first line is the header "
                "`id,region,amount`. What is the sum of the `amount` column over all rows? "
                + ANSWER.format(what="that number"),
                {"data/sales.csv": text(["id,region,amount"] + [f"{i},{r},{a}" for i, r, a in rows])},
                answer(sum(a for _, _, a in rows)),
                "awk -F, 'NR>1{s+=$3} END{print s}' data/sales.csv > answer.txt", 2)


def sum_filtered(rng):
    rows = _sales(rng)
    region = rng.choice(sorted({r for _, r, _ in rows}))
    return Task("The file `data/sales.csv` is comma-separated and its first line is the header "
                f"`id,region,amount`. What is the sum of the `amount` column over the rows whose "
                f"`region` is `{region}`? " + ANSWER.format(what="that number"),
                {"data/sales.csv": text(["id,region,amount"] + [f"{i},{r},{a}" for i, r, a in rows])},
                answer(sum(a for _, r, a in rows if r == region)),
                f"awk -F, 'NR>1 && $2==\"{region}\"{{s+=$3}} END{{print s+0}}' data/sales.csv > answer.txt", 3)


def largest_file(rng):
    names = [f"{w}.bin" for w in rng.sample(WORDS, rng.randint(4, 8))]
    sizes = rng.sample(range(200, 9000), len(names))
    files = {f"store/{n}": bytes(rng.getrandbits(8) for _ in range(s)) for n, s in zip(names, sizes)}
    return Task("Which file in the directory `store/` is the largest in bytes? "
                + ANSWER.format(what="its file name, without the directory,"),
                files, answer(names[sizes.index(max(sizes))]), "ls -S store | head -n 1 > answer.txt", 2)


def unique_sorted(rng):
    pool = rng.sample(WORDS, rng.randint(5, 10))
    lines = [rng.choice(pool) for _ in range(rng.randint(15, 40))]
    return Task("Write the distinct lines of the file `data/items.txt`, sorted alphabetically, one "
                "per line, to the file `out/unique.txt`. The directory `out/` does not exist yet.",
                {"data/items.txt": text(lines)},
                [{"type": "text", "path": "out/unique.txt", "expected": text(sorted(set(lines)))}],
                "mkdir -p out && sort -u data/items.txt > out/unique.txt", 2)


def replace_value(rng):
    keys = rng.sample(["port", "timeout", "retries", "workers", "cache_mb", "log_level"], 4)
    values = {k: str(rng.randint(2, 9000)) for k in keys}
    key = rng.choice(keys)
    new = str(int(values[key]) + rng.randint(1, 50))
    before = text([f"{k}={v}" for k, v in values.items()])
    after = text([f"{k}={new if k == key else v}" for k, v in values.items()])
    return Task(f"In the file `config/app.conf`, change the value of `{key}` from {values[key]} to "
                f"{new}. Do not change any other line.",
                {"config/app.conf": before}, [{"type": "text", "path": "config/app.conf", "expected": after}],
                f"sed -i 's/^{key}={values[key]}$/{key}={new}/' config/app.conf", 2)


def json_value(rng):
    outer = rng.sample(["database", "cache", "server", "queue"], 3)
    inner = rng.sample(["port", "size", "limit", "ttl"], 3)
    doc = {o: {i: rng.randint(10, 9999) for i in inner} for o in outer}
    o, i = rng.choice(outer), rng.choice(inner)
    return Task(f"The file `config/settings.json` contains JSON. Write the value stored at "
                f"`{o}` -> `{i}` to the file `answer.txt` in the working directory.",
                {"config/settings.json": json.dumps(doc, indent=2) + "\n"}, answer(doc[o][i]),
                f"python3 -c \"import json; print(json.load(open('config/settings.json'))['{o}']['{i}'])\" > answer.txt", 2)


def top_ip(rng):
    ips = [f"10.{rng.randint(0, 9)}.{rng.randint(0, 99)}.{last}" for last in rng.sample(range(1, 250), rng.randint(4, 7))]
    counts = rng.sample(range(3, 30), len(ips))            # distinct IPs + distinct counts -> a unique maximum
    lines = [f"{ip} GET /{rng.choice(WORDS)} 200" for ip, c in zip(ips, counts) for _ in range(c)]
    rng.shuffle(lines)
    return Task("Each line of the file `logs/access.log` starts with a client IP address. Which IP "
                "address appears on the most lines? " + ANSWER.format(what="that IP address"),
                {"logs/access.log": text(lines)}, answer(ips[counts.index(max(counts))]),
                "awk '{print $1}' logs/access.log | sort | uniq -c | sort -rn | head -n 1 | awk '{print $2}' > answer.txt", 3)


def find_token(rng):
    files = {}
    for _ in range(rng.randint(8, 14)):
        files[f"vault/{'/'.join(rng.sample(WORDS, rng.randint(1, 3)))}/{rng.choice(WORDS)}.cfg"] = \
            text([f"{rng.choice(WORDS)}={rng.randint(1, 999)}" for _ in range(rng.randint(2, 6))])
    token = "".join(rng.choice("abcdef0123456789") for _ in range(16))
    target = rng.choice(sorted(files))
    files[target] += f"TOKEN={token}\n"
    return Task("Exactly one file somewhere under the directory `vault/` contains a line that "
                "starts with `TOKEN=`. " + ANSWER.format(what="the value after `TOKEN=`"),
                files, answer(token), "grep -rh '^TOKEN=' vault | cut -d= -f2 > answer.txt", 2)


def extract_archive(rng):
    name = f"{rng.choice(WORDS)}.txt"
    members = {f"bundle/{rng.choice(WORDS)}/{rng.choice(WORDS)}{i}.txt": text(rng.sample(WORDS, rng.randint(2, 9)))
               for i in range(rng.randint(3, 6))}
    lines = rng.sample(WORDS, rng.randint(5, 20))
    members[f"bundle/{rng.choice(WORDS)}/{name}"] = text(lines)
    raw = io.BytesIO()
    with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz, tarfile.open(fileobj=gz, mode="w") as tar:
        for path in sorted(members):
            info = tarfile.TarInfo(path)
            info.size = len(members[path])
            tar.addfile(info, io.BytesIO(members[path].encode()))
    return Task(f"The archive `bundle.tar.gz` contains a file named `{name}` somewhere inside it. "
                f"How many lines does that file have? " + ANSWER.format(what="that number"),
                {"bundle.tar.gz": raw.getvalue()}, answer(len(lines)),
                f"mkdir -p unpacked && tar -xzf bundle.tar.gz -C unpacked && "
                f"wc -l < \"$(find unpacked -name {name})\" > answer.txt", 3)


def run_script(rng):
    a, b, c = rng.randint(11, 99), rng.randint(11, 99), rng.randint(100, 999)
    path = f"tools/{rng.choice(WORDS)}.sh"
    return Task(f"Run the shell script `{path}` and write what it prints to the file `answer.txt` "
                f"in the working directory.",
                {path: f"#!/bin/bash\nx={a}\ny={b}\necho $(( x * y + {c} ))\n"}, answer(a * b + c),
                f"bash {path} > answer.txt", 2, modes={path: 0o644})


def sha256(rng):
    path = f"data/{rng.choice(WORDS)}.bin"
    blob = bytes(rng.getrandbits(8) for _ in range(rng.randint(200, 2000)))
    return Task(f"Compute the SHA-256 checksum of the file `{path}`. "
                + ANSWER.format(what="the 64 hexadecimal characters of the checksum"),
                {path: blob},
                [{"type": "answer", "path": "answer.txt", "expected": hashlib.sha256(blob).hexdigest(),
                  "ignore_case": True}],
                f"sha256sum {path} | cut -d' ' -f1 > answer.txt", 2)


def move_by_ext(rng):
    names = [f"{w}.{rng.choice(['log', 'log', 'txt', 'csv'])}" for w in rng.sample(WORDS, rng.randint(6, 10))]
    if not any(n.endswith(".log") for n in names):
        names[0] = names[0].rsplit(".", 1)[0] + ".log"
    if all(n.endswith(".log") for n in names):
        names[-1] = names[-1].rsplit(".", 1)[0] + ".txt"
    logs = [n for n in names if n.endswith(".log")]
    return Task("Move every file whose name ends in `.log` from the directory `inbox/` into the "
                "directory `archive/logs/`, which does not exist yet. Files that do not end in "
                "`.log` must stay in `inbox/`.",
                {f"inbox/{n}": f"{rng.choice(WORDS)}\n" for n in names},
                [{"type": "listing", "path": "archive/logs", "expected": logs},
                 {"type": "listing", "path": "inbox", "expected": [n for n in names if n not in logs]}],
                "mkdir -p archive/logs && mv inbox/*.log archive/logs/", 2)


def sort_numeric(rng):
    nums = rng.sample(range(1, 5000), rng.randint(10, 30))
    return Task("The file `data/nums.txt` holds one integer per line. Write the same numbers sorted "
                "from largest to smallest, one per line, to the file `out/sorted.txt`. The "
                "directory `out/` does not exist yet.",
                {"data/nums.txt": text([str(n) for n in nums])},
                [{"type": "text", "path": "out/sorted.txt", "expected": text([str(n) for n in sorted(nums, reverse=True)])}],
                "mkdir -p out && sort -nr data/nums.txt > out/sorted.txt", 2)


def _events(rng):
    users = rng.sample(WORDS, rng.randint(4, 9))
    return [(rng.randint(1000, 9999), rng.choice(users), rng.choice(["login", "logout", "upload"]))
            for _ in range(rng.randint(20, 50))]


def distinct_count(rng):
    rows = _events(rng)
    return Task("The file `data/events.csv` is comma-separated and its first line is the header "
                "`id,user,action`. How many distinct values appear in the `user` column? "
                + ANSWER.format(what="that number"),
                {"data/events.csv": text(["id,user,action"] + [f"{i},{u},{a}" for i, u, a in rows])},
                answer(len({u for _, u, _ in rows})),
                "tail -n +2 data/events.csv | cut -d, -f2 | sort -u | wc -l > answer.txt", 3)


def csv_filter_count(rng):
    rows = [(rng.choice(WORDS), rng.choice(COUNTRIES), rng.randint(18, 70)) for _ in range(rng.randint(20, 45))]
    country, age = rng.choice(COUNTRIES), rng.randint(25, 50)
    return Task("The file `data/users.csv` is comma-separated and its first line is the header "
                f"`name,country,age`. How many rows have `country` equal to `{country}` and `age` "
                f"greater than {age}? " + ANSWER.format(what="that number"),
                {"data/users.csv": text(["name,country,age"] + [f"{n},{c},{a}" for n, c, a in rows])},
                answer(sum(c == country and a > age for _, c, a in rows)),
                f"awk -F, 'NR>1 && $2==\"{country}\" && $3>{age}' data/users.csv | wc -l > answer.txt", 3)


def total_size(rng):
    files = {}
    for _ in range(rng.randint(6, 12)):
        sub = "/".join(rng.sample(WORDS, rng.randint(0, 2)))
        files[f"store/{sub + '/' if sub else ''}{rng.choice(WORDS)}{rng.randint(1, 99)}.{rng.choice(['dat', 'dat', 'tmp'])}"] = \
            bytes(rng.getrandbits(8) for _ in range(rng.randint(50, 3000)))
    return Task("What is the total size in bytes of all files whose names end in `.dat` under the "
                "directory `store/`, counting all of its subdirectories? " + ANSWER.format(what="that number"),
                files, answer(sum(len(v) for k, v in files.items() if k.endswith(".dat"))),
                "find store -type f -name '*.dat' -printf '%s\\n' | awk '{s+=$1} END{print s+0}' > answer.txt", 3)


FAMILIES: Dict[str, Callable[[random.Random], Task]] = {
    "count-lines": count_lines, "count-files": count_files, "nth-line": nth_line,
    "copy-file": copy_file, "tail-lines": tail_lines, "word-count": word_count,
    "chmod-mode": chmod_mode, "symlink": symlink, "sum-column": sum_column,
    "sum-filtered": sum_filtered, "largest-file": largest_file, "unique-sorted": unique_sorted,
    "replace-value": replace_value, "json-value": json_value, "top-ip": top_ip,
    "find-token": find_token, "extract-archive": extract_archive, "run-script": run_script,
    "sha256": sha256, "move-by-ext": move_by_ext, "sort-numeric": sort_numeric,
    "distinct-count": distinct_count, "csv-filter-count": csv_filter_count, "total-size": total_size,
}


def write_task(root: Path, family: str, task: Task) -> None:
    (root / "instruction.md").parent.mkdir(parents=True, exist_ok=True)
    (root / "instruction.md").write_text(task.instruction + "\n")
    (root / "task.toml").write_text(TASK_TOML.format(family=family, difficulty=task.difficulty,
                                                     image=IMAGE, workdir=WORKDIR))
    for rel, content in task.files.items():
        path = root / "environment" / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content if isinstance(content, bytes) else content.encode())
        path.chmod(task.modes.get(rel, 0o644))
    (root / "tests").mkdir()
    (root / "tests" / "test.sh").write_text(TEST_SH)
    (root / "tests" / "check.py").write_text(CHECK_PY)
    (root / "tests" / "spec.json").write_text(json.dumps({"checks": task.checks}, indent=2) + "\n")
    (root / "solution").mkdir()
    (root / "solution" / "solve.sh").write_text(f"#!/bin/bash\ncd {WORKDIR}\n{task.solution}\n")
    for script in (root / "tests" / "test.sh", root / "solution" / "solve.sh"):
        script.chmod(0o755)


def generate(out: Path, train_per_family: int, val_per_family: int, seed: int) -> Dict[str, int]:
    if out.exists():
        shutil.rmtree(out)
    counts = {"train": 0, "val": 0}
    for family, build in FAMILIES.items():
        # one index space per family; the first train_per_family instances are train, the rest val
        for idx in range(train_per_family + val_per_family):
            split = "train" if idx < train_per_family else "val"
            task = build(random.Random(f"{seed}/{family}/{idx}"))
            write_task(out / split / f"{family}-{idx:03d}", family, task)
            counts[split] += 1
    return counts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=str(REPO_ROOT / "data" / "harbor" / "shell_v1"))
    ap.add_argument("--train-per-family", type=int, default=12)
    ap.add_argument("--val-per-family", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    counts = generate(Path(args.out), args.train_per_family, args.val_per_family, args.seed)
    print(f"wrote {counts['train']} train + {counts['val']} val tasks "
          f"({len(FAMILIES)} families) under {args.out}")


if __name__ == "__main__":
    main()
