"""The authoritative ALFWorld game list — a shipped asset, not a filesystem accident.

`AlfredTWEnv.collect_game_files` builds a split's game list by walking `$ALFWORLD_DATA` and
keeping the solvable, non-`movable`/`Sliced` trials of the configured task types. Everything
downstream of it is POSITIONAL — the seeded shuffle, the client index slices, the
`num_train_games`/`start_idx` caps, and `games[seed]` under `ALFWORLD_SEED_IS_INDEX` — so the
list defines which games each client trains on and which games the val curve is measured on.

Sorting the walk (2026-07-26) made the ORDER a pure function of the collected set. This module
closes the other half: it pins the SET. `game.tw-pddl` and its `solvable` flag are produced by
ALFWorld's preprocessing step, not shipped with the raw trajectories, so the walk's output
depends on how completely that step ran — on this machine 47 of the 187 eligible `valid_seen`
trials (25%) have no game file at all. A machine whose preprocessing went further collects a
LARGER set, every index shifts, and the val set / client shards silently become different games.

With a manifest the run reads a checked-in list of split-relative paths instead of walking:

* **Reproducible** — the same manifest yields the same games, in the same order, on any machine.
* **Loud on drift** — a listed game that is missing on disk is an error (with the count and
  examples), not a silent renumbering. `ALFWORLD_MANIFEST_STRICT=0` downgrades it to a warning
  that DROPS the missing entries (which does renumber — use it only to triage).
* **Extra games on disk are ignored by construction**, so a fuller preprocessing state cannot
  change a run's task set.
* **Faster** — it replaces the walk *and* its ~2 JSON reads per trial, so it subsumes the
  optional `ALFWORLD_MANIFEST_CACHE` speed-up (that cache stores an unvalidated pre-shuffle
  walk; this is a validated, versioned asset).

One manifest per split holds ALL six task types in canonical order; a run's `task_types` subset
(the eval breakdown pins one) is filtered from it by path, so the subset inherits the same
relative order. Generate/refresh with `tools/gen_alfworld_manifest.py`.

Stdlib only: the vendored engine imports it inside the ALFWorld service env, and the generator
runs under a bare interpreter.
"""
import hashlib
import json
import os

SCHEMA = 1
# The six AlfredTWEnv task families, i.e. the leading segment of a game's grandparent dir.
TASK_TYPES = (
    "pick_and_place_simple",
    "look_at_obj_in_light",
    "pick_clean_then_place_in_recep",
    "pick_heat_then_place_in_recep",
    "pick_cool_then_place_in_recep",
    "pick_two_obj_and_place",
)
GAME_FILE = "game.tw-pddl"
STRICT_ENV = "ALFWORLD_MANIFEST_STRICT"
MANIFEST_ENV = "ALFWORLD_GAME_MANIFEST"
SHIPPED_DIR = os.path.join("data", "alfworld_games")


def task_type_of(rel_path):
    """The task family of a split-relative game path, or None if it is not one.

    Layout: ``{task_type}-{obj}-...-{scene}/{trial_...}/game.tw-pddl`` — same derivation the
    env client uses for the eval breakdown tag."""
    parts = rel_path.replace("\\", "/").split("/")
    if len(parts) < 3 or parts[-1] != GAME_FILE:
        return None
    head = parts[-3].split("-", 1)[0]
    return head if head in TASK_TYPES else None


def digest(rel_paths):
    """Stable identity of a game list: sha256 over the newline-joined relative paths."""
    return hashlib.sha256("\n".join(rel_paths).encode()).hexdigest()


def collect(data_path, task_types=TASK_TYPES):
    """Walk a split root and return its game list as sorted split-relative paths.

    VERBATIM with the engine's own filter (`collect_game_files`): a trial counts when it has a
    `traj_data.json`, is not under a `movable`/`Sliced` path, has a task type in `task_types`,
    has a `game.tw-pddl`, and that file carries `solvable: true`. Sorted, so the result is a
    pure function of the collected set."""
    data_path = os.path.abspath(data_path)
    wanted = set(task_types)
    out = []
    for root, _dirs, files in os.walk(data_path, topdown=False):
        if "traj_data.json" not in files:
            continue
        if "movable" in root or "Sliced" in root:
            continue
        game_file = os.path.join(root, GAME_FILE)
        try:
            with open(os.path.join(root, "traj_data.json")) as f:
                if json.load(f).get("task_type") not in wanted:
                    continue
        except (OSError, ValueError):
            continue
        if not os.path.exists(game_file):
            continue
        try:
            with open(game_file) as f:
                if not json.load(f).get("solvable"):
                    continue
        except (OSError, ValueError):
            continue
        out.append(os.path.relpath(game_file, data_path).replace(os.sep, "/"))
    out.sort()
    return out


def build(data_path, split, task_types=TASK_TYPES):
    """A manifest dict for one split (see `write`/`load` for the on-disk form)."""
    games = collect(data_path, task_types)
    return {
        "schema": SCHEMA,
        "split": split,
        # provenance without leaking a home dir: the last two components of the split root
        "source": "/".join(os.path.abspath(data_path).split(os.sep)[-2:]),
        "task_types": list(task_types),
        "n": len(games),
        "sha256": digest(games),
        "games": games,
    }


def write(manifest, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(manifest, f, indent=1)
    return path


def load(path):
    """Read + schema-check a manifest. Raises ValueError on anything malformed: a manifest that
    cannot be trusted must not silently degrade to a walk."""
    with open(path) as f:
        m = json.load(f)
    if not isinstance(m, dict) or m.get("schema") != SCHEMA:
        raise ValueError(f"{path}: not an ALFWorld game manifest of schema {SCHEMA}")
    games = m.get("games")
    if not isinstance(games, list) or not games:
        raise ValueError(f"{path}: manifest carries no games")
    if len(games) != m.get("n"):
        raise ValueError(f"{path}: n={m.get('n')} but {len(games)} games listed")
    if m.get("sha256") and digest(games) != m["sha256"]:
        raise ValueError(f"{path}: sha256 does not match its own game list (edited by hand?)")
    return m


def default_path(split, start=None):
    """The shipped manifest for a split: ``<repo>/data/alfworld_games/<split>.json``.

    Found by walking up from this file, so the vendored engine (nine directories deep) and the
    tools agree without either hardcoding a level count. None if there is no shipped manifest."""
    d = os.path.dirname(os.path.abspath(start or __file__))
    for _ in range(16):
        cand = os.path.join(d, SHIPPED_DIR, f"{split}.json")
        if os.path.isfile(cand):
            return cand
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    return None


def resolve_path(split, env=None):
    """Which manifest to use for a split: ``$ALFWORLD_GAME_MANIFEST`` wins (``none`` disables
    manifests entirely and falls back to the walk), else the shipped one, else None."""
    env = os.environ if env is None else env
    explicit = (env.get(MANIFEST_ENV) or "").strip()
    if explicit.lower() in ("none", "off", "0"):
        return None
    if explicit:
        return explicit
    return default_path(split)


def game_files(data_path, split, task_types=TASK_TYPES, env=None):
    """The absolute game paths for a split, from the manifest — or None to fall back to the walk.

    Applies the run's `task_types` filter by path (one manifest serves every subset, in the same
    relative order) and validates that every listed game exists. Missing games raise by default;
    `ALFWORLD_MANIFEST_STRICT=0` warns and drops them instead."""
    env = os.environ if env is None else env
    path = resolve_path(split, env)
    if not path or not os.path.isfile(path):
        return None
    m = load(path)
    if m.get("split") != split:
        raise ValueError(f"{path}: manifest is for split {m.get('split')!r}, run wants {split!r}")
    wanted = set(task_types)
    rel = [g for g in m["games"] if task_type_of(g) in wanted]
    if not rel:
        raise ValueError(f"{path}: no games left after the task_types filter {sorted(wanted)}")
    root = os.path.abspath(data_path)
    absolute, missing = [], []
    for g in rel:
        p = os.path.join(root, g.replace("/", os.sep))
        (absolute if os.path.exists(p) else missing).append(p)
    if missing:
        head = ", ".join(os.path.relpath(p, root) for p in missing[:3])
        msg = (f"ALFWorld game manifest {path}: {len(missing)}/{len(rel)} listed games are "
               f"missing under {root} (e.g. {head}). The manifest defines the run's task set, so "
               f"dropping them would renumber every index — client shards and the val set would "
               f"stop matching other machines. Fix the data (ALFWorld's game.tw-pddl "
               f"preprocessing) or regenerate with tools/gen_alfworld_manifest.py; set "
               f"{STRICT_ENV}=0 to proceed anyway (NOT reproducible).")
        if str(env.get(STRICT_ENV, "1")).strip().lower() not in ("0", "false", "no", "off"):
            raise RuntimeError(msg)
        print(f"[alfworld-manifest] WARNING (non-strict): {msg}", flush=True)
    print(f"[alfworld-manifest] {os.path.basename(path)}: {len(absolute)} games "
          f"(split={split}, of {m['n']} listed, sha256={m['sha256'][:12]}); extra games on disk "
          f"are ignored by construction", flush=True)
    return absolute
