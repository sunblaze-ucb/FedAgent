"""The shipped ALFWorld game manifests, and the loader that makes them authoritative.

Sorting the engine's directory walk (2026-07-26) made the game ORDER a pure function of the
collected set. The manifest closes the other half — the SET. `game.tw-pddl` and its `solvable`
flag are produced by ALFWorld's preprocessing step, not shipped with the raw trajectories, so
the walk collects whatever that step happened to produce on a given machine: on this one, 47 of
the 187 eligible `valid_seen` trials (25%) have no game file at all. A machine whose
preprocessing went further collects a larger set, every index shifts, and client shards and the
val set silently become different games.

These tests cover the loader's contract (schema validation, task-type filtering by path, strict
vs non-strict handling of missing games) and the shipped assets themselves (present, valid,
self-consistent, and matching the counts the runs and the hardness labelling were built on).

Exercising the engine end-to-end needs alfworld + textworld; equivalence to the walk is instead
established by `tools/gen_alfworld_manifest.py --check`, which regenerates from the data and
compares sha256 (all three splits OK on the machine the manifests were cut from).

Stdlib only — `fedagent.envs.alfworld` re-exports its httpx client lazily so this module stays
importable under a bare interpreter (the generator has to run wherever the DATA is).
"""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fedagent.envs.alfworld import game_manifest as gm  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SPLITS = ("train", "eval_in_distribution", "eval_out_of_distribution")
# what the runs and the hardness labelling were built on (see docs/bugfixes.md 2026-07-27)
EXPECTED_N = {"train": 3553, "eval_in_distribution": 140, "eval_out_of_distribution": 134}


def _rel(task_type="pick_and_place_simple", trial="trial_T1", scene="1"):
    return f"{task_type}-Apple-None-Microwave-{scene}/{trial}/{gm.GAME_FILE}"


def _manifest(games, split="eval_in_distribution", **over):
    m = {"schema": gm.SCHEMA, "split": split, "source": "json_2.1.1/valid_seen",
         "task_types": list(gm.TASK_TYPES), "n": len(games), "sha256": gm.digest(games),
         "games": list(games)}
    m.update(over)
    return m


def _write(tmp_path, m, name="m.json"):
    p = tmp_path / name
    p.write_text(json.dumps(m))
    return str(p)


def _touch(root, rel):
    p = os.path.join(root, rel.replace("/", os.sep))
    os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, "w").close()
    return p


# --------------------------------------------------------------------- shipped assets

@pytest.mark.parametrize("split", SPLITS)
def test_shipped_manifest_is_present_and_self_consistent(split):
    path = gm.default_path(split)
    assert path and os.path.isfile(path), f"no shipped manifest for {split}"
    assert os.path.abspath(path).startswith(REPO)
    m = gm.load(path)                      # validates schema, n, and the sha256 of the list
    assert m["split"] == split
    assert m["n"] == EXPECTED_N[split]
    assert len(set(m["games"])) == m["n"], "duplicate game paths"
    assert m["games"] == sorted(m["games"]), "not in canonical order"
    assert all(g.endswith("/" + gm.GAME_FILE) for g in m["games"])
    assert all(gm.task_type_of(g) in gm.TASK_TYPES for g in m["games"])


def test_default_path_is_found_from_the_module_not_the_cwd(monkeypatch, tmp_path):
    """The vendored engine sits nine directories deep and the tools sit one; both must find the
    same file without hardcoding a level count."""
    monkeypatch.chdir(tmp_path)
    assert gm.default_path("train") == os.path.join(REPO, gm.SHIPPED_DIR, "train.json")
    assert gm.default_path("no_such_split") is None


# --------------------------------------------------------------------- loader contract

def test_task_type_of():
    assert gm.task_type_of(_rel("pick_two_obj_and_place")) == "pick_two_obj_and_place"
    assert gm.task_type_of("not/a/game.json") is None
    assert gm.task_type_of("bogus_type-A-None-B-1/trial_X/game.tw-pddl") is None


def test_load_rejects_a_manifest_it_cannot_trust(tmp_path):
    good = [_rel(scene=str(i)) for i in range(3)]
    for broken, why in [
        ({"schema": 999}, "wrong schema"),
        (_manifest(good, n=99), "n disagrees with the list"),
        (_manifest(good, sha256="deadbeef"), "digest disagrees with the list"),
        (_manifest([]), "empty"),
    ]:
        with pytest.raises(ValueError):
            gm.load(_write(tmp_path, broken, f"{why.replace(' ', '_')}.json"))


def test_game_files_filters_by_task_type_and_keeps_relative_order(tmp_path):
    games = sorted([_rel("pick_and_place_simple", scene="1"),
                    _rel("look_at_obj_in_light", scene="2"),
                    _rel("pick_and_place_simple", scene="3")])
    root = tmp_path / "data"
    for g in games:
        _touch(str(root), g)
    path = _write(tmp_path, _manifest(games))
    env = {gm.MANIFEST_ENV: path}

    got = gm.game_files(str(root), "eval_in_distribution", gm.TASK_TYPES, env=env)
    assert [os.path.relpath(p, str(root)).replace(os.sep, "/") for p in got] == games

    only_pick = gm.game_files(str(root), "eval_in_distribution",
                              ("pick_and_place_simple",), env=env)
    assert [os.path.basename(os.path.dirname(os.path.dirname(p))).split("-")[0]
            for p in only_pick] == ["pick_and_place_simple"] * 2
    assert only_pick == [p for p in got if "pick_and_place_simple" in p]   # order preserved

    with pytest.raises(ValueError):        # a subset that selects nothing is a config error
        gm.game_files(str(root), "eval_in_distribution", ("pick_two_obj_and_place",), env=env)


def test_missing_games_abort_by_default_and_can_be_downgraded(tmp_path, capsys):
    games = sorted([_rel(scene=str(i)) for i in range(4)])
    root = tmp_path / "data"
    for g in games[:2]:                    # two of the four never made it through preprocessing
        _touch(str(root), g)
    path = _write(tmp_path, _manifest(games))

    with pytest.raises(RuntimeError, match="2/4 listed games are missing"):
        gm.game_files(str(root), "eval_in_distribution", env={gm.MANIFEST_ENV: path})

    got = gm.game_files(str(root), "eval_in_distribution",
                        env={gm.MANIFEST_ENV: path, gm.STRICT_ENV: "0"})
    assert len(got) == 2
    assert "WARNING (non-strict)" in capsys.readouterr().out


def test_split_mismatch_is_an_error(tmp_path):
    games = [_rel()]
    _touch(str(tmp_path / "data"), games[0])
    path = _write(tmp_path, _manifest(games, split="train"))
    with pytest.raises(ValueError, match="manifest is for split"):
        gm.game_files(str(tmp_path / "data"), "eval_in_distribution",
                      env={gm.MANIFEST_ENV: path})


def test_manifest_can_be_switched_off_and_missing_files_fall_back(tmp_path):
    """`none` (and a path that does not exist) → None, i.e. the engine keeps its walk."""
    assert gm.game_files(str(tmp_path), "train", env={gm.MANIFEST_ENV: "none"}) is None
    assert gm.game_files(str(tmp_path), "train", env={gm.MANIFEST_ENV: "off"}) is None
    assert gm.game_files(str(tmp_path), "train",
                         env={gm.MANIFEST_ENV: str(tmp_path / "nope.json")}) is None


def test_collect_matches_a_manifest_built_from_the_same_tree(tmp_path):
    """The generator's filter is the engine's filter: solvable games of the wanted types only."""
    root = tmp_path / "d"
    def trial(task_type, scene, *, solvable=True, game=True, movable=False):
        d = os.path.join(str(root), ("movable_" if movable else "") + f"{task_type}-A-None-B-{scene}",
                         f"trial_{scene}")
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "traj_data.json"), "w") as f:
            json.dump({"task_type": task_type}, f)
        if game:
            with open(os.path.join(d, gm.GAME_FILE), "w") as f:
                json.dump({"solvable": solvable}, f)
    trial("pick_and_place_simple", "1")
    trial("look_at_obj_in_light", "2")
    trial("pick_and_place_simple", "3", solvable=False)   # dropped: unsolvable
    trial("pick_and_place_simple", "4", game=False)       # dropped: never preprocessed
    trial("pick_and_place_simple", "5", movable=True)     # dropped: movable
    trial("not_a_task_type", "6")                         # dropped: task type

    got = gm.collect(str(root))
    assert len(got) == 2 and got == sorted(got)
    built = gm.build(str(root), "eval_in_distribution")
    assert built["n"] == 2 and built["sha256"] == gm.digest(got)
