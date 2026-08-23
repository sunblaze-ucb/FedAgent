"""ALFWorld env-het suite: kernel-variant rewrites + the scene_disjoint partition.

Covers the trainer-side contracts that need no textworld install (the
fedagent-verl08 env): anchored rewrites apply with exact occurrence counts and
never touch the task rhs (tau red line), v_default is byte-pure, per-client
variant assignment is deterministic WebShop-math, and scene_disjoint delivers
its three headline properties -- fixed per-client size, env_div-0 byte identity
across clients, and a task-type mix matched to the global marginal.

Planner-level solvability of the rewritten kernels (fast-downward replan +
plan execution + won assert) needs alfworld/textworld and lives in
tools/env_heterogeneity/verify_alfworld_kernel_variants.py (run it in the
verl-agent-alfworld env). See docs/dev_doc/alfworld_env_heterogeneity.md.

Real game data is used when $ALFWORLD_DATA (or ~/.cache/alfworld) is present;
the rewrite tests otherwise fall back to a synthetic fixture built from the
measured 2026-08 anchor layout, so the suite stays runnable on data-less CI.
"""
import itertools
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "fedagent", "envs", "alfworld", "engine")))

np = pytest.importorskip("numpy")

from agent_system.environments.alfworld_kernel_variants import (  # noqa: E402
    KERNEL_VARIANT_STRATEGIES,
    VARIANT_POOLS,
    assignment_table,
    resolve_pool,
    rewrite_domain,
    rewrite_game_data,
    rewrite_grammar,
    rewrite_problem,
    variant_for_client,
)
from agent_system.environments.partition_strategy import (  # noqa: E402
    _scene_disjoint_partition_alfworld,
    partition_dataset,
)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MANIFEST = os.path.join(REPO_ROOT, "data", "alfworld_games", "train.json")
ALFWORLD_DATA = os.environ.get(
    "ALFWORLD_DATA", os.path.expanduser("~/.cache/alfworld"))
TRAIN_ROOT = os.path.join(ALFWORLD_DATA, "json_2.1.1", "train")


def _manifest_games():
    with open(MANIFEST) as f:
        return json.load(f)["games"]


def _have_game_data():
    if not os.path.isfile(MANIFEST) or not os.path.isdir(TRAIN_ROOT):
        return False
    return os.path.isfile(os.path.join(TRAIN_ROOT, _manifest_games()[0]))


needs_data = pytest.mark.skipif(
    not _have_game_data(), reason="ALFWorld game data not present")


def _one_game_per_task_type():
    seen = {}
    for rel in _manifest_games():
        tt = rel.split("/")[0].split("-")[0]
        if tt not in seen:
            with open(os.path.join(TRAIN_ROOT, rel)) as f:
                seen[tt] = json.load(f)
    assert len(seen) == 6
    return seen


# --------------------------------------------------------------------------- #
# variant assignment                                                          #
# --------------------------------------------------------------------------- #

def test_pools_have_control_arm_first():
    for strategy, pool in VARIANT_POOLS.items():
        assert pool[0] == "v_default", strategy
        assert len(pool) == 4, strategy
        assert resolve_pool(strategy, 0) == pool
        assert resolve_pool(strategy, 2) == pool[:2]


def test_variant_assignment_matches_webshop_math():
    # The exact WebShop env-variant formula: RandomState(42 + client_id).randint(N).
    for strategy in KERNEL_VARIANT_STRATEGIES:
        pool = VARIANT_POOLS[strategy]
        for cid in range(25):
            expect = pool[np.random.RandomState(42 + cid).randint(len(pool))]
            got = variant_for_client(strategy, cid)
            assert got == {"strategy": strategy, "key": expect, "n": 4}


def test_variant_assignment_deterministic_and_round_stable():
    t1 = assignment_table("goal_variant", 100)
    t2 = assignment_table("goal_variant", 100)
    assert t1 == t2
    # every pool member is actually exercised at federation scale
    assert set(t1.values()) == set(VARIANT_POOLS["goal_variant"])


def test_variant_n_bounds():
    with pytest.raises(ValueError):
        resolve_pool("goal_variant", 1)
    with pytest.raises(ValueError):
        resolve_pool("goal_variant", 5)
    with pytest.raises(ValueError):
        resolve_pool("no_such_strategy")


# --------------------------------------------------------------------------- #
# rewrites: synthetic fixture (anchor layout as measured 2026-08)             #
# --------------------------------------------------------------------------- #

SYN_GRAMMAR = (
    'grammar :: """\n    {\n        "intro": [\n            {\n'
    '                "rhs": "-= Welcome to TextWorld, ALFRED! =-'
    r'\n\n#look.feedback#\n\n#task#"'
    '\n            }\n        ],\n\n        "task": [\n            {\n'
    '                "rhs": "Your task is to: put a book in desk."\n'
    '            }\n        ],\n\n        "GotoLocation.feedback": [\n            {\n'
    '                "rhs": "You arrive at {r.name}. #examineReceptacle.feedback#"\n'
    '            }\n        ],\n\n        "OpenObject.feedback": [\n            {\n'
    '                "rhs": "You open the {r.name}. #examineReceptacle.feedback#"\n'
    '            }\n        ],\n\n        "CloseObject.feedback": [\n            {\n'
    '                "rhs": "You close the {r.name}."\n'
    '            }\n        ],\n\n        "PickupObject.feedback": [\n            {\n'
    '                "rhs": "You pick up the {o.name} from the {r.name}."\n'
    '            }\n        ],\n\n        "PutObject.feedback": [\n            {\n'
    '                "rhs": "You move the {o.name} to the {r.name}."\n'
    '            }\n        ]\n    }\n"""\n'
)

SYN_DOMAIN = (
    "(define (domain alfred)\n"
    " (:action PickupObject\n"
    "    :parameters (?a - agent ?l - location ?o - object ?r - receptacle)\n"
    "    :precondition\n"
    "        (and\n"
    "            (pickupable ?o)\n"
    "            (atLocation ?a ?l)\n"
    "        )\n"
    "    :effect (and (holds ?a ?o))\n"
    " )\n"
    " (:action PutObject\n"
    "    :effect (and\n"
    "                (inReceptacle ?o ?r)\n"
    "                (not (holds ?a ?o))\n"
    "                (not (holdsAny ?a))\n"
    "                (increase (total-cost) 1)\n"
    "            )\n"
    " )\n"
    ")\n"
)

SYN_PROBLEM = (
    "(define (problem plan_x)\n"
    "(:init (atLocation agent1 loc1))\n"
    "(:goal\n"
    "    (and\n"
    "        (exists (?r - receptacle)\n"
    "            (exists (?o - object)\n"
    "                (and\n"
    "                    (inReceptacle ?o ?r)\n"
    "                    (objectType ?o BookType)\n"
    "                    (receptacleType ?r DeskType)\n"
    "                )\n"
    "            )\n"
    "        )\n"
    "    )\n"
    ")\n"
    ")\n"
)

SYN_PROBLEM_LOOKAT = (
    "(define (problem plan_y)\n"
    "(:init (atLocation agent1 loc1))\n"
    "(:goal\n"
    "    (and\n"
    "        (exists (?ot - object)\n"
    "            (and (toggleable ?ot) (isToggled ?ot))\n"
    "        )\n"
    "        (exists (?o - object ?a - agent)\n"
    "            (and (objectType ?o BookType) (holds ?a ?o))\n"
    "        )\n"
    "    )\n"
    ")\n"
    ")\n"
)


def test_grammar_rewrites_synthetic():
    task = '"rhs": "Your task is to: put a book in desk."'
    for key in VARIANT_POOLS["obs_variant"]:
        out = rewrite_grammar(SYN_GRAMMAR, key)
        assert task in out, key                      # tau red line
        if key == "v_default":
            assert out == SYN_GRAMMAR
        else:
            assert out != SYN_GRAMMAR
    terse = rewrite_grammar(SYN_GRAMMAR, "v_terse_goto")
    assert '"rhs": "You arrive at {r.name}."' in terse
    assert "You arrive at {r.name}. #examineReceptacle.feedback#" not in terse
    # open feedback keeps its contents playback under terse-goto
    assert "You open the {r.name}. #examineReceptacle.feedback#" in terse
    blind = rewrite_grammar(SYN_GRAMMAR, "v_blind_intro")
    assert r"=-\n\n#task#" in blind
    assert "#look.feedback#" not in blind.split('"task"')[0].split("intro")[1]
    para = rewrite_grammar(SYN_GRAMMAR, "v_paraphrase")
    for gone in ("You arrive at", "You open the", "You close the",
                 "You pick up the", "You move the"):
        assert gone not in para
    assert "#examineReceptacle.feedback#" in para   # playback preserved


def test_domain_rewrites_synthetic():
    assert rewrite_domain(SYN_DOMAIN, "v_default") == SYN_DOMAIN
    gate = rewrite_domain(SYN_DOMAIN, "v_examine_gate")
    assert "(pickupable ?o)\n            (checked ?r)" in gate
    assert "(not (opened ?r))" not in gate
    close = rewrite_domain(SYN_DOMAIN, "v_autoclose")
    assert "(not (holdsAny ?a))\n                (not (opened ?r))" in close
    assert "(checked ?r)" not in close
    both = rewrite_domain(SYN_DOMAIN, "v_gate_autoclose")
    assert "(checked ?r)" in both and "(not (opened ?r))" in both


def test_goal_rewrites_synthetic():
    assert rewrite_problem(SYN_PROBLEM, "v_default") == SYN_PROBLEM
    ex = rewrite_problem(SYN_PROBLEM, "v_examined")
    assert "(receptacleType ?r DeskType) (checked ?r)" in ex
    cl = rewrite_problem(SYN_PROBLEM, "v_closed")
    assert "(receptacleType ?r DeskType) (not (opened ?r))" in cl
    ec = rewrite_problem(SYN_PROBLEM, "v_examined_closed")
    assert "(checked ?r) (not (opened ?r))" in ec
    # goal block only: init section untouched
    for out in (ex, cl, ec):
        assert out.split("(:goal")[0] == SYN_PROBLEM.split("(:goal")[0]
    # look_at shape: examined rides holds; closed is identity by design
    ex2 = rewrite_problem(SYN_PROBLEM_LOOKAT, "v_examined")
    assert "(holds ?a ?o) (checked ?o)" in ex2
    assert rewrite_problem(SYN_PROBLEM_LOOKAT, "v_closed") == SYN_PROBLEM_LOOKAT


def test_anchor_drift_fails_loud():
    with pytest.raises(ValueError):
        rewrite_grammar(SYN_GRAMMAR.replace("You arrive at", "You arrived at"),
                        "v_terse_goto")
    with pytest.raises(ValueError):
        rewrite_domain(SYN_DOMAIN.replace("(pickupable ?o)", "(pickupable ?x)"),
                       "v_examine_gate")
    with pytest.raises(ValueError):
        rewrite_problem(SYN_PROBLEM.replace("(:goal", "(:objective"), "v_examined")


# --------------------------------------------------------------------------- #
# rewrites: real game data (all six task types)                               #
# --------------------------------------------------------------------------- #

@needs_data
def test_rewrites_apply_on_all_task_types():
    games = _one_game_per_task_type()
    for tt, data in games.items():
        for strategy in KERNEL_VARIANT_STRATEGIES:
            for key in VARIANT_POOLS[strategy]:
                variant = {"strategy": strategy, "key": key}
                out = rewrite_game_data(data, variant)
                changed = any(out[k] != data[k]
                              for k in ("grammar", "pddl_domain", "pddl_problem"))
                if key == "v_default":
                    assert not changed, (tt, strategy)
                elif strategy == "goal_variant" and key == "v_closed" \
                        and tt == "look_at_obj_in_light":
                    assert not changed  # no goal receptacle -> identity by design
                else:
                    assert changed, (tt, strategy, key)


@needs_data
def test_real_task_line_untouched_by_every_grammar_variant():
    import re
    task_re = re.compile(r'"rhs": "Your task is to: [^"]*"')
    games = _one_game_per_task_type()
    for tt, data in games.items():
        before = task_re.findall(data["grammar"])
        assert len(before) == 1
        for key in VARIANT_POOLS["obs_variant"]:
            after = task_re.findall(rewrite_grammar(data["grammar"], key))
            assert after == before, (tt, key)


# --------------------------------------------------------------------------- #
# scene_disjoint partition                                                    #
# --------------------------------------------------------------------------- #

def _synthetic_paths():
    # 24 scenes x 4 room types, several specs per scene, deterministic counts.
    paths = []
    rooms = {"kitchen": (1, 13), "living": (201, 213),
             "bedroom": (301, 313), "bathroom": (401, 413)}
    tasks = {"kitchen": ["pick_heat_then_place_in_recep", "pick_and_place_simple"],
             "living": ["pick_and_place_simple", "pick_two_obj_and_place"],
             "bedroom": ["look_at_obj_in_light", "pick_two_obj_and_place"],
             "bathroom": ["pick_and_place_simple", "pick_two_obj_and_place"]}
    for room, (lo, hi) in rooms.items():
        for scene in range(lo, hi):
            for t_i, tt in enumerate(tasks[room]):
                for trial in range(6):
                    paths.append(
                        f"/data/train/{tt}-Obj{t_i}-None-Recep-{scene}"
                        f"/trial_T{scene:04d}{t_i}{trial:02d}/game.tw-pddl")
    return paths


def test_scene_disjoint_fixed_size_and_div0_identity(capsys):
    paths = _synthetic_paths()
    shards = [
        _scene_disjoint_partition_alfworld(
            paths, k, 10, min_samples_per_client=40,
            env_div=0.0, scenes_per_client=8)
        for k in range(10)
    ]
    assert all(s == shards[0] for s in shards)       # env_div=0 -> byte identity
    assert all(len(s) == 40 for s in shards)
    shards1 = [
        set(_scene_disjoint_partition_alfworld(
            paths, k, 10, min_samples_per_client=40,
            env_div=1.0, scenes_per_client=8))
        for k in range(10)
    ]
    assert all(len(s) == 40 for s in shards1)
    mean_j = np.mean([len(a & b) / len(a | b)
                      for a, b in itertools.combinations(shards1, 2)])
    assert mean_j < 0.9                              # divergence actually moves


def test_scene_disjoint_room_stratification():
    paths = _synthetic_paths()
    shard = _scene_disjoint_partition_alfworld(
        paths, 3, 10, min_samples_per_client=40, env_div=1.0, scenes_per_client=8)
    scenes = {p.split("/")[-3].rsplit("-", 1)[1] for p in shard}
    rooms = {"kitchen": 0, "living": 0, "bedroom": 0, "bathroom": 0}
    for s in scenes:
        n = int(s)
        rooms["kitchen" if n < 100 else "living" if n < 300
              else "bedroom" if n < 400 else "bathroom"] += 1
    assert all(v == 2 for v in rooms.values()), rooms


def test_scene_disjoint_task_quota_matches_global_marginal():
    paths = _synthetic_paths()
    from collections import Counter
    global_mix = Counter(p.split("/")[-3].split("-")[0] for p in paths)
    total = sum(global_mix.values())
    shard = _scene_disjoint_partition_alfworld(
        paths, 0, 10, min_samples_per_client=40, env_div=0.7, scenes_per_client=8)
    mix = Counter(p.split("/")[-3].split("-")[0] for p in shard)
    for tt, cnt in global_mix.items():
        assert abs(mix.get(tt, 0) / len(shard) - cnt / total) < 0.11, (tt, mix)


def test_scene_disjoint_holdout_and_determinism():
    paths = _synthetic_paths()
    holdout = ["1", "201", "301", "401"]
    a = _scene_disjoint_partition_alfworld(
        paths, 2, 10, min_samples_per_client=40, env_div=0.7,
        scenes_per_client=8, holdout_scenes=holdout)
    b = _scene_disjoint_partition_alfworld(
        paths, 2, 10, min_samples_per_client=40, env_div=0.7,
        scenes_per_client=8, holdout_scenes=holdout)
    assert a == b
    scenes = {p.split("/")[-3].rsplit("-", 1)[1] for p in a}
    assert not scenes & set(holdout)


def test_scene_disjoint_via_partition_dataset_dispatch():
    paths = _synthetic_paths()
    direct = _scene_disjoint_partition_alfworld(
        paths, 1, 4, min_samples_per_client=40, env_div=0.7, scenes_per_client=8)
    routed = partition_dataset(
        data=paths, strategy="scene_disjoint", client_id=1, client_num=4,
        min_samples_per_client=40, data_type="alfworld",
        env_div=0.7, scenes_per_client=8)
    assert routed == direct
    with pytest.raises(ValueError):
        partition_dataset(
            data=paths, strategy="scene_disjoint", client_id=0, client_num=4,
            min_samples_per_client=40, data_type="webshop")


@needs_data
def test_scene_disjoint_real_data_properties():
    games = _manifest_games()
    paths = [os.path.join(TRAIN_ROOT, g) for g in games]
    shard = _scene_disjoint_partition_alfworld(
        paths, 0, 20, min_samples_per_client=100, env_div=1.0, scenes_per_client=8)
    assert len(shard) == 100
    assert len({p.split("/")[-3].rsplit("-", 1)[1] for p in shard}) <= 8


# --------------------------------------------------------------------------- #
# task-text canonicalization (scene_disjoint's lexical-tau fix)               #
# --------------------------------------------------------------------------- #

from agent_system.environments.alfworld_kernel_variants import (  # noqa: E402
    canonicalize_task_text,
    task_type_from_path,
)


def _grammar_with_task(desc):
    return SYN_GRAMMAR.replace(
        '"rhs": "Your task is to: put a book in desk."',
        f'"rhs": "Your task is to: {desc}."')


def test_canonicalize_all_template_pairs():
    cases = {
        "pick_and_place_simple": ("put some book on desk", "put a book in desk"),
        "pick_clean_then_place_in_recep": (
            "clean some apple and put it in diningtable",
            "put a clean apple in diningtable"),
        "pick_heat_then_place_in_recep": (
            "heat some egg and put it in countertop", "put a hot egg in countertop"),
        "pick_cool_then_place_in_recep": (
            "cool some pot and put it in shelf", "put a cool pot in shelf"),
        "pick_two_obj_and_place": (
            "find two book and put them in desk", "put two book in desk"),
        "look_at_obj_in_light": (
            "examine the alarmclock with the desklamp",
            "look at alarmclock under the desklamp"),
    }
    for tt, (variant_desc, canon_desc) in cases.items():
        g = _grammar_with_task(variant_desc)
        out = canonicalize_task_text(g, tt)
        assert f'"Your task is to: {canon_desc}."' in out, tt
        # canonical input is a byte-pure no-op, and canonicalization is idempotent
        g2 = _grammar_with_task(canon_desc)
        assert canonicalize_task_text(g2, tt) is g2, tt
        assert canonicalize_task_text(out, tt) is out, tt


def test_canonicalize_rejects_unknown_wording():
    g = _grammar_with_task("do something entirely else")
    with pytest.raises(ValueError):
        canonicalize_task_text(g, "pick_and_place_simple")
    with pytest.raises(ValueError):
        task_type_from_path("/x/unknown_type-Book-None-Desk-1/t/game.tw-pddl")


def test_task_type_from_path():
    p = "/d/train/pick_two_obj_and_place-Book-None-Desk-313/trial_T1/game.tw-pddl"
    assert task_type_from_path(p) == "pick_two_obj_and_place"


@needs_data
def test_canonicalization_collapses_every_spec_to_one_wording():
    import re as _re
    task_re = _re.compile(r'"rhs": "Your task is to: ([^"]*)"')
    per_spec = {}
    for rel in _manifest_games():
        spec = rel.split("/")[0].rsplit("-", 1)[0]
        tt = rel.split("/")[0].split("-")[0]
        with open(os.path.join(TRAIN_ROOT, rel)) as f:
            grammar = json.load(f)["grammar"]
        out = canonicalize_task_text(grammar, tt)
        per_spec.setdefault(spec, set()).add(task_re.search(out).group(1))
    multi = {s: v for s, v in per_spec.items() if len(v) != 1}
    assert not multi, f"{len(multi)} specs kept >1 wording: {list(multi.items())[:3]}"
