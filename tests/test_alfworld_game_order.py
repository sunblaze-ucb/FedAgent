"""ALFWorld's game list must be a function of the dataset CONTENT, not of the filesystem.

``AlfredTWEnv.collect_game_files`` builds the split's game list with ``os.walk``, which returns
directory entries in filesystem order — different on every machine and every copy of the data.
Everything after that point is POSITIONAL: the seeded shuffle, the client index slices
(``slice_games_for_client`` → ``partition_dataset``), the ``num_train_games``/``start_idx`` caps,
and ``games[seed]`` selection under ``ALFWORLD_SEED_IS_INDEX``. So without a canonical order the
same config and the same seed produce a different partition and a different validation set on
another node — measured on the real valid_seen split (2026-07-26): two walk orders shared only
**29 of 64** val games; sorting first makes it 64/64.

The fix is one statement (`self.game_files.sort()`) placed after the walk *and* after the
manifest-cache load, before the seeded shuffle. Exercising it for real needs the alfworld +
textworld stack and an ~8810-game data walk — not something a unit test should do — so this
locks its STRUCTURE instead, via ast (no import, no deps): the sort exists in
``collect_game_files`` and precedes every shuffle there. That is exactly the failure mode worth
guarding: someone re-syncing the vendored engine from upstream and dropping the line.

Also pinned: the capped-eval subsample is drawn from a SEEDED generator. The upstream
``random.sample`` used the global RNG, whose state here depends on whatever ran earlier in the
process, so a capped eval set was not reproducible even on one machine.
"""
import ast
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

ENGINE = os.path.join(
    os.path.dirname(__file__), "..", "fedagent", "envs", "alfworld", "engine", "agent_system",
    "environments", "env_package", "alfworld", "alfworld", "agents", "environment",
    "alfred_tw_env.py")


def _collect_game_files_body():
    with open(ENGINE) as f:
        tree = ast.parse(f.read())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "collect_game_files":
            return node
    raise AssertionError("collect_game_files not found in the vendored alfred_tw_env.py")


def _line_of_first(fn, predicate):
    return next((n.lineno for n in ast.walk(fn) if isinstance(n, ast.Call) and predicate(n)), None)


def test_game_files_are_sorted_before_the_seeded_shuffle():
    fn = _collect_game_files_body()

    def is_sort(call):
        return (isinstance(call.func, ast.Attribute) and call.func.attr == "sort"
                and isinstance(call.func.value, ast.Attribute)
                and call.func.value.attr == "game_files")

    def is_shuffle(call):
        return isinstance(call.func, ast.Attribute) and call.func.attr == "shuffle"

    sort_line = _line_of_first(fn, is_sort)
    shuffle_line = _line_of_first(fn, is_shuffle)
    assert sort_line is not None, "self.game_files.sort() is gone -> game order is filesystem-dependent again"
    assert shuffle_line is not None, "the seeded shuffle is gone -> partitions changed meaning"
    assert sort_line < shuffle_line, "the canonical sort must run BEFORE the seeded shuffle"


def test_the_capped_eval_subsample_is_seeded():
    fn = _collect_game_files_body()
    calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call)]
    sampled = [c for c in calls
               if isinstance(c.func, ast.Attribute) and c.func.attr == "sample"]
    assert sampled, "the capped-eval subsample is gone -- update this guard with it"
    for call in sampled:
        # random.Random(42).sample(...) -> the receiver is itself a Call; random.sample(...) is not
        assert isinstance(call.func.value, ast.Call), \
            "random.sample() draws from the global RNG -> the capped eval set is not reproducible"
