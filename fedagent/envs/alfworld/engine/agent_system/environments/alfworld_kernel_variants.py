"""Per-client hidden-kernel variants for ALFWorld environment-level heterogeneity.

FedAgent's ALFWorld analog of the WebShop env-variant arms (bm25_field_subset /
bm25_reweight / lookalike / rank_wrapper): every client keeps the SAME uniform
game slice, the SAME goal text (tau invariance, the science red line), and gets a
deterministically assigned rewrite of one of the three strings inside
``game.tw-pddl``:

  - ``obs_variant``  -> rewrites ``grammar``      (rendering/encoding: how state
                        becomes text; the observation kernel O)
  - ``dyn_variant``  -> rewrites ``pddl_domain``  (dynamics: action preconditions
                        and effects; the transition kernel P)
  - ``goal_variant`` -> rewrites ``pddl_problem`` (the hidden success predicate;
                        the reward channel R -- the Lookalike-Injection analog)

The rewrite happens at episode load time inside ``make_kernel_wrapper``'s
textworld wrapper, inserted INNERMOST in ``AlfredTWEnv.init_env``'s wrapper list
(so the outer ``AlfredInfos`` keeps recording the ORIGINAL game path:
``extra.gamefile`` and the val service's seed==index mode are unaffected).
``GenericEnvironment.load`` guesses the backend from the path suffix, so the
rewritten dict is round-tripped through a per-wrapper-instance ``*.tw-pddl``
temp file rather than passed as a dict.

All rewrites are exact-substring anchor replacements with occurrence-count
asserts (fail loud). This is safe because the embedded ``pddl_domain`` is
byte-identical across all 3553 train games and ``grammar`` is byte-identical
modulo the per-game task line (measured 2026-08-22, see
docs/dev_doc/alfworld_env_heterogeneity.md). The task rhs
(``"Your task is to: ..."``) is asserted untouched by every grammar rewrite.

Per-client assignment copies the WebShop env-variant math verbatim
(``_bm25_variant_partition_webshop``): ``RandomState(base_seed + client_id)``
with ``base_seed = 42`` hardcoded; the same client keeps the same variant across
rounds, as FedAvg comparability requires.
"""
import json
import os
import re
import tempfile

import numpy as np

BASE_SEED = 42

KERNEL_VARIANT_STRATEGIES = ("obs_variant", "dyn_variant", "goal_variant")

# Pool order is load-bearing: N=2 truncation keeps [control, strongest-universal].
VARIANT_POOLS = {
    "obs_variant": ("v_default", "v_terse_goto", "v_blind_intro", "v_paraphrase"),
    "dyn_variant": ("v_default", "v_examine_gate", "v_autoclose", "v_gate_autoclose"),
    "goal_variant": ("v_default", "v_examined", "v_closed", "v_examined_closed"),
}


def resolve_pool(strategy, variant_n=0):
    """The first ``variant_n`` variants of ``strategy``'s pool (0 -> full pool)."""
    if strategy not in VARIANT_POOLS:
        raise ValueError(
            f"unknown kernel-variant strategy {strategy!r}; "
            f"supported: {sorted(VARIANT_POOLS)}"
        )
    pool = VARIANT_POOLS[strategy]
    n = int(variant_n) or len(pool)
    if not (2 <= n <= len(pool)):
        raise ValueError(
            f"variant_n for {strategy} must be in [2, {len(pool)}] (or 0 for the "
            f"pool default {len(pool)}); got {variant_n}"
        )
    return pool[:n]


def variant_for_client(strategy, client_id, variant_n=0, base_seed=BASE_SEED):
    """Deterministic per-client variant assignment (WebShop env-variant math).

    ``RandomState(base_seed + client_id)`` -> ``pool[rng.randint(N)]``: stable by
    ``client_id`` across rounds and processes, reproducible offline.
    """
    pool = resolve_pool(strategy, variant_n)
    rng = np.random.RandomState(base_seed + int(client_id))
    key = pool[rng.randint(len(pool))]
    return {"strategy": strategy, "key": key, "n": len(pool)}


def assignment_table(strategy, client_num, variant_n=0, base_seed=BASE_SEED):
    """client_id -> variant key for the whole federation (logging / analysis)."""
    return {
        c: variant_for_client(strategy, c, variant_n, base_seed)["key"]
        for c in range(int(client_num))
    }


# --------------------------------------------------------------------------- #
# anchored string surgery                                                     #
# --------------------------------------------------------------------------- #

def _replace_counted(text, old, new, expect, what):
    """text.replace with an exact occurrence-count assert (fail loud)."""
    found = text.count(old)
    if found != expect:
        raise ValueError(
            f"[alfworld_kernel_variants] anchor for {what} matched {found} times "
            f"(expected {expect}). The ALFWorld game data no longer matches the "
            f"measured 2026-08 layout; refusing to run a silently-wrong science "
            f"arm. Anchor: {old[:80]!r}"
        )
    return text.replace(old, new)


# ---- obs_variant: grammar rewrites ---------------------------------------- #
# NOTE on escaping: rhs VALUES inside the grammar's embedded JSON carry literal
# backslash-n two-char sequences (r"\n"); the JSON structure between rhs lines
# uses real newlines ("\n").

_INTRO_RHS = r'"rhs": "-= Welcome to TextWorld, ALFRED! =-\n\n#look.feedback#\n\n#task#"'
_INTRO_RHS_BLIND = r'"rhs": "-= Welcome to TextWorld, ALFRED! =-\n\n#task#"'

_GOTO_RHS = '"rhs": "You arrive at {r.name}. #examineReceptacle.feedback#"'
_GOTO_RHS_TERSE = '"rhs": "You arrive at {r.name}."'

# The pickup rhs string occurs 3x in the grammar; scope the paraphrase anchor to
# the PickupObject.feedback rule header (real newlines + the file's indentation).
_PICKUP_RULE = (
    '"PickupObject.feedback": [\n'
    '            {\n'
    '                "rhs": "You pick up the {o.name} from the {r.name}."'
)
_PICKUP_RULE_PARA = (
    '"PickupObject.feedback": [\n'
    '            {\n'
    '                "rhs": "You take the {o.name} from the {r.name}."'
)

_PARAPHRASE_RULES = (
    (_GOTO_RHS, '"rhs": "You are now at {r.name}. #examineReceptacle.feedback#"', 1),
    ('"rhs": "You open the {r.name}. #examineReceptacle.feedback#"',
     '"rhs": "The {r.name} swings open. #examineReceptacle.feedback#"', 1),
    ('"rhs": "You close the {r.name}."',
     '"rhs": "The {r.name} is now closed."', 1),
    (_PICKUP_RULE, _PICKUP_RULE_PARA, 1),
    ('"rhs": "You move the {o.name} to the {r.name}."',
     '"rhs": "You place the {o.name} at the {r.name}."', 1),
)

_TASK_RHS_RE = re.compile(r'"rhs": "Your task is to: [^"]*"')


def rewrite_grammar(grammar, key):
    if key == "v_default":
        return grammar
    task_before = _TASK_RHS_RE.findall(grammar)
    if len(task_before) != 1:
        raise ValueError(
            f"[alfworld_kernel_variants] expected exactly one task rhs in the "
            f"grammar, found {len(task_before)}"
        )
    if key == "v_terse_goto":
        out = _replace_counted(grammar, _GOTO_RHS, _GOTO_RHS_TERSE, 1, "v_terse_goto")
    elif key == "v_blind_intro":
        out = _replace_counted(grammar, _INTRO_RHS, _INTRO_RHS_BLIND, 1, "v_blind_intro")
    elif key == "v_paraphrase":
        out = grammar
        for old, new, n in _PARAPHRASE_RULES:
            out = _replace_counted(out, old, new, n, "v_paraphrase")
    else:
        raise ValueError(f"unknown obs_variant key: {key}")
    # tau red line: the goal text the agent reads is byte-identical.
    if _TASK_RHS_RE.findall(out) != task_before:
        raise AssertionError(
            "[alfworld_kernel_variants] grammar rewrite touched the task rhs -- "
            "tau invariance violated"
        )
    return out


# ---- dyn_variant: pddl_domain rewrites ------------------------------------ #
# Anchors measured unique (count==1) in the byte-identical embedded domain.

_PICKUP_PRE = "(pickupable ?o)\n            (atLocation ?a ?l)"
_PICKUP_PRE_GATED = (
    "(pickupable ?o)\n"
    "            (checked ?r)\n"
    "            (atLocation ?a ?l)"
)

_PUT_EFFECT = (
    "(not (holds ?a ?o))\n"
    "                (not (holdsAny ?a))\n"
    "                (increase (total-cost) 1)"
)
_PUT_EFFECT_AUTOCLOSE = (
    "(not (holds ?a ?o))\n"
    "                (not (holdsAny ?a))\n"
    "                (not (opened ?r))\n"
    "                (increase (total-cost) 1)"
)


def rewrite_domain(domain, key):
    if key == "v_default":
        return domain
    out = domain
    if key in ("v_examine_gate", "v_gate_autoclose"):
        # PickupObject additionally requires the source receptacle to have been
        # checked (examineReceptacle / OpenObject both set ``(checked ?r)``):
        # the learned "goto -> take" macro breaks until the agent inserts an
        # examine/open step.
        out = _replace_counted(out, _PICKUP_PRE, _PICKUP_PRE_GATED, 1, key)
    if key in ("v_autoclose", "v_gate_autoclose"):
        # PutObject closes the receptacle it just filled. For non-openable
        # receptacles ``opened`` is already false, so the extra literal is a
        # no-op there (no conditional effects needed); for openable ones the
        # agent must re-open before the next take (pick_two) and the feedback
        # string does NOT announce the closure -- sensed only via successor
        # state (admissible commands / examine).
        out = _replace_counted(out, _PUT_EFFECT, _PUT_EFFECT_AUTOCLOSE, 1, key)
    if out is domain:
        raise ValueError(f"unknown dyn_variant key: {key}")
    return out


# ---- goal_variant: pddl_problem (:goal) rewrites -------------------------- #

_RECEPTACLE_TYPE_RE = re.compile(r"\(receptacleType (\?\w+) (\w+)\)")
_HOLDS_RE = re.compile(r"\(holds (\?\w+) (\?\w+)\)")


def _extract_goal(problem):
    """(start, end) of the single balanced ``(:goal ...)`` block."""
    if problem.count("(:goal") != 1:
        raise ValueError(
            f"[alfworld_kernel_variants] expected exactly one (:goal block, "
            f"found {problem.count('(:goal')}"
        )
    i = problem.index("(:goal")
    depth = 0
    for j in range(i, len(problem)):
        ch = problem[j]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return i, j + 1
    raise ValueError("[alfworld_kernel_variants] unbalanced (:goal block")


def rewrite_problem(problem, key):
    """Inject hidden success conjuncts into the (:goal ...) block.

    The instruction text (grammar task rhs) is untouched by construction --
    only ``PddlState.check_goal()`` changes. Five of the six task types carry a
    ``(receptacleType ?r XType)`` anchor for the goal receptacle (pick_two has
    it twice for the SAME variable; injecting at every occurrence adds a
    duplicate conjunct, which is semantically idempotent). look_at_obj_in_light
    has no goal receptacle; its examined-variant rides the held object's
    ``(holds ?a ?o)`` anchor instead, and its closed-variant is identity.
    """
    if key == "v_default":
        return problem
    if key not in ("v_examined", "v_closed", "v_examined_closed"):
        raise ValueError(f"unknown goal_variant key: {key}")
    i, j = _extract_goal(problem)
    goal = problem[i:j]
    recs = _RECEPTACLE_TYPE_RE.findall(goal)
    if recs:
        rvars = {v for v, _ in recs}
        if len(rvars) != 1:
            raise ValueError(
                f"[alfworld_kernel_variants] goal binds multiple receptacle "
                f"variables {sorted(rvars)}; unsupported goal shape"
            )
        rv = rvars.pop()
        conj = {
            "v_examined": f"(checked {rv})",
            "v_closed": f"(not (opened {rv}))",
            "v_examined_closed": f"(checked {rv}) (not (opened {rv}))",
        }[key]
        goal2 = _RECEPTACLE_TYPE_RE.sub(lambda m: m.group(0) + " " + conj, goal)
    else:
        holds = _HOLDS_RE.findall(goal)
        if len(holds) != 1:
            raise ValueError(
                "[alfworld_kernel_variants] goal has neither a receptacleType "
                "anchor nor a single holds anchor; unsupported goal shape"
            )
        if key == "v_closed":
            return problem  # no goal receptacle to close -> identity by design
        av, ov = holds[0]
        goal2 = _replace_counted(
            goal, f"(holds {av} {ov})", f"(holds {av} {ov}) (checked {ov})", 1, key
        )
    return problem[:i] + goal2 + problem[j:]


# ---- dispatch -------------------------------------------------------------- #

def rewrite_game_data(data, variant):
    """Return a shallow-copied game dict with the variant's rewrite applied."""
    strategy, key = variant["strategy"], variant["key"]
    out = dict(data)
    if strategy == "obs_variant":
        out["grammar"] = rewrite_grammar(data["grammar"], key)
    elif strategy == "dyn_variant":
        out["pddl_domain"] = rewrite_domain(data["pddl_domain"], key)
    elif strategy == "goal_variant":
        out["pddl_problem"] = rewrite_problem(data["pddl_problem"], key)
    else:
        raise ValueError(f"unknown kernel-variant strategy: {strategy}")
    return out


def _make_rewrite_wrapper(rewrite_fn, prefix):
    """A textworld wrapper instance applying ``rewrite_fn(data, gamefile)`` at
    every episode load, round-tripped through a private ``*.tw-pddl`` temp file
    (``GenericEnvironment.load`` guesses the backend from the path suffix, so a
    dict cannot be passed down). ``rewrite_fn`` returning None means "load the
    original file untouched" (byte-pure control path).

    Insert FIRST in ``AlfredTWEnv.init_env``'s wrapper list (wrapper lists build
    inner->outer, so first == innermost): the outer ``AlfredInfos`` then records
    the ORIGINAL path while the backend loads the rewritten copy. Like
    ``AlfredDemangler``, one instance wraps one env -- valid because the service
    pins ``batch_size=1`` (SyncBatchEnv, same process).
    """
    import textworld.core

    class _AlfredRewriteWrapper(textworld.core.Wrapper):
        def __init__(self, env=None):
            super().__init__(env)
            fd, self._tmp_path = tempfile.mkstemp(
                suffix=".tw-pddl", prefix=prefix
            )
            os.close(fd)

        def load(self, gamefile):
            with open(gamefile) as f:
                data = json.load(f)
            rewritten = rewrite_fn(data, gamefile)
            if rewritten is None:
                return self._wrapped_env.load(gamefile)
            with open(self._tmp_path, "w") as f:
                json.dump(rewritten, f)
            return self._wrapped_env.load(self._tmp_path)

        def __del__(self):
            try:
                os.remove(self._tmp_path)
            except (OSError, AttributeError):
                pass

    return _AlfredRewriteWrapper()


def make_kernel_wrapper(variant):
    """Episode-load wrapper applying a kernel ``variant`` (obs/dyn/goal arm)."""
    variant = dict(variant)

    def _rewrite(data, gamefile):
        if variant["key"] == "v_default":
            return None                        # byte-pure control arm
        return rewrite_game_data(data, variant)

    return _make_rewrite_wrapper(_rewrite, "fedagent_kernel_")


# ---- task-text canonicalization (scene_disjoint's lexical-tau fix) --------- #
# ALFWorld's generator freezes ONE of TWO wording templates per goal type into
# each game's grammar (random.choice at generation time, scene-independent
# noise): 438/550 train specs carry both wordings (dev_doc/
# alfworld_query_env_decoupling.md section 3). Under scene_disjoint different
# clients hold different scenes, so the same spec can read "put two book in
# desk." on one client and "find two book and put them in desk." on another --
# a lexical-level tau leak in an arm whose contract is tau-invariant.
# Canonicalizing every task rhs to templates[0] (the goal_library.py order)
# closes it BY CONSTRUCTION. Patterns are selected by the task type parsed from
# the gamefile path, so no cross-type regex ambiguity is possible.

_TASK_CANON = {
    # task_type: (canonical regex, variant regex, variant -> canonical rewrite)
    "pick_and_place_simple": (
        re.compile(r"^put a (.+) in (.+)$"),
        re.compile(r"^put some (.+) on (.+)$"),
        r"put a \1 in \2",
    ),
    "pick_clean_then_place_in_recep": (
        re.compile(r"^put a clean (.+) in (.+)$"),
        re.compile(r"^clean some (.+) and put it in (.+)$"),
        r"put a clean \1 in \2",
    ),
    "pick_heat_then_place_in_recep": (
        re.compile(r"^put a hot (.+) in (.+)$"),
        re.compile(r"^heat some (.+) and put it in (.+)$"),
        r"put a hot \1 in \2",
    ),
    "pick_cool_then_place_in_recep": (
        re.compile(r"^put a cool (.+) in (.+)$"),
        re.compile(r"^cool some (.+) and put it in (.+)$"),
        r"put a cool \1 in \2",
    ),
    "pick_two_obj_and_place": (
        re.compile(r"^put two (.+) in (.+)$"),
        re.compile(r"^find two (.+) and put them in (.+)$"),
        r"put two \1 in \2",
    ),
    "look_at_obj_in_light": (
        re.compile(r"^look at (.+) under the (.+)$"),
        re.compile(r"^examine the (.+) with the (.+)$"),
        r"look at \1 under the \2",
    ),
}

_TASK_RHS_FULL_RE = re.compile(r'("rhs": "Your task is to: )([^"]*?)\.?(")')


def task_type_from_path(gamefile):
    """``.../<task_type>-<obj>-<mrecep>-<recep>-<scene>/<trial>/game.tw-pddl``"""
    parts = str(gamefile).rstrip("/").split("/")
    if len(parts) < 3:
        raise ValueError(f"cannot parse task type from path: {gamefile!r}")
    task_type = parts[-3].split("-")[0]
    if task_type not in _TASK_CANON:
        raise ValueError(
            f"unknown task type {task_type!r} in path {gamefile!r}; "
            f"known: {sorted(_TASK_CANON)}"
        )
    return task_type


def canonicalize_task_text(grammar, task_type):
    """Rewrite the grammar's task rhs to the canonical (templates[0]) wording.

    Returns the grammar unchanged when already canonical; raises on a wording
    that matches neither template of ``task_type`` (fail loud: an unknown
    wording means the upstream data no longer matches the measured layout).
    """
    canon_re, variant_re, repl = _TASK_CANON[task_type]
    m = _TASK_RHS_FULL_RE.search(grammar)
    if not m or len(_TASK_RHS_FULL_RE.findall(grammar)) != 1:
        raise ValueError(
            "[alfworld_kernel_variants] expected exactly one task rhs in the "
            "grammar for canonicalization"
        )
    desc = m.group(2)
    if canon_re.match(desc):
        return grammar
    vm = variant_re.match(desc)
    if not vm:
        raise ValueError(
            f"[alfworld_kernel_variants] task wording {desc!r} matches neither "
            f"template of {task_type}; refusing to guess"
        )
    canon_desc = variant_re.sub(repl, desc)
    return grammar[: m.start()] + m.group(1) + canon_desc + "." + m.group(3) \
        + grammar[m.end():]


def make_task_normalizer_wrapper():
    """Episode-load wrapper canonicalizing the task wording (scene_disjoint)."""

    def _rewrite(data, gamefile):
        task_type = task_type_from_path(gamefile)
        grammar = canonicalize_task_text(data["grammar"], task_type)
        if grammar is data["grammar"]:
            return None                        # already canonical: byte-pure
        out = dict(data)
        out["grammar"] = grammar
        return out

    return _make_rewrite_wrapper(_rewrite, "fedagent_taskcanon_")
