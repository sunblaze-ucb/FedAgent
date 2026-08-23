#!/usr/bin/env python
"""E2 stage 2 -- estimate the occupancy-weighted TV divergence (delta_eff) of every
environment-level heterogeneity variant by simulator replay.

Reads the (s,a) trace produced by `rollout_reference_occupancy.py`, replays each (s,a) in
EVERY client's environment backend, and reports the pairwise disagreement rate, which is the
plug-in estimate of the paper's policy-induced divergence

    delta_eff(pi; i, j) := E_{(s,a) ~ d^pi}[ D_TV( P_i(.|s,a), P_j(.|s,a) ) ]     (Appendix L)

Why the disagreement rate IS the divergence (and where it is only a bound). What we compute is
the independent-coupling statistic

    rho(i,j) = E_{(s,a)}[ P( o'_i != o'_j ) ]   with o'_i ~ P_i, o'_j ~ P_j drawn independently
             = E_{(s,a)}[ 1 - sum_x P_i(x) P_j(x) ]  >=  delta_eff(pi; i, j),

with EQUALITY at every (s,a) where both kernels are deterministic -- there the pointwise TV is
0/1 valued and rho is exact, not an approximation. WebShop's kernel is deterministic given (s,a)
for Catalog Split, Field-Subset Index, BM25 Reweighting and Lookalike Injection. It is NOT for
two of the four Rank Wrapper variants: `ShuffledTopKSearcher` and `PartialRandomSearcher` hold a
per-client-seeded `random.Random` consumed once per search call, so the page depends on how many
searches that backend has served rather than on (s,a) (engine.py:323-350). For those, rho
over-states the divergence -- most sharply for two clients that drew the SAME stochastic variant
with different seeds, where the induced distributions are identical (true TV = 0) while rho ~ 1.
For Rank Wrapper we therefore ALSO report the exact TV, computed in closed form from the known
variant distributions (point mass / uniform over ordered 10-subsets / 50-50 mixture).

Replay semantics
  * state forcing: WebShop's state is the session dict (`server.user_sessions[sid]`: goal,
    keywords, page, asin, asins, options, actions) plus the browser's current URL and page
    source. We copy the reference trajectory's state into each client backend and step it, so
    every backend is evaluated at the SAME s -- exactly the conditional the definition asks for.
    Replaying the action prefix instead would let the states diverge after the first difference.
  * unavailable products: a click whose ASIN is absent from that client's catalog (routine under
    Catalog Split) renders no page. It maps to a single sentinel observation, so such a client
    differs from every client that CAN render the product -- the kernels genuinely differ there --
    while two clients that both lack it agree. Per-backend rates are reported.
  * normalization: whitespace only (\\xa0 -> space, runs collapsed). Item ORDER is never touched:
    Rank Wrapper's whole effect is ordering and any set-wise comparison would zero it out.
  * retrieval detection: `search[...]` is not the only retrieval-invoking action -- 'next page' /
    'prev page' on the results page and 'prev page' from an item page all re-run the query
    (SimServer.receive). Every backend's searcher is wrapped in a probe that counts calls, so the
    search/click split is measured rather than guessed from the action string.

Backends mutate one warmed env rather than building one env per client, which is exact because
that is precisely how the service applies these variants: Catalog Split only swaps
`server.all_products` / `server.product_item_dict` (service/server.py `_apply_deferred_catalog`)
and the BM25 / rank variants only swap `server.search_engine`. Lookalike Injection alters the
product corpus itself, so it gets one real env per variant.

Runs in `verl-agent-webshop` (gym 0.24 / pyserini / the WebShop engine).

Usage:
    python tools/env_heterogeneity/measure_delta_occ.py \
        --trace e2_delta_occ/reference_rollout_val500.json \
        --out   e2_delta_occ/delta_occ_val64.json --episodes 64
"""
import argparse
import copy
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
_WEBSHOP = REPO_ROOT / "fedagent" / "envs" / "webshop" / "engine" / "webshop"
if str(_WEBSHOP) not in sys.path:
    sys.path.append(str(_WEBSHOP))

TOTAL_CLIENTS = 100          # every env_heterogeneity paper config: total_clients: 100
MIN_GOALS_PER_CLIENT = 100
VAL_SIZE = 500
BASE_SEED = 42
KEEP_RATIO = 0.7
UNAVAILABLE = "<<PRODUCT-UNAVAILABLE-IN-THIS-CLIENT-CATALOG>>"
PRODUCT_WINDOW = 10          # engine.PRODUCT_WINDOW: items rendered per results page

_WS_RE = re.compile(r"\s+")


def normalize(obs):
    """Whitespace-only normalization. MUST NOT reorder or set-ify."""
    if obs is UNAVAILABLE:
        return UNAVAILABLE
    return _WS_RE.sub(" ", (obs or "").replace("\xa0", " ")).strip()


class SearchProbe:
    """Counts retrieval calls and remembers the last hit list (for the analytic TV)."""

    def __init__(self, base):
        self.base = base
        self.calls = 0
        self.last_hits = None

    def search(self, query, k=50):
        self.calls += 1
        hits = self.base.search(query, k=k)
        self.last_hits = hits
        return hits

    def doc(self, docid):
        return self.base.doc(docid)


# --------------------------------------------------------------------------- #
# rendered-page recorder
# --------------------------------------------------------------------------- #
# The 0/1 pointwise TV saturates: once two kernels disagree at all, delta_eff equals the share of
# retrieval-invoking (s,a), so the four collapsing variants can land on the same number. To keep a
# MECHANISM signal (how far apart the pages are, not just whether they differ) we also record each
# backend's rendered results page. This is a diagnostic, NOT a divergence: TV is what the theory
# uses. Hooking `get_product_per_page` in the env module is the exact list SimServer renders.
_LAST_PAGE = {"asins": None, "total": None}


def _install_page_recorder():
    import web_agent_site.envs.web_agent_text_env as wate
    if getattr(wate, "_e2_page_recorder", False):
        return
    orig = wate.get_product_per_page

    def hooked(top_n_products, page):
        out = orig(top_n_products, page)
        _LAST_PAGE["asins"] = tuple(p["asin"] for p in out)
        _LAST_PAGE["total"] = len(top_n_products)
        return out

    wate.get_product_per_page = hooked
    wate._e2_page_recorder = True


def _jaccard(a, b):
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def make_env(seed=0, **kwargs):
    import gym
    from web_agent_site.envs import WebAgentTextEnv  # noqa: F401  (registers the gym id)
    kw = dict(observation_mode="text", num_products=None, seed=seed)
    kw.update(kwargs)
    env = gym.make("WebAgentTextEnv-v0", **kw)
    srv = env.unwrapped.server
    srv.search_engine = SearchProbe(srv.search_engine)
    return env


# --------------------------------------------------------------------------- #
# state forcing
# --------------------------------------------------------------------------- #
def snapshot(env):
    e = env.unwrapped
    sid = e.session
    return {
        "sid": sid,
        "session": copy.deepcopy(e.server.user_sessions[sid]),
        "url": e.browser.current_url,
        "html": e.browser.page_source,
        "instruction_text": e.instruction_text,
    }


def force_and_step(env, snap, action):
    """Put `env` into the reference state and apply `action`; return the rendered next
    observation, or the UNAVAILABLE sentinel when this client's catalog cannot render it."""
    e = env.unwrapped
    sid = snap["sid"]
    e.session = sid
    e.server.user_sessions[sid] = copy.deepcopy(snap["session"])
    e.browser.session_id = sid
    e.browser.current_url = snap["url"]
    e.browser.page_source = snap["html"]
    e.instruction_text = snap["instruction_text"]
    e.prev_obs, e.prev_actions = [], []
    try:
        obs, _r, _d, _i = e.step(action)
    except KeyError:
        obs = UNAVAILABLE
    # step() calls reset() on a terminal action, which mints a fresh random session; drop
    # everything so repeated forcing never accumulates state across (s,a) pairs.
    e.server.user_sessions.clear()
    return obs


# --------------------------------------------------------------------------- #
# arms
# --------------------------------------------------------------------------- #
class Arm:
    """One heterogeneity arm: client -> backend map, and how to configure a host env."""

    def __init__(self, name, client_backend, backends, svc_cfg=None):
        self.name = name
        self.client_backend = client_backend    # len TOTAL_CLIENTS -> backend index
        self.backends = backends                # list of (label, apply_fn, host_env_key)
        self.svc_cfg = svc_cfg or {}            # service env for the FAITHFUL rollout of this arm
        # representative client for each backend -- the client_id whose kernel that backend is,
        # which is what a faithful rollout has to be launched as
        self.backend_client = {}
        for cid, bi in enumerate(client_backend):
            self.backend_client.setdefault(bi, cid)

    @property
    def n_backends(self):
        return len(self.backends)


def _pristine(env):
    srv = env.unwrapped.server
    return (srv.all_products, srv.product_item_dict, srv.search_engine)


def _restore(env, snap3):
    srv = env.unwrapped.server
    srv.all_products, srv.product_item_dict, srv.search_engine = snap3


def build_uniform_arm():
    """Null check: two identical unperturbed backends must give delta = 0 exactly."""
    noop = lambda env: None  # noqa: E731
    return Arm("uniform_control", [0, 1],
               [("uniform_a", noop, "work"), ("uniform_b", noop, "work")])


def build_catalog_split_arm(env_goals, full_products, env_div, keep_ratio=KEEP_RATIO):
    """Variant 1. Each of the 100 clients gets its own catalog, derived exactly as the service
    does at runtime: the uniform goal shard -> its target ASINs -> `catalog_from_target_asins`."""
    from fedagent.hetero.webshop_uniform import uniform_for_client
    from fedagent.hetero.webshop_catalog_split import catalog_from_target_asins

    backends = []
    for cid in range(TOTAL_CLIENTS):
        idxs = uniform_for_client(cid, TOTAL_CLIENTS, MIN_GOALS_PER_CLIENT, env_goals, VAL_SIZE)
        targets = {env_goals[i]["asin"] for i in idxs}
        catalog = set(catalog_from_target_asins(
            targets, client_id=cid, env_div=env_div, keep_ratio=keep_ratio, holdout_file=None))
        prods = [p for p in full_products if p["asin"] in catalog]      # precomputed once
        item_dict = {p["asin"]: p for p in prods}

        def apply(env, prods=prods, item_dict=item_dict):
            srv = env.unwrapped.server
            srv.all_products = prods
            srv.product_item_dict = item_dict

        backends.append((f"client{cid}(|cat|={len(prods)})", apply, "work"))
    return Arm(f"catalog_split_div{env_div}_keep{keep_ratio}", list(range(TOTAL_CLIENTS)), backends,
           svc_cfg={"partition_strategy": "catalog_split", "client_num": TOTAL_CLIENTS,
                    "env_div": env_div, "keep_ratio": keep_ratio})


def build_bm25_arm(kind, N, full_products):
    """kind: 'bm25_field_subset' (Variant 2) | 'bm25_reweight' (Variant 3)."""
    from fedagent.hetero.webshop_env_variants import bm25_variant_for_client
    from web_agent_site.engine.engine import InMemoryBM25Searcher

    pool = "fields_only" if kind == "bm25_field_subset" else None
    keys, cfgs, client_backend = [], [], []
    for cid in range(TOTAL_CLIENTS):
        cfg = bm25_variant_for_client(
            cid, TOTAL_CLIENTS, N=N, variant_pool=pool)["bm25_in_memory_config"]
        key = (tuple(cfg["fields"]), cfg["k1"], cfg["b"])
        if key not in keys:
            keys.append(key)
            cfgs.append(cfg)
        client_backend.append(keys.index(key))

    backends = []
    for cfg in cfgs:
        probe = SearchProbe(InMemoryBM25Searcher(products=full_products,
                                                 fields=list(cfg["fields"]),
                                                 k1=cfg["k1"], b=cfg["b"]))

        def apply(env, probe=probe):
            env.unwrapped.server.search_engine = probe

        backends.append((cfg["_variant_name"], apply, "work"))
    return Arm(f"{kind}_N{N}", client_backend, backends,
               svc_cfg={"partition_strategy": kind, "client_num": TOTAL_CLIENTS, "variant_n": N})


def build_rank_wrapper_arm(N, full_products):
    """Variant 5. Deterministic types collapse to one backend each; the two STOCHASTIC types
    keep a per-client backend because their RNG seed is base_seed + client_id."""
    from fedagent.hetero.webshop_env_variants import rank_wrapper_for_client
    from web_agent_site.engine.engine import InMemoryBM25Searcher, _build_search_variant

    base = InMemoryBM25Searcher(
        products=full_products,
        fields=["name", "Title", "description", "features", "BulletPoints"], k1=1.2, b=0.75)
    stochastic = {"bm25_shuffle", "bm25_partial"}

    backends, client_backend, index, types = [], [], {}, []
    for cid in range(TOTAL_CLIENTS):
        cfg = rank_wrapper_for_client(cid, TOTAL_CLIENTS, N=N)["search_engine_variant"]
        vtype = cfg["type"]
        key = (vtype, cfg.get("seed")) if vtype in stochastic else (vtype, None)
        if key not in index:
            index[key] = len(backends)
            probe = SearchProbe(_build_search_variant(base, cfg))

            def apply(env, probe=probe):
                env.unwrapped.server.search_engine = probe

            backends.append((vtype if key[1] is None else f"{vtype}#seed{key[1]}", apply, "work"))
            types.append(vtype)
        client_backend.append(index[key])

    arm = Arm(f"rank_wrapper_N{N}", client_backend, backends,
              svc_cfg={"partition_strategy": "rank_wrapper", "client_num": TOTAL_CLIENTS,
                       "variant_n": N})
    arm.rw_types = types
    arm.rw_base = base
    return arm


def _lookalike_kwargs(host_key):
    """gym.make kwargs for a `lookalike:<variant index>` host env.

    DEEP-COPY, for the same reason service/server.py deep-copies ENV_VARIANT_KWARGS:
    `load_products` mutates each product dict in place (pricing str -> parsed list), and
    `_load_lookalikes` memoizes the parsed JSON. Handing the same list to a second env would
    re-process already-converted dicts -> `'list' object has no attribute 'split'`.
    """
    from fedagent.hetero.webshop_env_variants import LOOKALIKE_VARIANTS_DEFAULT, _load_lookalikes
    vi = int(host_key.split(":", 1)[1])
    fp = LOOKALIKE_VARIANTS_DEFAULT[vi]["lookalike_file"]
    if not os.path.isabs(fp):
        fp = str(REPO_ROOT / fp)
    return {"extra_products": copy.deepcopy(_load_lookalikes(fp))}


def lookalike_variant_of(client_id, N):
    """The shipped per-client assignment (`_lookalike_injection_partition_webshop`)."""
    return int(np.random.RandomState(BASE_SEED + client_id).randint(N))


def build_lookalike_arm(N, envs):
    """Variant 4. The corpus itself changes, so each variant is a separately-built env."""
    used, client_backend = [], []
    for cid in range(TOTAL_CLIENTS):
        vi = lookalike_variant_of(cid, N)
        if vi not in used:
            used.append(vi)
        client_backend.append(used.index(vi))
    from fedagent.hetero.webshop_env_variants import LOOKALIKE_VARIANTS_DEFAULT
    noop = lambda env: None  # noqa: E731
    backends = [(LOOKALIKE_VARIANTS_DEFAULT[vi]["name"], noop, f"lookalike:{vi}") for vi in used]
    for vi in used:
        assert f"lookalike:{vi}" in envs, f"lookalike env {vi} not built"
    return Arm(f"lookalike_N{N}", client_backend, backends,
               svc_cfg={"partition_strategy": "lookalike", "client_num": TOTAL_CLIENTS,
                        "variant_n": N})


# --------------------------------------------------------------------------- #
# analytic Rank-Wrapper TV
# --------------------------------------------------------------------------- #
def _n_orderings(n, w):
    out = 1.0
    for t in range(min(w, n)):
        out *= (n - t)
    return max(out, 1.0)


def rw_type_tv(t_i, t_j, n_hits, invert_equals_default, pool_size=1000, window=PRODUCT_WINDOW):
    """Exact D_TV(P_i, P_j) at one retrieval-invoking (s,a) for two Rank Wrapper types.

    Distributions over the rendered results page (the first `window` items returned):
      bm25_default : point mass on the top-`window` of the K-candidate list
      bm25_invert  : point mass on the reversed list's first `window`
      bm25_shuffle : uniform over ordered `window`-subsets of the K-candidate list
      bm25_partial : 0.5 * point mass(top-`window`) + 0.5 * uniform over ordered
                     `window`-subsets of the FULL product pool
    Two clients of the same stochastic type differ only in RNG seed, so their DISTRIBUTIONS
    coincide and the TV is 0 -- the case the empirical disagreement rate cannot see.
    """
    if t_i == t_j:
        return 0.0
    p_shuf = 1.0 / _n_orderings(n_hits, window)
    p_pool = 1.0 / _n_orderings(pool_size, window)
    pair = {t_i, t_j}
    if pair == {"bm25_default", "bm25_invert"}:
        return 0.0 if invert_equals_default else 1.0
    if pair == {"bm25_default", "bm25_shuffle"} or pair == {"bm25_invert", "bm25_shuffle"}:
        return 1.0 - p_shuf
    if pair == {"bm25_default", "bm25_partial"}:
        return 1.0 - (0.5 + 0.5 * p_pool)
    if pair == {"bm25_invert", "bm25_partial"}:
        return 1.0 - ((0.5 if invert_equals_default else 0.0) + 0.5 * p_pool)
    if pair == {"bm25_shuffle", "bm25_partial"}:
        return 1.0 - (0.5 * p_shuf + 0.5 * _n_orderings(n_hits, window) * p_pool)
    raise ValueError(f"unhandled Rank Wrapper pair {t_i}/{t_j}")


# --------------------------------------------------------------------------- #
# measurement
# --------------------------------------------------------------------------- #
def _render_all(arm, envs, pristine, snap, action):
    """Render P_i(.|s,a) once for every backend of the arm. Returns (ids, unavailable, n_hits)."""
    obs_ids, unavail, seen = [], [], {}
    pages, page_ids, page_seen = [], [], {}
    n_hits = None
    for _label, apply_fn, host in arm.backends:
        env = envs[host]
        _restore(env, pristine[host])
        apply_fn(env)
        se = env.unwrapped.server.search_engine
        before = se.calls
        _LAST_PAGE["asins"] = None
        obs = force_and_step(env, snap, action)
        if n_hits is None and se.calls > before and se.last_hits is not None:
            n_hits = len(se.last_hits)
        page = _LAST_PAGE["asins"]
        if page not in page_seen:
            page_seen[page] = len(pages)
            pages.append(page)
        page_ids.append(page_seen[page])
        key = normalize(obs)
        unavail.append(key == UNAVAILABLE)
        if key not in seen:
            seen[key] = len(seen)
        obs_ids.append(seen[key])
    return (np.asarray(obs_ids), np.asarray(unavail), n_hits,
            pages, np.asarray(page_ids))


def measure(arm, envs, ref_env, episodes, verbose_every=25):
    nb = arm.n_backends
    diff = np.zeros((nb, nb), dtype=np.int64)
    diff_search = np.zeros((nb, nb), dtype=np.int64)
    diff_click = np.zeros((nb, nb), dtype=np.int64)
    sup_hit = np.zeros((nb, nb), dtype=bool)
    n_unavailable = np.zeros(nb, dtype=np.int64)
    rw_tv = np.zeros((nb, nb), dtype=np.float64) if hasattr(arm, "rw_types") else None
    jaccard_sum = np.zeros((nb, nb), dtype=np.float64)
    n_jaccard = 0
    n_sa = n_search = n_anydiff = 0
    fidelity_ok = fidelity_tot = 0

    hosts = sorted({b[2] for b in arm.backends})
    pristine = {h: _pristine(envs[h]) for h in hosts}

    t0 = time.time()
    for ep_i, ep in enumerate(episodes):
        ref_env.unwrapped.reset(session=ep["goal_index"])
        for turn in ep["turns"]:
            snap = snapshot(ref_env)
            action = turn["action"]

            ids, unavail, n_hits, pages, page_ids = _render_all(
                arm, envs, pristine, snap, action)
            d = ids[:, None] != ids[None, :]
            diff += d
            sup_hit |= d
            n_unavailable += unavail.astype(np.int64)
            if n_hits is not None and all(p is not None for p in pages):
                nd = len(pages)
                jm = np.ones((nd, nd), dtype=np.float64)
                for a in range(nd):
                    for b in range(a + 1, nd):
                        jm[a, b] = jm[b, a] = _jaccard(pages[a], pages[b])
                jaccard_sum += jm[np.ix_(page_ids, page_ids)]
                n_jaccard += 1
            is_search = n_hits is not None
            if is_search:
                diff_search += d
                n_search += 1
            else:
                diff_click += d
            n_sa += 1
            if len(set(ids[arm.client_backend])) > 1:
                n_anydiff += 1

            if rw_tv is not None and is_search:
                base_hits = [h.docid for h in arm.rw_base.search(_query_of(snap, action), k=200)]
                inv_eq = base_hits[:PRODUCT_WINDOW] == list(reversed(base_hits))[:PRODUCT_WINDOW]
                tvt = {}
                for i in range(nb):
                    for j in range(nb):
                        k = (arm.rw_types[i], arm.rw_types[j])
                        if k not in tvt:
                            tvt[k] = rw_type_tv(k[0], k[1], len(base_hits), inv_eq)
                        rw_tv[i, j] += tvt[k]

            # advance the reference trajectory naturally and verify the replay reproduces the
            # page the rollout actually saw (end-to-end check on the whole state-forcing chain)
            obs, _r, done, _i = ref_env.unwrapped.step(action)
            fidelity_tot += 1
            if normalize(obs) == normalize(turn["obs_raw_after"]):
                fidelity_ok += 1
            if done:
                break

        if verbose_every and (ep_i + 1) % verbose_every == 0:
            print(f"[{arm.name}] {ep_i+1}/{len(episodes)} eps, N(s,a)={n_sa}, "
                  f"{time.time()-t0:.0f}s", flush=True)

    for h in hosts:
        _restore(envs[h], pristine[h])
    return {
        "arm": arm.name,
        "n_backends": nb,
        "backend_labels": [b[0] for b in arm.backends],
        "client_backend": arm.client_backend,
        "n_sa": n_sa, "n_search_sa": n_search, "n_anydiff_sa": n_anydiff,
        "diff": diff.tolist(), "diff_search": diff_search.tolist(),
        "diff_click": diff_click.tolist(), "sup_hit": sup_hit.tolist(),
        "n_unavailable": n_unavailable.tolist(),
        "rw_tv_sum": rw_tv.tolist() if rw_tv is not None else None,
        "jaccard_sum": jaccard_sum.tolist(), "n_jaccard": n_jaccard,
        "fidelity_ok": fidelity_ok, "fidelity_total": fidelity_tot,
        "seconds": round(time.time() - t0, 1),
    }


def measure_faithful(arm, envs, traces, verbose_every=25):
    """delta_eff exactly as Appendix L defines it: for the ordered pair (i, j) the occupancy is
    d^pi_{M_i}, so row i is measured on a rollout taken INSIDE client i's own environment.

    `traces` maps backend index -> the trace rolled out in that backend's kernel. Backends with
    no trace contribute no row (their column is still filled by every row that exists).
    """
    nb = arm.n_backends
    diff = np.zeros((nb, nb), dtype=np.float64)
    rows_n = np.zeros(nb, dtype=np.int64)
    rows_search = np.zeros(nb, dtype=np.int64)
    fidelity_ok = fidelity_tot = 0

    hosts = sorted({b[2] for b in arm.backends})
    pristine = {h: _pristine(envs[h]) for h in hosts}

    t0 = time.time()
    for bi, trace_path in sorted(traces.items()):
        trace = json.loads(Path(trace_path).read_text())
        _label, apply_i, host_i = arm.backends[bi]
        ref_key = f"faithref:{host_i}"
        ref_env = envs[ref_key]
        ref_pristine = _pristine(ref_env)
        _restore(ref_env, ref_pristine)
        apply_i(ref_env)                       # the reference env IS M_i for this row
        for ep_i, ep in enumerate(trace["episodes"]):
            ref_env.unwrapped.reset(session=ep["goal_index"])
            for turn in ep["turns"]:
                snap = snapshot(ref_env)
                ids, _unavail, n_hits, _pages, _pids = _render_all(
                    arm, envs, pristine, snap, turn["action"])
                diff[bi] += (ids != ids[bi])
                rows_n[bi] += 1
                if n_hits is not None:
                    rows_search[bi] += 1
                obs, _r, done, _i = ref_env.unwrapped.step(turn["action"])
                fidelity_tot += 1
                if normalize(obs) == normalize(turn["obs_raw_after"]):
                    fidelity_ok += 1
                if done:
                    break
            if verbose_every and (ep_i + 1) % verbose_every == 0:
                print(f"[{arm.name}|faithful b{bi}] {ep_i+1}/{len(trace['episodes'])} eps, "
                      f"{time.time()-t0:.0f}s", flush=True)
        _restore(ref_env, ref_pristine)

    for h in hosts:
        _restore(envs[h], pristine[h])
    rows = np.maximum(rows_n, 1)[:, None]
    return {
        "arm": arm.name,
        "occupancy": "faithful",
        "n_backends": nb,
        "backend_labels": [b[0] for b in arm.backends],
        "client_backend": arm.client_backend,
        "rows_measured": sorted(traces),
        "n_sa_per_row": rows_n.tolist(),
        "n_search_sa_per_row": rows_search.tolist(),
        "delta_matrix": (diff / rows).tolist(),     # row i = occupancy from M_i
        "fidelity_ok": fidelity_ok, "fidelity_total": fidelity_tot,
        "seconds": round(time.time() - t0, 1),
    }


def summarize_faithful(res):
    m = np.asarray(res["delta_matrix"], dtype=np.float64)
    rows = res["rows_measured"]
    vals = [m[i, j] for i in rows for j in rows if i != j]
    return {
        "arm": res["arm"],
        "occupancy": "faithful",
        "rows_measured": rows,
        "n_sa_total": int(sum(res["n_sa_per_row"])),
        "delta_max_pairwise": float(max(vals)) if vals else 0.0,
        "delta_mean_pairwise": float(np.mean(vals)) if vals else 0.0,
        "fidelity": f"{res['fidelity_ok']}/{res['fidelity_total']}",
        "backend_labels": res["backend_labels"],
        "delta_matrix": res["delta_matrix"],
        "seconds": res["seconds"],
    }


def emit_sweep(arms, path, max_backends):
    """Write the client configs a FAITHFUL run needs: one rollout per distinct kernel."""
    out = []
    for arm in arms:
        if not arm.svc_cfg:
            continue
        for bi in sorted(arm.backend_client)[:max_backends]:
            cfg = dict(arm.svc_cfg)
            cfg["client_id"] = arm.backend_client[bi]
            cfg["tag"] = f"{arm.name}__b{bi}"
            out.append(cfg)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(out, indent=1))
    print(f"[e2] wrote {path}: {len(out)} faithful rollout configs", flush=True)


def _query_of(snap, action):
    if action.startswith("search[") and action.endswith("]"):
        return action[len("search["):-1]
    return " ".join(snap["session"].get("keywords") or [])


def summarize(res):
    """Lift backend-level counts to the CLIENT-pair statistics delta_eff is defined over."""
    n = max(res["n_sa"], 1)
    cb = np.asarray(res["client_backend"])
    iu = np.triu_indices(len(cb), k=1)

    def client_pairs(mat, denom):
        m = np.asarray(mat, dtype=np.float64) / max(denom, 1)
        return m[np.ix_(cb, cb)][iu]

    pairs = client_pairs(res["diff"], n)
    sup = np.asarray(res["sup_hit"])[np.ix_(cb, cb)][iu]
    out = {
        "arm": res["arm"],
        "n_sa": res["n_sa"],
        "n_search_sa": res["n_search_sa"],
        "n_client_pairs": int(len(pairs)),
        "delta_max_pairwise": float(pairs.max()) if len(pairs) else 0.0,
        "delta_mean_pairwise": float(pairs.mean()) if len(pairs) else 0.0,
        "delta_min_pairwise": float(pairs.min()) if len(pairs) else 0.0,
        "any_pair_rate": res["n_anydiff_sa"] / n,
        "empirical_sup": float(sup.max()) if len(sup) else 0.0,
        "delta_max_search_only": float(client_pairs(res["diff_search"], res["n_search_sa"]).max())
        if len(pairs) else 0.0,
        "delta_max_click_only": float(client_pairs(res["diff_click"], n - res["n_search_sa"]).max())
        if len(pairs) else 0.0,
        "unavailable_rate_max": float(max(res["n_unavailable"]) / n),
        "fidelity": f"{res['fidelity_ok']}/{res['fidelity_total']}",
        "backend_labels": res["backend_labels"],
        "backend_matrix": (np.asarray(res["diff"], dtype=np.float64) / n).tolist(),
        "seconds": res["seconds"],
    }
    same_kernel = cb[iu[0]] == cb[iu[1]]
    out["frac_client_pairs_same_kernel"] = float(same_kernel.mean()) if len(pairs) else 0.0
    out["delta_mean_distinct_kernel_pairs"] = (
        float(pairs[~same_kernel].mean()) if (~same_kernel).any() else 0.0)
    if res.get("n_jaccard"):
        jac = client_pairs(res["jaccard_sum"], res["n_jaccard"])
        out["page_jaccard_mean_distinct_kernel_pairs"] = (
            float(jac[~same_kernel].mean()) if (~same_kernel).any() else 1.0)
        out["page_jaccard_min_pairwise"] = float(jac.min()) if len(jac) else 1.0
    if res["rw_tv_sum"] is not None:
        tvc = client_pairs(res["rw_tv_sum"], n)
        out["delta_max_pairwise_analytic"] = float(tvc.max())
        out["delta_mean_pairwise_analytic"] = float(tvc.mean())
    return out


def main():
    ap = argparse.ArgumentParser(description="E2 stage 2: delta_eff by simulator replay")
    ap.add_argument("--trace", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--emit-sweep", default=None,
                    help="write the faithful-occupancy rollout configs for the selected arms "
                         "and exit (feed to rollout_reference_occupancy.py --sweep)")
    ap.add_argument("--faithful-manifest", default=None,
                    help="manifest.json from a --sweep run; switches to the faithful occupancy "
                         "(row i measured on a rollout inside M_i)")
    ap.add_argument("--faithful-max-backends", type=int, default=12,
                    help="cap on distinct kernels rolled out per arm (Catalog Split has 100)")
    ap.add_argument("--arms", default="all",
                    help="comma list of: uniform,catalog_split,field_subset,bm25,lookalike,rank_wrapper")
    ap.add_argument("--episodes", type=int, default=None, help="limit episodes (default: all)")
    ap.add_argument("--catalog-divs", default="1.0", help="comma list of catalog_split env_div values")
    ap.add_argument("--search-return-n", type=int, default=200)
    args = ap.parse_args()

    os.environ["WEBSHOP_SEARCH_RETURN_N"] = str(args.search_return_n)
    trace = json.loads(Path(args.trace).read_text())
    episodes = trace["episodes"][:args.episodes] if args.episodes else trace["episodes"]
    print(f"[e2] trace {args.trace}: using {len(episodes)} episodes "
          f"({sum(e['n_turns'] for e in episodes)} (s,a) pairs)", flush=True)

    want = args.arms.split(",") if args.arms != "all" else [
        "uniform", "catalog_split", "field_subset", "bm25", "lookalike", "rank_wrapper"]

    print("[e2] building base envs ...", flush=True)
    envs = {"ref": make_env(seed=0), "work": make_env(seed=0)}
    _install_page_recorder()
    from web_agent_site.engine import engine as _eng
    assert _eng.SEARCH_RETURN_N == args.search_return_n, (
        f"engine imported with SEARCH_RETURN_N={_eng.SEARCH_RETURN_N}, wanted "
        f"{args.search_return_n}; export WEBSHOP_SEARCH_RETURN_N before launching")
    ref_env = envs["ref"]
    full_products = ref_env.unwrapped.server.full_products
    env_goals = ref_env.unwrapped.server.goals

    arms = []
    if "uniform" in want:
        arms.append(build_uniform_arm())
    if "catalog_split" in want:
        for div in [float(x) for x in args.catalog_divs.split(",")]:
            arms.append(build_catalog_split_arm(env_goals, full_products, div))
    if "field_subset" in want:
        for N in (4, 8):
            arms.append(build_bm25_arm("bm25_field_subset", N, full_products))
    if "bm25" in want:
        for N in (4, 8):
            arms.append(build_bm25_arm("bm25_reweight", N, full_products))
    if "rank_wrapper" in want:
        arms.append(build_rank_wrapper_arm(4, full_products))
    if "lookalike" in want:
        for N in (2, 4):
            for vi in sorted({lookalike_variant_of(c, N) for c in range(TOTAL_CLIENTS)}):
                key = f"lookalike:{vi}"
                if key not in envs:
                    print(f"[e2] building lookalike env {vi}", flush=True)
                    envs[key] = make_env(seed=0, **_lookalike_kwargs(key))
            arms.append(build_lookalike_arm(N, envs))

    if args.emit_sweep:
        emit_sweep(arms, args.emit_sweep, args.faithful_max_backends)
        return

    faithful = None
    if args.faithful_manifest:
        manifest = json.loads(Path(args.faithful_manifest).read_text())
        faithful = {}
        for tag, path in manifest.items():
            arm_name, _, bi = tag.rpartition("__b")
            faithful.setdefault(arm_name, {})[int(bi)] = path
        # the faithful rows need a reference env that is M_i and is never force-stepped
        for host in sorted({b[2] for a in arms for b in a.backends}):
            key = f"faithref:{host}"
            if key not in envs:
                print(f"[e2] building faithful reference env for host {host}", flush=True)
                envs[key] = (make_env(seed=0) if host == "work"
                             else make_env(seed=0, **_lookalike_kwargs(host)))

    results, summaries = [], []
    for arm in arms:
        print(f"[e2] === {arm.name}: {arm.n_backends} backends, "
              f"{len(set(arm.client_backend))} distinct kernels over {TOTAL_CLIENTS} clients ===",
              flush=True)
        if faithful is not None:
            traces = faithful.get(arm.name)
            if not traces:
                print(f"[e2] no faithful traces for {arm.name}; skipping", flush=True)
                continue
            res = measure_faithful(arm, envs, traces)
            results.append(res)
            s = summarize_faithful(res)
            summaries.append(s)
            print(json.dumps({k: v for k, v in s.items()
                              if k not in ("delta_matrix", "backend_labels")}, indent=1), flush=True)
            continue
        res = measure(arm, envs, ref_env, episodes)
        results.append(res)
        s = summarize(res)
        summaries.append(s)
        print(json.dumps({k: v for k, v in s.items()
                          if k not in ("backend_matrix", "backend_labels")}, indent=1), flush=True)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(
            {"meta": {"trace": args.trace, "trace_meta": trace["meta"],
                      "search_return_n": args.search_return_n,
                      "episodes_used": len(episodes)},
             "summaries": summaries, "raw": results}, ensure_ascii=False))
    print(f"[e2] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
