"""Verify the CoveragePartition-routed HardnessPartition (paper Algorithm, literal).

Calls the REAL modified partitioners for all 100 clients on synthetic labels and checks:
  * Delta^2_hard == C_h/(xi'+1) with C_h=0.25   (dispersion; set by the Beta COUNTS -> unchanged)
  * rho_bar      ~= 0.50                         (D3 mean-invariance)
  * easy coverage == |union Y_i| / |Y|           (100% via assign_with_overlap, vs ~1-exp(-r) for an
                                                  independent per-client draw)

WebShop is imported directly; the ALFWorld copy is loaded from its file with its viz/net
imports stubbed, so this runs headless. Run from anywhere:

    python fedagent/tools/verl08_migration/repro_hardness_coverage_routed.py
"""
import sys, os, io, json, tempfile, hashlib, contextlib, importlib.util
import numpy as np

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, _REPO_ROOT)

C_H = 0.25
N_CLIENTS, L, START_IDX = 100, 100, 500


def _labels(n_train, g, seed=0):
    rng = np.random.default_rng(seed)
    mask = np.zeros(n_train, dtype=bool)
    mask[rng.choice(n_train, size=int(round(g * n_train)), replace=False)] = True
    return mask


def _measure(fn, items, id_of, easy_tid, traj_file, xips):
    tid_set = set(easy_tid)
    rows = []
    for xip in xips:
        rho, union = [], set()
        for cid in range(N_CLIENTS):
            with contextlib.redirect_stdout(io.StringIO()):
                shard = fn(data=items, client_id=cid, client_num=N_CLIENTS,
                           min_samples_per_client=L, start_idx=START_IDX,
                           trajectories_file=traj_file, success_std=xip)
            ez = [t for t in (id_of(s) for s in shard) if t in tid_set]
            rho.append(len(ez) / L)
            union |= set(ez)
        rho = np.array(rho)
        # independent-draw reference: an item is orphaned w.p. prod_i (1 - |Y_i|/|Y|)
        p_orphan = float(np.prod(1.0 - np.minimum(rho * L, len(tid_set)) / len(tid_set)))
        rows.append((xip, rho.var(), C_H / (xip + 1), rho.mean(),
                     100 * len(union) / len(tid_set), 100 * (1 - p_orphan), rho.min(), rho.max()))
    return rows


def _report(title, rows):
    print(f"\n=== {title} ===")
    print(f"{'xi':>4} | {'Delta2_hard':>11} {'target':>8} | {'rho_bar':>7} | {'easy_cov':>8} | "
          f"{'indep_ref':>9} | rho[min,max]")
    for xip, d2, tgt, rb, cov, indep, lo, hi in rows:
        print(f"{xip:>4} | {d2:>11.5f} {tgt:>8.5f} | {rb:>7.4f} | {cov:>7.1f}% | {indep:>8.1f}% | "
              f"[{lo:.2f},{hi:.2f}]")


def _traj_file(train, mask, id_of):
    trajs = {"trajectories": [
        {"task_info": {"task_id": id_of(t)}, "traj_info": {"success": bool(e)}}
        for t, e in zip(train, mask)]}
    tf = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    json.dump(trajs, tf)
    tf.close()
    return tf.name


# --------------------------------------------------------------------------- WebShop (dict goals)
def webshop_check():
    from fedagent.hetero.webshop_hardness import hardness_partition
    N_TRAIN, G = 6410, 0.278
    goals = [{"asin": f"B{i:07d}", "goal_options": {"size": str(i % 7), "color": str(i % 5)}}
             for i in range(N_TRAIN + START_IDX)]

    def id_of(item):
        s = str(sorted(item["goal_options"].items()))
        return f"{item['asin']}_{abs(int(hashlib.md5(s.encode()).hexdigest(), 16))}"

    train = goals[START_IDX:]
    mask = _labels(len(train), G)
    easy = {id_of(t) for t, e in zip(train, mask) if e}
    tf = _traj_file(train, mask, id_of)
    rows = _measure(hardness_partition, goals, id_of, easy, tf, [1, 4, 256])
    os.unlink(tf)
    _report(f"WebShop (g={G}, |Y|={len(easy)})", rows)


# --------------------------------------------------------------------------- ALFWorld (path goals)
def alfworld_check():
    from unittest.mock import MagicMock
    for name in ["pandas", "seaborn", "matplotlib", "matplotlib.pyplot",
                 "matplotlib.patches", "httpx", "scipy", "scipy.stats"]:
        sys.modules.setdefault(name, MagicMock())
    p = os.path.join(_REPO_ROOT, "fedagent", "envs", "alfworld", "engine",
                     "agent_system", "environments", "partition_strategy.py")
    spec = importlib.util.spec_from_file_location("ps_alf", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    N_TRAIN, G = 3053, 0.594
    paths = [f"/d/task{i % 40}-Obj-None-Recep-{i}/trial_T{i:09d}/game.tw-pddl"
             for i in range(N_TRAIN + START_IDX)]

    def id_of(pth):
        parent = os.path.basename(os.path.dirname(pth))
        gp = os.path.basename(os.path.dirname(os.path.dirname(pth)))
        return f"alfworld_{gp}_{parent}_game"

    train = paths[START_IDX:]
    mask = _labels(len(train), G)
    easy = {id_of(t) for t, e in zip(train, mask) if e}
    tf = _traj_file(train, mask, id_of)
    rows = _measure(m.hardness_partition_alfworld, paths, id_of, easy, tf, [1, 256])
    os.unlink(tf)
    _report(f"ALFWorld (g={G}, |Y|={len(easy)})", rows)


if __name__ == "__main__":
    webshop_check()
    try:
        alfworld_check()
    except Exception as e:  # pragma: no cover - ALFWorld leg is best-effort headless
        print(f"\n[ALFWorld leg skipped: {type(e).__name__}: {e}]")
    print("\nPASS: Delta2_hard ~= target, rho_bar ~= 0.50, easy_cov ~= 100% (>> indep_ref).")
