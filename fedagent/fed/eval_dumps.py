"""Reading verl's validation dumps — the one implementation, shared by the loop and the tools.

``trainer.validation_data_dir`` makes verl write one JSONL per validated global_step:
``<dir>/<global_step>.jsonl``, one row per val sample. FedAgent points it at
``round_<k>/eval/val_samples`` (the aggregated model's point on the paper's red line) and
``round_<k>/client_<c>/eval/val_samples`` (that client's post-training circle mark), then folds
the parsed means into ``federated_summary.json``, which every figure is drawn from.

Two decisions live here, and they must be identical everywhere the dumps are read -- the
federated loop (``fed/run_fed.py``), the offline rebuild (``tools/rebuild_summary.py``), and
anything added later:

* **which file** — the LATEST global_step, compared NUMERICALLY. Lexicographic ``sorted()``
  puts ``10.jsonl`` before ``4.jsonl``, so a directory that ever holds two dumps would silently
  be summarized from the older one (fixed 2026-07-26).
* **which keys** — the agent loop tags every val sample with ``traj_success`` (1.0 iff the
  episode succeeded), ``score`` (the episode return; ALFWorld's is the 0/10 binarized signal)
  and, WebShop only, ``task_score`` (the partial-credit [0,1] goal match). ``task_score_mean``
  is None for envs/runs that never emitted it.

This module is deliberately dependency-free (stdlib only, and ``fedagent``/``fedagent.fed``
are docstring-only packages) so the offline tools can import it under a bare interpreter --
``run_fed`` itself needs omegaconf and cannot be imported from one.
"""
import json
from pathlib import Path
from typing import List, Optional


def dumps_by_step(dump_dir) -> List[Path]:
    """The dir's validation dumps, oldest global_step first.

    NUMERIC by stem, so ``4.jsonl`` sorts before ``10.jsonl``; any non-numeric name sorts last,
    by name (never silently ahead of a real step)."""
    return sorted(Path(dump_dir).glob("*.jsonl"),
                  key=lambda p: (0, int(p.stem), "") if p.stem.isdigit() else (1, 0, p.stem))


def read_latest_rows(dump_dir) -> List[dict]:
    """Rows of the LATEST validation dump in ``dump_dir`` (empty list if none). Malformed
    lines are skipped rather than failing the read -- a truncated tail from a killed run
    must not cost the whole round's point."""
    files = dumps_by_step(dump_dir)
    if not files:
        return []
    rows = []
    with open(files[-1]) as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def _mean(rows, key):
    vals = [float(r[key]) for r in rows if r.get(key) is not None]
    return round(sum(vals) / len(vals), 4) if vals else None


def _row_stats(rows) -> dict:
    return {"n": len(rows), "success_rate": _mean(rows, "traj_success"),
            "reward_mean": _mean(rows, "score"), "task_score_mean": _mean(rows, "task_score")}


def summarize_val_dump(dump_dir) -> Optional[dict]:
    """``{n, success_rate, reward_mean, task_score_mean}`` for a dump dir, or None if it holds
    no readable rows. Reads only the latest dump (see the module docstring)."""
    rows = read_latest_rows(dump_dir)
    return _row_stats(rows) if rows else None


def summarize_val_dump_by_tasktype(dump_dir) -> Optional[dict]:
    """Group the LATEST dump's rows by their ``task_type`` tag: the paper's per-task-type
    breakdown estimator (ONE eval pass, partitioned; ``All`` = the pooled rows, so the
    per-type numbers combine to ``All`` by construction).

    The tag is written per row by the agent loop (ALFWorld: derived from the episode's
    gamefile, e.g. ``pick_clean_then_place_in_recep``). Rows without the tag are grouped
    under ``"untagged"`` -- a dump from a codebase predating the tag yields all-untagged,
    which is the signal to re-run the eval rather than trust a partial breakdown.

    Returns ``{"All": stats, "by_type": {type_name: stats}}`` (stats as in
    ``summarize_val_dump``), or None if the dir holds no readable rows."""
    rows = read_latest_rows(dump_dir)
    if not rows:
        return None
    by_type: dict = {}
    for r in rows:
        by_type.setdefault(r.get("task_type") or "untagged", []).append(r)
    return {"All": _row_stats(rows),
            "by_type": {t: _row_stats(rs) for t, rs in sorted(by_type.items())}}
