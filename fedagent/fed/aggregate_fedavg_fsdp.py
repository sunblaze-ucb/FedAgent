#!/usr/bin/env python
"""Generalized matched-PG FedAvg for verl-0.8 FSDP checkpoints (the Phase 6 aggregator core).

FedAvg runs under a MATCHED-world-size process group (torchrun
--nproc_per_node = the save-time world_size): each rank loads its OWN rank shard from
every client, (weighted-)averages the LOCAL tensors IN PLACE (_get_local handles
DTensor / ShardedTensor / plain), and torch.saves the dict back. The output is
byte-structurally identical to a verl checkpoint (same tensor objects, only local
values changed), so the next round loads it with verl's own FSDP wrap unchanged.
Note: real verl-0.8 training checkpoints hold DTensor params that DO load
single-process (the "ShardedTensor cannot load single-process" belief came from a
synthetic spike's own save path) -- matched-PG write-back is kept for structural
identity, and because the re-wrap trap is real: verl's transformer auto-wrap policy
shards params differently than a whole-model wrap, so loading verl shards into a
basic-wrap model fails `assert type(tensor) is ShardedTensor`. Do NOT re-wrap.

Generalized from phase0a_ckpt_roundtrip.py (migrate/verl-0.8.0 branch) (FedAvg-exact + resume
validated). core/ will shell out to this (replacing utils/model_aggregation's single-
process load) for the verl-0.8 federated loop.

Usage (nproc_per_node MUST equal world_size in fsdp_config.json):
  torchrun --nproc_per_node=2 aggregate_fedavg_fsdp.py --phase aggregate \
      --client-actor-dirs A/global_step_X/actor,B/global_step_Y/actor \
      --output-actor-dir OUT/global_step_0/actor [--weights 0.5,0.5] [--global-step 0]
  torchrun --nproc_per_node=2 aggregate_fedavg_fsdp.py --phase verify \
      --output-actor-dir OUT/global_step_0/actor --client-actor-dirs A/...,B/...
"""
import argparse
import os
import shutil
from pathlib import Path

import torch
import torch.distributed as dist


def _get_local(t):
    """Writable local tensor backing a ShardedTensor / DTensor / plain tensor."""
    from torch.distributed._shard.sharded_tensor import ShardedTensor

    if isinstance(t, ShardedTensor):
        return t.local_shards()[0].tensor
    try:
        from torch.distributed.tensor import DTensor
    except Exception:
        from torch.distributed._tensor import DTensor
    if isinstance(t, DTensor):
        return t.to_local()
    return t


def _parse_weights(arg, n):
    w = [float(x) for x in arg.split(",")] if arg else [1.0 / n] * n
    assert len(w) == n, f"weights ({len(w)}) must match #clients ({n})"
    assert abs(sum(w) - 1.0) < 1e-6, f"weights must sum to 1, got {w}"
    return w


def aggregate(args):
    rank, ws = dist.get_rank(), dist.get_world_size()
    clients = [Path(p) for p in args.client_actor_dirs.split(",") if p]
    weights = _parse_weights(args.weights, len(clients))
    out = Path(args.output_actor_dir)
    out.mkdir(parents=True, exist_ok=True)
    rank_file = f"model_world_size_{ws}_rank_{rank}.pt"

    sds = [torch.load(c / rank_file, weights_only=False) for c in clients]
    base = sds[0]
    for k in base:
        acc = _get_local(base[k])
        acc.mul_(weights[0])
        for w, other in zip(weights[1:], sds[1:]):
            acc.add_(_get_local(other[k]), alpha=w)
    torch.save(base, out / rank_file)  # same ShardedTensor objects, averaged local values
    dist.barrier()

    if rank == 0:
        # Only the model is averaged; optimizer/extra_state are NOT carried over -- each
        # federated round starts a fresh optimizer from the aggregated model. Copy just
        # what a model-load needs: the FSDP metadata, the HF config/tokenizer, the tracker.
        src = clients[0]
        shutil.copy2(src / "fsdp_config.json", out / "fsdp_config.json")
        if (src / "huggingface").is_dir():
            shutil.copytree(src / "huggingface", out / "huggingface", dirs_exist_ok=True)
        (out.parent.parent / "latest_checkpointed_iteration.txt").write_text(str(int(args.global_step)))
        print(f"[aggregate] FedAvg {len(clients)} clients -> {out}  (ws={ws}, weights={weights})", flush=True)
    dist.barrier()


def verify(args):
    """Matched-PG load-back: re-load the written shards and confirm (a) they round-trip
    with each param in the SAME wrapper class the client shards carry (verl's strict load
    asserts the class, so a save/load that degraded a ShardedTensor to a plain tensor
    would kill the next round), and (b) the local values equal the weighted average of
    the clients (FedAvg correctness). FAILURE EXITS NONZERO -- torchrun propagates it, so
    a caller gating on the exit code actually gates (the 2026-07 audit found the old
    print-only FAIL let a broken aggregate pass a scripted check silently)."""
    rank, ws = dist.get_rank(), dist.get_world_size()
    out = Path(args.output_actor_dir)
    clients = [Path(p) for p in args.client_actor_dirs.split(",") if p]
    weights = _parse_weights(args.weights, len(clients))
    rank_file = f"model_world_size_{ws}_rank_{rank}.pt"

    from torch.distributed._shard.sharded_tensor import ShardedTensor

    got = torch.load(out / rank_file, weights_only=False)
    sds = [torch.load(c / rank_file, weights_only=False) for c in clients]

    n_sharded = sum(1 for k in got if isinstance(got[k], ShardedTensor))
    # structural check (was a print-only count): aggregate() re-saves the client-0 objects
    # with averaged local values, so every key must come back as the class client 0 carries.
    degraded = sorted(k for k in got if type(got[k]) is not type(sds[0][k]))
    maxdiff = 0.0
    worst = None
    for k in got:
        exp = _get_local(sds[0][k]).clone().float() * weights[0]
        for w, s in zip(weights[1:], sds[1:]):
            exp = exp + _get_local(s[k]).float() * w
        d = (_get_local(got[k]).float() - exp).abs().max().item()
        if d > maxdiff:
            maxdiff, worst = d, k
    # one all-reduce -> every rank holds the same verdict (a rank-local exit would strand
    # the other ranks at the final barrier with a different exit code)
    md = torch.tensor([maxdiff, float(len(degraded))],
                      device=f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}")
    dist.all_reduce(md, op=dist.ReduceOp.MAX)
    ok = md[0].item() < 1e-4 and md[1].item() == 0
    if rank == 0:
        print(f"[verify] round-trip: {len(got)} params, {n_sharded} ShardedTensor, "
              f"{len(degraded)} class-degraded", flush=True)
        print(f"[verify] FedAvg correctness: max|got - weighted_avg| = {md[0].item():.3e} "
              f"(worst@rank0 {worst})", flush=True)
        if degraded:
            print(f"[verify] degraded keys (first 5): {degraded[:5]}", flush=True)
        print("[verify] PASS" if ok else "[verify] FAIL", flush=True)
    if not ok:
        raise SystemExit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=["aggregate", "verify"])
    ap.add_argument("--client-actor-dirs", default="")
    ap.add_argument("--output-actor-dir", required=True)
    ap.add_argument("--weights", default="")
    ap.add_argument("--global-step", default="0")
    args = ap.parse_args()

    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    try:
        aggregate(args) if args.phase == "aggregate" else verify(args)
    finally:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
