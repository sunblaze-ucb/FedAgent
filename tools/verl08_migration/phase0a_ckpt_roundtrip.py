#!/usr/bin/env python
"""Phase 0(a) de-risk spike for the verl 0.8 migration.

Question: does FedAgent's FedAvg same-rank shard averaging + resume still work on
verl 0.8 (torch 2.8) FSDP checkpoints, for BOTH FSDP1 and FSDP2/DTensor?

Faithful 3-phase reproduction of production:
  save      (torchrun, 2 ranks)  -> two "clients" each write verl-0.8 FSDP shards
  aggregate (plain python, NO dist, like the fed server) -> ModelAggregator FedAvg
  resume    (torchrun, 2 ranks)  -> load aggregated shards into a fresh FSDP model
                                    + correctness check vs mean(A, B)

Run via tools/verl08_migration/run_phase0a.sh (handles torchrun vs python per phase).
"""
import argparse
import os
import sys
import traceback
from pathlib import Path

import torch

FEDAGENT = "/gpfs/projects/b1222/userdata/canyu/kangyu/fedagent"


# --------------------------------------------------------------------------- #
# model / wrapping helpers
# --------------------------------------------------------------------------- #
def build_model():
    """Tiny, download-free Llama with representative param names (q/k/v/o/gate/up/down)."""
    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig(
        vocab_size=512,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=64,
        tie_word_embeddings=False,
        attn_implementation="eager",  # avoid flash-attn (broken on this glibc)
    )
    torch.manual_seed(0)
    return LlamaForCausalLM(cfg)


def apply_perturb(model, seed):
    """Deterministic, rank-identical perturbation (CPU randn with a fixed generator)."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn(p.shape, generator=g) * 0.05)


def full_cpu_state(perturb_seed=None):
    m = build_model()
    if perturb_seed is not None:
        apply_perturb(m, perturb_seed)
    return {k: v.detach().clone().float() for k, v in m.state_dict().items()}


def wrap(model, strategy, device, mesh):
    model = model.to(device)
    if strategy == "fsdp":
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import ShardingStrategy

        return FSDP(
            model,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            device_id=torch.cuda.current_device(),
            use_orig_params=True,
        )
    elif strategy == "fsdp2":
        from torch.distributed.fsdp import fully_shard

        for layer in model.model.layers:
            fully_shard(layer, mesh=mesh)
        fully_shard(model, mesh=mesh)
        return model
    raise ValueError(strategy)


def make_manager(model):
    from omegaconf import OmegaConf

    from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager

    cfg = OmegaConf.create({"save_contents": ["model"], "load_contents": ["model"]})
    return FSDPCheckpointManager(
        model=model, optimizer=None, lr_scheduler=None, processing_class=None, checkpoint_config=cfg
    )


# --------------------------------------------------------------------------- #
# phases
# --------------------------------------------------------------------------- #
def do_save(args, device, mesh):
    rank = torch.distributed.get_rank()
    for name, seed in [("clientA", None), ("clientB", 123)]:
        m = build_model()
        if seed is not None:
            apply_perturb(m, seed)
        fm = wrap(m, args.strategy, device, mesh)
        mgr = make_manager(fm)
        out = Path(args.workdir) / args.strategy / name
        mgr.save_checkpoint(local_path=str(out), global_step=1)
        torch.distributed.barrier()
        if rank == 0:
            files = sorted(p.name for p in out.iterdir())
            print(f"[save] {name} -> {out}\n        files: {files}", flush=True)
        del fm, m
        torch.cuda.empty_cache()


def _try_load(path):
    """Load a shard dict; if DTensor needs a process group, retry under a 1-rank gloo group."""
    import torch.distributed as dist

    try:
        return torch.load(path, weights_only=False, map_location="cpu"), "no-dist"
    except Exception as e:
        print(f"[aggregate] raw torch.load failed ({type(e).__name__}: {str(e)[:100]}); retrying under gloo ws=1", flush=True)
        if not dist.is_initialized():
            os.environ.setdefault("MASTER_ADDR", "localhost")
            os.environ.setdefault("MASTER_PORT", "12399")
            os.environ.setdefault("RANK", "0")
            os.environ.setdefault("WORLD_SIZE", "1")
            dist.init_process_group(backend="gloo")
        return torch.load(path, weights_only=False, map_location="cpu"), "gloo-ws1"


def do_aggregate(args):
    sys.path.insert(0, FEDAGENT)
    wd = Path(args.workdir) / args.strategy
    ca, cb = wd / "clientA", wd / "clientB"

    print("=== RAW SHARD INSPECTION (no dist, like the fed server) ===", flush=True)
    f0 = ca / "model_world_size_2_rank_0.pt"
    try:
        sd, how = _try_load(f0)
        k0 = sorted(sd.keys())[0]
        v0 = sd[k0]
        print(f"[aggregate] loaded {f0.name} via {how}: {len(sd)} keys", flush=True)
        print(f"[aggregate] sample key={k0} type={type(v0).__name__} shape={tuple(getattr(v0, 'shape', ()))} dtype={getattr(v0,'dtype',None)}", flush=True)
        try:
            from torch.distributed.tensor import DTensor
        except Exception:
            from torch.distributed._tensor import DTensor
        is_dt = isinstance(v0, DTensor)
        print(f"[aggregate] is DTensor: {is_dt}", flush=True)
        if is_dt:
            print(f"[aggregate]   placements={v0.placements} mesh={v0.device_mesh} local_shape={tuple(v0.to_local().shape)}", flush=True)
    except Exception:
        traceback.print_exc()
        print("[aggregate] RAW LOAD FAILED -> FedAgent aggregator needs adaptation for 0.8 shards", flush=True)

    fc = ca / "fsdp_config.json"
    if fc.exists():
        print(f"[aggregate] fsdp_config.json: {fc.read_text().strip()}", flush=True)

    print("=== direct_shard_aggregation (FedAvg same-rank average) ===", flush=True)
    try:
        from utils.model_aggregation import ModelAggregator

        agg = ModelAggregator()
        out = wd / "aggregated" / "aggregated_actor_model.pth"
        res = agg.direct_shard_aggregation(
            [str(ca), str(cb)], str(out), expected_global_step=0, model_type="actor", n_gpus_per_node=2
        )
        print(f"[aggregate] result dir: {res}", flush=True)
        if res is None:
            print("[aggregate] FAIL: aggregation returned None", flush=True)
            sys.exit(2)
    except Exception:
        traceback.print_exc()
        print("[aggregate] FAIL: aggregation raised", flush=True)
        sys.exit(2)


def _get_local(t):
    """Return the writable local tensor backing a ShardedTensor / DTensor / plain tensor."""
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


def do_aggregate_dist(args, device, mesh):
    """The FIX: FedAvg in sharded space under a matched-world-size process group.

    Each rank loads its OWN rank shard from every client (valid because ws matches
    save time), averages the per-rank LOCAL tensors across clients, loads the result
    into a fresh FSDP model, and re-saves via verl's FSDPCheckpointManager so the
    output is a verl-native, resumable checkpoint.
    """
    import torch.distributed as dist

    rank, ws = dist.get_rank(), dist.get_world_size()
    wd = Path(args.workdir) / args.strategy
    clients = [wd / "clientA", wd / "clientB"]
    rank_file = f"model_world_size_{ws}_rank_{rank}.pt"

    sds = [torch.load(c / rank_file, weights_only=False) for c in clients]
    if rank == 0:
        print(f"[aggregate_dist] loaded {len(sds)} client shards under matched ws={ws}", flush=True)

    # average local tensors in place into sds[0]
    base = sds[0]
    for k in base:
        acc = _get_local(base[k])
        acc.mul_(1.0 / len(sds))
        for other in sds[1:]:
            acc.add_(_get_local(other[k]), alpha=1.0 / len(sds))

    m = build_model()
    fm = wrap(m, args.strategy, device, mesh)
    from torch.distributed.fsdp import ShardedStateDictConfig, StateDictType

    from verl.utils.fsdp_utils import get_fsdp_state_ctx

    with get_fsdp_state_ctx(fm, StateDictType.SHARDED_STATE_DICT, ShardedStateDictConfig(offload_to_cpu=False), None):
        fm.load_state_dict(base)
    dist.barrier()

    out = wd / "aggregated" / "checkpoints" / "global_step_0" / "actor"
    make_manager(fm).save_checkpoint(local_path=str(out), global_step=0)
    dist.barrier()
    if rank == 0:
        print(f"[aggregate_dist] wrote verl-native averaged checkpoint -> {out}", flush=True)


def do_resume(args, device, mesh):
    rank = torch.distributed.get_rank()
    wd = Path(args.workdir) / args.strategy
    aggdir = wd / "aggregated" / "checkpoints" / "global_step_0" / "actor"

    m = build_model()
    fm = wrap(m, args.strategy, device, mesh)
    mgr = make_manager(fm)
    mgr.load_checkpoint(local_path=str(aggdir))
    torch.distributed.barrier()
    if rank == 0:
        print(f"[resume] load_checkpoint OK from {aggdir}", flush=True)

    # correctness: gathered full params should equal mean(A, B)
    from verl.utils.fsdp_utils import get_fsdp_full_state_dict

    full = get_fsdp_full_state_dict(fm, offload_to_cpu=True, rank0_only=True)
    if rank == 0:
        a = full_cpu_state(None)
        b = full_cpu_state(123)
        maxdiff = 0.0
        worst = None
        for k in a:
            exp = (a[k] + b[k]) / 2.0
            got = full[k].detach().cpu().float()
            d = (got - exp).abs().max().item()
            if d > maxdiff:
                maxdiff, worst = d, k
        print(f"[resume] CORRECTNESS max|resumed - mean(A,B)| = {maxdiff:.3e} (worst key: {worst})", flush=True)
        print("[resume] PASS" if maxdiff < 1e-3 else "[resume] FAIL (diff too large)", flush=True)
        if maxdiff >= 1e-3:
            sys.exit(3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=["save", "aggregate", "aggregate_dist", "resume"])
    ap.add_argument("--strategy", default="fsdp", choices=["fsdp", "fsdp2"])
    ap.add_argument("--workdir", default="/tmp/xbb9020_phase0a")
    args = ap.parse_args()

    if args.phase == "aggregate":
        do_aggregate(args)
        return

    import torch.distributed as dist

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    mesh = None
    if args.strategy == "fsdp2":
        from torch.distributed.device_mesh import init_device_mesh

        mesh = init_device_mesh("cuda", (dist.get_world_size(),))

    if args.phase == "save":
        do_save(args, device, mesh)
    elif args.phase == "aggregate_dist":
        do_aggregate_dist(args, device, mesh)
    else:
        do_resume(args, device, mesh)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
