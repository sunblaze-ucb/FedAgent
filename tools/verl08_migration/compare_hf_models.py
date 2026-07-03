"""Compare two HF safetensors exports tensor-by-tensor (world-size independent).

Companion to compare_fsdp_checkpoints.py for arms whose FSDP world sizes differ
(e.g. parallel_clients lanes: ws=2 aggregated shards vs ws=4 baseline) -- the HF
export is the consolidated full state dict, so it compares across shardings.
Output format matches compare_fsdp_checkpoints.py (OVERALL / VERDICT lines).

    python compare_hf_models.py --a <hf_dir_A> --b <hf_dir_B> [--bar 1e-4]
"""
import argparse
import glob
import os

import torch
from safetensors.torch import load_file


def load_hf_state(d):
    state = {}
    files = sorted(glob.glob(os.path.join(d, "*.safetensors")))
    if not files:
        raise SystemExit(f"ERROR: no .safetensors under {d}")
    for f in files:
        state.update(load_file(f))
    return state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--bar", type=float, default=1e-4)
    args = ap.parse_args()

    sa, sb = load_hf_state(args.a), load_hf_state(args.b)
    print(f"compare A={args.a}\n        B={args.b}")
    keys_a, keys_b = set(sa), set(sb)
    mismatches = len(keys_a ^ keys_b)
    for k in sorted(keys_a ^ keys_b):
        print(f"  KEY ONLY IN {'A' if k in keys_a else 'B'}: {k}")

    max_d, mean_num, mean_den, worst = 0.0, 0.0, 0, None
    for k in sorted(keys_a & keys_b):
        ta, tb = sa[k].float(), sb[k].float()
        if ta.shape != tb.shape:
            mismatches += 1
            print(f"  SHAPE MISMATCH {k}: {tuple(ta.shape)} vs {tuple(tb.shape)}")
            continue
        d = (ta - tb).abs()
        m = d.max().item()
        mean_num += d.sum().item()
        mean_den += d.numel()
        if m > max_d:
            max_d, worst = m, k
    mean_d = mean_num / max(mean_den, 1)
    print(f"  OVERALL max|Δ|={max_d:.3e}  mean|Δ|={mean_d:.3e}  worst={worst} ({max_d:.3e})")
    ok = mismatches == 0 and max_d <= args.bar
    print(f"  VERDICT: {'EQUIVALENT' if ok else 'DIFFERENT'} (bar={args.bar:g}, key/shape mismatches={mismatches})")


if __name__ == "__main__":
    main()
