#!/bin/bash
# Phase 0(a) checkpoint round-trip spike runner.
# Usage (on the GPU node, via srun --jobid=<JID> --overlap):
#   bash run_phase0a.sh [fsdp|fsdp2]
set -e
source /software/miniconda3/4.10.3/etc/profile.d/conda.sh
conda activate fedagent-verl08

STRAT="${1:-fsdp}"
WD="/tmp/xbb9020_phase0a"
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
rm -rf "$WD/$STRAT"

echo "########## Phase 0(a) checkpoint round-trip  STRATEGY=$STRAT ##########"
echo "host=$(hostname)  glibc=$(ldd --version | head -1)"
python -c "import torch; print('torch', torch.__version__, 'ndev', torch.cuda.device_count())"

echo "----- [1/3] SAVE (torchrun ws=2) -----"
torchrun --nproc_per_node=2 --master_port=29577 phase0a_ckpt_roundtrip.py --phase save --strategy "$STRAT" --workdir "$WD"

echo "----- [2a/3] AGGREGATE single-process (expected to FAIL on 0.8 ShardedTensor) -----"
python phase0a_ckpt_roundtrip.py --phase aggregate --strategy "$STRAT" --workdir "$WD" || echo "(single-process aggregate failed as expected -> needs matched-PG path)"

echo "----- [2b/3] AGGREGATE matched-PG (the fix; torchrun ws=2) -----"
torchrun --nproc_per_node=2 --master_port=29579 phase0a_ckpt_roundtrip.py --phase aggregate_dist --strategy "$STRAT" --workdir "$WD"

echo "----- [3/3] RESUME (torchrun ws=2) -----"
torchrun --nproc_per_node=2 --master_port=29578 phase0a_ckpt_roundtrip.py --phase resume --strategy "$STRAT" --workdir "$WD"

echo "########## DONE  STRATEGY=$STRAT ##########"
