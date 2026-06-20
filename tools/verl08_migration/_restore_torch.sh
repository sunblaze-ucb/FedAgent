#!/bin/bash
# Restore torch after `flash-attn --force-reinstall` (no --no-deps) clobbered it
# from 2.8.0 -> 2.12.1+cu130. vllm 0.11.0 pins torch==2.8.0/vision0.23.0/audio2.8.0;
# node NVIDIA driver is CUDA 12.8 (12080) -> use the cu128 wheels (exact driver match).
set -e
source /software/miniconda3/4.10.3/etc/profile.d/conda.sh
conda activate fedagent-verl08

echo "=== before ==="
python -c "import torch;print('torch',torch.__version__)" 2>/dev/null || echo "torch import failed"

pip install --no-cache-dir \
  torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu128

echo "=== after (import; cuda_avail only meaningful on a GPU node) ==="
python -c "import torch;print('torch',torch.__version__,'cuda',torch.version.cuda)"
echo "=== RESTORE DONE ==="
