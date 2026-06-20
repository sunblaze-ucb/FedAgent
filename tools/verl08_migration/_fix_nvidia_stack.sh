#!/bin/bash
# The torch-2.12+cu130 clobber installed a parallel CUDA-13-era nvidia stack whose
# pip packages share the SAME `nvidia/<lib>/lib/` install paths as torch 2.8's cu12
# packages. The cu13 .so files (installed last) overwrote the cu12 ones on disk, so
# torch 2.8 loaded libnccl 2.29.7 (CUDA-13) on a CUDA-12.8 driver -> NCCL
# "unhandled cuda error" at FSDP param broadcast. The torch restore only reinstalled
# torch+triton, leaving these orphans in place.
# Fix: (1) uninstall the CUDA-13-era orphans; (2) force-reinstall torch 2.8.0+cu128
# WITH deps so every correct cu12 nvidia .so is rewritten.
set -e
source /software/miniconda3/4.10.3/etc/profile.d/conda.sh
conda activate fedagent-verl08

echo "=== libnccl size BEFORE (cu13=236524880 is the bad one) ==="
python - <<'PY' || true
import os, site, glob
hits = glob.glob(os.path.join(site.getsitepackages()[0], "nvidia", "nccl", "lib", "libnccl.so.2"))
print(hits[0], os.path.getsize(hits[0])) if hits else print("libnccl MISSING")
PY

echo "=== (1) uninstall CUDA-13-era orphan nvidia libs ==="
pip uninstall -y \
  nvidia-cublas nvidia-cuda-cupti nvidia-cuda-nvrtc nvidia-cuda-runtime \
  nvidia-cudnn-cu13 nvidia-cufft nvidia-cufile nvidia-curand \
  nvidia-cusolver nvidia-cusparse nvidia-cusparselt-cu13 nvidia-nccl-cu13 \
  nvidia-nvjitlink nvidia-nvtx nvidia-nvshmem-cu13 || true

echo "=== (2) force-reinstall torch trio (with deps) -> rewrite correct cu12 .so files ==="
pip install --force-reinstall --no-cache-dir \
  torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu128

echo "=== verify: libnccl should now be cu12 (429634192) ==="
python - <<'PY'
import os, site, glob
hits = glob.glob(os.path.join(site.getsitepackages()[0], "nvidia", "nccl", "lib", "libnccl.so.2"))
print("libnccl size", os.path.getsize(hits[0]), "(cu12 expected 429634192)") if hits else print("libnccl MISSING")
PY
python -c "import flash_attn; from flash_attn.bert_padding import unpad_input; print('flash_attn still OK', flash_attn.__version__)"
echo "=== leftover nvidia-*-cu13 / non-suffixed (should be empty) ==="
pip list 2>/dev/null | grep -iE "nvidia" | grep -ivE "cu12|ml-py|cudnn-frontend" || echo "  (none - clean)"
echo "=== NVIDIA FIX DONE ==="
