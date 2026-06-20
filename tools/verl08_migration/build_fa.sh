#!/bin/bash
# Build flash-attn from source against torch 2.8 on this RHEL8 (glibc 2.28) node.
# verl 0.8 unconditionally imports flash_attn.bert_padding.unpad_input in its training
# path; the prebuilt wheel needs GLIBC_2.32 (absent here), and system gcc 8.5 is too old
# for torch 2.8 (needs >=9). So: compile with the conda gcc-11 toolchain (>=9 for torch,
# <=12.2 for nvcc-12.1; pulls a matching libstdcxx-ng for runtime) + cuda-12.1 toolkit.
set -e
source /software/miniconda3/4.10.3/etc/profile.d/conda.sh
conda activate fedagent-verl08

export CUDA_HOME=/hpc/software/cuda/cuda-12.1.0
export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"
export PATH="$CUDA_HOME/bin:$CONDA_PREFIX/bin:$PATH"
export NVCC_PREPEND_FLAGS="-ccbin $CXX"
export MAX_JOBS=16
export FLASH_ATTENTION_FORCE_BUILD=TRUE
export FLASH_ATTN_CUDA_ARCHS=90   # H100 only -> much faster build

echo "=== toolchain ==="; $CXX --version | head -1; nvcc --version | tail -2
echo "=== torch ==="; python -c "import torch;print('torch',torch.__version__,'cuda',torch.version.cuda)"
echo "=== building flash-attn 2.7.4.post1 (source, sm_90) ==="
# CRITICAL: --no-deps so pip never reinstalls/upgrades torch (a bare --force-reinstall
# once cascaded into flash-attn's `torch` dep and pulled torch 2.12+cu130, breaking the
# env and producing an ABI-mismatched .so). We compile against the env's existing torch.
pip install flash-attn==2.7.4.post1 --no-build-isolation --no-deps --no-binary=flash-attn --force-reinstall --no-cache-dir 2>&1
echo "=== verify ==="
python -c "import flash_attn; from flash_attn.bert_padding import unpad_input; print('flash_attn OK', flash_attn.__version__)"
echo "=== FA BUILD DONE ==="
