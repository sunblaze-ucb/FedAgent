#!/bin/bash
# Diagnose env damage after flash-attn --force-reinstall clobbered torch.
source /software/miniconda3/4.10.3/etc/profile.d/conda.sh
conda activate fedagent-verl08
echo "=== current versions ==="
pip show torch vllm flashinfer-python sglang 2>/dev/null | grep -E "^Name|^Version"
echo "=== is env broken? import torch/vllm + cuda avail ==="
python - <<'PY' 2>&1 | grep -vE "FutureWarning|import pynvml|pynvml package"
import torch
print("torch", torch.__version__, "cuda_avail", torch.cuda.is_available())
try:
    import vllm; print("vllm import OK", vllm.__version__)
except Exception as e:
    print("vllm IMPORT BROKEN:", type(e).__name__, str(e)[:180])
PY
echo "=== vllm dist-info torch pin ==="
SP=$(python -c "import site;print(site.getsitepackages()[0])")
grep -iE "Requires-Dist: torch" "$SP"/vllm-*.dist-info/METADATA 2>/dev/null | head
echo "=== pip cache: torch wheels present ==="
pip cache list 2>/dev/null | grep -iE "torch-2" | head
echo "=== conda-meta torch record (original install) ==="
ls "$CONDA_PREFIX"/conda-meta/ 2>/dev/null | grep -iE "^pytorch|^torch" | head
echo "=== DIAG DONE ==="
