#!/bin/bash
# Durable relauncher for the GRPO hardness (xi'=1) re-run on job 7150586.
# run_fed's round-level resume (hf_export: every_round) continues at the first
# incomplete round. Launch DETACHED (nohup ... & disown) so no interactive-session
# teardown can kill the local srun client (that is what killed step .12).
LOG=/gpfs/projects/b1222/userdata/canyu/kangyu/fedagent/runs/hardness_rerun/grpo_ws_std1.log
echo "[relaunch $(date '+%F %T')] step .12 was killed 09:14:49 (session teardown); resuming" >> "$LOG"
exec srun --jobid=7150586 --overlap -N1 bash -lc '
  source /software/miniconda3/4.10.3/etc/profile.d/conda.sh
  conda activate fedagent-verl08
  cd /gpfs/projects/b1222/userdata/canyu/kangyu/fedagent/review/main
  export PYTHONPATH="$PWD:$PYTHONPATH"
  export VERL_CFG="$(python -c "import verl,os;print(os.path.join(os.path.dirname(verl.__file__),\"trainer\",\"config\"))")"
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_USE_V1=1
  export VLLM_USE_DEEP_GEMM=0 VLLM_SKIP_DEEP_GEMM_WARMUP=1
  export CUDA_HOME=/hpc/software/cuda/cuda-12.1.0
  python -c "import verl" || { echo "[relaunch] verl import FAILED (wrong env)"; exit 1; }
  exec python -m fedagent.fed.run_fed \
    --config fedagent/tools/verl08_migration/accel/webshop/hardness_rerun_ws_grpo_std1.yaml
' >> "$LOG" 2>&1
