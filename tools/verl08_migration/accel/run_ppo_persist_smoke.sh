#!/bin/bash
# PPO persistent smoke (re-run after fedagent_ppo.yaml critic block fix): adv_estimator=gae,
# persistent=true. Closing rc=0 means init_workers-once + per-client reload_client_model AND
# reload_critic_model both fired and actor+critic FedAvg'd. NO `set -u` (conda deactivate hook).
REPO=/gpfs/projects/b1222/userdata/canyu/kangyu/fedagent
for __c in "$CONDA_PREFIX_1" "$HOME/miniconda3" "$HOME/anaconda3" /opt/conda /software/miniconda3/4.10.3; do [ -f "$__c/etc/profile.d/conda.sh" ] && { . "$__c/etc/profile.d/conda.sh"; break; }; done
conda activate fedagent-verl08
cd "$REPO"
echo "PPO PERSIST SMOKE(v2) START $(date +%s)"
python -u -m fedagent.fed.run_fed --config _scratch/accel/ppo_persist_smoke.yaml
echo "PPO PERSIST SMOKE(v2) rc=$? end=$(date +%s)"
echo "=== checkpoints (actor + critic) ==="
ls -d _scratch/accel/ppo_persist_out/round_1/aggregated/checkpoints/global_step_*/{actor,critic} 2>/dev/null
