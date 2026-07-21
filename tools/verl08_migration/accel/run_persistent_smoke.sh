#!/bin/bash
# Lever #4 smoke: train 2 clients in ONE persistent process (init_workers once, fit per client).
# TinyGuess / concat / 0.5B / 4 GPU. Proves the mechanism (init once, reset, fit, reset, fit).
# NOTE: no `set -u` -- conda's proj4-deactivate.sh references an unbound var during activate.
REPO=/gpfs/projects/b1222/userdata/canyu/kangyu/fedagent
BASE=/projects/b1222/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775
OUT=$REPO/_scratch/accel/persist_out
STEPS=${STEPS:-2}
rm -rf "$OUT"; mkdir -p "$OUT"

cat > "$OUT/plan.json" <<JSON
[
 {"client":0,"model_path":"$BASE","critic_path":null,"seed":142,"out_dir":"$OUT/client_0/checkpoints","exp":"persist_c0"},
 {"client":1,"model_path":"$BASE","critic_path":null,"seed":143,"out_dir":"$OUT/client_1/checkpoints","exp":"persist_c1"}
]
JSON

for __c in "$CONDA_PREFIX_1" "$HOME/miniconda3" "$HOME/anaconda3" /opt/conda /software/miniconda3/4.10.3; do [ -f "$__c/etc/profile.d/conda.sh" ] && { . "$__c/etc/profile.d/conda.sh"; break; }; done
conda activate fedagent-verl08
export PYTHONPATH=$REPO:${PYTHONPATH:-}
export VERL_CFG=$(python -c "import verl,os;print(os.path.join(os.path.dirname(verl.__file__),'trainer','config'))")
export TOKENIZERS_PARALLELISM=false
export FEDAGENT_PERSISTENT=1
export FEDAGENT_PERSISTENT_PLAN=$OUT/plan.json
cd "$REPO"

echo "wall_start_epoch=$(date +%s)"
python -u -m fedagent.fed.persistent_main \
  data.train_files=$REPO/fedagent/config/envs/tiny_guess.yaml \
  data.val_files=$REPO/fedagent/config/envs/tiny_guess.yaml \
  data.custom_cls.path=$REPO/fedagent/data/agentic_dataset.py \
  actor_rollout_ref.model.path=$BASE \
  +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
  actor_rollout_ref.rollout.agent.agent_loop_config_path=$REPO/fedagent/config/agent.yaml \
  trainer.default_local_dir=$OUT/client_0/checkpoints \
  trainer.n_gpus_per_node=4 \
  trainer.total_epochs=1 \
  trainer.save_freq=$STEPS \
  trainer.val_before_train=false \
  trainer.resume_mode=disable \
  trainer.total_training_steps=$STEPS \
  trainer.project_name=fedagent_persist \
  trainer.experiment_name=persist_c0
echo "wall_end_epoch=$(date +%s) rc=$?"
echo "=== checkpoints produced ==="
ls -d "$OUT"/client_*/checkpoints/global_step_* 2>/dev/null
