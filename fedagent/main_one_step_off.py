"""FedAgent entry for verl's EXPERIMENTAL one_step_off_policy (ADDITIONAL OPTION).

    python -m fedagent.main_one_step_off <hydra overrides...>

Same thin-overlay shape as ``fedagent.main_ppo_fed`` (stock verl, no fork), but the task
runner is upstream's ``OneStepTaskRunner``: rollout workers live on a dedicated GPU split
(``rollout.n_gpus_per_node``) and generate batch t+1 WHILE batch t trains on the remaining
GPUs -- step wall goes from ``gen + train`` to ``max(gen, train)``.

>>> OFF-POLICY by one step (the training batch was sampled from the previous weights):
>>> outside the paper-reproduction bar. Wired as run_fed's ``one_step_off: true`` for the
>>> SUBPROCESS client path only, as a timing probe / sign-off-gated new-experiments track.
"""
import hydra

# Import the agent-loop module so its @register("gym_text") runs in this process too.
from fedagent.agent_loops import gym_text_agent_loop  # noqa: F401
from verl.experimental.one_step_off_policy.main_ppo import OneStepTaskRunner
from verl.trainer.main_ppo import run_ppo


@hydra.main(config_path="config", config_name="fedagent_one_step_off", version_base=None)
def main(config):
    # Upstream keeps the gen-split resources under top-level `rollout:` and copies them into
    # actor_rollout_ref.rollout in ITS main() (not in OneStepTaskRunner) -- mirror that here,
    # or the standalone server manager sees rollout.nnodes=0 and asserts.
    config.actor_rollout_ref.rollout.nnodes = config.rollout.nnodes
    config.actor_rollout_ref.rollout.n_gpus_per_node = config.rollout.n_gpus_per_node
    run_ppo(config, task_runner_class=OneStepTaskRunner)


if __name__ == "__main__":
    main()
