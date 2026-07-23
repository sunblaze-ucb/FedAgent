# verl local patches

FedAgent runs **stock verl 0.8 as a thin overlay, no fork**; the `others/verl` checkout stays
pristine upstream. The one deliberate exception is captured here as a patch so the change is
reproducible without forking verl. Apply it into the editable `others/verl` checkout after env setup.

## `verl_weight_transfer_jobid.patch`

- **Base:** verl commit `7aed6b2` (`others/verl`), files
  `verl/workers/rollout/vllm_rollout/{vllm_rollout.py, vllm_async_server.py}`, **2 lines**.
- **Why:** verl derives the FSDP→vLLM weight-transfer **ZMQ IPC socket** path from the Ray job id
  (`ipc:///tmp/rl-colocate-zmq-<job_id>-replica-<r>-rank-<lr>.sock`) *specifically* to keep concurrent
  jobs disjoint. But FedAgent runs each client/eval as its **own isolated Ray cluster** (`RAY_TMPDIR`),
  and every fresh cluster assigns the **same first job id `01000000`**, so concurrent clients/eval on
  one node compute the **same** `/tmp` socket path and the weight sync **deadlocks** (GPU-confirmed:
  two trainers hung 44 min at 0 % util in `update_weights`). See
  [archive acceleration.md §Lever #3 / §7.7](https://github.com/sunblaze-ucb/FedAgent/tree/migrate/verl-0.8.0/fedagent/docs/acceleration.md)
  (a `migrate/verl-0.8.0` branch archive, not shipped in this tree).
- **What it does:** makes both the sender (`vllm_rollout.py`) and the receiver-source
  (`vllm_async_server.py`) **honor a driver-supplied `VERL_RAY_JOB_ID` override** instead of the
  colliding job id. `fedagent/fed/run_fed.py` sets that env var **uniquely per launched verl
  subprocess** (`_RUN_TAG` uuid + role/client/round). Stock single-cluster runs leave the override
  unset → fall back to the real Ray job id → **byte-for-byte unchanged**.

### Apply

```bash
# from the editable verl checkout root (others/verl)
VERL_ROOT=$(python -c 'import verl, os; print(os.path.dirname(os.path.dirname(verl.__file__)))')
cd "$VERL_ROOT"
git apply /path/to/fedagent/tools/setup/patches/verl_weight_transfer_jobid.patch
# verify:
grep -n 'VERL_RAY_JOB_ID' verl/workers/rollout/vllm_rollout/vllm_rollout.py
```

Without the patch, the `run_fed`-side `VERL_RAY_JOB_ID` injection is simply ignored (verl keeps using
`get_job_id()`), so single-job runs still work and only **concurrent same-node** jobs (client-parallel
#3, eval∥train) risk the deadlock.
