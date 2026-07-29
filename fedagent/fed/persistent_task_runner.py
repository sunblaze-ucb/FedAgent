"""PersistentFedTaskRunner -- lever #4 (docs/acceleration.md).

Build RayPPOTrainer + init_workers() ONCE, then loop over a federated PLAN (a list of
per-client specs) calling ``_reset_for_client`` + the stock ``trainer.fit()`` -- instead of
the subprocess-per-client cold-start. Each ``fit()`` already does global_steps=0 ->
update_weights -> [val] -> train -> save (ray_trainer.py:1362), so reusing it per client is
faithful; the reset reproduces what a fresh subprocess gets for free.

Wired via ``run_ppo(config, task_runner_class=ray.remote(...)(PersistentFedTaskRunner))``
(main_ppo.py:52,99-101). The plan is read from the JSON file at $FEDAGENT_PERSISTENT_PLAN:
  [{"client":0,"model_path":...,"critic_path":null,"seed":4200,"out_dir":...,"exp":...}, ...]
All clients of the plan share the SAME architecture (FedAvg requires identical shapes), so the
tokenizer/hf_config built once stay valid; only weights (local_path) + data (seed) change.
"""
import contextlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf, open_dict

from verl.trainer.main_ppo import (
    TaskRunner,
    create_rl_dataset,
    create_rl_sampler,
    need_critic,
    need_reference_policy,
    validate_config,
)
from verl.trainer.ppo.ray_trainer import RayPPOTrainer


@contextlib.contextmanager
def unseeded_eval_data():
    """Build a VALIDATION dataset with FEDAGENT_BASE_SEED cleared (bugfix 2026-07-26).

    ``AgenticDataset`` seeds every row from the process-global ``FEDAGENT_BASE_SEED``
    (agentic_dataset.py:58), and this runner sets that var to the CLIENT's training seed
    (``base_seed + round*100 + client``) before building datasets. The worker-eval val
    dataloader was built inside that window, so its episode seeds inherited the (round,
    client) training seed -- and with a per-round worker process (persistent + cross_round
    off) EVERY ROUND SCORED A DIFFERENT VAL DRAW:

      * ALFWorld: the service maps seed -> ``RandomState(seed).shuffle(gamefiles)[0]``, so 64
        row seeds are 64 draws WITH REPLACEMENT from the 140-game valid_seen split -- a fresh
        ~53-unique multiset per process. Observed in alfworld_ppo_hardness_std1: 48 rounds,
        48 different game sets, and round k's aggregated point measured on a different set
        than round k's client circles (the circles come from process k, the point from
        process k+1, which evals round k's model at its i==0).
      * WebShop was accidentally immune: its val branch is ``seed % VAL_SIZE`` and
        500 | 100_000, so ``base*100_000 + i`` always lands on goals[0:n_envs] regardless of
        base -- which is why only the ALFWorld curve carries the noise.

    The orchestrator's subprocess eval path (run_fed._build_eval) never sets the var, and
    ``config/envs/*_val.yaml`` documents "the same fixed set every round". Clearing it here
    makes eval_mode=worker agree with both. Training is untouched: the var is restored on
    exit, and ``_reset_for_client`` re-sets it per client anyway."""
    saved = os.environ.pop("FEDAGENT_BASE_SEED", None)
    try:
        yield
    finally:
        if saved is not None:
            os.environ["FEDAGENT_BASE_SEED"] = saved


class PersistentFedTaskRunner(TaskRunner):
    """Train N federated clients against ONE persistent trainer (init_workers once)."""

    def run(self, config):
        from verl.utils import hf_processor, hf_tokenizer
        from verl.utils.dataset.rl_dataset import collate_fn
        from verl.utils.fs import copy_to_local

        OmegaConf.resolve(config)
        plan = json.load(open(os.environ["FEDAGENT_PERSISTENT_PLAN"]))
        assert plan, "empty persistent plan"
        print(f"[persistent] plan: {len(plan)} client(s) -> "
              f"{[(s['client'], s['seed']) for s in plan]}", flush=True)

        # --- one-time setup (mirrors stock TaskRunner.run, main_ppo.py:244-312) ----------
        actor_rollout_cls, ray_worker_group_cls = self.add_actor_rollout_worker(config)
        self.add_critic_worker(config)
        self.add_reward_model_resource_pool(config)
        self.add_teacher_model_resource_pool(config)
        self.add_ref_policy_worker(config, actor_rollout_cls)
        validate_config(
            config=config,
            use_reference_policy=need_reference_policy(config),
            use_critic=need_critic(config),
        )

        # seed the FIRST client's env + model BEFORE building dataset/trainer
        # (FEDAGENT_BASE_SEED is read in AgenticDataset.__init__).
        os.environ["FEDAGENT_BASE_SEED"] = str(plan[0]["seed"])
        with open_dict(config):
            config.actor_rollout_ref.model.path = plan[0]["model_path"]
            config.trainer.default_local_dir = plan[0]["out_dir"]
            config.trainer.experiment_name = plan[0]["exp"]

        local_path = copy_to_local(
            config.actor_rollout_ref.model.path,
            use_shm=config.actor_rollout_ref.model.get("use_shm", False),
        )
        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)
        resource_pool_manager = self.init_resource_pool_mgr(config)

        train_dataset = create_rl_dataset(
            config.data.train_files, config.data, tokenizer, processor,
            is_train=True, max_samples=config.data.get("train_max_samples", -1),
        )
        val_dataset = create_rl_dataset(
            config.data.val_files, config.data, tokenizer, processor,
            is_train=False, max_samples=config.data.get("val_max_samples", -1),
        )
        train_sampler = create_rl_sampler(config.data, train_dataset)

        self.trainer = RayPPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=self.role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
        )
        self.trainer.init_workers()  # ONCE: Ray + FSDP + vLLM + kernels (binds reload_client_model)

        # --- eval_mode=worker setup (docs §7.4): build a val dataloader from the UNPERTURBED
        # val_env_spec so THIS hot trainer can eval the merged model itself each round -- no second
        # vLLM, no eval cold-start, no OOM (the answer for a GPU-saturated node). Gated on the env var. -
        self._worker_eval_spec = os.environ.get("FEDAGENT_WORKER_EVAL")
        self._worker_eval_dir = os.environ.get("FEDAGENT_WORKER_EVAL_DIR")
        self._worker_eval_url = os.environ.get("FEDAGENT_WORKER_EVAL_URL")
        # eval cadence: the per-round GLOBAL eval (paper red line, server-aggregated model) runs EVERY
        # round; only the round-0 BASE point is gated by val_before_train. cfg.test_freq is inert in this
        # stack (client jobs pin trainer.test_freq=-1); per-client circle marks need client_end_eval.
        self._worker_vbt = os.environ.get("FEDAGENT_WORKER_EVAL_VBT", "1") == "1"
        # client-end circles: eval EACH client's post-training model on the hot engine after its fit().
        self._worker_client_end_eval = os.environ.get("FEDAGENT_WORKER_CLIENT_END_EVAL", "0") == "1"
        self._worker_val_dl = None
        if self._worker_eval_spec:
            from torchdata.stateful_dataloader import StatefulDataLoader
            # unseeded: the val set is FIXED across rounds/clients, not a per-client draw
            # (see unseeded_eval_data -- this dataset is built once and reused every round).
            with unseeded_eval_data():
                wval = create_rl_dataset(self._worker_eval_spec, config.data, tokenizer, processor,
                                         is_train=False,
                                         max_samples=config.data.get("val_max_samples", -1))
            # honor data.val_batch_size like stock verl (_create_dataloader): only fall back to the
            # whole val set when it's unset. Hardcoding len(wval) would fire ALL val episodes at the
            # env service in one batch -> the connection/VRAM/time storm on full WebShop/ALFWorld.
            val_bs = config.data.get("val_batch_size", None) or len(wval)
            self._worker_val_dl = StatefulDataLoader(
                dataset=wval, batch_size=val_bs, shuffle=False, drop_last=False,
                num_workers=config.data.get("dataloader_num_workers", 8), collate_fn=collate_fn)
            with open_dict(config):
                config.actor_rollout_ref.rollout.val_kwargs.temperature = float(
                    os.environ.get("FEDAGENT_WORKER_EVAL_TEMP", "0.4"))
                config.actor_rollout_ref.rollout.val_kwargs.do_sample = True
            print(f"[persistent] worker-eval armed: {self._worker_eval_spec} ({len(wval)} samples) "
                  f"-> reuse the hot engine each round (no eval cold-start)", flush=True)

        # --- cross-round outer loop (lever #4 extended, docs §7.2) -----------------------
        # cross_round=off: run this round's plan once, return (process exits -> next round is a
        # fresh process). cross_round=on: after each round, signal the orchestrator (which runs
        # the SAME external FedAvg/merge), then wait for the next round's merged-model plan and
        # keep going IN THE SAME PROCESS -- paying the cold-start once for the whole run.
        cross_round = os.environ.get("FEDAGENT_CROSS_ROUND") == "1"
        xdir = Path(os.environ["FEDAGENT_XROUND_DIR"]) if cross_round else None
        r = int(os.environ.get("FEDAGENT_XROUND_START_ROUND", "1"))
        first_ever = True
        while True:
            # --- per-client loop: the whole point of lever #4 ---------------------------
            for i, spec in enumerate(plan):
                if not (first_ever and i == 0):
                    self._reset_for_client(spec)  # very first client already configured above
                # worker-eval: at i==0 the round's STARTING model (base for r=1, else model_{r-1}
                # merged) is loaded -> eval it on the hot engine, label round r-1, BEFORE training.
                # Gated on the orchestrator's cadence so worker matches inline/parallel/shared.
                if i == 0 and self._worker_val_dl is not None and self._should_worker_eval(r - 1):
                    self._worker_validate(r - 1)
                # final_eval_mode=worker (Tier-2c): an eval-only plan carries the FINAL aggregated
                # model as round T+1's "starting model" -- the worker-eval above just scored it on
                # the hot engine (label r-1 == T, dump round_T/eval/val_samples); there is nothing
                # to train, so skip fit()/routing/client-end for this pseudo-spec.
                if spec.get("eval_only"):
                    print(f"[persistent] eval-only plan: scored round {r - 1} model on the hot "
                          f"engine; no fit", flush=True)
                    continue
                self._route_service(spec)         # point shared workers at THIS client's env service
                print(f"[persistent] >>> round {r} client {spec['client']} (idx {i}) fit() -> "
                      f"{spec['out_dir']}", flush=True)
                self.trainer.fit()
                # fit() EXITS with the rollout AWAKE: its last op is checkpoint_manager.update_weights
                # (ray_trainer.py:1675; the loop's last sleep_replicas was at 1471, BEFORE that sync).
                # Every later update_weights (client-end _worker_validate below, next round's
                # round-level eval) assumes verl's precondition "rollout asleep before update_weights"
                # (engine_workers.py:672). On an awake engine the two resume() wakes are no-ops
                # ("Executor is not sleeping" x2) and the FSDP full_tensor() gather runs with the
                # whole gpu_memory_utilization pool still resident -> weight-sync OOM after a
                # headroom-dependent number of rounds (446MiB bf16 embed cast; GRPO r11 / PPO r8,
                # both at awake-sync #17 == fits-in-process + 1). Re-sleep here (level 2: weights+kv
                # released; the next update_weights re-syncs everything -- verl's own per-step
                # sleep(1471)->update(1675) cycle). Never a no-op: fit() always exits awake.
                cm = getattr(self.trainer, "checkpoint_manager", None)
                if cm is not None:
                    cm.sleep_replicas()
                print(f"[persistent] <<< round {r} client {spec['client']} done", flush=True)
                # client-end circle: score THIS client's just-trained model on the val service using the
                # hot engine (no second vLLM). _worker_validate routes to the val service + syncs the
                # current (client-trained) weights; the next client's reset/route/fit restores training.
                if self._worker_client_end_eval and self._worker_val_dl is not None:
                    self._worker_validate(r, client_id=spec["client"])
            first_ever = False
            if not cross_round:
                return
            # signal the orchestrator that round r's checkpoints are saved, then wait for either
            # the next round's plan (merged model) or STOP. The worker idles here (holding GPUs)
            # while FedAvg/merge/eval run -- they coexist (separate NCCL world, ample VRAM).
            (xdir / f"done_{r}").write_text("ok")
            print(f"[persistent] round {r} done -> signalled; waiting for round {r + 1} / stop",
                  flush=True)
            plan = self._wait_next_round(xdir, r)
            if plan is None:
                print("[persistent] STOP received; exiting cross-round loop", flush=True)
                return
            r += 1

    @staticmethod
    def _wait_next_round(xdir: Path, r: int, poll_s: float = 2.0):
        """Block until the orchestrator publishes round r+1's plan (-> load + return it) or STOP
        (-> return None). File handshake (GPFS-safe: go_{r+1} is touched only AFTER the plan is
        fully written)."""
        go, stop, plan_f = xdir / f"go_{r + 1}", xdir / "stop", xdir / f"plan_round_{r + 1}.json"
        while True:
            if stop.exists():
                return None
            if go.exists() and plan_f.exists():
                return json.load(open(plan_f))
            time.sleep(poll_s)

    def _reset_for_client(self, spec):
        """Reproduce, per client, everything the subprocess-per-client path got for free."""
        from verl.utils.fs import copy_to_local

        t = self.trainer
        cfg = t.config
        # (e) re-point output dir / experiment name / model path
        with open_dict(cfg):
            cfg.trainer.default_local_dir = spec["out_dir"]
            cfg.trainer.experiment_name = spec["exp"]
            cfg.actor_rollout_ref.model.path = spec["model_path"]

        # (b) rebuild dataloader for this client's seed (read in AgenticDataset.__init__).
        # BEFORE the weight reload so the fresh LR scheduler sees the right total_training_steps
        # (ray_trainer.py:438-452); harmless for constant-LR (paper) schedules.
        os.environ["FEDAGENT_BASE_SEED"] = str(spec["seed"])
        t._create_dataloader(None, None, None, None)

        # (a)+(c)+(FedProx) reload weights + fresh optimizer/scheduler + drop FedProx anchor.
        # hf_export=final (Tier-2d): model_path is the aggregated FSDP SHARD dir (no per-round HF
        # merge) -> rebuild the engine from the BASE model (fresh optimizer/scheduler as always),
        # then overwrite the weights in place from the shards (model-only load).
        mp = str(spec["model_path"])
        shard_reload = os.path.isdir(mp) and bool(list(Path(mp).glob("model_world_size_*_rank_*.pt")))
        if shard_reload:
            base_local = copy_to_local(spec["base_model"])
            with open_dict(cfg):
                cfg.actor_rollout_ref.model.path = spec["base_model"]  # initialize() reads this
            t.actor_rollout_wg.reload_client_model(base_local, shard_dir=mp)
        else:
            actor_local = copy_to_local(mp)
            t.actor_rollout_wg.reload_client_model(actor_local)
        # PPO/gae: reload the federated critic too (fresh value weights + fresh critic optimizer)
        if getattr(t, "use_critic", False) and spec.get("critic_path"):
            cp = str(spec["critic_path"])
            critic_shards = os.path.isdir(cp) and bool(list(Path(cp).glob("model_world_size_*_rank_*.pt")))
            if critic_shards:
                t.critic_wg.reload_critic_model(copy_to_local(spec["base_model"]), shard_dir=cp)
            else:
                t.critic_wg.reload_critic_model(copy_to_local(cp))

        # (d) deterministic driver-side RNG (advantage/uuid) + GPU hygiene (audit #14)
        seed = int(spec["seed"])
        random.seed(seed)
        np.random.seed(seed % (2**32))
        torch.manual_seed(seed)
        torch.cuda.empty_cache()
        print(f"[persistent] reset client {spec['client']}: model={spec['model_path']} seed={seed}",
              flush=True)

    @staticmethod
    def _route_service(spec):
        """Per-client env-service routing (webshop/alfworld). Rewrite FEDAGENT_SERVICE_URL_FILE with
        this client's service URL BEFORE its fit(), so the SHARED agent-loop workers (one process for
        all clients) build envs that hit the right per-client service -- which process-env routing
        can't do within one process. No-op for in-process envs (tinyguess): the plan carries no
        service_url, or FEDAGENT_SERVICE_URL_FILE is unset (subprocess path)."""
        url = spec.get("service_url")
        url_file = os.environ.get("FEDAGENT_SERVICE_URL_FILE")
        if url and url_file:
            Path(url_file).write_text(url)
            print(f"[persistent] route client {spec['client']} -> {url}", flush=True)

    def _should_worker_eval(self, eval_round: int) -> bool:
        """The per-round GLOBAL eval (paper red line, server-aggregated model) runs EVERY round; only
        the round-0 BASE point is gated by val_before_train. test_freq is inert in this stack (client
        jobs pin trainer.test_freq=-1) -- don't gate on it. The worker evals the
        STARTING model of each round (round r-1 at round r), covering 0..T-1; the FINAL round is evaled
        by the orchestrator after the worker stops."""
        return self._worker_vbt if eval_round == 0 else True

    def _worker_validate(self, eval_round: int, client_id=None) -> None:
        """eval_mode=worker: score the ALREADY-LOADED model on the unperturbed val set using verl's
        ``_validate`` + the HOT rollout engine (no second vLLM -> no OOM, no eval cold-start). Dumps
        val_samples in eval_global's layout so the orchestrator reads it the same way
        (summarize_val_dump). Routes the env to the val service for the pass, then training reroutes."""
        t = self.trainer
        rdir = Path(self._worker_eval_dir) / (f"round_{eval_round}" if eval_round > 0 else "round_0")
        # client_id set => client-end circle (this client's post-training model) -> client_<c>/eval;
        # else the aggregated round point -> round_<r>/eval.
        dump = (rdir / f"client_{client_id}" if client_id is not None else rdir) / "eval" / "val_samples"
        dump.mkdir(parents=True, exist_ok=True)
        url_file = os.environ.get("FEDAGENT_SERVICE_URL_FILE")
        if url_file and self._worker_eval_url:                  # eval hits the VAL service, not a client's
            Path(url_file).write_text(self._worker_eval_url)
        saved_dl, saved_dump = t.val_dataloader, t.config.trainer.get("validation_data_dir", None)
        t.val_dataloader = self._worker_val_dl
        with open_dict(t.config):
            t.config.trainer.validation_data_dir = str(dump)
        _what = "client-end" if client_id is not None else "worker"
        _who = f" client {client_id}" if client_id is not None else ""
        print(f"[persistent] {_what}-eval round {eval_round}{_who} on the hot engine -> {dump}", flush=True)
        # verl's _validate() reads self.global_steps for the logged step label; only fit() sets it, and
        # the FIRST worker-eval (round r=1, i==0) runs BEFORE any fit() -> seed it when missing. fit()
        # resets global_steps at its own start, so this never leaks into training; the dump path is
        # overridden above (validation_data_dir) so the label value doesn't affect what we parse.
        if not hasattr(t, "global_steps"):
            t.global_steps = 0
        # re-init the dump executor if a prior fit() shut it down -- exactly what verl's own fit() does
        # (ray_trainer.py:1369-1370). Each fit() calls _shutdown_dump_executor at its end (1770), so by
        # the next round's worker-eval the executor is dead and _validate()'s _dump_generations would
        # raise "cannot schedule new futures after shutdown". (No-op on the first eval: still alive.)
        dex = getattr(t, "_dump_executor", None)
        if dex is not None and dex._shutdown:
            t._init_dump_executor()
        # CRITICAL: mirror fit()'s pre-validate engine prep (ray_trainer.py:1386-1387). verl inits the
        # vLLM rollout with DUMMY weights and leaves the replicas ASLEEP at the end of init_workers
        # (ray_trainer.py:972); the real weights are synced from FSDP by checkpoint_manager.update_weights
        # at each rollout. The worker-eval runs BEFORE this round's fit(), so without the sync vLLM still
        # holds dummy weights -> CUDA illegal-memory-access / invalid-argument (EngineDeadError). The engine
        # is asleep here (after init_workers, after the previous _worker_validate's finally, or after
        # run()'s post-fit re-sleep -- NOT fit() itself: fit() exits AWAKE, its last op is the 1675
        # update_weights; relying on "fit's last sleep_replicas" was the r11/r8 weight-sync OOM), so
        # the update_weights precondition (rollout asleep) holds. _validate() leaves it AWAKE, so
        # re-sleep in finally to restore the state fit()'s own update_weights (1387) assumes.
        cm = getattr(t, "checkpoint_manager", None)
        if cm is not None:
            cm.update_weights(t.global_steps)
        try:
            t._validate()
        finally:
            if cm is not None:
                cm.sleep_replicas()
            t.val_dataloader = saved_dl
            with open_dict(t.config):
                t.config.trainer.validation_data_dir = saved_dump
