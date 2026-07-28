"""Per-turn manager/worker for the WINDOWED (faithful) rollout mode.

verl 0.8's stock agent-loop path HARD-ENFORCES "1 training sample per input prompt":
fit() does `combined_gen_output.slice(0, num_sampled_prompts)` (truncate) + `batch.repeat(n)
.union(gen_out)` (equal-size assert); _validate() does `test_batch.union(test_output_gen_batch)`;
and `make_iterator` asserts `batch_size % mini_batch_size == 0` (UNCONDITIONAL — use_dynamic_bsz
relaxes only the MICRO split, never this one). The windowed/faithful mode trains on K per-turn
samples PER episode, so a naive expansion is silently TRUNCATED in train (corrupt) and CRASHES
eval. There is no native multi-sample/step-level support in verl 0.8.

verl-agent (the paper code) solved the identical problem by forking ray_trainer: it does
`del batch; batch = gen_batch_output` (use the per-turn batch directly) then `adjust_batch(...)`
(pad the dynamic batch up to lcm(micro*n_gpus, ppo_mini) by duplicating rows). We reproduce that
WITHOUT forking verl, via a tag-gated scoped monkeypatch:

  * WindowedAgentLoopWorker.run -> per-episode List[AgentLoopOutput] (one per turn); reuse the
    STOCK per-output _agent_loop_postprocess on each turn; _postprocess flattens + np.repeats the
    per-row non_tensor (uid/raw_prompt/index/data_source) to per-turn so GRPO groups by `uid`
    (task) and the per-turn `traj_uid` distinguishes trajectories (for the opt-in grpo_traj).
  * For EVAL (validate=True) each episode is COLLAPSED to 1 row (the last turn carries the
    broadcast episode return + traj_success) so eval stays strictly 1:1 — _validate's pad/unpad
    + union are then untouched (the metric is per-episode success, which is what eval needs).
  * TRAIN outputs are tagged `meta_info["__windowed_expanded__"] = size_divisor`. The monkeypatch
    then (a) makes `DataProto.slice(0, k)` a no-op on a tagged batch (no truncation), and
    (b) makes `batch.union(tagged_other_of_different_len)` ADOPT `other` (it is self-sufficient:
    `_get_gen_batch` re-adds {data_source,reward_model,extra_info,uid} so np.repeat carries every
    non_tensor per-turn) and pad it to a multiple of `size_divisor` (== verl-agent adjust_batch).
    Both are gated on the tag + a `len(self)!=len(other)` guard, so matched-size unions, ALL eval,
    REMAX's start!=0 baseline slice, and downstream mini-batch slicing are byte-for-byte original.

Select via rollout_mode=windowed ->
    actor_rollout_ref.rollout.agent.agent_loop_manager_class=fedagent.agent_loops.windowed_manager.WindowedAgentLoopManager
The stock full-concat manager/worker are untouched (the other, opt-in mode). NOTE: windowed mode
is incompatible with adv_estimator=REMAX (its second-slice baseline path); use grpo/ppo.
"""
from __future__ import annotations

import numpy as np
import ray

from verl.experimental.agent_loop.agent_loop import AgentLoopManager, AgentLoopWorker
from verl.protocol import DataProto

_TAG = "__windowed_expanded__"


def _compute_size_divisor(config) -> int:
    """LCM of every batch-size constraint the expanded per-turn batch must satisfy so verl's
    DP-chunk + mini-batch `% == 0` asserts all pass. Mirrors verl-agent adjust_batch:
    lcm(micro*n_gpus, ppo_mini). The mini split uses ppo_mini_batch_size*rollout.n samples
    (ray_trainer.py:1311). use_dynamic_bsz relaxes only the MICRO split, so its micro term is
    dropped when enabled."""
    arr = config.actor_rollout_ref
    n = int(arr.rollout.n)
    nnodes = int(config.trainer.get("nnodes", 1) or 1)
    ws = int(config.trainer.n_gpus_per_node) * nnodes
    divs = [int(arr.actor.ppo_mini_batch_size) * n, ws]
    if not bool(arr.actor.get("use_dynamic_bsz", False)):
        amicro = arr.actor.get("ppo_micro_batch_size_per_gpu", None)
        if amicro:
            divs.append(int(amicro) * ws)
    if not bool(arr.rollout.get("log_prob_use_dynamic_bsz", False)):
        lmicro = arr.rollout.get("log_prob_micro_batch_size_per_gpu", None)
        if lmicro:
            divs.append(int(lmicro) * ws)
    # critic split too (only when adv_estimator uses a value function, e.g. gae/PPO; ray_trainer.py:1339)
    algo = config.get("algorithm", None)
    adv = str(algo.get("adv_estimator", "grpo")) if algo is not None else "grpo"
    crit = config.get("critic", None)
    if adv == "gae" and crit is not None:
        cmini = crit.get("ppo_mini_batch_size", None)
        if cmini:
            divs.append(int(cmini) * n)
        if not bool(crit.get("use_dynamic_bsz", False)):
            cmicro = crit.get("ppo_micro_batch_size_per_gpu", None)
            if cmicro:
                divs.append(int(cmicro) * ws)
    return int(np.lcm.reduce(np.array([d for d in divs if d > 0], dtype=np.int64)))


def _adjust_to_divisor(data: DataProto, divisor: int) -> DataProto:
    """Pad the dynamic per-turn batch up to a multiple of `divisor` by duplicating rows
    (== verl-agent adjust_batch mode='copy'). Deterministic dup-indices (reproducible). Dup'd
    rows keep their uid, so GRPO grouping is unaffected — they only slightly reweight their
    group's mean/std (the same accepted approximation the paper used).

    Dup indices are EVENLY SPACED over the batch (2026-07-28): the old head slice
    (``arange(to_add) % bs``) always cloned the FIRST rows, and the per-turn batch is
    episode-major, so every training step systematically over-weighted episode 0's turns.
    Spacing keeps determinism (no RNG) without the head bias. (Executed paper runs used the
    head slice; the reweight is a few rows of one batch, so curves are comparable.)"""
    bs = len(data)
    if divisor <= 1 or bs == 0 or bs % divisor == 0:
        return data
    to_add = divisor - (bs % divisor)
    dup_idx = np.linspace(0, bs - 1, num=to_add, dtype=np.int64)   # repeats if to_add > bs
    dup = data.select_idxs(dup_idx)
    return DataProto.concat([data, dup])


# ---- scoped monkeypatch (applied once at import; this module is imported only when
#      rollout_mode=windowed selects WindowedAgentLoopManager) ----------------------------------
if not getattr(DataProto, "_windowed_patched", False):
    _orig_slice = DataProto.slice
    _orig_union = DataProto.union

    def _windowed_slice(self, start=None, end=None, step=None):
        mi = getattr(self, "meta_info", None)
        # neutralize ONLY fit()'s `combined_gen_output.slice(0, num_sampled_prompts)` truncation
        # of the tagged TRAIN expansion (from-start, contiguous, shorter than the real length).
        if (mi and mi.get(_TAG) and start in (0, None) and step in (None, 1)
                and end is not None and end < len(self)):
            return self
        return _orig_slice(self, start=start, end=end, step=step)

    def _windowed_union(self, other):
        omi = getattr(other, "meta_info", None)
        # neutralize fit()'s `batch.repeat(n).union(gen_out)` when `other` is the tagged,
        # self-sufficient per-turn batch of a DIFFERENT row count: adopt it + pad to the divisor.
        #
        # The tag is consumed WHENEVER it is present -- including the equal-length case (which
        # happens iff every episode of the batch produced exactly one turn). There the stock
        # union runs and MERGES meta_info, so a surviving tag would ride onto the merged TRAINING
        # batch, and _windowed_slice above would then silently no-op every later legitimate
        # `slice(0, k<len)` on it (REMAX's baseline slice, any future trainer slicing). Popping
        # first keeps the tag's lifetime exactly one union, which is all it is for.
        if omi and omi.get(_TAG):
            divisor = int(omi.pop(_TAG))
            if len(self) != len(other):
                adjusted = _adjust_to_divisor(other, divisor)
                # the stock union MERGES meta_info; we adopt other's ROWS but must keep self's meta
                # keys (e.g. 'temperature', set at ray_trainer.py:1436 and required by compute_log_prob,
                # plus global_steps etc.). other's keys (timing) win on the (equal-valued) overlap.
                adjusted.meta_info = {**(self.meta_info or {}), **(adjusted.meta_info or {})}
                print(f"[windowed] train batch: adopted {len(other)} per-turn rows "
                      f"(stock would have truncated to {len(self)}) -> padded to {len(adjusted)} "
                      f"(divisor {divisor})", flush=True)
                return adjusted
        return _orig_union(self, other)

    DataProto.slice = _windowed_slice
    DataProto.union = _windowed_union
    DataProto._windowed_patched = True


class WindowedAgentLoopWorker(AgentLoopWorker):
    """Worker that expands each episode into its per-turn samples (windowed mode)."""

    async def _run_agent_loop(self, sampling_params, trajectory, *, agent_name, trace=True, **kwargs):
        import hydra
        from verl.experimental.agent_loop.agent_loop import (
            _agent_loop_registry, DictConfigWrap, ToolListWrap,
        )
        from verl.utils.rollout_trace import rollout_trace_attr

        with rollout_trace_attr(step=trajectory["step"], sample_index=trajectory["sample_index"],
                                rollout_n=trajectory["rollout_n"], validate=trajectory["validate"],
                                name="agent_loop", trace=trace):
            # windowed manager -> use the WINDOWED variant of the agent loop. The env spec carries
            # agent_name=gym_text (shared with concat); gym_text_windowed is the subclass that adds
            # run_episode_windowed. Auto-map gym_text -> gym_text_windowed so no per-mode agent_name
            # / config change is needed; fall back to agent_name if no _windowed variant exists (then
            # run_episode_windowed must be on it, else it errors as before -- the release blocker).
            _wname = agent_name if str(agent_name).endswith("_windowed") else f"{agent_name}_windowed"
            _wcfg = _agent_loop_registry[_wname] if _wname in _agent_loop_registry \
                else _agent_loop_registry[agent_name]
            agent_loop = hydra.utils.instantiate(
                config=_wcfg,
                trainer_config=DictConfigWrap(config=self.config),
                server_manager=self.llm_client, tokenizer=self.tokenizer, processor=self.processor,
                dataset_cls=self.dataset_cls, data_config=DictConfigWrap(self.config.data),
                tools=ToolListWrap(self.tools),
            )
            turn_outputs = await agent_loop.run_episode_windowed(
                sampling_params, validate=trajectory["validate"], **kwargs)
            if trajectory["validate"] and turn_outputs:
                # EVAL: collapse episode -> 1 row. The metric is per-episode success; reward_score
                # is already the broadcast episode return and traj_success is in reward_extra_info,
                # both present on every turn -> taking the last turn suffices. Keeps eval 1:1 so
                # _validate's pad/unpad + union are untouched (no expansion, never tagged below).
                last = turn_outputs[-1]
                last.num_turns = len(turn_outputs)
                turn_outputs = [last]
            # reuse the STOCK per-output postprocess (correct padding/masks/position_ids) per turn
            return [await self._agent_loop_postprocess(o, trajectory["validate"], **kwargs)
                    for o in turn_outputs]

    def _postprocess(self, inputs, input_non_tensor_batch=None, validate=False):
        # inputs is a list of per-episode lists -> flatten, and expand the per-episode
        # non_tensor rows to per-turn so uid/raw_prompt/index stay aligned with samples.
        flat, repeats = [], []
        for ep in inputs:
            ep_list = ep if isinstance(ep, list) else [ep]
            flat.extend(ep_list)
            repeats.append(len(ep_list))
        expanded_nt = None
        if input_non_tensor_batch is not None:
            expanded_nt = {k: np.repeat(v, repeats, axis=0) for k, v in input_non_tensor_batch.items()}
        out = super()._postprocess(flat, input_non_tensor_batch=expanded_nt, validate=validate)
        # The stock _postprocess only folds input_non_tensor_batch into the output when there are
        # NO streaming reward-loop workers (agent_loop.py: `if reward_loop_worker_handles is None`).
        # The concat path doesn't care (it gets uid/data_source/index from `self` in batch.union),
        # but the windowed path ADOPTS this batch wholesale, so it must carry them itself. Force-add
        # any per-turn non_tensor key the stock path skipped (sizes match: len(flat)==sum(repeats)).
        # Required by compute_advantage (groups by `uid`).
        if expanded_nt:
            for k, v in expanded_nt.items():
                if k not in out.non_tensor_batch:
                    out.non_tensor_batch[k] = v
        if not validate:
            # tag the TRAIN expansion (carries the pad divisor); eval stays 1:1 and untagged.
            out.meta_info[_TAG] = _compute_size_divisor(self.config)
        return out


class WindowedAgentLoopManager(AgentLoopManager):
    """Manager that uses the per-turn windowed worker. Select via
    ``actor_rollout_ref.rollout.agent.agent_loop_manager_class=fedagent.agent_loops.windowed_manager.WindowedAgentLoopManager``."""

    def __init__(self, *args, **kwargs):
        # mirror verl's pattern: set the worker class BEFORE super().__init__
        self.agent_loop_workers_class = ray.remote(WindowedAgentLoopWorker)
        super().__init__(*args, **kwargs)
