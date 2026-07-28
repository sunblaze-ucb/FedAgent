"""AgenticDataset — verl ``custom_cls`` dataset that emits one row per env instance.

verl 0.8 loads this via ``data.custom_cls.{path,name}``; its non-tensor columns are
passed as ``**kwargs`` to ``AgentLoop.run()`` (so ``env_name``/``seed``/``config``/
``max_turns`` reach the agent-loop). This is the verl-0.8 equivalent of verl-agent's
``AgenticDataset`` / ``fed_make_envs`` env enumeration.

Input: an env-spec YAML (``data.train_files``) of the form::

    envs:
      - name: TinyGuess
        n_envs: 64
        max_turns: 6
        agent_name: gym_text     # optional (default: gym_text)
        config: {lo: 1, hi: 50}

Output: ``n_envs`` rows per spec, each a distinct env instance (distinct ``seed``);
with ``FEDAGENT_DATA_EPOCHS=E`` (set by run_fed together with ``trainer.total_epochs=1``),
``E x n_envs`` rows per spec -- one fresh n_envs draw per local epoch, restoring the
original fed sampler's per-epoch resampling (see the inline comment).
GRPO grouping is handled downstream by verl's ``rollout.n`` (each row is repeated n
times -> one GRPO group per env instance).

Stock-trainer contract (validated in Phase 0(b)): stock verl ``_get_gen_batch`` does
NOT pop tensor keys before unioning the agent-loop output back onto the batch, so the
dataset must NOT emit ``input_ids``/``attention_mask``/``position_ids`` (the agent-loop
generates them). We emit a single non-colliding ``ds_dummy`` tensor purely for batch
sizing. (VAGEN can emit dummy input_ids only because it forks ``_get_gen_batch``; we
don't fork the trainer.)

PHASE 4 SEAM: ``_partition_specs`` is where FedAgent heterogeneity plugs in --
``partition_strategy.py`` will turn the global env pool into this client's slice,
driven by env vars (PARTITION_STRATEGY, CLIENT_ID, CLIENT_NUM, OMEGA, SIZE_STD,
SUCCESS_STD, ENV_DIV, HOLDOUT_FILE, ...), preserving deterministic-by-client-id
assignment. For now (non-federated) it is identity.
"""
import os
from typing import Any, Dict, List

import torch
from omegaconf import OmegaConf
from torch.utils.data import Dataset

DEFAULT_AGENT_LOOP = "gym_text"


class AgenticDataset(Dataset):
    def __init__(self, data_files, tokenizer=None, processor=None, config=None, **kwargs):
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config

        specs = self._load_specs(data_files)
        specs = self._partition_specs(specs)

        # base seed for per-instance variety; the federated bridge sets this per
        # client-round (Phase 4/6) so each client sees a distinct, reproducible draw.
        base_seed = int(os.environ.get("FEDAGENT_BASE_SEED", 0))
        # additive per-instance seed shift (default 0 = unchanged). Used by chunked/resumable
        # labeling: the WebShop service maps seed -> goal (goal = VAL_SIZE + seed % train_pool),
        # so shifting every row's seed by K makes a run cover a DISJOINT contiguous goal window
        # goals[VAL_SIZE+K : VAL_SIZE+K+n_envs]. Lets 6410 goals be labeled in resilient chunks.
        seed_offset = int(os.environ.get("FEDAGENT_SEED_OFFSET", 0))

        # Per-epoch goal resampling (paper fidelity; audit data-dataset-seeds-6): the ORIGINAL
        # fed sampler drew a FRESH goal batch at every local epoch (WebShop: persistent
        # RandomState advanced by each envs.reset; ALFWorld: the game iterator advanced), so E
        # epochs/round saw up to E x n_envs distinct tasks. A verl dataset replays IDENTICAL
        # rows every epoch, which silently shrank that to n_envs (same goals x E). run_fed
        # therefore launches clients with trainer.total_epochs=1 + FEDAGENT_DATA_EPOCHS=E, and
        # this dataset emits each spec's rows E times with a DISTINCT per-epoch-slot seed
        # (e * n_envs stride, epoch-major order) -- same optimizer-step count, original
        # diversity restored. FEDAGENT_DATA_EPOCHS_FILE (when set) restricts the expansion to
        # that one spec file: the persistent worker builds its worker-eval dataloader (the val
        # spec) in the SAME process, and the val set must stay exactly n_envs episodes.
        data_epochs = max(1, int(os.environ.get("FEDAGENT_DATA_EPOCHS", 1) or 1))
        epochs_file = os.environ.get("FEDAGENT_DATA_EPOCHS_FILE")
        if data_epochs > 1 and epochs_file:
            my_path = data_files[0] if isinstance(data_files, (list, tuple)) else data_files
            if os.path.realpath(str(my_path)) != os.path.realpath(epochs_file):
                data_epochs = 1

        self.items: List[Dict[str, Any]] = []
        # Seed layout gives each spec a 1000-wide window (si * 1_000 + e*n_envs + i): a non-last
        # spec whose EXPANDED row count (n_envs * data_epochs) exceeds 1000 would silently ALIAS
        # into the next spec's seed range (two rows -> the same env seed -> the same served
        # goal). Refuse loudly.
        for si, s in enumerate(specs[:-1]):
            if int(s.get("n_envs", 64)) * data_epochs > 1000:
                raise ValueError(
                    f"spec {si} ({s.get('name')}) has n_envs={s.get('n_envs')} x "
                    f"FEDAGENT_DATA_EPOCHS={data_epochs} > 1000 rows with later specs present; "
                    f"row seeds would collide across specs (si*1000 + e*n_envs + i layout)")
        for si, s in enumerate(specs):
            n = int(s.get("n_envs", 64))
            max_turns = int(s.get("max_turns", 6))
            env_cfg = s.get("config", {}) or {}
            name = s.get("name", "TinyGuess")
            agent_name = s.get("agent_name", DEFAULT_AGENT_LOOP)
            # `validate: true` in a spec marks its rows as VALIDATION episodes. Stock verl
            # passes row columns -- not trajectory["validate"] -- into AgentLoop.run(), so this
            # column is how the CONCAT loop learns it is scoring a val row (skips the invalid-
            # action penalty; the windowed manager gets `validate` natively). Set in the shared
            # val specs (config/envs/*_val.yaml) only: client in-job validation reuses the TRAIN
            # spec and deliberately keeps train reward semantics.
            is_validation = bool(s.get("validate", False))
            for e in range(data_epochs):        # e=0 with data_epochs=1 == the historical layout
                for i in range(n):
                    self.items.append(
                        {
                            "env_name": name,
                            "seed": base_seed * 100_000 + si * 1_000 + e * n + i + seed_offset,
                            "config": env_cfg,
                            "max_turns": max_turns,
                            "agent_name": agent_name,
                            "is_validation": is_validation,
                            "data_source": name.lower(),
                            # verl's agent-loop postprocess stores kwargs["raw_prompt"]; our
                            # loop builds its own prompt from the env, so this is a placeholder.
                            "raw_prompt": [{"role": "user", "content": f"<{name} episode>"}],
                            # single non-colliding dummy tensor (see module docstring).
                            "ds_dummy": torch.tensor([0]),
                        }
                    )

    @staticmethod
    def _load_specs(data_files) -> List[Dict[str, Any]]:
        path = data_files[0] if isinstance(data_files, (list, tuple)) else data_files
        # No path at all -> the built-in TinyGuess smoke (the genuine "no env-spec" case).
        if not path:
            return [{"name": "TinyGuess", "n_envs": 64, "max_turns": 6, "config": {"lo": 1, "hi": 50}}]
        # A path WAS given: FAIL FAST on a missing file / unparseable YAML / no `envs:` list,
        # rather than silently falling back to the TinyGuess toy env. A misconfigured paper run
        # (typo'd env_spec, broken YAML) must NOT "succeed" against the wrong training objective.
        if not os.path.exists(str(path)):
            raise FileNotFoundError(f"AgenticDataset env-spec not found: {path!r}")
        try:
            cfg = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
        except Exception as e:
            raise ValueError(f"AgenticDataset could not parse env-spec {path!r}: {e}") from e
        specs = cfg.get("envs", []) if isinstance(cfg, dict) else []
        if not specs:
            raise ValueError(
                f"AgenticDataset env-spec {path!r} has no `envs:` list -- refusing to silently "
                f"fall back to TinyGuess (pass an empty data_files for the smoke default)."
            )
        return specs

    def _partition_specs(self, specs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """PHASE 4 hook: apply FedAgent heterogeneity partitioning. Identity for now."""
        return specs

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.items[idx]
