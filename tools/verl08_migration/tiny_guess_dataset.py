"""Phase 0(b) spike: a minimal verl-0.8 custom dataset that emits env-spec rows.

Mirrors VAGEN's AgenticDataset contract: each row carries env metadata + an
`agent_name` (which agent loop drives it) + dummy tensors. verl loads this via
`data.custom_cls.path/name` and passes the non-tensor columns as **kwargs to the
AgentLoop.run().
"""
from typing import Any, Dict

import torch
from omegaconf import OmegaConf
from torch.utils.data import Dataset


class TinyGuessDataset(Dataset):
    def __init__(self, data_files, config: Dict[str, Any], **kwargs):
        df = data_files[0] if isinstance(data_files, (list, tuple)) else data_files
        try:
            cfg = OmegaConf.to_container(OmegaConf.load(df), resolve=True)
            specs = cfg.get("envs", [])
        except Exception:
            specs = []
        if not specs:
            specs = [{"name": "TinyGuess", "n_envs": 64, "max_turns": 6, "config": {"lo": 1, "hi": 50}}]

        base_seed = 0
        try:
            base_seed = int(config.get("base_seed", 0))
        except Exception:
            pass

        self.items = []
        for si, s in enumerate(specs):
            n = int(s.get("n_envs", 64))
            mt = int(s.get("max_turns", 6))
            ec = s.get("config", {}) or {}
            name = s.get("name", "TinyGuess")
            for i in range(n):
                self.items.append(
                    {
                        "env_name": name,
                        "seed": base_seed * 100000 + si * 1000 + i,
                        "config": ec,
                        "max_turns": mt,
                        "agent_name": "gym_text",
                        "data_source": "tiny_guess",
                        # verl's agent-loop postprocess stores kwargs["raw_prompt"]; our loop
                        # builds its own prompt from the env, so this is a placeholder.
                        "raw_prompt": [{"role": "user", "content": "Let's play guess-the-number."}],
                        # Stock verl _get_gen_batch does NOT pop tensor keys, then unions
                        # batch with the agent-loop output. So the dataset must NOT provide
                        # input_ids/attention_mask/position_ids (the loop generates them).
                        # Keep one non-colliding dummy tensor purely for batch sizing.
                        "ds_dummy": torch.tensor([0]),
                    }
                )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]
