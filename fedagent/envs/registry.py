"""Environment registry: ``env_name`` (carried on each dataset row) -> env class.

The agent-loop looks up the env class by the ``env_name`` of the dataset row it is
driving. New envs register here: ``TinyGuess`` (in-process), ``WebShop`` / ``ALFWorld``
(thin HTTP clients to their per-env ``service/`` backends), ``Harbor`` (in-process; one
Harbor trial per episode).
"""
from typing import Any, Dict, Optional, Type

from fedagent.envs.alfworld import AlfworldEnv
from fedagent.envs.base import BaseTextEnv
from fedagent.envs.harbor import HarborEnv
from fedagent.envs.tiny_guess import TinyGuessEnv
from fedagent.envs.webshop import WebShopEnv

ENV_REGISTRY: Dict[str, Type[BaseTextEnv]] = {
    "TinyGuess": TinyGuessEnv,
    "WebShop": WebShopEnv,  # HTTP client -> envs/webshop/service (verl-agent-webshop env)
    "ALFWorld": AlfworldEnv,  # HTTP client -> envs/alfworld/service (verl-agent-alfworld env)
    "Harbor": HarborEnv,  # in-process; imports harbor lazily (pip install harbor in the trainer env)
}


def make_env(env_name: str, env_config: Optional[Dict[str, Any]] = None) -> BaseTextEnv:
    if env_name not in ENV_REGISTRY:
        raise KeyError(f"Unknown env '{env_name}'. Registered: {sorted(ENV_REGISTRY)}")
    return ENV_REGISTRY[env_name](env_config or {})
