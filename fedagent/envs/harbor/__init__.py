"""Harbor environment -- Harbor tasks as FedAgent episodes, in-process.

- ``harbor_env.py``  the trainer-side ``HarborEnv`` (BaseTextEnv): picks a task per seed and
                     drives one Harbor ``Trial`` per episode.
- ``bridge.py``      ``BridgeAgent``, the Harbor external agent inside that trial; loaded by
                     Harbor's agent factory, never imported from here.

Only ``HarborEnv`` is re-exported, and it imports ``harbor`` lazily (at the first ``reset``), so
``import fedagent.envs`` keeps working in a trainer env without Harbor installed.
"""
from fedagent.envs.harbor.harbor_env import HarborEnv

__all__ = ["HarborEnv"]
