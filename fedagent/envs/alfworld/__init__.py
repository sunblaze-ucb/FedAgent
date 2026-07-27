"""ALFWorld environment — two halves of one env.

- ``alfworld_env.py``  the trainer-side ``AlfworldEnv`` (BaseTextEnv): a thin async HTTP
                       client, imported in the trainer env (``fedagent-verl08``).
- ``service/``         the out-of-process backend (FastAPI) that holds the real
                       ALFWorld/TextWorld env; runs in the ``verl-agent-alfworld`` conda
                       env and is **never** imported trainer-side.

- ``game_manifest.py``  the authoritative per-split game list (stdlib only): read by the
                       service's engine and by ``tools/gen_alfworld_manifest.py``.

Only the client is re-exported here, and **lazily**, so ``import fedagent.envs.alfworld``
pulls nothing — never ALFWorld's conflicting deps (they live behind the HTTP boundary in
``service/``), and not even the client's ``httpx``. That matters because ``game_manifest`` is
deliberately dependency-free: the generator has to run wherever the DATA is, which is not
necessarily a machine with the trainer env installed. Eagerly importing the client here would
have made a stdlib-only module require httpx for nothing.
"""
from typing import TYPE_CHECKING

__all__ = ["AlfworldEnv"]

if TYPE_CHECKING:  # keep the symbol visible to type checkers / IDEs
    from fedagent.envs.alfworld.alfworld_env import AlfworldEnv


def __getattr__(name):  # PEP 562 lazy re-export
    if name == "AlfworldEnv":
        from fedagent.envs.alfworld.alfworld_env import AlfworldEnv
        return AlfworldEnv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
