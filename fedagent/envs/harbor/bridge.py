"""The Harbor half of ``HarborEnv``: an external agent whose turn loop is driven from outside.

A Harbor ``Trial`` owns the episode lifecycle (start the sandbox -> ``agent.run()`` -> task
verifier -> teardown) and expects the AGENT to own the turn loop. FedAgent's agent-loop owns the
turn loop instead: it has to generate every token on the policy being trained, which is where the
exact token ids / log-probs / response mask come from. This agent inverts control so both hold:
``run()`` publishes the instruction, then executes whatever command ``HarborEnv.step()`` sends
until it is told the episode is over. Returning from ``run()`` hands control back to Harbor, which
runs the task's verifier and tears the sandbox down -- the unmodified Harbor trial path.

Commands are executed HERE, inside ``run()``, rather than from the env's own task: the trial wraps
``run()`` in the agent-phase context (default user, exec env, network policy), and only code
running inside it is guaranteed to see that context.

Imported only by Harbor's agent factory (``AgentConfig.import_path``), so the module-level
``import harbor`` never reaches a trainer env that does not have Harbor installed.
"""
import asyncio
from typing import Optional, Tuple

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

# (command, timeout_sec) to execute, or None to end the episode.
Request = Optional[Tuple[str, int]]


class BridgeAgent(BaseAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Built inside Trial.create(), i.e. on the event loop that will run both this agent and
        # the HarborEnv driving it (one loop per agent-loop worker process).
        self.to_agent: "asyncio.Queue[Request]" = asyncio.Queue()
        self.to_env: asyncio.Queue = asyncio.Queue()

    @staticmethod
    def name() -> str:
        return "fedagent-bridge"

    def version(self) -> str:
        return "1"

    async def setup(self, environment: BaseEnvironment) -> None:
        return None   # nothing to install: the "agent" is the trainer's policy, outside the sandbox

    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        await self.to_env.put(("ready", instruction))
        while True:
            request = await self.to_agent.get()
            if request is None:
                return
            command, timeout_sec = request
            try:
                result = await environment.exec(command, timeout_sec=timeout_sec)
            except Exception as exc:   # exec backstop timeout / dead container: report, keep serving
                await self.to_env.put(("error", exc))
            else:
                await self.to_env.put(("result", result))
