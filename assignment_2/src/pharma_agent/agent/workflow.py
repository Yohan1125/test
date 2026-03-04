"""
High-level agent workflow: plan → retrieve → synthesize.

AgentWorkflow is the main entry point for the agentic pipeline.
"""

from __future__ import annotations

import logging
import os

from openai import OpenAI

from pharma_agent.agent.base import AgentStep, BaseAgent
from pharma_agent.agent.tools import retrieve_context

logger = logging.getLogger(__name__)


class OpenAIAgent(BaseAgent):
    """Concrete agent implementation backed by OpenAI Chat Completions."""

    def __init__(
        self,
        model: str | None = None,
        max_iterations: int | None = None,
        temperature: float | None = None,
    ) -> None:
        super().__init__(
            max_iterations=max_iterations
            or int(os.getenv("AGENT_MAX_ITERATIONS", "10"))
        )
        self._client = OpenAI()
        self._model = model or os.getenv("OPENAI_MODEL", "gpt-4o")
        self._temperature = temperature or float(os.getenv("TEMPERATURE", "0.0"))

    def _build_system_prompt(self) -> str:
        tools_desc = "\n".join(
            f"- {name}: {fn.__doc__ or '(no description)'}"
            for name, fn in self._tools.items()
        )
        return (
            "You are a pharmaceutical research assistant with access to the "
            "following tools:\n\n"
            f"{tools_desc}\n\n"
            "Use the tools to answer questions accurately and cite your sources. "
            "When you have a complete answer, prefix it with 'FINAL ANSWER:'."
        )

    def _call_llm(self, messages: list[dict[str, str]]) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,  # type: ignore[arg-type]
            temperature=self._temperature,
        )
        return response.choices[0].message.content or ""

    def _parse_response(self, iteration: int, raw: str) -> AgentStep:
        if "FINAL ANSWER:" in raw:
            answer = raw.split("FINAL ANSWER:", 1)[-1].strip()
            return AgentStep(iteration=iteration, thought=raw, final_answer=answer)
        # No final answer yet — continue the loop
        return AgentStep(iteration=iteration, thought=raw)


class AgentWorkflow:
    """
    Composes an agent with the retrieval pipeline.
    Exposes a simple `run(query)` interface.
    """

    def __init__(self, agent: BaseAgent | None = None) -> None:
        self._agent = agent or OpenAIAgent()
        self._agent.register_tool("retrieve_context", retrieve_context)

    def run(self, query: str) -> str:
        """Execute the full agentic workflow for the given query."""
        logger.info("AgentWorkflow.run: query=%r", query)
        return self._agent.run(query)

    @property
    def trace(self) -> list[AgentStep]:
        """Return the execution trace from the most recent run."""
        return self._agent.trace
