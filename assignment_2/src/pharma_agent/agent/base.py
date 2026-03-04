"""
Abstract base agent with a generic tool-calling loop.

Subclass BaseAgent and implement `_build_system_prompt` and `_call_llm`.
Register tools with `register_tool`. The `run` method drives the ReAct loop.
"""

from __future__ import annotations

import abc
import logging
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ToolResult(BaseModel):
    """Typed result returned by any registered tool."""

    tool_name: str
    output: Any
    error: str | None = None


class AgentStep(BaseModel):
    """Single step in the agent's execution trace."""

    iteration: int
    thought: str = ""
    tool_name: str | None = None
    tool_input: dict[str, Any] = Field(default_factory=dict)
    tool_result: ToolResult | None = None
    final_answer: str | None = None


class BaseAgent(abc.ABC):
    """
    Abstract ReAct-style agent skeleton.

    Concrete subclasses must implement:
        _build_system_prompt() -> str
        _call_llm(messages) -> str
        _parse_response(iteration, raw) -> AgentStep
    """

    def __init__(self, max_iterations: int = 10) -> None:
        self.max_iterations = max_iterations
        self._tools: dict[str, Callable[..., Any]] = {}
        self._steps: list[AgentStep] = []

    def register_tool(self, name: str, fn: Callable[..., Any]) -> None:
        """Register a callable as a named tool available to the agent."""
        self._tools[name] = fn
        logger.debug("Registered tool: %s", name)

    @abc.abstractmethod
    def _build_system_prompt(self) -> str:
        """Return the system prompt that describes available tools."""

    @abc.abstractmethod
    def _call_llm(self, messages: list[dict[str, str]]) -> str:
        """Send messages to the LLM and return the raw text response."""

    @abc.abstractmethod
    def _parse_response(self, iteration: int, raw: str) -> AgentStep:
        """Parse the raw LLM response into an AgentStep."""

    def run(self, user_query: str) -> str:
        """
        Execute the agent loop for a given query.

        Returns the agent's final answer string.
        Raises RuntimeError if max_iterations is exceeded without a final answer.
        """
        self._steps = []
        messages: list[dict[str, str]] = [
            {"role": "system", "content": self._build_system_prompt()},
            {"role": "user", "content": user_query},
        ]

        for i in range(1, self.max_iterations + 1):
            logger.info("Agent iteration %d/%d", i, self.max_iterations)
            raw = self._call_llm(messages)
            step = self._parse_response(i, raw)
            self._steps.append(step)

            if step.final_answer is not None:
                return step.final_answer

            if step.tool_name is not None:
                result = self._execute_tool(step.tool_name, step.tool_input)
                step.tool_result = result
                messages.append({"role": "assistant", "content": raw})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"Tool '{result.tool_name}' returned:\n{result.output}"
                            if result.error is None
                            else f"Tool '{result.tool_name}' raised an error: {result.error}"
                        ),
                    }
                )

        raise RuntimeError(
            f"Agent did not produce a final answer within max_iterations={self.max_iterations}."
        )

    def _execute_tool(self, name: str, inputs: dict[str, Any]) -> ToolResult:
        if name not in self._tools:
            return ToolResult(
                tool_name=name,
                output=None,
                error=f"Unknown tool '{name}'. Available: {list(self._tools)}",
            )
        try:
            output = self._tools[name](**inputs)
            return ToolResult(tool_name=name, output=output)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Tool '%s' raised an exception", name)
            return ToolResult(tool_name=name, output=None, error=str(exc))

    @property
    def trace(self) -> list[AgentStep]:
        """Return the full execution trace from the most recent `run` call."""
        return list(self._steps)
