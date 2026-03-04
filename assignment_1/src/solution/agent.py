"""ReAct-style tool-calling agent."""

from __future__ import annotations

import logging
import os
from typing import Any

from openai import OpenAI

from .models import AgentConfig, Message, Role, ToolResult
from .tools import execute_tool, get_tool_schemas

logger = logging.getLogger(__name__)


class Agent:
    """Minimal tool-calling agent over the OpenAI chat completions API."""

    def __init__(self, config: AgentConfig | None = None) -> None:
        self.config = config or AgentConfig(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            max_iterations=int(os.getenv("MAX_ITERATIONS", "10")),
            temperature=float(os.getenv("TEMPERATURE", "0.0")),
        )
        self._client = OpenAI()
        self._history: list[Message] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, user_message: str) -> str:
        """Run the agent loop until a final answer is produced."""
        self._history = [
            Message(role=Role.SYSTEM, content=self.config.system_prompt),
            Message(role=Role.USER, content=user_message),
        ]

        for iteration in range(self.config.max_iterations):
            logger.debug("Agent iteration %d", iteration + 1)
            response = self._call_llm()
            choice = response.choices[0]

            if choice.finish_reason == "stop":
                return choice.message.content or ""

            if choice.finish_reason == "tool_calls":
                tool_results = self._execute_tool_calls(choice.message.tool_calls)
                self._append_tool_results(tool_results)
                continue

            raise RuntimeError(f"Unexpected finish_reason: {choice.finish_reason!r}")

        raise RuntimeError("Agent exceeded maximum iteration budget.")

    @property
    def history(self) -> list[Message]:
        return list(self._history)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _call_llm(self) -> Any:
        return self._client.chat.completions.create(
            model=self.config.model,
            temperature=self.config.temperature,
            messages=[m.model_dump(exclude_none=True) for m in self._history],
            tools=get_tool_schemas(),
            tool_choice="auto",
        )

    def _execute_tool_calls(self, tool_calls: list[Any]) -> list[ToolResult]:
        results: list[ToolResult] = []
        for tc in tool_calls:
            try:
                output = execute_tool(tc.function.name, tc.function.arguments)
                results.append(ToolResult(tool_call_id=tc.id, output=output))
            except Exception as exc:  # noqa: BLE001
                logger.warning("Tool %r failed: %s", tc.function.name, exc)
                results.append(
                    ToolResult(tool_call_id=tc.id, output=None, error=str(exc))
                )
        return results

    def _append_tool_results(self, results: list[ToolResult]) -> None:
        for r in results:
            self._history.append(
                Message(
                    role=Role.TOOL,
                    content=(
                        str(r.output) if r.error is None else f"ERROR: {r.error}"
                    ),
                    tool_call_id=r.tool_call_id,
                )
            )
