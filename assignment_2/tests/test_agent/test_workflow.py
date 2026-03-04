"""Tests for BaseAgent and AgentWorkflow."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from pharma_agent.agent.base import AgentStep, BaseAgent, ToolResult


# ---------------------------------------------------------------------------
# Concrete stub for testing BaseAgent
# ---------------------------------------------------------------------------


class AlwaysAnswerAgent(BaseAgent):
    """Agent that always returns a final answer immediately."""

    def _build_system_prompt(self) -> str:
        return "You are a test agent."

    def _call_llm(self, messages: list[dict]) -> str:
        return "FINAL ANSWER: done"

    def _parse_response(self, iteration: int, raw: str) -> AgentStep:
        return AgentStep(iteration=iteration, thought=raw, final_answer="done")


class LoopingAgent(BaseAgent):
    """Agent that never produces a final answer (for testing iteration limit)."""

    def _build_system_prompt(self) -> str:
        return ""

    def _call_llm(self, messages: list[dict]) -> str:
        return "still thinking..."

    def _parse_response(self, iteration: int, raw: str) -> AgentStep:
        return AgentStep(iteration=iteration, thought=raw)


# ---------------------------------------------------------------------------
# BaseAgent tests
# ---------------------------------------------------------------------------


class TestBaseAgent:
    def test_run_returns_final_answer(self) -> None:
        agent = AlwaysAnswerAgent()
        assert agent.run("What is metformin?") == "done"

    def test_trace_populated_after_run(self) -> None:
        agent = AlwaysAnswerAgent()
        agent.run("test")
        assert len(agent.trace) == 1
        assert agent.trace[0].iteration == 1

    def test_register_and_execute_tool(self) -> None:
        agent = AlwaysAnswerAgent()
        agent.register_tool("add", lambda a, b: a + b)
        result = agent._execute_tool("add", {"a": 1, "b": 2})
        assert result.output == 3
        assert result.error is None

    def test_unknown_tool_returns_error_result(self) -> None:
        agent = AlwaysAnswerAgent()
        result = agent._execute_tool("nonexistent", {})
        assert result.error is not None
        assert "Unknown tool" in result.error

    def test_max_iterations_raises_runtime_error(self) -> None:
        agent = LoopingAgent(max_iterations=2)
        with pytest.raises(RuntimeError, match="max_iterations"):
            agent.run("loop forever")

    def test_tool_exception_captured_in_result(self) -> None:
        agent = AlwaysAnswerAgent()
        agent.register_tool("boom", lambda: (_ for _ in ()).throw(ValueError("oops")))
        result = agent._execute_tool("boom", {})
        assert result.error is not None
        assert "oops" in result.error


# ---------------------------------------------------------------------------
# AgentWorkflow tests
# ---------------------------------------------------------------------------


class TestAgentWorkflow:
    def test_run_returns_string(self, mock_openai_client: MagicMock) -> None:
        from pharma_agent.agent.workflow import AgentWorkflow

        with patch("pharma_agent.agent.workflow.OpenAI", return_value=mock_openai_client):
            workflow = AgentWorkflow()
            result = workflow.run("What is metformin?")

        assert isinstance(result, str)
        assert len(result) > 0

    def test_trace_available_after_run(self, mock_openai_client: MagicMock) -> None:
        from pharma_agent.agent.workflow import AgentWorkflow

        with patch("pharma_agent.agent.workflow.OpenAI", return_value=mock_openai_client):
            workflow = AgentWorkflow()
            workflow.run("test query")

        assert isinstance(workflow.trace, list)
