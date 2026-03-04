"""Unit tests for agent loop, tool registry, and config models."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from solution.models import AgentConfig
from solution.tools import ToolDefinition, execute_tool, get_tool_schemas, tool


# ---------------------------------------------------------------------------
# Tool registry tests
# ---------------------------------------------------------------------------


def test_tool_registration_adds_schema() -> None:
    @tool(
        ToolDefinition(
            name="test_dummy_tool",
            description="A dummy tool for testing.",
            parameters={"type": "object", "properties": {}, "required": []},
        )
    )
    def dummy() -> str:
        return "ok"

    schemas = get_tool_schemas()
    names = [s["function"]["name"] for s in schemas]
    assert "test_dummy_tool" in names


def test_execute_unknown_tool_raises() -> None:
    with pytest.raises(ValueError, match="Unknown tool"):
        execute_tool("nonexistent_xyz", "{}")


def test_execute_tool_passes_arguments() -> None:
    @tool(
        ToolDefinition(
            name="test_add_numbers",
            description="Add two numbers.",
            parameters={
                "type": "object",
                "properties": {
                    "a": {"type": "number"},
                    "b": {"type": "number"},
                },
                "required": ["a", "b"],
            },
        )
    )
    def add_numbers(a: float, b: float) -> float:
        return a + b

    result = execute_tool("test_add_numbers", json.dumps({"a": 2, "b": 3}))
    assert result == 5.0


# ---------------------------------------------------------------------------
# AgentConfig tests
# ---------------------------------------------------------------------------


def test_agent_config_defaults() -> None:
    config = AgentConfig()
    assert config.model == "gpt-4o-mini"
    assert config.max_iterations == 10
    assert config.temperature == 0.0


def test_agent_config_validation_rejects_zero_iterations() -> None:
    with pytest.raises(Exception):
        AgentConfig(max_iterations=0)  # ge=1 constraint


def test_agent_config_validation_rejects_negative_temperature() -> None:
    with pytest.raises(Exception):
        AgentConfig(temperature=-0.1)


# ---------------------------------------------------------------------------
# Agent loop tests
# ---------------------------------------------------------------------------


def test_agent_run_stop_response(mock_openai_client: MagicMock) -> None:
    from solution.agent import Agent

    with patch("solution.agent.OpenAI", return_value=mock_openai_client):
        agent = Agent()
        result = agent.run("Hello")

    assert result == "Test answer"


def test_agent_run_exceeds_max_iterations() -> None:
    from solution.agent import Agent

    looping_client = MagicMock()
    looping_client.chat.completions.create.return_value = MagicMock(
        choices=[
            MagicMock(
                finish_reason="tool_calls",
                message=MagicMock(
                    content=None,
                    tool_calls=[
                        MagicMock(
                            id="call_1",
                            function=MagicMock(
                                name="nonexistent_tool",
                                arguments="{}",
                            ),
                        )
                    ],
                ),
            )
        ]
    )

    with patch("solution.agent.OpenAI", return_value=looping_client):
        agent = Agent(AgentConfig(max_iterations=2))
        with pytest.raises(RuntimeError, match="exceeded maximum"):
            agent.run("loop forever")
