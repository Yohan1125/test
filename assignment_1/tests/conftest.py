"""Shared pytest fixtures for assignment 1."""

from __future__ import annotations

import os
from unittest.mock import MagicMock

import pytest


@pytest.fixture(autouse=True)
def _fake_api_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent tests from accidentally using real API keys."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-fake")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-fake")


@pytest.fixture()
def agent_config():
    from solution.models import AgentConfig

    return AgentConfig(model="gpt-4o-mini", max_iterations=3)


@pytest.fixture()
def mock_openai_client() -> MagicMock:
    """Mock OpenAI client that returns a simple stop response."""
    mock = MagicMock()
    mock.chat.completions.create.return_value = MagicMock(
        choices=[
            MagicMock(
                finish_reason="stop",
                message=MagicMock(content="Test answer", tool_calls=None),
            )
        ]
    )
    return mock
