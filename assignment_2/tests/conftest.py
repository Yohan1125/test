"""Shared pytest fixtures for assignment 2."""

from __future__ import annotations

from typing import Generator
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _fake_api_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent tests from accidentally using real API keys."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-fake")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-fake")


@pytest.fixture()
def mock_openai_client() -> MagicMock:
    """Mock OpenAI client that returns a FINAL ANSWER response."""
    mock = MagicMock()
    mock.chat.completions.create.return_value = MagicMock(
        choices=[
            MagicMock(
                message=MagicMock(content="FINAL ANSWER: test answer")
            )
        ]
    )
    return mock


@pytest.fixture()
def mock_embedding_model() -> MagicMock:
    """Mock that always produces 384-dim zero vectors."""
    mock = MagicMock()
    mock.embed_documents.side_effect = lambda texts: [[0.0] * 384] * len(texts)
    mock.embed_query.return_value = [0.0] * 384
    return mock


@pytest.fixture()
def mock_vector_store() -> MagicMock:
    """Mock VectorStore that returns canned query results."""
    mock = MagicMock()
    mock.query.return_value = [
        {
            "content": "Metformin reduces hepatic glucose production.",
            "source": "doc1.pdf",
            "score": 0.92,
            "metadata": {},
        },
        {
            "content": "Common side effects include nausea.",
            "source": "doc1.pdf",
            "score": 0.85,
            "metadata": {},
        },
    ]
    mock.count.return_value = 2
    return mock
