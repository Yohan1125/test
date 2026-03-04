"""Tests for the RAG retrieval pipeline."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from pharma_agent.retrieval.pipeline import RetrievalPipeline, _chunk_text


class TestChunkText:
    def test_basic_chunking(self) -> None:
        text = "a" * 100
        chunks = _chunk_text(text, chunk_size=40, overlap=10)
        assert len(chunks) > 1
        assert all(len(c) <= 40 for c in chunks)

    def test_overlap_creates_shared_content(self) -> None:
        text = "abcdefghij"  # 10 chars
        chunks = _chunk_text(text, chunk_size=6, overlap=2)
        # chunk[0][-2:] should equal chunk[1][:2]
        assert chunks[0][-2:] == chunks[1][:2]

    def test_short_text_single_chunk(self) -> None:
        chunks = _chunk_text("hello", chunk_size=100, overlap=0)
        assert chunks == ["hello"]

    def test_empty_text_returns_empty(self) -> None:
        assert _chunk_text("", chunk_size=10, overlap=2) == []

    def test_zero_chunk_size_raises(self) -> None:
        with pytest.raises(ValueError, match="chunk_size"):
            _chunk_text("hello", chunk_size=0, overlap=0)


class TestRetrievalPipeline:
    def test_ingest_returns_chunk_count(
        self,
        mock_embedding_model: MagicMock,
        mock_vector_store: MagicMock,
    ) -> None:
        pipeline = RetrievalPipeline(
            embedding_model=mock_embedding_model,
            vector_store=mock_vector_store,
            chunk_size=50,
            chunk_overlap=10,
        )
        count = pipeline.ingest_texts(["word " * 50], sources=["test.pdf"])
        assert count > 0
        mock_vector_store.upsert.assert_called_once()

    def test_ingest_empty_list_returns_zero(
        self,
        mock_embedding_model: MagicMock,
        mock_vector_store: MagicMock,
    ) -> None:
        pipeline = RetrievalPipeline(
            embedding_model=mock_embedding_model,
            vector_store=mock_vector_store,
        )
        count = pipeline.ingest_texts([])
        assert count == 0
        mock_vector_store.upsert.assert_not_called()

    def test_query_calls_embedder_and_store(
        self,
        mock_embedding_model: MagicMock,
        mock_vector_store: MagicMock,
    ) -> None:
        pipeline = RetrievalPipeline(
            embedding_model=mock_embedding_model,
            vector_store=mock_vector_store,
        )
        results = pipeline.query("metformin side effects", top_k=2)
        mock_embedding_model.embed_query.assert_called_once_with(
            "metformin side effects"
        )
        mock_vector_store.query.assert_called_once()
        assert isinstance(results, list)

    def test_score_threshold_filters_results(
        self,
        mock_embedding_model: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        low_score_store = MagicMock()
        low_score_store.query.return_value = [
            {"content": "x", "source": "s", "score": 0.3, "metadata": {}},
        ]
        monkeypatch.setenv("RETRIEVAL_SCORE_THRESHOLD", "0.9")
        pipeline = RetrievalPipeline(
            embedding_model=mock_embedding_model,
            vector_store=low_score_store,
        )
        results = pipeline.query("test query")
        assert results == []
