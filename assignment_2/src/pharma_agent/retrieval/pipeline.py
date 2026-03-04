"""
End-to-end RAG pipeline: ingest → chunk → embed → store → query.

Usage:
    pipeline = RetrievalPipeline()
    pipeline.ingest_texts(texts, sources=["doc_a.pdf"])
    results = pipeline.query("What are the contraindications of metformin?")
"""

from __future__ import annotations

import hashlib
import logging
import os
from typing import Any

from pharma_agent.retrieval.embeddings import EmbeddingModel, get_default_embedding_model
from pharma_agent.retrieval.store import VectorStore

logger = logging.getLogger(__name__)

_default_pipeline: RetrievalPipeline | None = None


def _chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into overlapping character-level chunks."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not text:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start += chunk_size - overlap
    return chunks


class RetrievalPipeline:
    """Orchestrates chunking, embedding, storage, and retrieval."""

    def __init__(
        self,
        embedding_model: EmbeddingModel | None = None,
        vector_store: VectorStore | None = None,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> None:
        self._embedder = embedding_model or get_default_embedding_model()
        self._store = vector_store or VectorStore()
        self._chunk_size = chunk_size or int(os.getenv("CHUNK_SIZE", "512"))
        self._chunk_overlap = chunk_overlap or int(os.getenv("CHUNK_OVERLAP", "64"))

    @classmethod
    def get_default(cls) -> RetrievalPipeline:
        """Return (or lazily create) the process-global default pipeline."""
        global _default_pipeline
        if _default_pipeline is None:
            _default_pipeline = cls()
        return _default_pipeline

    def ingest_texts(
        self,
        texts: list[str],
        sources: list[str] | None = None,
        extra_metadata: list[dict[str, Any]] | None = None,
    ) -> int:
        """Chunk, embed, and store texts. Returns number of chunks ingested."""
        if not texts:
            return 0

        sources = sources or ["unknown"] * len(texts)
        extra_metadata = extra_metadata or [{} for _ in texts]

        all_ids: list[str] = []
        all_documents: list[str] = []
        all_metadatas: list[dict[str, Any]] = []

        for text, source, meta in zip(texts, sources, extra_metadata):
            chunks = _chunk_text(text, self._chunk_size, self._chunk_overlap)
            for i, chunk in enumerate(chunks):
                chunk_id = hashlib.sha256(
                    f"{source}::{i}::{chunk}".encode()
                ).hexdigest()
                all_ids.append(chunk_id)
                all_documents.append(chunk)
                all_metadatas.append({"source": source, "chunk_index": i, **meta})

        if not all_ids:
            return 0

        all_embeddings = self._embedder.embed_documents(all_documents)
        self._store.upsert(all_ids, all_embeddings, all_documents, all_metadatas)
        logger.info("Ingested %d chunks from %d documents", len(all_ids), len(texts))
        return len(all_ids)

    def query(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve top-k most relevant chunks for a query."""
        k = top_k or int(os.getenv("RETRIEVAL_TOP_K", "5"))
        query_embedding = self._embedder.embed_query(query)
        results = self._store.query(query_embedding, top_k=k)

        threshold = float(os.getenv("RETRIEVAL_SCORE_THRESHOLD", "0.0"))
        filtered = [r for r in results if r["score"] >= threshold]
        logger.debug(
            "query=%r: %d/%d results passed threshold %.2f",
            query,
            len(filtered),
            len(results),
            threshold,
        )
        return filtered
