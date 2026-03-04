"""
Embedding model abstraction.

Backends:
    sentence-transformers  Local model, no API key required (default)
    openai                 text-embedding-3-* models

Switch via EMBEDDING_BACKEND env var.
"""

from __future__ import annotations

import os
from typing import Protocol, runtime_checkable


@runtime_checkable
class EmbeddingModel(Protocol):
    """Structural protocol for any embedding backend."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


class SentenceTransformerEmbeddings:
    """Local embedding model via sentence-transformers (no API key needed)."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer  # type: ignore[import]

        self._model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(texts, convert_to_numpy=True).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self._model.encode([text], convert_to_numpy=True)[0].tolist()


class OpenAIEmbeddings:
    """OpenAI embedding model (requires OPENAI_API_KEY)."""

    def __init__(self, model: str | None = None) -> None:
        from openai import OpenAI

        self._client = OpenAI()
        self._model = model or os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        response = self._client.embeddings.create(input=texts, model=self._model)
        return [item.embedding for item in response.data]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def get_default_embedding_model() -> EmbeddingModel:
    """Factory: choose backend based on EMBEDDING_BACKEND env var."""
    backend = os.getenv("EMBEDDING_BACKEND", "sentence-transformers").lower()
    if backend == "openai":
        return OpenAIEmbeddings()
    return SentenceTransformerEmbeddings(
        model_name=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    )
