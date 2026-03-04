"""ChromaDB-backed vector store with a simple upsert/query API."""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_COLLECTION = "pharma_docs"


class VectorStore:
    """Persistent ChromaDB vector store."""

    def __init__(
        self,
        persist_dir: str | None = None,
        collection_name: str = DEFAULT_COLLECTION,
    ) -> None:
        import chromadb  # type: ignore[import]

        persist_dir = persist_dir or os.getenv("CHROMA_PERSIST_DIR", "./.chroma_db")
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "VectorStore: collection=%r, persist_dir=%r",
            collection_name,
            persist_dir,
        )

    def upsert(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None:
        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas or [{} for _ in ids],
        )
        logger.debug("Upserted %d documents", len(ids))

    def query(
        self,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Return top-k most similar documents as list of dicts."""
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        output: list[dict[str, Any]] = []
        docs = (results.get("documents") or [[]])[0]
        metas = (results.get("metadatas") or [[]])[0]
        dists = (results.get("distances") or [[]])[0]

        for doc, meta, dist in zip(docs, metas, dists):
            output.append(
                {
                    "content": doc,
                    "source": (meta or {}).get("source", "unknown"),
                    "score": 1.0 - float(dist),  # cosine similarity
                    "metadata": meta or {},
                }
            )
        return output

    def count(self) -> int:
        return self._collection.count()
