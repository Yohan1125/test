"""Tool definitions wired to the retrieval pipeline and external APIs."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def retrieve_context(query: str, top_k: int = 5) -> list[dict[str, Any]]:
    """
    Retrieve the most relevant document chunks for a query.

    Args:
        query:  Natural-language query string.
        top_k:  Number of chunks to return.

    Returns:
        List of dicts with keys: 'content', 'source', 'score'.
    """
    from pharma_agent.retrieval.pipeline import RetrievalPipeline

    pipeline = RetrievalPipeline.get_default()
    results = pipeline.query(query, top_k=top_k)
    logger.debug("retrieve_context: %d results for query=%r", len(results), query)
    return results


def summarize_document(content: str, focus: str = "") -> str:
    """
    Summarize a long document, optionally focusing on a specific aspect.

    Args:
        content:  Raw document text.
        focus:    Optional aspect to emphasize (e.g., "side effects").

    Returns:
        Summary string.
    """
    raise NotImplementedError("Implement with an LLM call")


def lookup_drug_info(drug_name: str) -> dict[str, Any]:
    """
    Look up structured information about a pharmaceutical compound.

    Args:
        drug_name:  IUPAC name, brand name, or common name.

    Returns:
        Dict with keys: 'name', 'mechanism', 'indications', 'contraindications'.
    """
    raise NotImplementedError("Implement with PubChem / RxNorm API")
