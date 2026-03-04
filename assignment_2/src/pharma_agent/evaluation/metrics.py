"""
Evaluation metrics for the RAG / agentic pipeline.

Metrics:
    token_f1                Token-level F1 vs. reference answer (offline)
    context_recall_lexical  Fraction of reference tokens in retrieved context (offline)
    measure_latency         Wall-clock time wrapper
"""

from __future__ import annotations

import re
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any


@dataclass
class EvalResult:
    """Container for all metrics on a single query."""

    query: str
    answer: str
    context_chunks: list[str] = field(default_factory=list)
    reference_answer: str = ""

    answer_correctness: float = 0.0
    context_recall: float = 0.0
    latency_s: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class EvalMetrics:
    """Collection of evaluation metric functions."""

    @staticmethod
    def token_f1(prediction: str, reference: str) -> float:
        """Token-level F1 between prediction and reference (offline)."""
        pred_tokens = Counter(_tokenize(prediction))
        ref_tokens = Counter(_tokenize(reference))

        common = pred_tokens & ref_tokens
        num_common = sum(common.values())

        if num_common == 0:
            return 0.0

        precision = num_common / sum(pred_tokens.values())
        recall = num_common / sum(ref_tokens.values())
        return 2 * precision * recall / (precision + recall)

    @staticmethod
    def context_recall_lexical(
        reference_answer: str,
        context_chunks: list[str],
    ) -> float:
        """
        Estimate context recall as the fraction of reference answer tokens
        present in any of the retrieved context chunks (offline approximation).
        """
        ref_tokens = set(_tokenize(reference_answer))
        if not ref_tokens:
            return 0.0
        context_text = " ".join(context_chunks)
        context_tokens = set(_tokenize(context_text))
        return len(ref_tokens & context_tokens) / len(ref_tokens)

    @staticmethod
    def measure_latency(fn: Any, *args: Any, **kwargs: Any) -> tuple[Any, float]:
        """
        Call fn(*args, **kwargs) and return (result, elapsed_seconds).

        Usage:
            answer, latency = EvalMetrics.measure_latency(workflow.run, query)
        """
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        elapsed = time.perf_counter() - start
        return result, elapsed


def _tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, split on whitespace."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return text.split()
