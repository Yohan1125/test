"""
Evaluation harness: runs (query, reference_answer) pairs through
AgentWorkflow and aggregates metrics.

Dataset format (JSONL):
    {"query": "...", "reference_answer": "..."}

Output (JSON):
    {"summary": {...}, "results": [...]}
"""

from __future__ import annotations

import json
import logging
import statistics
from dataclasses import asdict
from pathlib import Path
from typing import Any

from pharma_agent.agent.workflow import AgentWorkflow
from pharma_agent.evaluation.metrics import EvalMetrics, EvalResult

logger = logging.getLogger(__name__)


class Evaluator:
    """Runs AgentWorkflow over an evaluation dataset and computes metrics."""

    def __init__(self, workflow: AgentWorkflow | None = None) -> None:
        self._workflow = workflow or AgentWorkflow()

    def evaluate_dataset(
        self,
        dataset_path: str | Path,
        output_path: str | Path | None = None,
    ) -> dict[str, Any]:
        """Load JSONL dataset, evaluate each row, return aggregated results."""
        rows = _load_jsonl(Path(dataset_path))
        logger.info("Evaluating %d examples from %s", len(rows), dataset_path)

        eval_results: list[EvalResult] = []
        for i, row in enumerate(rows, start=1):
            query = row["query"]
            reference = row.get("reference_answer", "")
            logger.info("Example %d/%d: %r", i, len(rows), query[:60])

            answer, latency = EvalMetrics.measure_latency(self._workflow.run, query)

            # Collect retrieved context from trace
            flat_chunks: list[str] = []
            for step in self._workflow.trace:
                if (
                    step.tool_name == "retrieve_context"
                    and step.tool_result is not None
                    and step.tool_result.error is None
                ):
                    raw = step.tool_result.output
                    if isinstance(raw, list):
                        flat_chunks.extend(
                            r.get("content", "") for r in raw if isinstance(r, dict)
                        )

            result = EvalResult(
                query=query,
                answer=answer,
                context_chunks=flat_chunks,
                reference_answer=reference,
                answer_correctness=EvalMetrics.token_f1(answer, reference),
                context_recall=EvalMetrics.context_recall_lexical(
                    reference, flat_chunks
                ),
                latency_s=latency,
            )
            eval_results.append(result)

        summary = _summarize(eval_results)
        output: dict[str, Any] = {
            "summary": summary,
            "results": [asdict(r) for r in eval_results],
        }

        if output_path is not None:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            Path(output_path).write_text(json.dumps(output, indent=2))
            logger.info("Results written to %s", output_path)

        return output


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _summarize(results: list[EvalResult]) -> dict[str, float]:
    if not results:
        return {}
    return {
        "mean_answer_correctness": statistics.mean(
            r.answer_correctness for r in results
        ),
        "mean_context_recall": statistics.mean(r.context_recall for r in results),
        "mean_latency_s": statistics.mean(r.latency_s for r in results),
        "num_examples": float(len(results)),
    }
