"""Tests for evaluation metrics."""

from __future__ import annotations

import pytest

from pharma_agent.evaluation.metrics import EvalMetrics, _tokenize


class TestTokenize:
    def test_lowercases(self) -> None:
        assert _tokenize("Hello World") == ["hello", "world"]

    def test_strips_punctuation(self) -> None:
        assert _tokenize("Hello, world!") == ["hello", "world"]

    def test_empty_string(self) -> None:
        assert _tokenize("") == []


class TestTokenF1:
    def test_identical_strings(self) -> None:
        score = EvalMetrics.token_f1("hello world", "hello world")
        assert score == pytest.approx(1.0)

    def test_no_overlap(self) -> None:
        score = EvalMetrics.token_f1("foo bar", "baz qux")
        assert score == pytest.approx(0.0)

    def test_partial_overlap(self) -> None:
        score = EvalMetrics.token_f1("hello world", "hello everyone")
        assert 0.0 < score < 1.0

    def test_empty_prediction_returns_zero(self) -> None:
        assert EvalMetrics.token_f1("", "reference answer") == pytest.approx(0.0)

    def test_empty_reference_returns_zero(self) -> None:
        assert EvalMetrics.token_f1("prediction", "") == pytest.approx(0.0)


class TestContextRecallLexical:
    def test_full_recall(self) -> None:
        score = EvalMetrics.context_recall_lexical(
            reference_answer="metformin reduces glucose",
            context_chunks=["metformin reduces hepatic glucose production"],
        )
        assert score == pytest.approx(1.0)

    def test_zero_recall(self) -> None:
        score = EvalMetrics.context_recall_lexical(
            reference_answer="metformin reduces glucose",
            context_chunks=["aspirin relieves pain"],
        )
        assert score == pytest.approx(0.0)

    def test_empty_reference_returns_zero(self) -> None:
        assert EvalMetrics.context_recall_lexical("", ["some context"]) == pytest.approx(0.0)

    def test_partial_recall(self) -> None:
        score = EvalMetrics.context_recall_lexical(
            reference_answer="drug x shows efficacy in trials",
            context_chunks=["drug x was tested"],
        )
        assert 0.0 < score < 1.0


class TestMeasureLatency:
    def test_returns_result_and_positive_float(self) -> None:
        result, elapsed = EvalMetrics.measure_latency(lambda a, b: a + b, 3, 4)
        assert result == 7
        assert elapsed >= 0.0
        assert isinstance(elapsed, float)
