"""Evaluation framework for DEICIDE analysis quality, precision, recall, latency, and tokens."""

from deicide.evaluation.models import (
    CaseEvaluationResult,
    EvaluationFixture,
    EvaluationSummary,
    ExpectedFindingSpec,
)
from deicide.evaluation.runner import evaluate_case, evaluate_corpus, load_fixtures_from_file

__all__ = [
    "CaseEvaluationResult",
    "EvaluationFixture",
    "EvaluationSummary",
    "ExpectedFindingSpec",
    "evaluate_case",
    "evaluate_corpus",
    "load_fixtures_from_file",
]
