"""Tests for the DEICIDE evaluation framework and curated fixture corpus."""

from __future__ import annotations

from pathlib import Path

from deicide.evaluation.runner import evaluate_corpus, load_fixtures_from_file
from deicide.providers.fake import FakeProvider


def test_curated_evaluation_corpus_metrics() -> None:
    corpus_path = Path("evaluations/fixtures/code_review_corpus.json")
    assert corpus_path.exists()

    fixtures = load_fixtures_from_file(corpus_path)
    assert len(fixtures) == 6

    provider = FakeProvider()
    summary = evaluate_corpus(fixtures, provider=provider)

    assert summary.dataset_size == 6
    assert summary.positive_cases == 4
    assert summary.negative_cases == 2
    assert summary.true_positives == 4
    assert summary.false_positives == 0
    assert summary.false_negatives == 0
    assert summary.precision == 1.0
    assert summary.recall == 1.0
    assert summary.false_positive_rate == 0.0
    assert summary.usage.total_tokens > 0


def test_evaluation_cli_and_provider_validation() -> None:
    from typer.testing import CliRunner

    from deicide.cli import app
    from deicide.core.errors import ExitCode

    runner = CliRunner()
    # Evaluating with fake succeeds
    res_fake = runner.invoke(app, ["evaluate", "--provider", "fake"])
    assert res_fake.exit_code == int(ExitCode.SUCCESS)
    assert "DEICIDE Evaluation Summary" in res_fake.stdout

    # Evaluating with unsupported gemini fails
    res_gemini = runner.invoke(app, ["evaluate", "--provider", "gemini"])
    assert res_gemini.exit_code == int(ExitCode.CONFIGURATION_ERROR)
    assert "Unsupported provider 'gemini'" in res_gemini.stderr
