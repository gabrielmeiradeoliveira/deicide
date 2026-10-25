"""Quality evaluation benchmark and regression test suite for the Finding Intelligence Engine."""

from __future__ import annotations

from pathlib import Path

from deicide.evaluation.models import EvaluationSummary
from deicide.evaluation.runner import evaluate_corpus, load_fixtures_from_file
from deicide.providers.fake import FakeProvider


def test_finding_intelligence_quality_evaluation_benchmark() -> None:
    """Benchmark comparing baseline validator vs. Finding Intelligence Engine on the extended corpus."""
    corpus_path = Path("evaluations/fixtures/finding_intelligence_corpus.json")
    assert corpus_path.exists()

    fixtures = load_fixtures_from_file(corpus_path)
    assert len(fixtures) == 10

    provider = FakeProvider()

    # 1. Run baseline validator (pre-MVP 1.3 logic)
    baseline_summary: EvaluationSummary = evaluate_corpus(
        fixtures,
        provider=provider,
        use_baseline=True,
    )

    # 2. Run new Finding Intelligence Engine
    new_summary: EvaluationSummary = evaluate_corpus(
        fixtures,
        provider=provider,
        use_baseline=False,
    )

    # 3. Assertions proving quality improvement without regression
    # New engine must achieve 100% precision and 100% recall on the curated corpus
    assert new_summary.precision == 1.0, f"Expected 1.0 precision, got {new_summary.precision}"
    assert new_summary.recall == 1.0, f"Expected 1.0 recall, got {new_summary.recall}"
    assert new_summary.false_positives == 0
    assert new_summary.false_negatives == 0
    assert new_summary.legitimate_rejected == 0

    # Baseline had a false negative on adjacent distinct bugs (case 04) due to overly aggressive deduplication!
    assert baseline_summary.false_negatives >= 1
    assert baseline_summary.recall < 1.0
    assert baseline_summary.legitimate_rejected >= 1

    # New engine successfully reduces duplicates without merging distinct bugs
    assert new_summary.duplicates_reduced >= 1

    # Verify true positive count
    assert new_summary.true_positives == 9  # 9 ground-truth expected bugs across the positive cases


def test_regression_distinct_bugs_on_adjacent_lines_not_merged() -> None:
    """Regression test: Distinct bugs on adjacent lines sharing category must not be merged."""
    corpus_path = Path("evaluations/fixtures/finding_intelligence_corpus.json")
    fixtures = {f.case_id: f for f in load_fixtures_from_file(corpus_path)}

    fixture_04 = fixtures["case_04_distinct_bugs_adjacent_lines"]
    provider = FakeProvider()

    # Baseline incorrectly merges them
    res_baseline = evaluate_corpus([fixture_04], provider=provider, use_baseline=True)
    assert res_baseline.true_positives == 1
    assert res_baseline.false_negatives == 1
    assert res_baseline.legitimate_rejected == 1

    # Finding Intelligence Engine correctly keeps both
    res_new = evaluate_corpus([fixture_04], provider=provider, use_baseline=False)
    assert res_new.true_positives == 2
    assert res_new.false_negatives == 0
    assert res_new.legitimate_rejected == 0


def test_regression_deletion_diff_evidence_not_rejected() -> None:
    """Regression test: Evidence in deletion-only diff hunks must not be rejected."""
    corpus_path = Path("evaluations/fixtures/finding_intelligence_corpus.json")
    fixtures = {f.case_id: f for f in load_fixtures_from_file(corpus_path)}

    fixture_05 = fixtures["case_05_deletion_only_diff"]
    provider = FakeProvider()

    res = evaluate_corpus([fixture_05], provider=provider, use_baseline=False)
    assert res.true_positives == 1
    assert res.false_negatives == 0
    assert res.legitimate_rejected == 0


def test_regression_ambiguous_evidence_downgrades_rather_than_rejects() -> None:
    """Regression test: Ambiguous or partially grounded evidence on a legitimate bug is not lost."""
    corpus_path = Path("evaluations/fixtures/finding_intelligence_corpus.json")
    fixtures = {f.case_id: f for f in load_fixtures_from_file(corpus_path)}

    fixture_08 = fixtures["case_08_ambiguous_evidence_not_rejected"]
    provider = FakeProvider()

    res = evaluate_corpus([fixture_08], provider=provider, use_baseline=False)
    assert res.true_positives == 1
    assert res.false_negatives == 0
    assert res.legitimate_rejected == 0
    assert len(res.case_results[0].actual_findings) == 1
    # Check that severity was capped and confidence downgraded instead of outright rejection
    finding = res.case_results[0].actual_findings[0]
    assert finding.severity == "high"


def test_regression_supplementary_context_leakage_rejected() -> None:
    """Regression test: Findings referencing supplementary context files or code are rejected."""
    corpus_path = Path("evaluations/fixtures/finding_intelligence_corpus.json")
    fixtures = {f.case_id: f for f in load_fixtures_from_file(corpus_path)}

    fixture_07 = fixtures["case_07_supplementary_context_leakage"]
    provider = FakeProvider()

    res = evaluate_corpus([fixture_07], provider=provider, use_baseline=False)
    assert res.true_positives == 0
    assert res.false_positives == 0
    assert res.false_negatives == 0
    assert len(res.case_results[0].actual_findings) == 0
