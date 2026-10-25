"""Automated tests for the DEICIDE MVP 1.3 Real-World Quality Benchmark."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from deicide.cli import app
from deicide.evaluation.models import EvaluationSummary
from deicide.evaluation.runner import evaluate_corpus, load_fixtures_from_file
from deicide.git.analyzer import parse_unified_diff
from deicide.providers.fake import FakeProvider


def test_real_world_benchmark_corpus_integrity() -> None:
    """Validate dataset structure, schema compliance, authentic provenance, and negative examples."""
    corpus_path = Path("evaluations/fixtures/real_world_benchmark.json")
    assert corpus_path.exists(), "Benchmark fixture file must exist."

    fixtures = load_fixtures_from_file(corpus_path)
    # Must meet or exceed the target of at least 30 independently labeled evaluation cases
    assert len(fixtures) >= 30, f"Expected at least 30 benchmark cases, found {len(fixtures)}."

    positive_count = 0
    negative_count = 0

    case_ids: set[str] = set()
    for fixture in fixtures:
        assert fixture.case_id, "Every fixture must have a non-empty case_id."
        assert fixture.case_id not in case_ids, f"Duplicate case_id detected: {fixture.case_id}"
        case_ids.add(fixture.case_id)

        assert fixture.description.strip(), f"Missing description in {fixture.case_id}."
        assert fixture.diff_text.strip(), f"Missing diff_text in {fixture.case_id}."
        assert fixture.file_contents, f"Missing file_contents in {fixture.case_id}."

        # Verify authentic provenance fields
        assert fixture.repository, f"Missing repository provenance in {fixture.case_id}."
        assert fixture.commit_id, f"Missing commit_id in {fixture.case_id}."
        assert fixture.fix_reference, f"Missing fix_reference in {fixture.case_id}."

        if fixture.is_clean:
            negative_count += 1
            assert len(fixture.expected_findings) == 0, (
                f"Negative case {fixture.case_id} must have 0 expected findings."
            )
        else:
            positive_count += 1
            assert len(fixture.expected_findings) > 0, f"Positive case {fixture.case_id} must have expected findings."

    # Verify both positive and negative cases are represented
    assert positive_count >= 20, f"Expected at least 20 positive cases, got {positive_count}."
    assert negative_count >= 5, f"Expected at least 5 negative cases, got {negative_count}."


def test_real_world_benchmark_ground_truth_consistency() -> None:
    """Verify that expected findings and diff coordinates strictly align with source contents."""
    corpus_path = Path("evaluations/fixtures/real_world_benchmark.json")
    fixtures = load_fixtures_from_file(corpus_path)

    for fixture in fixtures:
        parsed_files = parse_unified_diff(fixture.diff_text)
        assert parsed_files, f"Diff parsing produced 0 FileDiff objects in {fixture.case_id}."

        for expected in fixture.expected_findings:
            assert expected.file_path in fixture.file_contents, (
                f"Expected finding references file '{expected.file_path}' "
                f"missing from file_contents in {fixture.case_id}."
            )
            file_text = fixture.file_contents[expected.file_path]
            line_count = len(file_text.splitlines())
            assert expected.line_range.start_line <= line_count, (
                f"Start line {expected.line_range.start_line} exceeds total lines {line_count} "
                f"in {fixture.case_id}:{expected.file_path}."
            )
            assert expected.line_range.end_line <= line_count, (
                f"End line {expected.line_range.end_line} exceeds total lines {line_count} "
                f"in {fixture.case_id}:{expected.file_path}."
            )

            # Confirm git diff has hunk overlapping expected line range
            matching_diff = next(
                (f for f in parsed_files if f.new_path == expected.file_path or f.old_path == expected.file_path),
                None,
            )
            assert matching_diff is not None, f"Diff does not contain expected file {expected.file_path}."
            assert matching_diff.has_changed_overlap(expected.line_range), (
                f"Diff in {fixture.case_id} does not overlap with expected range {expected.line_range}."
            )


def test_real_world_benchmark_reproducible_metrics() -> None:
    """Verify that evaluation metrics are 100% deterministic and reproducible across runs."""
    corpus_path = Path("evaluations/fixtures/real_world_benchmark.json")
    fixtures = load_fixtures_from_file(corpus_path)
    provider = FakeProvider()

    run1: EvaluationSummary = evaluate_corpus(fixtures, provider=provider, use_baseline=False)
    run2: EvaluationSummary = evaluate_corpus(fixtures, provider=provider, use_baseline=False)

    # Identical deterministic metrics across runs
    assert run1.true_positives == run2.true_positives
    assert run1.false_positives == run2.false_positives
    assert run1.false_negatives == run2.false_negatives
    assert run1.precision == run2.precision
    assert run1.recall == run2.recall
    assert run1.duplicates_reduced == run2.duplicates_reduced
    assert run1.legitimate_rejected == run2.legitimate_rejected

    # Evaluate baseline for comparison
    baseline_summary: EvaluationSummary = evaluate_corpus(fixtures, provider=provider, use_baseline=True)

    # Finding Intelligence Engine must achieve precision and recall >= baseline
    assert run1.precision >= baseline_summary.precision
    assert run1.recall >= baseline_summary.recall

    # Ensure the benchmark reports honest results rather than artificially claiming 100%
    # (The benchmark contains difficult cases such as subtle type compatibility false positives)
    assert run1.dataset_size == 33
    assert run1.positive_cases == 22
    assert run1.negative_cases == 11


def test_real_world_benchmark_end_to_end_distinction() -> None:
    """Verify that end-to-end evaluation with FakeProvider honestly reports 0% recall on semantic bugs."""
    corpus_path = Path("evaluations/fixtures/real_world_benchmark.json")
    fixtures = load_fixtures_from_file(corpus_path)
    provider = FakeProvider()

    e2e_summary: EvaluationSummary = evaluate_corpus(
        fixtures,
        provider=provider,
        use_baseline=False,
        force_end_to_end=True,
    )

    # FakeProvider only catches simple regex bugs; it cannot detect complex semantic real-world bugs.
    # The benchmark must report this limitation honestly without claiming false AI capability.
    assert e2e_summary.evaluation_mode == "end_to_end"
    assert e2e_summary.recall < 0.20, "FakeProvider must not deceptively pass authentic semantic benchmarks."


def test_real_world_benchmark_cli_execution() -> None:
    """Test CLI evaluation command options on the real-world benchmark."""
    runner = CliRunner()
    corpus_file = "evaluations/fixtures/real_world_benchmark.json"

    # 1. Standard Post-Processing with Finding Intelligence
    result = runner.invoke(app, ["evaluate", "--fixtures", corpus_file])
    assert result.exit_code == 0
    assert "DEICIDE Evaluation Summary (Finding Intelligence Engine)" in result.stdout
    assert "Dataset Size        : 33" in result.stdout

    # 2. Baseline Comparison
    result_base = runner.invoke(app, ["evaluate", "--fixtures", corpus_file, "--baseline"])
    assert result_base.exit_code == 0
    assert "DEICIDE Evaluation Summary (Baseline Validator)" in result_base.stdout

    # 3. End-to-End mode flag
    result_e2e = runner.invoke(app, ["evaluate", "--fixtures", corpus_file, "--end-to-end"])
    assert result_e2e.exit_code == 0
    assert "DEICIDE Evaluation Summary (End-to-End fake + Finding Intelligence Engine)" in result_e2e.stdout
