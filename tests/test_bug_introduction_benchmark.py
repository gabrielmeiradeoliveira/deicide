"""Tests for DEICIDE MVP 1.4 Valid Bug Introduction Benchmark."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from deicide.cli import app
from deicide.core.errors import ExitCode
from deicide.core.models import FindingCategory
from deicide.evaluation.models import ChangeType, EvaluationSummary
from deicide.evaluation.runner import evaluate_corpus, load_fixtures_from_file
from deicide.git.analyzer import parse_unified_diff
from deicide.providers.fake import FakeProvider

BENCHMARK_PATH = Path("evaluations/fixtures/bug_introduction_benchmark.json")


def test_bug_introduction_benchmark_corpus_integrity() -> None:
    """Validate structure, classification, authenticity, and composition of the benchmark."""
    assert BENCHMARK_PATH.exists(), f"Benchmark file missing: {BENCHMARK_PATH}"
    fixtures = load_fixtures_from_file(BENCHMARK_PATH)

    assert len(fixtures) == 8, f"Expected 8 benchmark cases, found {len(fixtures)}"

    positive_cases = [f for f in fixtures if not f.is_clean]
    negative_cases = [f for f in fixtures if f.is_clean]

    assert len(positive_cases) == 5, f"Expected 5 positive cases, got {len(positive_cases)}"
    assert len(negative_cases) == 3, f"Expected 3 negative cases, got {len(negative_cases)}"

    # Verify all 8 cases are marked suitable for live benchmark
    for f in fixtures:
        assert f.case_id.strip()
        assert f.description.strip()
        assert f.diff_text.strip()
        assert f.file_contents
        assert f.repository
        assert f.fix_reference
        assert f.provenance
        assert f.is_suitable_for_live_benchmark is True, f"Case {f.case_id} must be suitable for live benchmark"

    # Verify all positive cases are ChangeType.DEFECT_INTRODUCTION
    for f in positive_cases:
        assert f.resolved_change_type == ChangeType.DEFECT_INTRODUCTION
        assert len(f.expected_findings) == 1, f"Expected 1 finding for {f.case_id}"
        assert len(f.candidate_findings) >= 1, f"Expected candidates for {f.case_id}"

    # Verify negative cases composition: legitimate bug fix, safe refactor, docs
    neg_types = {f.resolved_change_type for f in negative_cases}
    assert ChangeType.CLEAN_BUG_FIX in neg_types, "Must include a legitimate bug fix negative case"
    assert ChangeType.CLEAN_UNRELATED in neg_types, "Must include clean unrelated negative cases"

    # Verify expected categories for positive cases:
    # 1. Division by zero (correctness)
    # 2. Boundary condition underflow
    # 3. Missing exception handling
    # 4. State mutation
    # 5. Cross-file contract violation (correctness)
    categories = {f.expected_findings[0].category for f in positive_cases}
    assert FindingCategory.CORRECTNESS in categories
    assert FindingCategory.BOUNDARY_CONDITION in categories
    assert FindingCategory.ERROR_HANDLING in categories
    assert FindingCategory.STATE_MUTATION in categories


def test_bug_introduction_benchmark_ground_truth_consistency() -> None:
    """Verify ground-truth line ranges and file coordinates strictly overlap modified diff hunks."""
    fixtures = load_fixtures_from_file(BENCHMARK_PATH)

    for f in fixtures:
        parsed_files = parse_unified_diff(f.diff_text)
        assert parsed_files, f"Failed parsing unified diff in {f.case_id}"

        for expected in f.expected_findings:
            assert expected.file_path in f.file_contents, (
                f"{expected.file_path} missing in file_contents for {f.case_id}"
            )
            file_lines = len(f.file_contents[expected.file_path].splitlines())
            assert expected.line_range.start_line <= file_lines, (
                f"Start line {expected.line_range.start_line} exceeds file length {file_lines} in {f.case_id}"
            )
            assert expected.line_range.end_line <= file_lines, (
                f"End line {expected.line_range.end_line} exceeds file length {file_lines} in {f.case_id}"
            )

            diff_file = next(
                (d for d in parsed_files if d.new_path == expected.file_path or d.old_path == expected.file_path),
                None,
            )
            assert diff_file is not None, f"File {expected.file_path} not found in diff for {f.case_id}"
            assert diff_file.has_changed_overlap(expected.line_range), (
                f"Diff in {f.case_id} does not overlap expected range {expected.line_range}"
            )


def test_bug_introduction_benchmark_post_processing_metrics() -> None:
    """Verify FindingIntelligenceEngine achieves 100% precision and recall on candidate findings."""
    fixtures = load_fixtures_from_file(BENCHMARK_PATH)
    provider = FakeProvider()

    summary: EvaluationSummary = evaluate_corpus(
        fixtures,
        provider=provider,
        use_baseline=False,
        force_end_to_end=False,
    )

    assert summary.dataset_size == 8
    assert summary.positive_cases == 5
    assert summary.negative_cases == 3
    assert summary.successful_cases == 8
    assert summary.failed_cases == 0
    assert summary.true_positives == 5
    assert summary.false_positives == 0
    assert summary.false_negatives == 0
    assert summary.precision == 1.0
    assert summary.recall == 1.0
    assert summary.false_positive_rate == 0.0


def test_bug_introduction_benchmark_end_to_end_reports_honest_misses() -> None:
    """Verify end-to-end evaluation with FakeProvider honestly reports 0% recall on authentic defects."""
    fixtures = load_fixtures_from_file(BENCHMARK_PATH)
    provider = FakeProvider()

    e2e_summary: EvaluationSummary = evaluate_corpus(
        fixtures,
        provider=provider,
        use_baseline=False,
        force_end_to_end=True,
    )

    assert e2e_summary.evaluation_mode == "end_to_end"
    assert e2e_summary.dataset_size == 8
    assert e2e_summary.positive_cases == 5
    assert e2e_summary.negative_cases == 3
    assert e2e_summary.successful_cases == 8
    assert e2e_summary.true_positives == 0
    assert e2e_summary.false_positives == 0
    assert e2e_summary.false_negatives == 5
    assert e2e_summary.precision == 0.0
    assert e2e_summary.recall == 0.0
    assert e2e_summary.false_positive_rate == 0.0


def test_bug_introduction_benchmark_cli_invocation() -> None:
    """Verify CLI evaluate command correctly evaluates the new benchmark dataset."""
    runner = CliRunner()

    # Post-processing evaluation
    res_post = runner.invoke(
        app,
        ["evaluate", "--provider", "fake", "--fixtures", str(BENCHMARK_PATH)],
    )
    assert res_post.exit_code == int(ExitCode.SUCCESS)
    assert "Dataset Size        : 8 (positive=5, negative=3)" in res_post.stdout
    assert "TP=5, FP=0, FN=0" in res_post.stdout
    assert "Precision           : 100.00%" in res_post.stdout
    assert "Recall              : 100.00%" in res_post.stdout

    # End-to-end evaluation
    res_e2e = runner.invoke(
        app,
        ["evaluate", "--provider", "fake", "--fixtures", str(BENCHMARK_PATH), "--end-to-end"],
    )
    assert res_e2e.exit_code == int(ExitCode.SUCCESS)
    assert "TP=0, FP=0, FN=5" in res_e2e.stdout
    assert "Recall              : 0.00%" in res_e2e.stdout
    assert "False Positive Rate : 0.00%" in res_e2e.stdout


def test_behavioral_reproduction_case_01_division_by_zero() -> None:
    """Case 1: Prove before-change code avoids division by zero while after-change crashes."""

    # Before-change logic (safe guard)
    def compute_rate_before(dn: float, delta_t: float) -> float:
        if delta_t <= 0:
            return 0.0
        return dn / delta_t

    # After-change logic (unguarded division)
    def compute_rate_after(dn: float, delta_t: float) -> float:
        return dn / delta_t

    # Normal execution works in both
    assert compute_rate_before(100.0, 5.0) == 20.0
    assert compute_rate_after(100.0, 5.0) == 20.0

    # Defect condition (delta_t == 0.0)
    assert compute_rate_before(100.0, 0.0) == 0.0
    with pytest.raises(ZeroDivisionError):
        compute_rate_after(100.0, 0.0)


def test_behavioral_reproduction_case_02_boundary_condition_underflow() -> None:
    """Case 2: Prove before-change clamps token budget to >= 0 while after-change underflows negative."""

    # Before-change logic (clamped lower bound)
    def calculate_budget_before(max_context_tokens: int, baseline_tokens: int) -> int:
        return max(0, max_context_tokens - baseline_tokens)

    # After-change logic (unclamped subtraction)
    def calculate_budget_after(max_context_tokens: int, baseline_tokens: int) -> int:
        return max_context_tokens - baseline_tokens

    # Standard positive budget works in both
    assert calculate_budget_before(10000, 3000) == 7000
    assert calculate_budget_after(10000, 3000) == 7000

    # Defect condition: baseline exceeds max context tokens
    assert calculate_budget_before(4000, 5000) == 0
    underflow_result = calculate_budget_after(4000, 5000)
    assert underflow_result == -1000
    assert underflow_result < 0, "Introduced code permits negative token budgets"


def test_behavioral_reproduction_case_03_missing_exception_handling() -> None:
    """Case 3: Prove before-change catches malformed cookies while after-change causes 500 crash."""

    class MockSerializer:
        def loads(self, val: str) -> dict[str, str]:
            if "\x00" in val:
                raise ValueError("Embedded null byte in session cookie payload")
            if val == "corrupt":
                raise KeyError("Missing secret signature")
            return {"user_id": "authenticated_user"}

    serializer = MockSerializer()

    # Before-change logic (handles ValueError and KeyError safely)
    def open_session_before(val: str) -> dict[str, str]:
        if not val:
            return {}
        try:
            return serializer.loads(val)
        except (ValueError, KeyError):
            return {}
        except Exception:
            return {}

    # After-change logic (omitted ValueError and KeyError handlers)
    def open_session_after(val: str) -> dict[str, str]:
        if not val:
            return {}
        try:
            return serializer.loads(val)
        except Exception as exc:
            if type(exc).__name__ == "BadSignature":
                return {}
            raise

    # Valid cookies decode in both
    assert open_session_before("valid_cookie") == {"user_id": "authenticated_user"}
    assert open_session_after("valid_cookie") == {"user_id": "authenticated_user"}

    # Malformed null-byte cookie safely handled before, crashes after
    assert open_session_before("malicious\x00payload") == {}
    with pytest.raises(ValueError, match="Embedded null byte"):
        open_session_after("malicious\x00payload")


def test_behavioral_reproduction_case_04_client_state_mutation() -> None:
    """Case 4: Prove before-change preserves session defaults while after-change leaks headers."""

    # Before-change logic (shallow copy preserves original instance defaults)
    class SessionBefore:
        def __init__(self, default_headers: dict[str, str]) -> None:
            self.default_headers = default_headers

        def prepare_headers(self, custom: dict[str, str] | None = None) -> dict[str, str]:
            headers = dict(self.default_headers)
            if custom:
                headers.update(custom)
            return headers

    # After-change logic (aliasing self.default_headers mutates instance defaults)
    class SessionAfter:
        def __init__(self, default_headers: dict[str, str]) -> None:
            self.default_headers = default_headers

        def prepare_headers(self, custom: dict[str, str] | None = None) -> dict[str, str]:
            headers = self.default_headers
            if custom:
                headers.update(custom)
            return headers

    # Before: first request does not contaminate subsequent requests
    s_before = SessionBefore({"User-Agent": "Deicide/1.0"})
    res1 = s_before.prepare_headers({"Authorization": "Bearer secret123"})
    assert res1 == {"User-Agent": "Deicide/1.0", "Authorization": "Bearer secret123"}
    res2 = s_before.prepare_headers()
    assert res2 == {"User-Agent": "Deicide/1.0"}
    assert "Authorization" not in s_before.default_headers

    # After: first request mutates default_headers in-place, leaking secrets into second request
    s_after = SessionAfter({"User-Agent": "Deicide/1.0"})
    res1_after = s_after.prepare_headers({"Authorization": "Bearer secret123"})
    assert res1_after == {"User-Agent": "Deicide/1.0", "Authorization": "Bearer secret123"}
    res2_after = s_after.prepare_headers()
    assert "Authorization" in res2_after, "State mutation leaked Authorization header into subsequent request"
    assert s_after.default_headers["Authorization"] == "Bearer secret123"


def test_behavioral_reproduction_case_05_cross_file_signature_break() -> None:
    """Case 5: Prove before-change caller succeeds with default currency while after-change raises TypeError."""

    # Before-change logic (default parameter present)
    def charge_card_before(amount_cents: int, currency: str = "USD") -> dict[str, str]:
        return {"status": "success", "charged": str(amount_cents), "currency": currency}

    def process_checkout_before(order_total: int) -> dict[str, str]:
        return charge_card_before(order_total)

    # After-change logic (required positional argument without default)
    def charge_card_after(amount_cents: int, currency: str) -> dict[str, str]:
        return {"status": "success", "charged": str(amount_cents), "currency": currency}

    def process_checkout_after(order_total: int) -> dict[str, str]:
        return charge_card_after(order_total)

    # Before: 1-argument invocation succeeds using default currency
    result_before = process_checkout_before(5000)
    assert result_before["status"] == "success"
    assert result_before["currency"] == "USD"

    # After: 1-argument invocation fails with missing required positional argument
    with pytest.raises(TypeError, match="missing 1 required positional argument: 'currency'"):
        process_checkout_after(5000)


def test_behavioral_validation_negative_cases() -> None:
    """Negative cases: Prove clean changes introduce no behavioral regressions."""

    # Clean Case 6: tqdm fix prevents ZeroDivisionError
    def compute_rate_fixed(dn: float, delta_t: float) -> float:
        if delta_t <= 0:
            return 0.0
        return dn / delta_t

    assert compute_rate_fixed(50.0, 0.0) == 0.0
    assert compute_rate_fixed(50.0, 2.0) == 25.0

    # Clean Case 7: refactor preserves exact dictionary transformation
    def format_headers_original(raw: dict[str, str]) -> dict[str, str]:
        formatted = {}
        for k, v in raw.items():
            formatted[k.strip().lower()] = v.strip()
        return formatted

    def _normalize_key(key: str) -> str:
        return key.strip().lower()

    def format_headers_refactored(raw: dict[str, str]) -> dict[str, str]:
        formatted = {}
        for k, v in raw.items():
            formatted[_normalize_key(k)] = v.strip()
        return formatted

    sample = {" CONTENT-TYPE ": " text/html ", " X-REQUEST-ID ": " 12345 "}
    assert format_headers_original(sample) == format_headers_refactored(sample)

    # Clean Case 8: docstring update does not modify execution behavior
    def execute_before(module_name: str, timeout_seconds: float = 30.0) -> bool:
        return True

    def execute_with_docs(module_name: str, timeout_seconds: float = 30.0) -> bool:
        # Execute code analysis pipeline for the selected module.
        # module_name: Unique registered identifier for the analysis module.
        # timeout_seconds: Per-module execution deadline in seconds.
        return True

    assert execute_before("code_review") is True
    assert execute_with_docs("code_review") is True
