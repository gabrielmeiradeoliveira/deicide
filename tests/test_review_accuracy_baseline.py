"""Tests for DEICIDE MVP 1.4 Review Accuracy Baseline evaluation infrastructure."""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from pydantic import SecretStr
from typer.testing import CliRunner

from deicide.cli import app
from deicide.core.models import (
    CandidateFinding,
    Confidence,
    ExecutionBudget,
    FindingCategory,
    LineRange,
    Severity,
)
from deicide.evaluation.models import (
    EvaluationFixture,
    ExpectedFindingSpec,
)
from deicide.evaluation.runner import evaluate_case, evaluate_corpus
from deicide.providers.fake import FakeProvider
from deicide.providers.nvidia import NvidiaProvider

runner = CliRunner()


def test_provenance_classification_resolution() -> None:
    """Ensure fixtures correctly classify fully_verified, reference_modeled, and unverified cases."""
    # 1. Authentic positive bug fix
    positive_fixture = EvaluationFixture(
        case_id="pos-test",
        description="Bug fix",
        diff_text="diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1,1 +1,1 @@\n-old\n+new\n",
        file_contents={"a.py": "new\n"},
        expected_findings=[ExpectedFindingSpec(file_path="a.py", line_range=LineRange(start_line=1, end_line=1))],
        repository="psf/requests",
        commit_id="74ea7cf7a6a27a4eeb2ae24e162bcc942a6706d5",
        is_clean=False,
        fix_reference="CVE-2023-32681",
    )
    assert positive_fixture.resolved_provenance == "fully_verified"

    # 2. Authentic local DEICIDE fix
    local_fixture = EvaluationFixture(
        case_id="local-test",
        description="Local fix",
        diff_text="diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1,1 +1,1 @@\n-old\n+new\n",
        file_contents={"a.py": "new\n"},
        expected_findings=[],
        repository="deicide",
        commit_id="cfc826d",
        is_clean=True,
        fix_reference="Commit cfc826d",
    )
    assert local_fixture.resolved_provenance == "fully_verified"

    # 3. Clean reference-modeled upstream example
    clean_modeled_fixture = EvaluationFixture(
        case_id="clean-modeled-test",
        description="Clean refactoring",
        diff_text="diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1,1 +1,1 @@\n-old\n+new\n",
        file_contents={"a.py": "new\n"},
        expected_findings=[],
        repository="pallets/flask",
        commit_id="8f4b005e836b2d28b12f6a978f6c38a2eefea8f1",
        is_clean=True,
        fix_reference="PR #4512",
    )
    assert clean_modeled_fixture.resolved_provenance == "reference_modeled"


def test_multiple_findings_do_not_inflate_true_positives() -> None:
    """Ensure duplicate findings targeting the same expected bug only count 1 TP and remaining as FP."""
    fixture = EvaluationFixture(
        case_id="dup-inflation-test",
        description="Test case with single bug",
        diff_text="diff --git a/src/math.py b/src/math.py\n--- a/src/math.py\n+++ b/src/math.py\n@@ -1,2 +1,2 @@\n-def f(x):\n+def f(x):\n-    return x\n+    return 10 / x\n",
        file_contents={"src/math.py": "def f(x):\n    return 10 / x\n"},
        expected_findings=[
            ExpectedFindingSpec(
                file_path="src/math.py",
                line_range=LineRange(start_line=2, end_line=2),
                category=None,
                severity=Severity.HIGH,
            )
        ],
        candidate_findings=[
            # Candidate 1: boundary_condition bug on line 2
            CandidateFinding(
                category=FindingCategory.BOUNDARY_CONDITION,
                severity=Severity.HIGH,
                confidence=Confidence.HIGH,
                file_path="src/math.py",
                start_line=2,
                end_line=2,
                title="ZeroDivisionError in f()",
                description="Dividing by x when x=0 causes ZeroDivisionError.",
                evidence="return 10 / x",
            ),
            # Candidate 2: distinct category resource_management within tolerance range
            CandidateFinding(
                category=FindingCategory.RESOURCE_MANAGEMENT,
                severity=Severity.HIGH,
                confidence=Confidence.HIGH,
                file_path="src/math.py",
                start_line=2,
                end_line=2,
                title="Unclosed socket connection",
                description="Socket handle not closed before function return.",
                evidence="return 10 / x",
            ),
        ],
    )

    provider = FakeProvider()
    res = evaluate_case(fixture, provider=provider, force_end_to_end=False)

    # Incompatible categories are preserved as distinct findings by Finding Intelligence.
    # The first finding matches the single expected bug (TP=1). The second finding attempts to match,
    # but expected finding 0 is already matched, so the second finding CANNOT count as TP:
    assert len(res.actual_findings) == 2
    assert res.true_positives == 1
    assert res.false_positives == 1  # Cannot inflate TP to 2!
    assert res.false_negatives == 0


def test_zero_findings_do_not_receive_artificial_100_percent_precision() -> None:
    """Ensure that an evaluation producing 0 findings yields 0.0% precision, not 100%."""
    fixture = EvaluationFixture(
        case_id="empty-provider-test",
        description="Fixture with bug but provider outputs nothing",
        diff_text="diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1,1 +1,1 @@\n-old\n+new\n",
        file_contents={"a.py": "new\n"},
        expected_findings=[ExpectedFindingSpec(file_path="a.py", line_range=LineRange(start_line=1, end_line=1))],
        candidate_findings=[],
    )
    empty_provider = FakeProvider(scripted_responses=['{"findings": []}'])
    summary = evaluate_corpus([fixture], provider=empty_provider, force_end_to_end=True)

    assert summary.true_positives == 0
    assert summary.false_positives == 0
    assert summary.false_negatives == 1
    assert summary.precision == 0.0  # Must be 0.0, never 1.0!
    assert summary.recall == 0.0


def test_clean_cases_drive_false_positive_rate() -> None:
    """Ensure clean cases that produce findings correctly calculate false positive rate."""
    clean_fixture1 = EvaluationFixture(
        case_id="clean-1",
        description="Clean case with hallucinated finding",
        diff_text="diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1,1 +1,1 @@\n-old\n+new\n",
        file_contents={"a.py": "new\n"},
        expected_findings=[],
        is_clean=True,
        candidate_findings=[
            CandidateFinding(
                category=FindingCategory.CORRECTNESS,
                severity=Severity.INFO,
                confidence=Confidence.HIGH,
                file_path="a.py",
                start_line=1,
                end_line=1,
                title="Unnecessary change",
                description="Clean change flagged improperly.",
                evidence="new",
            )
        ],
    )
    clean_fixture2 = EvaluationFixture(
        case_id="clean-2",
        description="Clean case without any findings",
        diff_text="diff --git a/b.py b/b.py\n--- a/b.py\n+++ b/b.py\n@@ -1,1 +1,1 @@\n-old\n+new\n",
        file_contents={"b.py": "new\n"},
        expected_findings=[],
        is_clean=True,
        candidate_findings=[],
    )

    provider = FakeProvider()
    summary = evaluate_corpus([clean_fixture1, clean_fixture2], provider=provider, force_end_to_end=False)

    assert summary.positive_cases == 0
    assert summary.negative_cases == 2
    assert summary.true_positives == 0
    assert summary.false_positives == 1
    assert summary.false_positive_rate == 0.50  # 1 out of 2 clean cases had FP


def test_real_provider_evaluation_with_mocked_nvidia_transport() -> None:
    """Demonstrate genuine provider evaluation using NvidiaProvider with deterministic transport mock."""
    secret_key = "nvapi-TestBenchmarkKey9876543210"
    requests_captured: list[str] = []

    def mock_transport(req: urllib.request.Request, _timeout: float) -> tuple[int, dict[str, str], str]:
        body = req.data.decode("utf-8") if req.data else ""
        requests_captured.append(body)

        # Mock LLM response identifying the bug
        response_payload = {
            "id": "chatcmpl-mock-benchmark",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": json.dumps(
                            {
                                "findings": [
                                    {
                                        "category": "boundary_condition",
                                        "severity": "high",
                                        "confidence": "high",
                                        "file_path": "src/stats.py",
                                        "start_line": 2,
                                        "end_line": 2,
                                        "title": "ZeroDivisionError on empty values",
                                        "description": "Division by len(values) without guard.",
                                        "evidence": "return sum(values) / len(values)",
                                        "suggested_remediation": "Guard against empty collection.",
                                    }
                                ]
                            }
                        ),
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 450,
                "completion_tokens": 85,
                "total_tokens": 535,
            },
        }
        return 200, {"content-type": "application/json"}, json.dumps(response_payload)

    provider = NvidiaProvider(
        api_key=SecretStr(secret_key),
        sleep_fn=lambda _: None,
        transport_fn=mock_transport,
    )

    fixture = EvaluationFixture(
        case_id="nvidia-eval-test",
        description="Division by zero bug",
        diff_text="diff --git a/src/stats.py b/src/stats.py\n--- a/src/stats.py\n+++ b/src/stats.py\n@@ -1,2 +1,2 @@\n def avg(values):\n-    return 0\n+    return sum(values) / len(values)\n",
        file_contents={"src/stats.py": "def avg(values):\n    return sum(values) / len(values)\n"},
        expected_findings=[
            ExpectedFindingSpec(
                file_path="src/stats.py",
                line_range=LineRange(start_line=2, end_line=2),
                category=FindingCategory.BOUNDARY_CONDITION,
                severity=Severity.HIGH,
            )
        ],
    )

    summary = evaluate_corpus(
        [fixture],
        provider=provider,
        force_end_to_end=True,
    )

    assert len(requests_captured) == 1
    assert summary.true_positives == 1
    assert summary.false_positives == 0
    assert summary.false_negatives == 0
    assert summary.precision == 1.0
    assert summary.recall == 1.0
    assert summary.usage.total_tokens == 535
    assert not summary.usage.is_estimated
    assert summary.evaluation_mode == "end_to_end"


def test_external_provider_requires_explicit_opt_in() -> None:
    """Ensure deicide evaluate rejects external provider execution without --allow-external."""
    # Running without --allow-external must fail with credentials/configuration error
    result = runner.invoke(
        app,
        ["evaluate", "--provider", "nvidia"],
        env={"NVIDIA_API_KEY": "nvapi-test12345"},
    )
    assert result.exit_code != 0
    assert "External code transmission to provider 'nvidia' is not enabled" in result.output or (
        "Missing API credentials" in result.output
    )


def test_budget_cutoff_stops_corpus_safely() -> None:
    """Ensure evaluation stops safely when max requests or token budget is reached."""
    fixture1 = EvaluationFixture(
        case_id="case-1",
        description="First case",
        diff_text="diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1,1 +1,1 @@\n-old\n+new\n",
        file_contents={"a.py": "new\n"},
        expected_findings=[],
    )
    fixture2 = EvaluationFixture(
        case_id="case-2",
        description="Second case",
        diff_text="diff --git a/b.py b/b.py\n--- a/b.py\n+++ b/b.py\n@@ -1,1 +1,1 @@\n-old\n+new\n",
        file_contents={"b.py": "new\n"},
        expected_findings=[],
    )

    # Budget allows only 1 request
    budget = ExecutionBudget(max_requests=1)
    provider = FakeProvider(scripted_responses=['{"findings": []}', '{"findings": []}'])

    summary = evaluate_corpus(
        [fixture1, fixture2],
        provider=provider,
        budget=budget,
        force_end_to_end=True,
    )

    assert summary.budget_exhausted is True
    # Evaluated fixture 1, hit request budget on fixture 2:
    assert summary.dataset_size == 2
    assert summary.case_results[0].status == "completed"
    assert summary.case_results[1].status == "budget_exhausted"
    assert "request budget exceeded" in (summary.case_results[1].error or "").lower()


def test_api_error_preserves_failures_and_redacts_secrets() -> None:
    """Ensure provider API errors record per-case failure, redact secrets, and are not counted as passes."""
    secret_key = "nvapi-SuperSecretToken123456789"

    def failing_transport(req: urllib.request.Request, _timeout: float) -> tuple[int, dict[str, str], str]:
        raise urllib.error.HTTPError(
            url=req.full_url,
            code=500,
            msg=f"Internal Server Error for key {secret_key}",
            hdrs=None,  # type: ignore[arg-type]
            fp=None,
        )

    provider = NvidiaProvider(
        api_key=SecretStr(secret_key),
        max_retries=0,
        sleep_fn=lambda _: None,
        transport_fn=failing_transport,
    )

    fixture = EvaluationFixture(
        case_id="failing-api-case",
        description="Test case encountering 500 error",
        diff_text="diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1,1 +1,1 @@\n-old\n+new\n",
        file_contents={"a.py": "new\n"},
        expected_findings=[ExpectedFindingSpec(file_path="a.py", line_range=LineRange(start_line=1, end_line=1))],
    )

    summary = evaluate_corpus([fixture], provider=provider, force_end_to_end=True)

    assert summary.failed_cases == 1
    case_res = summary.case_results[0]
    assert case_res.status == "failed"
    assert case_res.error is not None
    # Verify secret is redacted:
    assert secret_key not in case_res.error
    assert "[REDACTED]" in case_res.error
    # The expected finding was NOT found due to API failure -> counted as false negative!
    assert case_res.false_negatives == 1
    assert summary.false_negatives == 1
    assert summary.recall == 0.0


def test_cli_max_cases_limits_evaluation() -> None:
    """Ensure --max-cases limits the number of fixtures evaluated via CLI."""
    result = runner.invoke(
        app,
        [
            "evaluate",
            "--fixtures",
            "evaluations/fixtures/real_world_benchmark.json",
            "--max-cases",
            "3",
            "--format",
            "json",
        ],
    )
    assert result.exit_code == 0
    parsed = json.loads(result.output)
    assert parsed["dataset_size"] == 3
    assert parsed["max_cases_applied"] == 3
    assert len(parsed["case_results"]) == 3
