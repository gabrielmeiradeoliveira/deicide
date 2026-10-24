"""Comprehensive unit and integration tests for the DEICIDE MVP 1.3 Finding Intelligence Engine."""

from __future__ import annotations

import json
from pathlib import Path

from deicide.cli import build_default_registry
from deicide.context.context_models import (
    ContextRelationshipCategory,
    ContextSelectionResult,
    SelectedSnippet,
)
from deicide.core.config import DeicideConfig
from deicide.core.models import (
    CandidateFinding,
    Confidence,
    Finding,
    FindingCategory,
    LineRange,
    Severity,
)
from deicide.core.orchestrator import AnalysisOrchestrator
from deicide.git.analyzer import parse_unified_diff
from deicide.git.models import ContextSnippet, DiffBundle, FileContext
from deicide.intelligence import (
    ChangeRelevance,
    EvidenceAssessment,
    EvidenceQuality,
    FindingIntelligenceEngine,
    are_findings_duplicates,
    assess_evidence,
    compute_finding_id,
    deduplicate_findings_semantic,
    normalize_finding_severity,
    prioritize_findings,
    validate_and_assess_candidates,
)
from deicide.providers.fake import FakeProvider


def _make_diff_and_context() -> tuple[DiffBundle, dict[str, FileContext]]:
    diff_text = (
        "diff --git a/src/service.py b/src/service.py\n"
        "--- a/src/service.py\n"
        "+++ b/src/service.py\n"
        "@@ -1,4 +1,12 @@\n"
        " def process_items(items: list[int], cache=[]) -> float:\n"
        "+    if items is 'invalid':\n"
        "+        return 0.0\n"
        "+    try:\n"
        "+        ratio = sum(items) / len(items)\n"
        "+    except Exception:\n"
        "+        pass\n"
        "+    bonus = 100 / 0\n"
        "+    adjusted = ratio - bonus\n"
        "     return adjusted\n"
    )
    files = parse_unified_diff(diff_text)
    bundle = DiffBundle(
        repository_root="/repo",
        base_revision="HEAD",
        target_description="working-tree",
        files=files,
    )
    lines = {
        1: "def process_items(items: list[int], cache=[]) -> float:",
        2: "    if items is 'invalid':",
        3: "        return 0.0",
        4: "    try:",
        5: "        ratio = sum(items) / len(items)",
        6: "    except Exception:",
        7: "        pass",
        8: "    bonus = 100 / 0",
        9: "    adjusted = ratio - bonus",
        10: "    return adjusted",
        25: "def unrelated_helper(val: int) -> int:",
        26: "    return val * 2",
    }
    contexts = {
        "src/service.py": FileContext(
            file_path="src/service.py",
            total_lines=30,
            snippets=[
                ContextSnippet(start_line=1, end_line=10, lines={k: v for k, v in lines.items() if k <= 10}),
                ContextSnippet(start_line=25, end_line=26, lines={k: v for k, v in lines.items() if k >= 25}),
            ],
        )
    }
    return bundle, contexts


# ============================================================================
# 1. EVIDENCE VALIDATION & QUALITY ASSESSMENT TESTS
# ============================================================================


def test_evidence_validation_exact_and_invented_symbols() -> None:
    bundle, contexts = _make_diff_and_context()
    fdiff = bundle.get_file_diff("src/service.py")
    assert fdiff is not None
    fctx = contexts["src/service.py"]

    # 1. Exact evidence on added line
    exact_assessment = assess_evidence(
        evidence="ratio = sum(items) / len(items)",
        line_range=LineRange(start_line=5, end_line=5),
        file_diff=fdiff,
        file_context=fctx,
    )
    assert exact_assessment.is_valid is True
    assert exact_assessment.quality == EvidenceQuality.EXACT
    assert exact_assessment.change_relevance == ChangeRelevance.DIRECT_ADDED_OR_MODIFIED
    assert exact_assessment.invented_symbols == []

    # 2. Completely invented symbol/code not in file -> rejected as UNSUPPORTED
    invented_assessment = assess_evidence(
        evidence="result = compute_nonexistent_matrix(items)",
        line_range=LineRange(start_line=5, end_line=5),
        file_diff=fdiff,
        file_context=fctx,
    )
    assert invented_assessment.is_valid is False
    assert invented_assessment.quality == EvidenceQuality.UNSUPPORTED
    assert "compute_nonexistent_matrix" in invented_assessment.invented_symbols


def test_evidence_validation_supplementary_context_not_reportable() -> None:
    bundle, contexts = _make_diff_and_context()
    supp_context = ContextSelectionResult(
        selected_snippets=[
            SelectedSnippet(
                snippet_id="snip-caller",
                file_path="src/caller.py",
                start_line=10,
                end_line=14,
                content="def invoke_service(data):\n    return process_items(data)",
                relationship=ContextRelationshipCategory.DIRECT_CALLER,
            )
        ],
        serialized_text="=== CONTEXT SNIPPET: src/caller.py ===",
    )

    candidates = [
        # Finding targeting supplementary file src/caller.py directly
        CandidateFinding(
            category=FindingCategory.CORRECTNESS,
            severity=Severity.HIGH,
            confidence=Confidence.HIGH,
            file_path="src/caller.py",
            start_line=11,
            end_line=11,
            title="Caller passes unvalidated data",
            description="Caller invokes process_items without checking data.",
            evidence="return process_items(data)",
        ),
        # Finding claiming changed file src/service.py:5, but citing evidence from src/caller.py
        CandidateFinding(
            category=FindingCategory.CORRECTNESS,
            severity=Severity.HIGH,
            confidence=Confidence.HIGH,
            file_path="src/service.py",
            start_line=5,
            end_line=5,
            title="Misattributed supplementary snippet evidence",
            description="Cites caller code as if it were in service.py.",
            evidence="def invoke_service(data):",
        ),
    ]

    validated, assessments, rejected, downgraded, warnings = validate_and_assess_candidates(
        module_id="code_review",
        candidates=candidates,
        diff_bundle=bundle,
        contexts=contexts,
        context_result=supp_context,
    )

    assert validated == []
    assert rejected == 2
    assert any(w.code == "FINDING_REJECTED_UNCHANGED_FILE" and "supplementary context" in w.message for w in warnings)
    assert any(
        w.code == "FINDING_REJECTED_UNSUPPORTED_EVIDENCE" and "supplementary context" in w.message for w in warnings
    )


def test_evidence_validation_downgrades_partial_or_weak_evidence() -> None:
    bundle, contexts = _make_diff_and_context()

    candidates = [
        # Candidate citing line 5, but its evidence actually comes from line 25 (far outside hunk)
        CandidateFinding(
            category=FindingCategory.LOGIC_ERROR,
            severity=Severity.HIGH,
            confidence=Confidence.HIGH,
            file_path="src/service.py",
            start_line=5,
            end_line=5,
            title="Mismatched line citation",
            description="Cites helper code from line 26 while reporting line 5.",
            evidence="return val * 2",
        ),
        # Candidate with multi-line evidence where 1 of 2 lines is accurate and 1 is fabricated
        CandidateFinding(
            category=FindingCategory.BOUNDARY_CONDITION,
            severity=Severity.CRITICAL,
            confidence=Confidence.HIGH,
            file_path="src/service.py",
            start_line=5,
            end_line=5,
            title="ZeroDivisionError on empty items",
            description="Dividing sum(items) by len(items) fails when items is empty.",
            evidence="ratio = sum(items) / len(items)\nraise_custom_alarm(ratio)",
        ),
    ]

    engine = FindingIntelligenceEngine()
    result = engine.process_candidates(
        module_id="code_review",
        candidates=candidates,
        diff_bundle=bundle,
        contexts=contexts,
    )

    assert len(result.findings) == 2
    assert result.downgraded_count == 2
    assert any(w.code == "FINDING_DOWNGRADED_WEAK_EVIDENCE" for w in result.warnings)

    for finding in result.findings:
        # Both started with Confidence.HIGH and were downgraded
        assert finding.confidence in {Confidence.MEDIUM, Confidence.LOW}
        # The CRITICAL finding with weak/partial evidence was also capped in severity
        assert finding.severity != Severity.CRITICAL


def test_evidence_validation_preserves_deletion_only_diff_behavior() -> None:
    deletion_diff = (
        "diff --git a/calculator.py b/calculator.py\n"
        "--- a/calculator.py\n"
        "+++ b/calculator.py\n"
        "@@ -1,4 +1,2 @@\n"
        " def calculate_average(values):\n"
        "-    if not values:\n"
        "-        return 0\n"
        "     return sum(values) / len(values)\n"
    )
    files = parse_unified_diff(deletion_diff)
    bundle = DiffBundle(
        repository_root="/repo",
        base_revision="HEAD",
        target_description="deletion-test",
        files=files,
    )
    contexts = {
        "calculator.py": FileContext(
            file_path="calculator.py",
            total_lines=2,
            snippets=[
                ContextSnippet(
                    start_line=1,
                    end_line=2,
                    lines={
                        1: "def calculate_average(values):",
                        2: "    return sum(values) / len(values)",
                    },
                )
            ],
        )
    }

    cand = CandidateFinding(
        category=FindingCategory.BOUNDARY_CONDITION,
        severity=Severity.HIGH,
        confidence=Confidence.HIGH,
        file_path="calculator.py",
        start_line=2,
        end_line=2,
        title="Potential ZeroDivisionError when 'values' is empty",
        description="Removing the empty guard causes sum(values) / len(values) to fail on empty input.",
        evidence="return sum(values) / len(values)",
    )

    engine = FindingIntelligenceEngine()
    result = engine.process_candidates(
        module_id="code_review",
        candidates=[cand],
        diff_bundle=bundle,
        contexts=contexts,
    )

    assert len(result.findings) == 1
    finding = result.findings[0]
    assert finding.severity == Severity.HIGH
    assert finding.confidence == Confidence.HIGH
    assessment = result.assessments[finding.id]
    assert assessment.quality == EvidenceQuality.EXACT
    assert assessment.change_relevance == ChangeRelevance.DELETION_ADJACENT


# ============================================================================
# 2. SEMANTIC DEDUPLICATION TESTS
# ============================================================================


def test_semantic_deduplication_merges_paraphrased_same_issue_and_enriches_remediation() -> None:
    f1 = Finding(
        id="dcd-111111111111",
        module_id="code_review",
        category=FindingCategory.BOUNDARY_CONDITION,
        severity=Severity.HIGH,
        confidence=Confidence.HIGH,
        file_path="src/service.py",
        line_range=LineRange(start_line=5, end_line=5),
        title="Potential ZeroDivisionError when 'items' is empty",
        description="Dividing by len(items) without checking emptiness raises ZeroDivisionError.",
        evidence="ratio = sum(items) / len(items)",
        suggested_remediation=None,
    )
    f2 = Finding(
        id="dcd-222222222222",
        module_id="code_review",
        category=FindingCategory.CORRECTNESS,
        severity=Severity.MEDIUM,
        confidence=Confidence.HIGH,
        file_path="src/service.py",
        line_range=LineRange(start_line=4, end_line=5),
        title="Division by zero on empty list in process_items",
        description="When items is an empty list, len(items) is zero and causes division by zero.",
        evidence="5: ratio = sum(items) / len(items)",
        suggested_remediation="Check 'if not items:' before dividing.",
    )

    kept, deduped_count, groups = deduplicate_findings_semantic([f2, f1])
    assert len(kept) == 1
    assert deduped_count == 1
    assert kept[0].id == f1.id
    assert kept[0].severity == Severity.HIGH
    # Survivor inherited remediation from f2
    assert kept[0].suggested_remediation == "Check 'if not items:' before dividing."
    assert len(groups) == 1
    assert groups[0].kept_finding_id == f1.id
    assert groups[0].merged_finding_ids == [f2.id]


def test_semantic_deduplication_preserves_distinct_bugs_on_adjacent_and_same_lines() -> None:
    # Distinct Bug 1 on line 8: Division by literal zero
    f_div_zero = Finding(
        id="dcd-aaaa11112222",
        module_id="code_review",
        category=FindingCategory.CORRECTNESS,
        severity=Severity.CRITICAL,
        confidence=Confidence.HIGH,
        file_path="src/service.py",
        line_range=LineRange(start_line=8, end_line=8),
        title="Division by literal zero",
        description="Computing 100 / 0 always raises ZeroDivisionError at runtime.",
        evidence="bonus = 100 / 0",
    )
    # Distinct Bug 2 on adjacent line 9 with the SAME category (CORRECTNESS): Wrong arithmetic operator
    f_wrong_op = Finding(
        id="dcd-bbbb33334444",
        module_id="code_review",
        category=FindingCategory.CORRECTNESS,
        severity=Severity.HIGH,
        confidence=Confidence.HIGH,
        file_path="src/service.py",
        line_range=LineRange(start_line=9, end_line=9),
        title="Bonus subtracted instead of added to ratio",
        description="The calculation subtracts bonus from ratio instead of adding it.",
        evidence="adjusted = ratio - bonus",
    )
    # Distinct Bug 3 on line 1 (same file): Mutable default argument
    f_mutable_default = Finding(
        id="dcd-cccc55556666",
        module_id="code_review",
        category=FindingCategory.STATE_MUTATION,
        severity=Severity.HIGH,
        confidence=Confidence.HIGH,
        file_path="src/service.py",
        line_range=LineRange(start_line=1, end_line=1),
        title="Mutable default argument shared across calls",
        description="Default list cache=[] persists state across function invocations.",
        evidence="def process_items(items: list[int], cache=[]) -> float:",
    )

    is_dup, _score, reason = are_findings_duplicates(f_div_zero, f_wrong_op)
    assert is_dup is False
    assert reason == "distinct_adjacent_lines_and_evidence"

    kept, deduped_count, groups = deduplicate_findings_semantic([f_div_zero, f_wrong_op, f_mutable_default])
    assert len(kept) == 3
    assert deduped_count == 0
    assert groups == []


# ============================================================================
# 3. SEVERITY NORMALIZATION TESTS
# ============================================================================


def test_severity_normalization_rules_and_no_arbitrary_upgrades() -> None:
    # 1. Low confidence with CRITICAL severity -> capped at MEDIUM
    f_low_conf = Finding(
        id="dcd-lowconf00001",
        module_id="code_review",
        category=FindingCategory.LOGIC_ERROR,
        severity=Severity.CRITICAL,
        confidence=Confidence.LOW,
        file_path="src/service.py",
        line_range=LineRange(start_line=2, end_line=2),
        title="Possible logic branch issue",
        description="Condition may fail depending on caller input types.",
        evidence="if items is 'invalid':",
    )
    norm_low, adj_low = normalize_finding_severity(f_low_conf)
    assert norm_low.severity == Severity.MEDIUM
    assert adj_low is not None
    assert adj_low.rule_id == "SEV_LOW_CONFIDENCE_CAP"

    # 2. ERROR_HANDLING claiming CRITICAL without unconditional crash -> capped at HIGH
    f_err_cat = Finding(
        id="dcd-errcat000002",
        module_id="code_review",
        category=FindingCategory.ERROR_HANDLING,
        severity=Severity.CRITICAL,
        confidence=Confidence.HIGH,
        file_path="src/service.py",
        line_range=LineRange(start_line=6, end_line=7),
        title="Silent exception swallowing",
        description="Bare except pass masks runtime exceptions.",
        evidence="except Exception:\n    pass",
    )
    norm_cat, adj_cat = normalize_finding_severity(f_err_cat)
    assert norm_cat.severity == Severity.HIGH
    assert adj_cat is not None
    assert adj_cat.rule_id == "SEV_CATEGORY_CEILING"

    # 3. Minor framing in description -> capped at LOW
    f_minor = Finding(
        id="dcd-minor0000003",
        module_id="code_review",
        category=FindingCategory.LOGIC_ERROR,
        severity=Severity.HIGH,
        confidence=Confidence.HIGH,
        file_path="src/service.py",
        line_range=LineRange(start_line=2, end_line=2),
        title="Redundant check",
        description="This is a minor issue that is mostly trivial in practice.",
        evidence="if items is 'invalid':",
    )
    norm_minor, adj_minor = normalize_finding_severity(f_minor)
    assert norm_minor.severity == Severity.LOW
    assert adj_minor is not None
    assert adj_minor.rule_id == "SEV_MINOR_FRAMING_CAP"

    # 4. Valid LOW/MEDIUM finding is NEVER arbitrarily upgraded
    f_valid_low = Finding(
        id="dcd-validlow0004",
        module_id="code_review",
        category=FindingCategory.CORRECTNESS,
        severity=Severity.LOW,
        confidence=Confidence.HIGH,
        file_path="src/service.py",
        line_range=LineRange(start_line=8, end_line=8),
        title="Division by zero",
        description="Dividing 100 by 0 raises ZeroDivisionError.",
        evidence="bonus = 100 / 0",
    )
    norm_unchanged, adj_none = normalize_finding_severity(f_valid_low)
    assert norm_unchanged.severity == Severity.LOW
    assert adj_none is None


# ============================================================================
# 4. DETERMINISTIC PRIORITIZATION TESTS
# ============================================================================


def test_deterministic_prioritization_order_invariance() -> None:
    f_crit = Finding(
        id="dcd-0001",
        module_id="code_review",
        category=FindingCategory.CORRECTNESS,
        severity=Severity.CRITICAL,
        confidence=Confidence.HIGH,
        file_path="src/z_last.py",
        line_range=LineRange(start_line=50, end_line=50),
        title="Critical bug in z_last.py",
        description="Unconditional crash at runtime due to zero division.",
        evidence="x = 1 / 0",
    )
    f_high_exact = Finding(
        id="dcd-0002",
        module_id="code_review",
        category=FindingCategory.BOUNDARY_CONDITION,
        severity=Severity.HIGH,
        confidence=Confidence.HIGH,
        file_path="src/b.py",
        line_range=LineRange(start_line=20, end_line=20),
        title="High bug with exact evidence",
        description="Off-by-one error causes IndexError.",
        evidence="for i in range(len(a) + 1):",
    )
    f_high_partial = Finding(
        id="dcd-0003",
        module_id="code_review",
        category=FindingCategory.BOUNDARY_CONDITION,
        severity=Severity.HIGH,
        confidence=Confidence.HIGH,
        file_path="src/a.py",
        line_range=LineRange(start_line=5, end_line=5),
        title="High bug with partial evidence",
        description="Empty list causes division by zero.",
        evidence="sum(a) / len(a)",
    )
    f_med = Finding(
        id="dcd-0004",
        module_id="code_review",
        category=FindingCategory.ERROR_HANDLING,
        severity=Severity.MEDIUM,
        confidence=Confidence.HIGH,
        file_path="src/a.py",
        line_range=LineRange(start_line=1, end_line=1),
        title="Medium bug in a.py",
        description="Silent exception swallowing masks errors.",
        evidence="except Exception: pass",
    )

    assessments = {
        "dcd-0001": EvidenceAssessment(
            is_valid=True,
            quality=EvidenceQuality.EXACT,
            change_relevance=ChangeRelevance.DIRECT_ADDED_OR_MODIFIED,
        ),
        "dcd-0002": EvidenceAssessment(
            is_valid=True,
            quality=EvidenceQuality.EXACT,
            change_relevance=ChangeRelevance.DIRECT_ADDED_OR_MODIFIED,
        ),
        "dcd-0003": EvidenceAssessment(
            is_valid=True,
            quality=EvidenceQuality.PARTIAL,
            change_relevance=ChangeRelevance.HUNK_CONTEXT,
        ),
        "dcd-0004": EvidenceAssessment(
            is_valid=True,
            quality=EvidenceQuality.EXACT,
            change_relevance=ChangeRelevance.DIRECT_ADDED_OR_MODIFIED,
        ),
    }

    order_1 = prioritize_findings([f_med, f_high_partial, f_crit, f_high_exact], assessments)
    order_2 = prioritize_findings([f_high_exact, f_med, f_crit, f_high_partial], assessments)

    expected_ids = ["dcd-0001", "dcd-0002", "dcd-0003", "dcd-0004"]
    assert [f.id for f in order_1] == expected_ids
    assert [f.id for f in order_2] == expected_ids


# ============================================================================
# 5. END-TO-END ORCHESTRATOR INTEGRATION TEST
# ============================================================================


def test_orchestrator_end_to_end_finding_intelligence_pipeline(temp_git_repo: Path) -> None:
    calc_file = temp_git_repo / "src" / "calc.py"
    calc_file.write_text(
        "def process(items: list[int]) -> float:\n"
        "    try:\n"
        "        avg = sum(items) / len(items)\n"
        "    except Exception:\n"
        "        pass\n"
        "    return avg / 0\n",
        encoding="utf-8",
    )

    scripted_payload = json.dumps(
        {
            "findings": [
                # 1. Lower-priority medium finding (except pass), inflated to CRITICAL -> normalized to HIGH
                {
                    "category": "error_handling",
                    "severity": "critical",
                    "confidence": "high",
                    "file_path": "src/calc.py",
                    "start_line": 4,
                    "end_line": 5,
                    "title": "Silent exception swallowing",
                    "description": "Catching Exception with pass hides errors.",
                    "evidence": "except Exception:\n    pass",
                },
                # 2. Duplicate of finding #3 with lower severity
                {
                    "category": "correctness",
                    "severity": "medium",
                    "confidence": "medium",
                    "file_path": "src/calc.py",
                    "start_line": 6,
                    "end_line": 6,
                    "title": "Division by zero in return",
                    "description": "Dividing avg by 0 fails at runtime.",
                    "evidence": "return avg / 0",
                    "suggested_remediation": "Remove '/ 0' divisor.",
                },
                # 3. Highest-priority CRITICAL finding (division by 0)
                {
                    "category": "correctness",
                    "severity": "critical",
                    "confidence": "high",
                    "file_path": "src/calc.py",
                    "start_line": 6,
                    "end_line": 6,
                    "title": "Unconditional ZeroDivisionError on return",
                    "description": "Expression avg / 0 always raises ZeroDivisionError.",
                    "evidence": "return avg / 0",
                },
                # 4. Hallucinated finding with invented symbol -> rejected
                {
                    "category": "correctness",
                    "severity": "high",
                    "confidence": "high",
                    "file_path": "src/calc.py",
                    "start_line": 3,
                    "end_line": 3,
                    "title": "Hallucinated call",
                    "description": "Calls nonexistent_helper(items).",
                    "evidence": "nonexistent_helper(items)",
                },
            ]
        }
    )

    provider = FakeProvider(scripted_responses=[scripted_payload])
    config = DeicideConfig.from_env_and_args(repo_path=temp_git_repo, provider="fake", environ={})
    orchestrator = AnalysisOrchestrator(
        config=config,
        provider=provider,
        registry=build_default_registry(),
    )
    report = orchestrator.run()

    # Only 1 provider request was made (zero extra LLM calls)
    assert report.provider.requests_made == 1
    assert report.rejected_findings_count == 1
    assert report.deduplicated_findings_count == 1
    assert len(report.findings) == 2

    # First finding is CRITICAL (return avg / 0), enriched with remediation from its merged duplicate
    assert report.findings[0].severity == Severity.CRITICAL
    assert report.findings[0].line_range.start_line == 6
    assert report.findings[0].suggested_remediation == "Remove '/ 0' divisor."

    # Second finding was normalized from CRITICAL to HIGH due to ERROR_HANDLING category ceiling
    assert report.findings[1].severity == Severity.HIGH
    assert report.findings[1].category == FindingCategory.ERROR_HANDLING
    assert any(w.code == "FINDING_SEVERITY_NORMALIZED" for w in report.warnings)
    assert (
        compute_finding_id(
            "code_review",
            "src/calc.py",
            "correctness",
            report.findings[0].line_range,
            "return avg / 0",
        )
        == report.findings[0].id
    )


def test_proximate_multiline_silent_exception_deduplication_and_same_line_distinct_bugs() -> None:
    # 1. Proximate lines (6 and 7) referring to the same silent exception swallowing defect
    f_exc_header = Finding(
        id="dcd-exc001",
        module_id="code_review",
        category=FindingCategory.ERROR_HANDLING,
        severity=Severity.MEDIUM,
        confidence=Confidence.HIGH,
        file_path="src/service.py",
        line_range=LineRange(start_line=6, end_line=6),
        title="Silent exception swallowing",
        description="Broad except Exception block swallows runtime errors.",
        evidence="except Exception:",
    )
    f_exc_pass = Finding(
        id="dcd-exc002",
        module_id="code_review",
        category=FindingCategory.ERROR_HANDLING,
        severity=Severity.LOW,
        confidence=Confidence.MEDIUM,
        file_path="src/service.py",
        line_range=LineRange(start_line=7, end_line=7),
        title="Ignored exception in except block",
        description="Ignoring exception with pass hides unexpected failures.",
        evidence="pass",
    )

    kept_exc, deduped_exc, _ = deduplicate_findings_semantic([f_exc_header, f_exc_pass])
    assert len(kept_exc) == 1
    assert deduped_exc == 1
    assert kept_exc[0].id == "dcd-exc001"

    # 2. Same line (line 1) with two completely distinct defect concepts (mutable default vs zero division)
    f_same_line_mut = Finding(
        id="dcd-same001",
        module_id="code_review",
        category=FindingCategory.STATE_MUTATION,
        severity=Severity.HIGH,
        confidence=Confidence.HIGH,
        file_path="src/one_liner.py",
        line_range=LineRange(start_line=1, end_line=1),
        title="Mutable default argument",
        description="Default list cache=[] persists across calls.",
        evidence="def bad(x, cache=[]): return x / 0",
    )
    f_same_line_div = Finding(
        id="dcd-same002",
        module_id="code_review",
        category=FindingCategory.CORRECTNESS,
        severity=Severity.CRITICAL,
        confidence=Confidence.HIGH,
        file_path="src/one_liner.py",
        line_range=LineRange(start_line=1, end_line=1),
        title="Division by literal zero",
        description="Dividing x by 0 raises ZeroDivisionError.",
        evidence="return x / 0",
    )

    kept_same, deduped_same, _ = deduplicate_findings_semantic([f_same_line_mut, f_same_line_div])
    assert len(kept_same) == 2
    assert deduped_same == 0


def test_severity_normalization_critical_confidence_gate_and_speculative_cap() -> None:
    # 1. CRITICAL with MEDIUM confidence without literal / 0 -> capped at HIGH
    f_crit_med = Finding(
        id="dcd-critmed01",
        module_id="code_review",
        category=FindingCategory.NULL_HANDLING,
        severity=Severity.CRITICAL,
        confidence=Confidence.MEDIUM,
        file_path="src/service.py",
        line_range=LineRange(start_line=5, end_line=5),
        title="Unchecked optional return value",
        description="Calling method on optional value raises AttributeError if None.",
        evidence="ratio = sum(items) / len(items)",
    )
    norm_cm, adj_cm = normalize_finding_severity(f_crit_med)
    assert norm_cm.severity == Severity.HIGH
    assert adj_cm is not None
    assert adj_cm.rule_id == "SEV_CRITICAL_CONFIDENCE_GATE"

    # 2. Speculative framing ("might possibly") with HIGH severity -> capped at MEDIUM
    f_spec = Finding(
        id="dcd-spec0002",
        module_id="code_review",
        category=FindingCategory.CONCURRENCY,
        severity=Severity.HIGH,
        confidence=Confidence.HIGH,
        file_path="src/service.py",
        line_range=LineRange(start_line=5, end_line=5),
        title="Hypothetical thread contention",
        description="This might possibly cause contention in an unlikely edge case.",
        evidence="ratio = sum(items) / len(items)",
    )
    norm_sp, adj_sp = normalize_finding_severity(f_spec)
    assert norm_sp.severity == Severity.MEDIUM
    assert adj_sp is not None
    assert adj_sp.rule_id == "SEV_SPECULATIVE_FRAMING_CAP"
