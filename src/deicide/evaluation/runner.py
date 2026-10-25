"""Runner for evaluating DEICIDE analysis modules against curated ground-truth fixtures."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

from deicide.context.context_models import (
    ContextRelationshipCategory,
    ContextSelectionResult,
    SelectedSnippet,
)
from deicide.core.errors import BudgetExceededError, QuotaExhaustedError
from deicide.core.models import BudgetTracker, CandidateFinding, ExecutionBudget, LineRange
from deicide.core.registry import AnalysisModule
from deicide.evaluation.baseline import baseline_process_candidates
from deicide.evaluation.models import (
    CaseEvaluationResult,
    EvaluationFixture,
    EvaluationSummary,
)
from deicide.git.analyzer import parse_unified_diff
from deicide.git.models import ContextSnippet, DiffBundle, FileContext
from deicide.intelligence.pipeline import FindingIntelligenceEngine
from deicide.modules.code_review import CodeReviewModule
from deicide.providers.base import LLMProvider


def _sanitize_error_message(msg: str, provider: LLMProvider | None = None) -> str:
    """Remove potential API keys or authorization secrets from error strings."""
    cleaned = msg
    api_key = getattr(provider, "_api_key", None)
    if api_key and hasattr(api_key, "get_secret_value"):
        secret_val = api_key.get_secret_value().strip()
        if secret_val:
            cleaned = cleaned.replace(secret_val, "[REDACTED]")
    cleaned = re.sub(r"nvapi-[A-Za-z0-9_-]+", "[REDACTED]", cleaned)
    cleaned = re.sub(r"Bearer\s+[A-Za-z0-9_.-]+", "Bearer [REDACTED]", cleaned, flags=re.IGNORECASE)
    return cleaned


def _build_fixture_contexts(
    fixture: EvaluationFixture,
) -> dict[str, FileContext]:
    contexts: dict[str, FileContext] = {}
    for file_path, content in fixture.file_contents.items():
        lines = content.splitlines()
        line_map = {idx + 1: line for idx, line in enumerate(lines)}
        snippets = [ContextSnippet(start_line=1, end_line=len(lines), lines=line_map)] if line_map else []
        contexts[file_path] = FileContext(
            file_path=file_path,
            total_lines=len(lines),
            snippets=snippets,
            truncated=False,
        )
    return contexts


def evaluate_case(
    fixture: EvaluationFixture,
    provider: LLMProvider,
    module: AnalysisModule | None = None,
    budget: ExecutionBudget | None = None,
    use_baseline: bool = False,
    force_end_to_end: bool = False,
    strict_severity: bool = False,
    tracker: BudgetTracker | None = None,
) -> CaseEvaluationResult:
    """Run a single EvaluationFixture and compute TP, FP, FN, latency, and token consumption."""
    active_module = module or CodeReviewModule()
    active_budget = budget or ExecutionBudget()
    active_tracker = tracker or BudgetTracker(
        active_budget,
        provider_name=provider.provider_name,
        model_name=provider.model_name,
    )

    start_ns = time.perf_counter_ns()
    parsed_files = parse_unified_diff(fixture.diff_text)
    diff_bundle = DiffBundle(
        repository_root="/evaluation/fixture",
        base_revision="HEAD",
        target_description=f"evaluation:{fixture.case_id}",
        files=parsed_files,
        truncated=False,
    )
    contexts = _build_fixture_contexts(fixture)

    context_result: ContextSelectionResult | None = None
    if fixture.supplementary_snippets:
        selected_snippets = []
        serialized_sections = []
        for snip in fixture.supplementary_snippets:
            snippet_obj = SelectedSnippet(
                snippet_id=str(snip.get("snippet_id", "supp-1")),
                file_path=str(snip.get("file_path", "unknown.py")),
                start_line=int(snip.get("start_line", 1)),
                end_line=int(snip.get("end_line", 1)),
                content=str(snip.get("content", "")),
                relationship=ContextRelationshipCategory(snip.get("relationship", "direct_caller")),
            )
            selected_snippets.append(snippet_obj)
            serialized_sections.append(f"=== CONTEXT SNIPPET: {snippet_obj.file_path} ===\n{snippet_obj.content}")

        context_result = ContextSelectionResult(
            selected_snippets=selected_snippets,
            serialized_text="\n\n".join(serialized_sections),
            selected_snippets_count=len(selected_snippets),
        )

    candidates: list[CandidateFinding] = []
    case_status = "completed"
    case_error: str | None = None

    if not force_end_to_end and fixture.candidate_findings:
        candidates = list(fixture.candidate_findings)
    else:
        try:
            active_tracker.check_can_make_request()
            try:
                candidates = active_module.analyze(
                    diff_bundle=diff_bundle,
                    contexts=contexts,
                    provider=provider,
                    tracker=active_tracker,
                    context_result=context_result,
                )
            except TypeError:
                candidates = active_module.analyze(
                    diff_bundle=diff_bundle,
                    contexts=contexts,
                    provider=provider,
                    tracker=active_tracker,
                )
        except (BudgetExceededError, QuotaExhaustedError) as exc:
            case_status = "budget_exhausted"
            case_error = _sanitize_error_message(str(exc), provider)
        except Exception as exc:
            case_status = "failed"
            case_error = _sanitize_error_message(str(exc), provider)

    if case_status != "completed":
        elapsed_ms = max(0, int((time.perf_counter_ns() - start_ns) / 1_000_000))
        return CaseEvaluationResult(
            case_id=fixture.case_id,
            true_positives=0,
            false_positives=0,
            false_negatives=len(fixture.expected_findings),
            duplicates_reduced=0,
            legitimate_rejected=0,
            severity_mismatches=0,
            latency_ms=elapsed_ms,
            usage=active_tracker.usage,
            actual_findings=[],
            status=case_status,
            error=case_error,
            provenance=fixture.resolved_provenance,
        )

    if use_baseline:
        actual_findings, rejected_count, deduped_count, _warnings = baseline_process_candidates(
            module_id=active_module.module_id,
            candidates=candidates,
            diff_bundle=diff_bundle,
            contexts=contexts,
            repo_root=None,
        )
    else:
        intel_result = FindingIntelligenceEngine().process_candidates(
            module_id=active_module.module_id,
            candidates=candidates,
            diff_bundle=diff_bundle,
            contexts=contexts,
            repo_root=None,
            context_result=context_result,
        )
        actual_findings = intel_result.findings
        deduped_count = intel_result.deduplicated_count

    elapsed_ms = max(0, int((time.perf_counter_ns() - start_ns) / 1_000_000))

    matched_expected_indices: set[int] = set()
    tp = 0
    fp = 0
    severity_mismatches = 0

    for actual in actual_findings:
        matched = False
        for idx, expected in enumerate(fixture.expected_findings):
            if idx not in matched_expected_indices and expected.matches(actual, strict_severity=strict_severity):
                matched_expected_indices.add(idx)
                matched = True
                if not expected.severity_matches(actual):
                    severity_mismatches += 1
                break
        if matched:
            tp += 1
        else:
            fp += 1

    fn = len(fixture.expected_findings) - len(matched_expected_indices)

    # Calculate legitimate findings incorrectly rejected (ground truth in candidates but missing from actual)
    legitimate_rejected = 0
    for idx, expected in enumerate(fixture.expected_findings):
        if idx not in matched_expected_indices:
            candidate_present = any(
                c.file_path == expected.file_path
                and LineRange(start_line=c.start_line, end_line=c.end_line or c.start_line).overlaps_or_adjacent(
                    expected.line_range, tolerance=2
                )
                and (expected.category is None or c.category == expected.category)
                for c in candidates
            )
            if candidate_present:
                legitimate_rejected += 1

    return CaseEvaluationResult(
        case_id=fixture.case_id,
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        duplicates_reduced=deduped_count,
        legitimate_rejected=legitimate_rejected,
        severity_mismatches=severity_mismatches,
        latency_ms=elapsed_ms,
        usage=active_tracker.usage,
        actual_findings=actual_findings,
        status="completed",
        error=None,
        provenance=fixture.resolved_provenance,
    )


def evaluate_corpus(
    fixtures: list[EvaluationFixture],
    provider: LLMProvider,
    module: AnalysisModule | None = None,
    budget: ExecutionBudget | None = None,
    use_baseline: bool = False,
    force_end_to_end: bool = False,
    strict_severity: bool = False,
    max_cases: int | None = None,
    shared_tracker: BudgetTracker | None = None,
) -> EvaluationSummary:
    """Evaluate a corpus of fixtures and aggregate precision, recall, latency, and tokens."""
    evaluated_fixtures = fixtures
    max_cases_applied: int | None = None
    if max_cases is not None and max_cases > 0:
        max_cases_applied = max_cases
        evaluated_fixtures = fixtures[:max_cases]

    active_budget = budget or ExecutionBudget(
        max_requests=max(len(fixtures) * 2, 100),
        max_total_tokens=10_000_000,
    )
    active_tracker = shared_tracker or BudgetTracker(
        active_budget,
        provider_name=provider.provider_name,
        model_name=provider.model_name,
    )

    case_results: list[CaseEvaluationResult] = []
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_duplicates_reduced = 0
    total_legitimate_rejected = 0
    total_severity_mismatches = 0
    total_latency_ms = 0
    positive_cases = 0
    negative_cases = 0
    negative_cases_with_fp = 0
    verified_cases = 0
    reference_modeled_cases = 0
    unverified_cases = 0
    failed_cases = 0
    budget_exhausted = False

    for fixture in evaluated_fixtures:
        prov = fixture.resolved_provenance
        if prov == "fully_verified":
            verified_cases += 1
        elif prov == "reference_modeled":
            reference_modeled_cases += 1
        else:
            unverified_cases += 1

        is_positive = len(fixture.expected_findings) > 0
        if is_positive:
            positive_cases += 1
        else:
            negative_cases += 1

        res = evaluate_case(
            fixture,
            provider=provider,
            module=module,
            budget=active_budget,
            use_baseline=use_baseline,
            force_end_to_end=force_end_to_end,
            strict_severity=strict_severity,
            tracker=active_tracker,
        )
        case_results.append(res)
        total_tp += res.true_positives
        total_fp += res.false_positives
        total_fn += res.false_negatives
        total_duplicates_reduced += res.duplicates_reduced
        total_legitimate_rejected += res.legitimate_rejected
        total_severity_mismatches += res.severity_mismatches
        total_latency_ms += res.latency_ms

        if not is_positive and res.false_positives > 0:
            negative_cases_with_fp += 1

        if res.status == "budget_exhausted":
            budget_exhausted = True
            failed_cases += 1
            break
        elif res.status != "completed":
            failed_cases += 1

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 1.0
    fp_rate = negative_cases_with_fp / negative_cases if negative_cases > 0 else 0.0
    avg_latency = total_latency_ms / len(case_results) if case_results else 0.0

    return EvaluationSummary(
        dataset_size=len(case_results),
        positive_cases=positive_cases,
        negative_cases=negative_cases,
        true_positives=total_tp,
        false_positives=total_fp,
        false_negatives=total_fn,
        precision=round(precision, 4),
        recall=round(recall, 4),
        false_positive_rate=round(fp_rate, 4),
        duplicates_reduced=total_duplicates_reduced,
        legitimate_rejected=total_legitimate_rejected,
        severity_mismatches=total_severity_mismatches,
        evaluation_mode="end_to_end" if force_end_to_end else "post_processing",
        total_latency_ms=total_latency_ms,
        average_latency_ms=round(avg_latency, 2),
        usage=active_tracker.usage,
        verified_cases=verified_cases,
        reference_modeled_cases=reference_modeled_cases,
        unverified_cases=unverified_cases,
        failed_cases=failed_cases,
        budget_exhausted=budget_exhausted,
        max_cases_applied=max_cases_applied,
        case_results=case_results,
    )


def load_fixtures_from_file(path: Path) -> list[EvaluationFixture]:
    """Load a list of EvaluationFixture objects from a JSON file."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("Evaluation fixture file must contain a JSON list of fixtures.")
    return [EvaluationFixture.model_validate(item) for item in raw]
