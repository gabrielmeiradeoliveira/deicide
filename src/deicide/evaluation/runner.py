"""Runner for evaluating DEICIDE analysis modules against curated ground-truth fixtures."""

from __future__ import annotations

import json
import time
from pathlib import Path

from deicide.core.models import BudgetTracker, ExecutionBudget, TokenUsage
from deicide.core.registry import AnalysisModule
from deicide.core.validation import deduplicate_findings, validate_candidates
from deicide.evaluation.models import (
    CaseEvaluationResult,
    EvaluationFixture,
    EvaluationSummary,
)
from deicide.git.analyzer import parse_unified_diff
from deicide.git.models import ContextSnippet, DiffBundle, FileContext
from deicide.modules.code_review import CodeReviewModule
from deicide.providers.base import LLMProvider


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
) -> CaseEvaluationResult:
    """Run a single EvaluationFixture and compute TP, FP, FN, latency, and token consumption."""
    active_module = module or CodeReviewModule()
    active_budget = budget or ExecutionBudget()
    tracker = BudgetTracker(active_budget)

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

    candidates = active_module.analyze(
        diff_bundle=diff_bundle,
        contexts=contexts,
        provider=provider,
        tracker=tracker,
    )
    validated, _rejected, _warnings = validate_candidates(
        module_id=active_module.module_id,
        candidates=candidates,
        diff_bundle=diff_bundle,
        contexts=contexts,
        repo_root=None,
    )
    actual_findings, _deduped = deduplicate_findings(validated)
    elapsed_ms = max(0, int((time.perf_counter_ns() - start_ns) / 1_000_000))

    matched_expected_indices: set[int] = set()
    tp = 0
    fp = 0

    for actual in actual_findings:
        matched = False
        for idx, expected in enumerate(fixture.expected_findings):
            if idx not in matched_expected_indices and expected.matches(actual):
                matched_expected_indices.add(idx)
                matched = True
                break
        if matched:
            tp += 1
        else:
            fp += 1

    fn = len(fixture.expected_findings) - len(matched_expected_indices)

    return CaseEvaluationResult(
        case_id=fixture.case_id,
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        latency_ms=elapsed_ms,
        usage=tracker.usage,
        actual_findings=actual_findings,
    )


def evaluate_corpus(
    fixtures: list[EvaluationFixture],
    provider: LLMProvider,
    module: AnalysisModule | None = None,
    budget: ExecutionBudget | None = None,
) -> EvaluationSummary:
    """Evaluate a corpus of fixtures and aggregate precision, recall, latency, and tokens."""
    case_results: list[CaseEvaluationResult] = []
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_latency_ms = 0
    total_usage = TokenUsage()
    positive_cases = 0
    negative_cases = 0
    negative_cases_with_fp = 0

    for fixture in fixtures:
        is_positive = len(fixture.expected_findings) > 0
        if is_positive:
            positive_cases += 1
        else:
            negative_cases += 1

        res = evaluate_case(fixture, provider=provider, module=module, budget=budget)
        case_results.append(res)
        total_tp += res.true_positives
        total_fp += res.false_positives
        total_fn += res.false_negatives
        total_latency_ms += res.latency_ms
        total_usage = total_usage.add(res.usage)
        if not is_positive and res.false_positives > 0:
            negative_cases_with_fp += 1

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 1.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 1.0
    fp_rate = negative_cases_with_fp / negative_cases if negative_cases > 0 else 0.0
    avg_latency = total_latency_ms / len(fixtures) if fixtures else 0.0

    return EvaluationSummary(
        dataset_size=len(fixtures),
        positive_cases=positive_cases,
        negative_cases=negative_cases,
        true_positives=total_tp,
        false_positives=total_fp,
        false_negatives=total_fn,
        precision=round(precision, 4),
        recall=round(recall, 4),
        false_positive_rate=round(fp_rate, 4),
        total_latency_ms=total_latency_ms,
        average_latency_ms=round(avg_latency, 2),
        usage=total_usage,
        case_results=case_results,
    )


def load_fixtures_from_file(path: Path) -> list[EvaluationFixture]:
    """Load a list of EvaluationFixture objects from a JSON file."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("Evaluation fixture file must contain a JSON list of fixtures.")
    return [EvaluationFixture.model_validate(item) for item in raw]
