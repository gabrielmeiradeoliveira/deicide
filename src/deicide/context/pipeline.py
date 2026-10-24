"""Context Engine pipeline integration connecting Git diffs, AST analysis, and review prompts."""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

from deicide.context.changed_symbols import identify_changed_symbols
from deicide.context.context_models import (
    ContextSelectionLimits,
    ContextSelectionResult,
)
from deicide.context.dependency_graph import build_dependency_graph
from deicide.context.models import ChangedSymbol, FileAnalysisStatus
from deicide.context.project_index import IndexLimits, build_project_index
from deicide.context.python_ast import PythonAstAnalyzer
from deicide.context.snippet_selector import select_context
from deicide.core.errors import SecurityViolationError
from deicide.core.models import ContextSummary, ExecutionBudget
from deicide.git.models import DiffBundle, FileChangeStatus


class ContextPreparationOutcome(NamedTuple):
    """Result of context preparation containing selection result and observable summary."""

    result: ContextSelectionResult | None
    summary: ContextSummary


def prepare_review_context(
    repo_root: Path,
    diff_bundle: DiffBundle,
    budget: ExecutionBudget,
    limits: ContextSelectionLimits | None = None,
    context_enabled: bool = True,
    baseline_prompt_tokens: int = 0,
) -> ContextPreparationOutcome:
    """Safely prepare budgeted supplementary cross-file context for code review."""
    if not context_enabled:
        return ContextPreparationOutcome(
            result=None,
            summary=ContextSummary(status="disabled", fallback=False),
        )

    python_files = [
        f for f in diff_bundle.reviewable_files if f.new_path.endswith(".py") and f.status != FileChangeStatus.DELETED
    ]
    if not python_files:
        return ContextPreparationOutcome(
            result=None,
            summary=ContextSummary(status="no_python_changes", fallback=False),
        )

    active_limits = limits or ContextSelectionLimits()
    ast_analyzer = PythonAstAnalyzer()
    all_changed_symbols: list[ChangedSymbol] = []

    for fdiff in python_files:
        try:
            analysis = ast_analyzer.analyze_file(repo_root, fdiff.new_path)
            if analysis.status == FileAnalysisStatus.PARSED:
                changed_res = identify_changed_symbols(fdiff, analysis)
                all_changed_symbols.extend(changed_res.changed_symbols)
        except SecurityViolationError:
            raise
        except Exception:  # noqa: S112
            continue

    if not all_changed_symbols:
        return ContextPreparationOutcome(
            result=None,
            summary=ContextSummary(status="no_changed_symbols", fallback=False),
        )

    try:
        index_limits = IndexLimits(
            max_files=min(500, active_limits.max_context_files * 20),
            max_file_size_bytes=budget.max_file_bytes,
            max_total_bytes=budget.max_total_context_bytes,
        )
        index = build_project_index(repo_root, limits=index_limits, analyzer=ast_analyzer)
        graph = build_dependency_graph(index)

        # Budget allowance calculation
        output_reservation = 1000
        safety_margin = 150
        remaining_capacity = max(
            0,
            budget.max_total_tokens - baseline_prompt_tokens - output_reservation - safety_margin,
        )
        context_token_allowance = min(active_limits.max_context_tokens, remaining_capacity)

        context_result = select_context(
            repo_root=repo_root,
            changed_symbols=all_changed_symbols,
            dependency_graph=graph,
            project_index=index,
            limits=active_limits,
            max_context_tokens=context_token_allowance,
        )

        summary = ContextSummary(
            status="ready",
            indexed_files=len(index.file_analyses),
            selected_snippets=context_result.selected_snippets_count,
            related_files=context_result.selected_files_count,
            estimated_tokens=context_result.total_estimated_tokens,
            omitted_candidates=context_result.omitted_snippets_count,
            fallback=False,
        )
        return ContextPreparationOutcome(result=context_result, summary=summary)

    except SecurityViolationError:
        raise
    except Exception as exc:
        summary = ContextSummary(
            status="fallback",
            fallback=True,
            fallback_reason=f"Recoverable context failure: {exc}",
        )
        return ContextPreparationOutcome(result=None, summary=summary)
