"""Deterministic identification of symbols overlapping modified Git diff lines."""

from __future__ import annotations

from deicide.context.models import (
    ChangedSymbol,
    ChangedSymbolsResult,
    FileAnalysis,
    FileAnalysisStatus,
    ModuleLevelChange,
    OverlapClassification,
    Symbol,
)
from deicide.git.models import FileChangeStatus, FileDiff


def _classify_overlap(symbol: Symbol, overlapping_lines: list[int]) -> OverlapClassification:
    """Classify the nature of changed line overlap against a symbol's definition."""
    start = symbol.location.start_line
    end = symbol.location.end_line
    full_span = set(range(start, end + 1))
    overlap_set = set(overlapping_lines)

    if overlap_set == full_span:
        return OverlapClassification.EXACT
    # If any changed line touches the definition header or decorators
    if start in overlap_set:
        return OverlapClassification.SIGNATURE
    return OverlapClassification.BODY


def identify_changed_symbols(
    file_diff: FileDiff,
    file_analysis: FileAnalysis,
) -> ChangedSymbolsResult:
    """Map modified lines in a FileDiff to extracted AST symbols with innermost resolution."""
    file_path = file_diff.new_path

    # Count deleted lines from diff hunks
    deleted_lines_count = 0
    for hunk in file_diff.hunks:
        deleted_lines_count += sum(1 for line in hunk.lines if line.startswith("-") and not line.startswith("---"))

    deleted_note: str | None = None
    if deleted_lines_count > 0:
        deleted_note = "Deleted lines cannot be mapped to working tree AST symbols without base revision source"

    # Handle deleted files
    if file_diff.status == FileChangeStatus.DELETED:
        return ChangedSymbolsResult(
            file_path=file_path,
            changed_symbols=[],
            module_level_changes=[],
            deleted_lines_count=deleted_lines_count,
            deleted_lines_note="File was deleted in Git diff; working tree AST is unavailable.",
        )

    # Handle non-parsed or error files
    if file_analysis.status != FileAnalysisStatus.PARSED:
        return ChangedSymbolsResult(
            file_path=file_path,
            changed_symbols=[],
            module_level_changes=[],
            deleted_lines_count=deleted_lines_count,
            deleted_lines_note=f"AST analysis incomplete: {file_analysis.status.value}",
        )

    added_lines = sorted(file_diff.added_lines)
    if not added_lines:
        return ChangedSymbolsResult(
            file_path=file_path,
            changed_symbols=[],
            module_level_changes=[],
            deleted_lines_count=deleted_lines_count,
            deleted_lines_note=deleted_note,
        )

    # For each added line, find all matching symbols and pick the innermost
    line_to_symbols: dict[int, list[Symbol]] = {line: [] for line in added_lines}
    for sym in file_analysis.symbols:
        for line in added_lines:
            if sym.location.contains_line(line):
                line_to_symbols[line].append(sym)

    line_to_innermost: dict[int, Symbol | None] = {}
    for line, candidate_symbols in line_to_symbols.items():
        if not candidate_symbols:
            line_to_innermost[line] = None
        else:
            # Deepest scope path, then smallest line span
            innermost = max(
                candidate_symbols,
                key=lambda s: (
                    len(s.scope_path),
                    -(s.location.end_line - s.location.start_line),
                ),
            )
            line_to_innermost[line] = innermost

    # Collect module-level changes
    module_lines = [line for line, sym in line_to_innermost.items() if sym is None]
    module_changes: list[ModuleLevelChange] = []
    if module_lines:
        module_changes.append(
            ModuleLevelChange(
                file_path=file_path,
                changed_lines=module_lines,
                description="Module-level changes outside symbol definitions",
            )
        )

    # Build ChangedSymbol objects
    changed_symbols: list[ChangedSymbol] = []
    all_innermost_ids = {sym.id for sym in line_to_innermost.values() if sym is not None}

    for sym in file_analysis.symbols:
        sym_overlapping = [line for line in added_lines if sym.location.contains_line(line)]
        if not sym_overlapping:
            continue

        classification = _classify_overlap(sym, sym_overlapping)
        is_inner = sym.id in all_innermost_ids

        changed_symbols.append(
            ChangedSymbol(
                symbol_id=sym.id,
                symbol_name=sym.name,
                qualified_name=sym.qualified_name,
                kind=sym.kind,
                location=sym.location,
                parent_id=sym.parent_id,
                overlapping_lines=sym_overlapping,
                classification=classification,
                is_innermost=is_inner,
            )
        )

    # Deterministic sorting: innermost symbols prioritized, then by start line, then qualified name
    sorted_changed = sorted(
        changed_symbols,
        key=lambda cs: (
            0 if cs.is_innermost else 1,
            cs.location.start_line,
            cs.location.start_col or 0,
            cs.qualified_name,
            cs.symbol_id,
        ),
    )

    return ChangedSymbolsResult(
        file_path=file_path,
        changed_symbols=sorted_changed,
        module_level_changes=module_changes,
        deleted_lines_count=deleted_lines_count,
        deleted_lines_note=deleted_note,
    )
