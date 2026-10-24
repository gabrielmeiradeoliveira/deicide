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

    # Extract deletion events: (new_line, del_run)
    deletion_events: list[tuple[int, list[str]]] = []
    for hunk in file_diff.hunks:
        current_new_line = hunk.new_start
        i = 0
        hunk_lines = hunk.lines
        while i < len(hunk_lines):
            hline = hunk_lines[i]
            if (hline.startswith("+") and not hline.startswith("+++")) or hline.startswith(" "):
                current_new_line += 1
                i += 1
            elif hline.startswith("-") and not hline.startswith("---"):
                del_run: list[str] = []
                while i < len(hunk_lines) and hunk_lines[i].startswith("-") and not hunk_lines[i].startswith("---"):
                    del_run.append(hunk_lines[i][1:])
                    i += 1
                deletion_events.append((current_new_line, del_run))
            else:
                i += 1

    # Map deletions to symbols and line numbers in the new file
    symbol_deleted_lines: dict[str, set[int]] = {sym.id: set() for sym in file_analysis.symbols}
    unmapped_deletion_lines: set[int] = set()

    for new_line, del_run in deletion_events:
        code_lines = [line for line in del_run if line.strip() and not line.strip().startswith("#")]
        if not code_lines:
            continue

        first_code = code_lines[0]
        first_indent = len(first_code) - len(first_code.lstrip(" "))
        has_decorator = any(line.lstrip().startswith("@") for line in code_lines)

        mapped_to_sym = False

        for sym in file_analysis.symbols:
            start_ln = sym.location.start_line
            end_ln = sym.location.end_line
            sym_indent = sym.location.start_col or 0

            # Condition A: Deletion strictly inside symbol body
            if start_ln < new_line <= end_ln:
                symbol_deleted_lines[sym.id].add(new_line)
                mapped_to_sym = True
            # Condition B: Deletion at the end of symbol body
            elif new_line == end_ln + 1:
                if first_indent > sym_indent and not first_code.lstrip().startswith(("def ", "class ", "async def ")):
                    symbol_deleted_lines[sym.id].add(end_ln)
                    mapped_to_sym = True
            # Condition C: Deletion of decorator at start of symbol
            elif new_line == start_ln and has_decorator:
                if first_indent == sym_indent:
                    symbol_deleted_lines[sym.id].add(start_ln)
                    mapped_to_sym = True

        if not mapped_to_sym and new_line > 0:
            unmapped_deletion_lines.add(new_line)

    all_changed_lines: set[int] = set(added_lines)
    for del_lines in symbol_deleted_lines.values():
        all_changed_lines.update(del_lines)

    if not all_changed_lines and not unmapped_deletion_lines:
        return ChangedSymbolsResult(
            file_path=file_path,
            changed_symbols=[],
            module_level_changes=[],
            deleted_lines_count=deleted_lines_count,
            deleted_lines_note=deleted_note,
        )

    # For each changed line, find all matching symbols and pick the innermost
    line_to_symbols: dict[int, list[Symbol]] = {line: [] for line in all_changed_lines}
    for sym in file_analysis.symbols:
        for line in all_changed_lines:
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
    module_lines = sorted(line for line, sym in line_to_innermost.items() if sym is None)
    for u_line in unmapped_deletion_lines:
        if not any(sym.location.contains_line(u_line) for sym in file_analysis.symbols) and u_line not in module_lines:
            module_lines.append(u_line)
    module_lines.sort()

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
        sym_added = [line for line in added_lines if sym.location.contains_line(line)]
        sym_deleted = list(symbol_deleted_lines.get(sym.id, set()))
        sym_overlapping = sorted(set(sym_added) | set(sym_deleted))
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
