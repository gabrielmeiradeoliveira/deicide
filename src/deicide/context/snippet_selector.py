"""Deterministic source snippet slicing, deduplication, and context selection."""

from __future__ import annotations

from pathlib import Path

from deicide.context.context_budget import (
    enforce_context_budget,
    estimate_snippet_tokens,
)
from deicide.context.context_models import (
    ContextCandidate,
    ContextSelectionLimits,
    ContextSelectionResult,
    OmissionReason,
    OmittedCandidate,
    SelectedSnippet,
)
from deicide.context.context_ranker import ContextRanker
from deicide.context.dependency_graph import DependencyGraph
from deicide.context.models import ChangedSymbol
from deicide.context.project_index import ProjectIndex
from deicide.core.errors import SecurityViolationError
from deicide.git.analyzer import ensure_safe_repo_file, normalize_repo_relative_path


class SnippetExtractor:
    """Safely reads repository source files and slices bounded, numbered code snippets."""

    def __init__(self, repo_root: Path, max_snippet_lines: int = 60) -> None:
        self.repo_root = repo_root
        self.max_snippet_lines = max_snippet_lines
        self._file_lines_cache: dict[str, list[str]] = {}

    def get_file_lines(self, relative_path: str) -> tuple[list[str] | None, OmissionReason | None, str | None]:
        """Safely load raw file lines with security validation and path confinement."""
        try:
            norm_path = normalize_repo_relative_path(relative_path)
            safe_path = ensure_safe_repo_file(self.repo_root, norm_path)
        except SecurityViolationError as exc:
            return None, OmissionReason.UNSAFE_PATH, exc.message
        except Exception as exc:
            return None, OmissionReason.UNSAFE_PATH, str(exc)

        if norm_path in self._file_lines_cache:
            return self._file_lines_cache[norm_path], None, None

        if not safe_path.exists() or not safe_path.is_file():
            return None, OmissionReason.SOURCE_UNAVAILABLE, f"File not found: '{norm_path}'"

        try:
            source_text = safe_path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            return None, OmissionReason.SOURCE_UNAVAILABLE, f"Unicode decoding failed: {exc}"
        except Exception as exc:
            return None, OmissionReason.SOURCE_UNAVAILABLE, str(exc)

        lines = source_text.splitlines()
        self._file_lines_cache[norm_path] = lines
        return lines, None, None

    def slice_candidate(
        self,
        candidate: ContextCandidate,
    ) -> tuple[SelectedSnippet | None, OmittedCandidate | None]:
        """Extract a bounded, numbered source snippet for a ContextCandidate."""
        lines, reason, err_detail = self.get_file_lines(candidate.file_path)
        if lines is None:
            omitted = OmittedCandidate(
                candidate_id=candidate.candidate_id,
                file_path=candidate.file_path,
                symbol_name=candidate.qualified_name,
                reason=reason or OmissionReason.SOURCE_UNAVAILABLE,
                detail=err_detail or "Source unavailable",
                priority=candidate.priority,
                estimated_tokens=candidate.estimated_tokens,
            )
            return None, omitted

        total_lines = len(lines)
        start = candidate.start_line
        end = candidate.end_line

        if start > total_lines or end < 1 or start > end:
            omitted = OmittedCandidate(
                candidate_id=candidate.candidate_id,
                file_path=candidate.file_path,
                symbol_name=candidate.qualified_name,
                reason=OmissionReason.INVALID_SOURCE_SPAN,
                detail=f"Requested lines {start}-{end} outside file total {total_lines}",
                priority=candidate.priority,
                estimated_tokens=candidate.estimated_tokens,
            )
            return None, omitted

        symbol_line_count = end - start + 1
        is_truncated = False
        original_lines: int | None = None

        if symbol_line_count <= self.max_snippet_lines:
            slice_start = start
            slice_end = end
        else:
            # Large function handling: window around primary focus line (e.g. call line)
            is_truncated = True
            original_lines = symbol_line_count

            focus_line = candidate.focus_lines[0] if candidate.focus_lines else start
            focus_line = max(start, min(end, focus_line))

            window_half = self.max_snippet_lines // 2
            slice_start = max(start, focus_line - window_half)
            slice_end = min(end, slice_start + self.max_snippet_lines - 1)

            if slice_end == end:
                slice_start = max(start, slice_end - self.max_snippet_lines + 1)
            elif slice_start == start:
                slice_end = min(end, slice_start + self.max_snippet_lines - 1)

        # Slice 1-based lines preserving exact indentation and content
        sliced_lines: list[str] = []
        for line_no in range(slice_start, slice_end + 1):
            if 1 <= line_no <= total_lines:
                sliced_lines.append(f"{line_no:4d} | {lines[line_no - 1]}")

        content_str = "\n".join(sliced_lines)
        snippet_id = f"snip::{candidate.file_path}::{slice_start}:{slice_end}"

        snippet = SelectedSnippet(
            snippet_id=snippet_id,
            file_path=candidate.file_path,
            start_line=slice_start,
            end_line=slice_end,
            symbol_name=candidate.qualified_name.split(".")[-1] if candidate.qualified_name else None,
            qualified_name=candidate.qualified_name,
            content=content_str,
            relationship=candidate.relationship,
            provenance=candidate.provenance,
            estimated_tokens=0,  # Computed below
            is_truncated=is_truncated,
            truncated_original_lines=original_lines,
            relevant_call_lines=[ln for ln in candidate.focus_lines if slice_start <= ln <= slice_end],
        )

        tokens = estimate_snippet_tokens(snippet)
        snippet = snippet.model_copy(update={"estimated_tokens": tokens})
        return snippet, None


def _deduplicate_and_merge_snippets(
    snippets: list[SelectedSnippet],
    max_snippet_lines: int,
) -> list[SelectedSnippet]:
    """Merge overlapping or adjacent snippets from the same file preserving all provenance."""
    if not snippets:
        return []

    # Group by file path
    by_file: dict[str, list[SelectedSnippet]] = {}
    for s in snippets:
        by_file.setdefault(s.file_path, []).append(s)

    merged_snippets: list[SelectedSnippet] = []

    for file_path in sorted(by_file.keys()):
        file_snippets = sorted(by_file[file_path], key=lambda s: (s.start_line, s.end_line))
        current = file_snippets[0]

        for nxt in file_snippets[1:]:
            # Exact duplicate or identical span
            if current.start_line == nxt.start_line and current.end_line == nxt.end_line:
                # Merge provenance records
                combined_prov = list(current.provenance)
                for p in nxt.provenance:
                    if p not in combined_prov:
                        combined_prov.append(p)

                combined_calls = sorted(set(current.relevant_call_lines + nxt.relevant_call_lines))
                # Prioritize higher-priority relationship
                rel = current.relationship if current.relationship.value <= nxt.relationship.value else nxt.relationship
                current = current.model_copy(
                    update={
                        "provenance": combined_prov,
                        "relevant_call_lines": combined_calls,
                        "relationship": rel,
                    }
                )
                continue

            # Overlapping or adjacent (within 2 lines)
            span_size = max(current.end_line, nxt.end_line) - min(current.start_line, nxt.start_line) + 1
            if nxt.start_line <= current.end_line + 2 and span_size <= max_snippet_lines:
                # Merge into unified snippet
                combined_prov = list(current.provenance)
                for p in nxt.provenance:
                    if p not in combined_prov:
                        combined_prov.append(p)

                combined_calls = sorted(set(current.relevant_call_lines + nxt.relevant_call_lines))
                # Re-slice lines
                m_start = min(current.start_line, nxt.start_line)
                m_end = max(current.end_line, nxt.end_line)

                # Merge content lines by line number
                lines_by_num: dict[int, str] = {}
                for snip in (current, nxt):
                    for line in snip.content.splitlines():
                        parts = line.split(" | ", 1)
                        if len(parts) == 2 and parts[0].strip().isdigit():
                            lines_by_num[int(parts[0].strip())] = line

                sorted_lines = [lines_by_num[n] for n in sorted(lines_by_num.keys())]
                new_content = "\n".join(sorted_lines)

                current = SelectedSnippet(
                    snippet_id=f"snip::{file_path}::{m_start}:{m_end}",
                    file_path=file_path,
                    start_line=m_start,
                    end_line=m_end,
                    symbol_name=current.symbol_name,
                    qualified_name=current.qualified_name,
                    content=new_content,
                    relationship=current.relationship,
                    provenance=combined_prov,
                    estimated_tokens=0,
                    is_truncated=current.is_truncated or nxt.is_truncated,
                    truncated_original_lines=current.truncated_original_lines,
                    relevant_call_lines=combined_calls,
                )
                current = current.model_copy(update={"estimated_tokens": estimate_snippet_tokens(current)})
            else:
                merged_snippets.append(current)
                current = nxt

        merged_snippets.append(current)

    return merged_snippets


def select_context(
    repo_root: Path,
    changed_symbols: list[ChangedSymbol],
    dependency_graph: DependencyGraph,
    project_index: ProjectIndex,
    limits: ContextSelectionLimits | None = None,
    max_context_tokens: int | None = None,
) -> ContextSelectionResult:
    """High-level entrypoint: discovers, ranks, slices, deduplicates, and budgets context."""
    active_limits = limits.model_copy() if limits else ContextSelectionLimits()
    if max_context_tokens is not None:
        active_limits = active_limits.model_copy(update={"max_context_tokens": max_context_tokens})

    diagnostics: list[str] = list(project_index.diagnostics)
    if project_index.limits_reached:
        diagnostics.append("Repository indexing was bounded by safety limits; dependency coverage may be partial.")

    if not changed_symbols:
        diagnostics.append("No changed symbols provided for context discovery.")
        return ContextSelectionResult(
            selected_snippets=[],
            omitted_candidates=[],
            total_estimated_tokens=0,
            max_allowed_tokens=active_limits.max_context_tokens,
            selected_files_count=0,
            selected_snippets_count=0,
            omitted_snippets_count=0,
            is_truncated=False,
            serialized_text="",
            diagnostics=diagnostics,
            limits=active_limits,
        )

    # 1. Discover and rank candidates
    ranker = ContextRanker(
        dependency_graph=dependency_graph,
        project_index=project_index,
        limits=active_limits,
    )
    ranked_candidates = ranker.discover_candidates(changed_symbols)

    # 2. Extract snippets
    extractor = SnippetExtractor(
        repo_root=repo_root,
        max_snippet_lines=active_limits.max_snippet_lines,
    )
    raw_snippets: list[SelectedSnippet] = []
    omitted_candidates: list[OmittedCandidate] = []

    for cand in ranked_candidates:
        snip, omitted = extractor.slice_candidate(cand)
        if snip:
            raw_snippets.append(snip)
        elif omitted:
            omitted_candidates.append(omitted)

    # 3. Deduplicate and merge overlapping snippets
    merged_snippets = _deduplicate_and_merge_snippets(
        raw_snippets,
        max_snippet_lines=active_limits.max_snippet_lines,
    )

    # 4. Enforce strict context budget
    selected, budget_omitted, serialized, final_tokens, is_truncated = enforce_context_budget(
        merged_snippets,
        limits=active_limits,
    )

    all_omitted = [*omitted_candidates, *budget_omitted]
    selected_files = {s.file_path for s in selected}

    return ContextSelectionResult(
        selected_snippets=selected,
        omitted_candidates=all_omitted,
        total_estimated_tokens=final_tokens,
        max_allowed_tokens=active_limits.max_context_tokens,
        selected_files_count=len(selected_files),
        selected_snippets_count=len(selected),
        omitted_snippets_count=len(all_omitted),
        is_truncated=is_truncated,
        serialized_text=serialized,
        diagnostics=diagnostics,
        limits=active_limits,
    )
