"""Deterministic serialization, conservative token estimation, and strict budget enforcement."""

from __future__ import annotations

from deicide.context.context_models import (
    ContextSelectionLimits,
    OmissionReason,
    OmittedCandidate,
    SelectedSnippet,
)
from deicide.core.models import estimate_token_count


def serialize_snippet_block(snippet: SelectedSnippet) -> str:
    """Format a single selected snippet with its metadata and boundary delimiters."""
    prov_notes = "; ".join(p.explanation for p in snippet.provenance if p.explanation)
    trunc_tag = " [TRUNCATED]" if snippet.is_truncated else ""
    sym_name = snippet.qualified_name or snippet.symbol_name or snippet.file_path

    lines = [
        f"=== CONTEXT SNIPPET: {snippet.file_path} ({snippet.relationship.value}) ===",
        f"Symbol: {sym_name} (lines {snippet.start_line}-{snippet.end_line}{trunc_tag})",
    ]
    if prov_notes:
        lines.append(f"Provenance: {prov_notes}")
    lines.append(snippet.content)
    lines.append("============================================================")
    return "\n".join(lines)


def serialize_context_bundle(snippets: list[SelectedSnippet]) -> str:
    """Deterministically format all selected snippets into unified context text."""
    # Deterministic sorting for output formatting
    sorted_snippets = sorted(snippets, key=lambda s: (s.file_path, s.start_line, s.snippet_id))
    blocks = [serialize_snippet_block(s) for s in sorted_snippets]
    return "\n\n".join(blocks)


def estimate_snippet_tokens(snippet: SelectedSnippet) -> int:
    """Estimate token count for a snippet including its serialized metadata overhead."""
    text = serialize_snippet_block(snippet)
    return estimate_token_count(text)


def enforce_context_budget(
    snippets: list[SelectedSnippet],
    limits: ContextSelectionLimits,
) -> tuple[list[SelectedSnippet], list[OmittedCandidate], str, int, bool]:
    """Strictly enforce token, file, and snippet count limits with omission diagnostics.

    Returns: (selected_snippets, omitted_candidates, serialized_text, total_tokens, is_truncated).
    """
    if limits.max_context_tokens <= 0:
        omitted = [
            OmittedCandidate(
                candidate_id=s.snippet_id,
                file_path=s.file_path,
                symbol_name=s.symbol_name,
                reason=OmissionReason.TOKEN_BUDGET_EXCEEDED,
                detail="Context token budget is zero or negative",
                estimated_tokens=s.estimated_tokens,
            )
            for s in snippets
        ]
        return [], omitted, "", 0, bool(snippets)

    selected: list[SelectedSnippet] = []
    omitted: list[OmittedCandidate] = []
    selected_files: set[str] = set()

    for snippet in snippets:
        # 1. Enforce snippet count limit
        if len(selected) >= limits.max_snippets:
            omitted.append(
                OmittedCandidate(
                    candidate_id=snippet.snippet_id,
                    file_path=snippet.file_path,
                    symbol_name=snippet.symbol_name,
                    reason=OmissionReason.SNIPPET_LIMIT_EXCEEDED,
                    detail=f"Exceeded max snippet limit ({limits.max_snippets})",
                    estimated_tokens=snippet.estimated_tokens,
                )
            )
            continue

        # 2. Enforce file count limit
        is_new_file = snippet.file_path not in selected_files
        if is_new_file and len(selected_files) >= limits.max_context_files:
            omitted.append(
                OmittedCandidate(
                    candidate_id=snippet.snippet_id,
                    file_path=snippet.file_path,
                    symbol_name=snippet.symbol_name,
                    reason=OmissionReason.FILE_LIMIT_EXCEEDED,
                    detail=f"Exceeded max context files limit ({limits.max_context_files})",
                    estimated_tokens=snippet.estimated_tokens,
                )
            )
            continue

        # 3. Tentatively add and check total serialized tokens
        candidate_selection = [*selected, snippet]
        tentative_serialized = serialize_context_bundle(candidate_selection)
        tentative_tokens = estimate_token_count(tentative_serialized)

        if tentative_tokens > limits.max_context_tokens:
            omitted.append(
                OmittedCandidate(
                    candidate_id=snippet.snippet_id,
                    file_path=snippet.file_path,
                    symbol_name=snippet.symbol_name,
                    reason=OmissionReason.TOKEN_BUDGET_EXCEEDED,
                    detail=f"Exceeds remaining context token budget ({tentative_tokens} > {limits.max_context_tokens})",
                    estimated_tokens=snippet.estimated_tokens,
                )
            )
            continue

        # Accepted
        selected.append(snippet)
        selected_files.add(snippet.file_path)

    # 4. Final verification: ensure final serialized bundle strictly fits
    final_serialized = serialize_context_bundle(selected)
    final_tokens = estimate_token_count(final_serialized)

    while final_tokens > limits.max_context_tokens and selected:
        dropped = selected.pop()
        final_serialized = serialize_context_bundle(selected)
        final_tokens = estimate_token_count(final_serialized)
        omitted.append(
            OmittedCandidate(
                candidate_id=dropped.snippet_id,
                file_path=dropped.file_path,
                symbol_name=dropped.symbol_name,
                reason=OmissionReason.TOKEN_BUDGET_EXCEEDED,
                detail="Dropped during final serialization budget verification",
                estimated_tokens=dropped.estimated_tokens,
            )
        )

    is_truncated = len(omitted) > 0 or any(s.is_truncated for s in selected)
    return selected, omitted, final_serialized, final_tokens, is_truncated
