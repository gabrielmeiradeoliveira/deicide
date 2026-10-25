"""Exact baseline finding validation and deduplication from DEICIDE prior to MVP 1.3.

Preserved verbatim for reproducible, scientific side-by-side quality evaluations.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from deicide.core.errors import SecurityViolationError
from deicide.core.models import (
    AnalysisWarning,
    CandidateFinding,
    Finding,
    LineRange,
)
from deicide.git.analyzer import ensure_safe_repo_file, normalize_repo_relative_path
from deicide.git.models import DiffBundle, FileContext, FileDiff

_WHITESPACE_RE = re.compile(r"\s+")
_STYLE_NITPICK_RE = re.compile(
    r"\b(pep\s*8|line length|trailing whitespace|missing docstring|rename variable for style|"
    r"formatting nit|alphabetical import|indentation style)\b",
    re.IGNORECASE,
)


def _normalize_ws(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text).strip()


def baseline_compute_finding_id(
    module_id: str,
    file_path: str,
    category: str,
    line_range: LineRange,
    evidence: str,
) -> str:
    """Compute deterministic finding ID using the pre-MVP 1.3 baseline algorithm."""
    fingerprint = (
        f"{module_id.strip().lower()}|"
        f"{file_path.strip()}|"
        f"{category.strip().lower()}|"
        f"{line_range.start_line}:{line_range.end_line}|"
        f"{_normalize_ws(evidence)}"
    )
    digest = hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()[:12]
    return f"dcd-{digest}"


def baseline_is_evidence_supported(
    evidence: str,
    file_diff: FileDiff,
    file_context: FileContext | None,
) -> bool:
    """Verify that evidence is grounded using the pre-MVP 1.3 baseline algorithm."""
    cleaned_evidence = evidence.strip()
    if not cleaned_evidence:
        return False

    available_lines: list[str] = []
    if file_context is not None:
        for snippet in file_context.snippets:
            available_lines.extend(snippet.lines.values())

    for hunk in file_diff.hunks:
        for hline in hunk.lines:
            if hline.startswith(("+", " ")) and not hline.startswith("+++"):
                available_lines.append(hline[1:])

    if not available_lines:
        return False

    normalized_corpus = "\n".join(_normalize_ws(line) for line in available_lines if line.strip())
    normalized_flat_corpus = _normalize_ws(" ".join(available_lines))

    evidence_lines = [
        re.sub(r"^(?:\+|\-|\d+\s*:\s*)", "", line).strip() for line in cleaned_evidence.splitlines() if line.strip()
    ]
    if not evidence_lines:
        return False

    norm_evidence_flat = _normalize_ws(" ".join(evidence_lines))
    if norm_evidence_flat and norm_evidence_flat in normalized_flat_corpus:
        return True

    matched_lines = 0
    for ev_line in evidence_lines:
        norm_ev_line = _normalize_ws(ev_line)
        if len(norm_ev_line) >= 3 and norm_ev_line in normalized_corpus:
            matched_lines += 1

    return matched_lines > 0 and matched_lines >= len(evidence_lines) // 2 + (1 if len(evidence_lines) == 1 else 0)


def baseline_validate_candidates(
    module_id: str,
    candidates: list[CandidateFinding],
    diff_bundle: DiffBundle,
    contexts: dict[str, FileContext],
    repo_root: Path | None = None,
) -> tuple[list[Finding], int, list[AnalysisWarning]]:
    """Validate candidate findings using the pre-MVP 1.3 baseline algorithm."""
    validated: list[Finding] = []
    rejected_count = 0
    warnings: list[AnalysisWarning] = []

    for candidate in candidates:
        try:
            norm_path = normalize_repo_relative_path(candidate.file_path)
        except SecurityViolationError as exc:
            rejected_count += 1
            warnings.append(
                AnalysisWarning(
                    code="FINDING_REJECTED_UNSAFE_PATH",
                    message=f"Rejected candidate finding with unsafe file path: {exc.message}",
                    file_path=candidate.file_path,
                )
            )
            continue

        if repo_root is not None:
            try:
                safe_target = ensure_safe_repo_file(repo_root, norm_path)
            except SecurityViolationError as exc:
                rejected_count += 1
                warnings.append(
                    AnalysisWarning(
                        code="FINDING_REJECTED_UNSAFE_PATH",
                        message=f"Rejected candidate finding outside repository root: {exc.message}",
                        file_path=norm_path,
                    )
                )
                continue
            if not safe_target.exists() or not safe_target.is_file():
                rejected_count += 1
                warnings.append(
                    AnalysisWarning(
                        code="FINDING_REJECTED_MISSING_FILE",
                        message=f"Rejected finding referencing non-existent file '{norm_path}'.",
                        file_path=norm_path,
                    )
                )
                continue

        file_diff = diff_bundle.get_file_diff(norm_path)
        if file_diff is None:
            rejected_count += 1
            warnings.append(
                AnalysisWarning(
                    code="FINDING_REJECTED_UNCHANGED_FILE",
                    message=f"Rejected finding on '{norm_path}' because file is not part of the diff.",
                    file_path=norm_path,
                )
            )
            continue

        end_line = candidate.end_line if candidate.end_line is not None else candidate.start_line
        line_range = LineRange(start_line=candidate.start_line, end_line=end_line)

        file_ctx = contexts.get(norm_path)
        if file_ctx is not None and file_ctx.total_lines > 0 and line_range.end_line > file_ctx.total_lines:
            rejected_count += 1
            warnings.append(
                AnalysisWarning(
                    code="FINDING_REJECTED_OUT_OF_BOUNDS_LINE",
                    message=(
                        f"Rejected finding on '{norm_path}:{line_range.start_line}-{line_range.end_line}' "
                        f"exceeding file length ({file_ctx.total_lines} lines)."
                    ),
                    file_path=norm_path,
                )
            )
            continue

        if not file_diff.has_changed_overlap(line_range):
            rejected_count += 1
            warnings.append(
                AnalysisWarning(
                    code="FINDING_REJECTED_UNMAPPED_LINE",
                    message=(
                        f"Rejected finding on '{norm_path}:{line_range.start_line}-{line_range.end_line}' "
                        "because location does not correspond to changed lines."
                    ),
                    file_path=norm_path,
                )
            )
            continue

        if _STYLE_NITPICK_RE.search(candidate.title) or _STYLE_NITPICK_RE.search(candidate.description):
            rejected_count += 1
            warnings.append(
                AnalysisWarning(
                    code="FINDING_REJECTED_STYLE_NOISE",
                    message=f"Rejected stylistic recommendation '{candidate.title}' in '{norm_path}'.",
                    file_path=norm_path,
                )
            )
            continue

        if not baseline_is_evidence_supported(candidate.evidence, file_diff, file_ctx):
            rejected_count += 1
            warnings.append(
                AnalysisWarning(
                    code="FINDING_REJECTED_UNSUPPORTED_EVIDENCE",
                    message=(
                        f"Rejected finding '{candidate.title}' in '{norm_path}' because evidence "
                        "is not supported by the source context or diff."
                    ),
                    file_path=norm_path,
                )
            )
            continue

        stable_id = baseline_compute_finding_id(
            module_id=module_id,
            file_path=norm_path,
            category=candidate.category.value,
            line_range=line_range,
            evidence=candidate.evidence,
        )

        validated.append(
            Finding(
                id=stable_id,
                module_id=module_id,
                category=candidate.category,
                severity=candidate.severity,
                confidence=candidate.confidence,
                file_path=norm_path,
                line_range=line_range,
                title=candidate.title,
                description=candidate.description,
                evidence=candidate.evidence,
                suggested_remediation=candidate.suggested_remediation,
            )
        )

    return validated, rejected_count, warnings


def baseline_deduplicate_findings(findings: list[Finding]) -> tuple[list[Finding], int]:
    """Deduplicate findings using the pre-MVP 1.3 baseline algorithm."""
    if not findings:
        return [], 0

    sorted_findings = sorted(
        findings,
        key=lambda f: (
            f.severity.rank,
            f.confidence.rank,
            len(f.description),
            -f.line_range.start_line,
        ),
        reverse=True,
    )

    kept: list[Finding] = []
    seen_ids: set[str] = set()
    removed_count = 0

    for finding in sorted_findings:
        if finding.id in seen_ids:
            removed_count += 1
            continue

        is_duplicate = False
        for existing in kept:
            if (
                existing.module_id == finding.module_id
                and existing.file_path == finding.file_path
                and existing.line_range.overlaps_or_adjacent(finding.line_range, tolerance=1)
                and (
                    existing.category == finding.category
                    or _normalize_ws(existing.title.lower()) == _normalize_ws(finding.title.lower())
                    or _normalize_ws(existing.evidence) == _normalize_ws(finding.evidence)
                )
            ):
                is_duplicate = True
                break

        if is_duplicate:
            removed_count += 1
        else:
            seen_ids.add(finding.id)
            kept.append(finding)

    kept.sort(
        key=lambda f: (
            f.file_path,
            f.line_range.start_line,
            -f.severity.rank,
            f.id,
        )
    )
    return kept, removed_count


def baseline_process_candidates(
    module_id: str,
    candidates: list[CandidateFinding],
    diff_bundle: DiffBundle,
    contexts: dict[str, FileContext],
    repo_root: Path | None = None,
) -> tuple[list[Finding], int, int, list[AnalysisWarning]]:
    """Run the complete pre-MVP 1.3 baseline validation and deduplication pipeline."""
    validated, rejected_count, warnings = baseline_validate_candidates(
        module_id=module_id,
        candidates=candidates,
        diff_bundle=diff_bundle,
        contexts=contexts,
        repo_root=repo_root,
    )
    kept, deduped_count = baseline_deduplicate_findings(validated)
    return kept, rejected_count, deduped_count, warnings
