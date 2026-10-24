"""Evidence validation, location grounding, symbol verification, and downgrade/rejection gates."""

from __future__ import annotations

import hashlib
import keyword
import re
from pathlib import Path
from typing import Any

from deicide.core.errors import SecurityViolationError
from deicide.core.models import (
    AnalysisWarning,
    CandidateFinding,
    Confidence,
    Finding,
    LineRange,
)
from deicide.git.analyzer import ensure_safe_repo_file, normalize_repo_relative_path
from deicide.git.models import DiffBundle, FileChangeStatus, FileContext, FileDiff
from deicide.intelligence.models import (
    ChangeRelevance,
    EvidenceAssessment,
    EvidenceQuality,
)

_WHITESPACE_RE = re.compile(r"\s+")
_LINE_PREFIX_RE = re.compile(r"^(?:\+|\-|\d+\s*:\s*)")
_STRING_LITERAL_RE = re.compile(r"(?:'[^']*'|\"[^\"]*\")")
_IDENTIFIER_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")

_STYLE_NITPICK_RE = re.compile(
    r"\b(pep\s*8|line length|trailing whitespace|missing docstring|rename variable for style|"
    r"formatting nit|alphabetical import|indentation style)\b",
    re.IGNORECASE,
)

_IGNORED_IDENTIFIERS: frozenset[str] = frozenset(
    set(keyword.kwlist)
    | {
        "self",
        "cls",
        "int",
        "float",
        "str",
        "bool",
        "bytes",
        "list",
        "dict",
        "set",
        "tuple",
        "frozenset",
        "object",
        "type",
        "Any",
        "Optional",
        "Union",
        "Callable",
        "Iterable",
        "Iterator",
        "Sequence",
        "Mapping",
        "len",
        "sum",
        "min",
        "max",
        "abs",
        "all",
        "any",
        "range",
        "enumerate",
        "zip",
        "map",
        "filter",
        "sorted",
        "reversed",
        "open",
        "print",
        "isinstance",
        "issubclass",
        "getattr",
        "setattr",
        "hasattr",
        "super",
        "property",
        "staticmethod",
        "classmethod",
        "Exception",
        "BaseException",
        "ValueError",
        "TypeError",
        "KeyError",
        "IndexError",
        "ZeroDivisionError",
        "AttributeError",
        "RuntimeError",
        "OSError",
        "IOError",
        "FileNotFoundError",
        "PermissionError",
        "StopIteration",
        "NotImplementedError",
        "AssertionError",
    }
)


def normalize_ws(text: str) -> str:
    """Collapse consecutive whitespace into single spaces and strip ends."""
    return _WHITESPACE_RE.sub(" ", text).strip()


def strip_evidence_line_prefixes(evidence: str) -> list[str]:
    """Strip diff markers (+, -) and numbered prefixes (42:) from evidence lines."""
    cleaned = evidence.strip()
    if not cleaned:
        return []
    return [
        _LINE_PREFIX_RE.sub("", line).strip() for line in cleaned.splitlines() if _LINE_PREFIX_RE.sub("", line).strip()
    ]


def _extract_code_identifiers(lines: list[str]) -> set[str]:
    """Extract non-builtin, non-keyword code identifiers outside string literals."""
    identifiers: set[str] = set()
    for line in lines:
        # Remove inline comments and string literals before scanning identifiers
        code_part = line.split("#", 1)[0]
        code_without_strings = _STRING_LITERAL_RE.sub(" ", code_part)
        for match in _IDENTIFIER_RE.findall(code_without_strings):
            if match not in _IGNORED_IDENTIFIERS and len(match) >= 2:
                identifiers.add(match)
    return identifiers


def compute_finding_id(
    module_id: str,
    file_path: str,
    category: str,
    line_range: LineRange,
    evidence: str,
) -> str:
    """Compute a deterministic, reproducible finding ID."""
    fingerprint = (
        f"{module_id.strip().lower()}|"
        f"{file_path.strip()}|"
        f"{category.strip().lower()}|"
        f"{line_range.start_line}:{line_range.end_line}|"
        f"{normalize_ws(evidence)}"
    )
    digest = hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()[:12]
    return f"dcd-{digest}"


def classify_change_relevance(file_diff: FileDiff, line_range: LineRange) -> ChangeRelevance:
    """Classify how directly `line_range` maps to changed lines in `file_diff`."""
    for line in file_diff.added_lines:
        if line_range.contains_line(line):
            return ChangeRelevance.DIRECT_ADDED_OR_MODIFIED

    for hunk in file_diff.hunks:
        hunk_range = hunk.new_line_range
        if hunk_range is not None and line_range.overlaps_or_adjacent(hunk_range, tolerance=1):
            if hunk.old_count > 0 and not hunk.added_line_numbers:
                return ChangeRelevance.DELETION_ADJACENT
            if hunk.old_count > 0:
                return ChangeRelevance.DELETION_ADJACENT

    return ChangeRelevance.HUNK_CONTEXT


def _collect_file_lines(
    file_diff: FileDiff,
    file_context: FileContext | None,
    line_range: LineRange | None = None,
) -> tuple[list[str], dict[int, str], list[str]]:
    """Return (all_available_lines, numbered_lines, local_range_lines) for a changed file."""
    numbered_lines: dict[int, str] = {}
    all_lines: list[str] = []
    local_lines: list[str] = []

    if file_context is not None:
        for snippet in file_context.snippets:
            for ln, text in snippet.lines.items():
                numbered_lines[ln] = text
                all_lines.append(text)
                if line_range is not None and (line_range.start_line - 1 <= ln <= line_range.end_line + 1):
                    local_lines.append(text)

    for hunk in file_diff.hunks:
        hunk_range = hunk.new_line_range
        overlaps_local = (
            line_range is not None
            and hunk_range is not None
            and line_range.overlaps_or_adjacent(hunk_range, tolerance=2)
        )
        cur_new = hunk.new_start
        for hline in hunk.lines:
            if hline.startswith(("+", " ")) and not hline.startswith("+++"):
                content = hline[1:]
                all_lines.append(content)
                numbered_lines.setdefault(cur_new, content)
                if overlaps_local or (
                    line_range is not None and (line_range.start_line - 1 <= cur_new <= line_range.end_line + 1)
                ):
                    local_lines.append(content)
                cur_new += 1
            elif hline.startswith("-") and not hline.startswith("---"):
                # Preserve deletion-only and modified diff evidence
                content = hline[1:]
                all_lines.append(content)
                if overlaps_local:
                    local_lines.append(content)

    return all_lines, numbered_lines, local_lines


def _is_in_supplementary_only(
    norm_evidence_flat: str,
    evidence_lines: list[str],
    context_result: Any | None,
) -> bool:
    """Check if unsupported evidence was lifted from supplementary context snippets."""
    if context_result is None:
        return False
    selected_snippets = getattr(context_result, "selected_snippets", None)
    if not selected_snippets:
        return False

    supp_lines: list[str] = []
    for snip in selected_snippets:
        content = getattr(snip, "content", "")
        if content:
            for raw_ln in content.splitlines():
                stripped = _LINE_PREFIX_RE.sub("", raw_ln).strip()
                if stripped:
                    supp_lines.append(stripped)

    if not supp_lines:
        return False

    supp_flat = normalize_ws(" ".join(supp_lines))
    if norm_evidence_flat and norm_evidence_flat in supp_flat:
        return True

    supp_corpus = "\n".join(normalize_ws(ln) for ln in supp_lines)
    for ev_ln in evidence_lines:
        norm_ev = normalize_ws(ev_ln)
        if len(norm_ev) >= 4 and norm_ev in supp_corpus:
            return True
    return False


def assess_evidence(
    evidence: str,
    line_range: LineRange,
    file_diff: FileDiff,
    file_context: FileContext | None,
    context_result: Any | None = None,
) -> EvidenceAssessment:
    """Evaluate whether `evidence` is grounded in the diff/context and classify its quality."""
    relevance = classify_change_relevance(file_diff, line_range)
    evidence_lines = strip_evidence_line_prefixes(evidence)
    if not evidence_lines:
        return EvidenceAssessment(
            is_valid=False,
            quality=EvidenceQuality.UNSUPPORTED,
            change_relevance=relevance,
            rejection_code="FINDING_REJECTED_UNSUPPORTED_EVIDENCE",
            rejection_reason="Evidence is empty after stripping prefixes.",
        )

    all_lines, numbered_lines, local_lines = _collect_file_lines(file_diff, file_context, line_range)
    if not all_lines:
        return EvidenceAssessment(
            is_valid=False,
            quality=EvidenceQuality.UNSUPPORTED,
            change_relevance=relevance,
            rejection_code="FINDING_REJECTED_UNSUPPORTED_EVIDENCE",
            rejection_reason="No source lines available in diff or context to support evidence.",
        )

    normalized_corpus = "\n".join(normalize_ws(line) for line in all_lines if line.strip())
    normalized_flat_corpus = normalize_ws(" ".join(all_lines))
    normalized_local_corpus = "\n".join(normalize_ws(line) for line in local_lines if line.strip())
    normalized_local_flat = normalize_ws(" ".join(local_lines))

    norm_evidence_flat = normalize_ws(" ".join(evidence_lines))

    # Check identifier grounding to detect invented symbols inside evidence
    file_identifiers = _extract_code_identifiers(all_lines)
    evidence_identifiers = _extract_code_identifiers(evidence_lines)
    invented_symbols = sorted(evidence_identifiers - file_identifiers) if file_identifiers else []

    # Find which numbered lines match the evidence
    matched_line_numbers: list[int] = []
    for ln, text in sorted(numbered_lines.items()):
        norm_text = normalize_ws(text)
        if not norm_text:
            continue
        for ev_line in evidence_lines:
            norm_ev_line = normalize_ws(ev_line)
            if norm_ev_line and (norm_ev_line in norm_text or norm_text in norm_ev_line):
                matched_line_numbers.append(ln)
                break

    flat_in_all = bool(norm_evidence_flat and norm_evidence_flat in normalized_flat_corpus)
    flat_in_local = bool(norm_evidence_flat and norm_evidence_flat in normalized_local_flat)

    matched_ev_lines_all = 0
    matched_ev_lines_local = 0
    for ev_line in evidence_lines:
        norm_ev_line = normalize_ws(ev_line)
        if len(norm_ev_line) >= 3 and norm_ev_line in normalized_corpus:
            matched_ev_lines_all += 1
        if len(norm_ev_line) >= 3 and norm_ev_line in normalized_local_corpus:
            matched_ev_lines_local += 1

    required_matches = len(evidence_lines) // 2 + (1 if len(evidence_lines) == 1 else 0)
    supported_in_all = flat_in_all or (matched_ev_lines_all > 0 and matched_ev_lines_all >= required_matches)

    if not supported_in_all:
        from_supp = _is_in_supplementary_only(norm_evidence_flat, evidence_lines, context_result)
        reason = (
            "Evidence comes from unchanged supplementary context rather than the target diff."
            if from_supp
            else "Evidence is not supported by the source context or diff."
        )
        return EvidenceAssessment(
            is_valid=False,
            quality=EvidenceQuality.UNSUPPORTED,
            change_relevance=relevance,
            matched_lines=matched_line_numbers,
            invented_symbols=invented_symbols,
            from_supplementary_only=from_supp,
            rejection_code="FINDING_REJECTED_UNSUPPORTED_EVIDENCE",
            rejection_reason=reason,
        )

    # If all code identifiers in the evidence were invented (e.g. matched only punctuation/keywords)
    if evidence_identifiers and len(invented_symbols) == len(evidence_identifiers):
        return EvidenceAssessment(
            is_valid=False,
            quality=EvidenceQuality.UNSUPPORTED,
            change_relevance=relevance,
            matched_lines=matched_line_numbers,
            invented_symbols=invented_symbols,
            rejection_code="FINDING_REJECTED_UNSUPPORTED_EVIDENCE",
            rejection_reason=f"Evidence references invented symbol(s) not in file: {', '.join(invented_symbols)}.",
        )

    # If some identifiers in multi-part evidence were invented, downgrade to WEAK
    if invented_symbols:
        return EvidenceAssessment(
            is_valid=True,
            quality=EvidenceQuality.WEAK,
            change_relevance=relevance,
            matched_lines=matched_line_numbers,
            invented_symbols=invented_symbols,
            downgrade_reason=(
                f"Evidence partially matches file but includes unverified identifier(s): {', '.join(invented_symbols)}."
            ),
        )

    # Check if evidence is overly trivial compared to the target line(s)
    max_local_len = max((len(normalize_ws(ln)) for ln in local_lines), default=0)
    if len(norm_evidence_flat) < 4 and max_local_len >= 12:
        return EvidenceAssessment(
            is_valid=True,
            quality=EvidenceQuality.WEAK,
            change_relevance=relevance,
            matched_lines=matched_line_numbers,
            downgrade_reason="Evidence fragment is overly brief compared to the target source statement.",
        )

    # Classify between EXACT, STRONG, and PARTIAL
    if flat_in_local or matched_ev_lines_local == len(evidence_lines):
        return EvidenceAssessment(
            is_valid=True,
            quality=EvidenceQuality.EXACT,
            change_relevance=relevance,
            matched_lines=matched_line_numbers,
        )

    if matched_ev_lines_all == len(evidence_lines) or flat_in_all:
        # Check distance of matched lines from reported line_range
        if matched_line_numbers:
            min_dist = min(
                0
                if line_range.contains_line(ln)
                else min(abs(ln - line_range.start_line), abs(ln - line_range.end_line))
                for ln in matched_line_numbers
            )
            if min_dist <= 3:
                return EvidenceAssessment(
                    is_valid=True,
                    quality=EvidenceQuality.STRONG,
                    change_relevance=relevance,
                    matched_lines=matched_line_numbers,
                )
            return EvidenceAssessment(
                is_valid=True,
                quality=EvidenceQuality.PARTIAL,
                change_relevance=relevance,
                matched_lines=matched_line_numbers,
                downgrade_reason=(
                    f"Evidence matches lines {matched_line_numbers} outside reported range "
                    f"{line_range.start_line}-{line_range.end_line}."
                ),
            )
        return EvidenceAssessment(
            is_valid=True,
            quality=EvidenceQuality.STRONG,
            change_relevance=relevance,
            matched_lines=matched_line_numbers,
        )

    # Partial line match (e.g., subset of multi-line evidence matched)
    return EvidenceAssessment(
        is_valid=True,
        quality=EvidenceQuality.PARTIAL,
        change_relevance=relevance,
        matched_lines=matched_line_numbers,
        downgrade_reason=(
            f"Only {matched_ev_lines_all} of {len(evidence_lines)} evidence lines matched the diff context."
        ),
    )


def is_evidence_supported(
    evidence: str,
    file_diff: FileDiff,
    file_context: FileContext | None,
) -> bool:
    """Verify that `evidence` is grounded in the retrieved file context or diff hunks."""
    dummy_range = LineRange(start_line=1, end_line=max(1, file_context.total_lines if file_context else 1))
    assessment = assess_evidence(
        evidence=evidence,
        line_range=dummy_range,
        file_diff=file_diff,
        file_context=file_context,
    )
    return assessment.is_valid


def _downgrade_confidence(confidence: Confidence) -> Confidence:
    if confidence == Confidence.HIGH:
        return Confidence.MEDIUM
    return Confidence.LOW


def validate_and_assess_candidates(
    module_id: str,
    candidates: list[CandidateFinding],
    diff_bundle: DiffBundle,
    contexts: dict[str, FileContext],
    repo_root: Path | None = None,
    context_result: Any | None = None,
) -> tuple[list[Finding], dict[str, EvidenceAssessment], int, int, list[AnalysisWarning]]:
    """Validate candidates against diff/context, returning (validated, assessments, rejected_count, downgraded_count, warnings)."""
    validated: list[Finding] = []
    assessments: dict[str, EvidenceAssessment] = {}
    rejected_count = 0
    downgraded_count = 0
    warnings: list[AnalysisWarning] = []

    supplementary_files: set[str] = set()
    if context_result is not None:
        for snip in getattr(context_result, "selected_snippets", []) or []:
            f_path = getattr(snip, "file_path", None)
            if f_path:
                supplementary_files.add(f_path.replace("\\", "/").lstrip("./"))

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
        if file_diff is None or file_diff.status == FileChangeStatus.DELETED or file_diff.is_binary:
            rejected_count += 1
            supp_note = (
                " (supplementary context files are not independently reportable)"
                if norm_path in supplementary_files
                else ""
            )
            warnings.append(
                AnalysisWarning(
                    code="FINDING_REJECTED_UNCHANGED_FILE",
                    message=f"Rejected finding on '{norm_path}' because file is not part of the diff{supp_note}.",
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

        assessment = assess_evidence(
            evidence=candidate.evidence,
            line_range=line_range,
            file_diff=file_diff,
            file_context=file_ctx,
            context_result=context_result,
        )

        if not assessment.is_valid:
            rejected_count += 1
            warnings.append(
                AnalysisWarning(
                    code=assessment.rejection_code or "FINDING_REJECTED_UNSUPPORTED_EVIDENCE",
                    message=(
                        f"Rejected finding '{candidate.title}' in '{norm_path}' because "
                        f"{(assessment.rejection_reason or 'evidence is not supported by the source context or diff').rstrip('.')}."
                    ),
                    file_path=norm_path,
                )
            )
            continue

        effective_confidence = candidate.confidence
        if assessment.quality in {EvidenceQuality.PARTIAL, EvidenceQuality.WEAK}:
            downgraded_conf = _downgrade_confidence(candidate.confidence)
            if downgraded_conf != candidate.confidence or assessment.downgrade_reason:
                effective_confidence = downgraded_conf
                downgraded_count += 1
                warnings.append(
                    AnalysisWarning(
                        code="FINDING_DOWNGRADED_WEAK_EVIDENCE",
                        message=(
                            f"Downgraded confidence of '{candidate.title}' in '{norm_path}' to "
                            f"'{effective_confidence.value}': {assessment.downgrade_reason}"
                        ),
                        file_path=norm_path,
                    )
                )

        stable_id = compute_finding_id(
            module_id=module_id,
            file_path=norm_path,
            category=candidate.category.value,
            line_range=line_range,
            evidence=candidate.evidence,
        )

        finding = Finding(
            id=stable_id,
            module_id=module_id,
            category=candidate.category,
            severity=candidate.severity,
            confidence=effective_confidence,
            file_path=norm_path,
            line_range=line_range,
            title=candidate.title,
            description=candidate.description,
            evidence=candidate.evidence,
            suggested_remediation=candidate.suggested_remediation,
        )
        validated.append(finding)
        assessments[stable_id] = assessment

    return validated, assessments, rejected_count, downgraded_count, warnings
