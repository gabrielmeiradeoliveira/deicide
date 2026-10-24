"""Deterministic prioritization and sorting of validated DEICIDE findings."""

from __future__ import annotations

from deicide.core.models import Finding
from deicide.intelligence.models import (
    ChangeRelevance,
    EvidenceAssessment,
    EvidenceQuality,
)


def compute_finding_priority_key(
    finding: Finding,
    assessment: EvidenceAssessment | None = None,
) -> tuple[int, int, int, int, str, int, int, str]:
    """Compute a deterministic sort key for prioritizing findings.

    Ordering criteria (highest priority first):
    1. Severity rank descending (critical > high > medium > low > info).
    2. Confidence rank descending (high > medium > low).
    3. Evidence quality rank descending (exact > strong > partial > weak).
    4. Change relevance rank descending (direct_added_or_modified > deletion_adjacent > hunk_context).
    5. File path ascending (lexicographical tie-breaker).
    6. Start line ascending.
    7. End line ascending.
    8. Stable finding ID ascending.
    """
    quality = assessment.quality if assessment is not None else EvidenceQuality.EXACT
    relevance = assessment.change_relevance if assessment is not None else ChangeRelevance.DIRECT_ADDED_OR_MODIFIED
    return (
        -finding.severity.rank,
        -finding.confidence.rank,
        -quality.rank,
        -relevance.rank,
        finding.file_path,
        finding.line_range.start_line,
        finding.line_range.end_line,
        finding.id,
    )


def prioritize_findings(
    findings: list[Finding],
    assessments: dict[str, EvidenceAssessment] | None = None,
) -> list[Finding]:
    """Return findings sorted deterministically by priority criteria."""
    if not findings:
        return []
    assessment_map = assessments or {}
    return sorted(
        findings,
        key=lambda f: compute_finding_priority_key(f, assessment_map.get(f.id)),
    )
