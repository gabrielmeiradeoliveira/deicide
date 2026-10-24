"""Finding validation, evidence verification, stable ID computation, and deduplication.

Delegates to the DEICIDE MVP 1.3 Finding Intelligence Engine (`deicide.intelligence`)
while preserving backward-compatible function signatures.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from deicide.core.models import (
    AnalysisWarning,
    CandidateFinding,
    Finding,
)
from deicide.git.models import DiffBundle, FileContext
from deicide.intelligence.deduplicator import deduplicate_findings_semantic
from deicide.intelligence.evidence_validator import (
    compute_finding_id,
    is_evidence_supported,
    validate_and_assess_candidates,
)
from deicide.intelligence.severity_normalizer import normalize_findings

__all__ = [
    "compute_finding_id",
    "deduplicate_findings",
    "is_evidence_supported",
    "validate_candidates",
]


def validate_candidates(
    module_id: str,
    candidates: list[CandidateFinding],
    diff_bundle: DiffBundle,
    contexts: dict[str, FileContext],
    repo_root: Path | None = None,
    context_result: Any | None = None,
) -> tuple[list[Finding], int, list[AnalysisWarning]]:
    """Validate candidate findings against the repository, diff mappings, and source context."""
    validated, assessments, rejected_count, _downgraded_count, warnings = validate_and_assess_candidates(
        module_id=module_id,
        candidates=candidates,
        diff_bundle=diff_bundle,
        contexts=contexts,
        repo_root=repo_root,
        context_result=context_result,
    )
    normalized, adjustments = normalize_findings(validated, assessments=assessments)
    for adj in adjustments:
        if adj.normalized_severity != adj.original_severity:
            target_file = next(
                (f.file_path for f in normalized if f.id == adj.finding_id),
                None,
            )
            warnings.append(
                AnalysisWarning(
                    code="FINDING_SEVERITY_NORMALIZED",
                    message=(
                        f"Normalized finding '{adj.finding_id}' severity from "
                        f"'{adj.original_severity.value}' to '{adj.normalized_severity.value}' "
                        f"[{adj.rule_id}]: {adj.reason}"
                    ),
                    file_path=target_file,
                )
            )
    return normalized, rejected_count, warnings


def deduplicate_findings(findings: list[Finding]) -> tuple[list[Finding], int]:
    """Deduplicate identical or semantically equivalent findings and return prioritized list."""
    kept, removed_count, _groups = deduplicate_findings_semantic(findings)
    return kept, removed_count
