"""DEICIDE MVP 1.3 Finding Intelligence Engine package."""

from __future__ import annotations

from deicide.intelligence.deduplicator import (
    are_categories_compatible,
    are_findings_duplicates,
    deduplicate_findings_semantic,
    extract_code_tokens,
    extract_defect_concepts,
    extract_semantic_tokens,
    normalize_code_evidence,
    token_jaccard,
)
from deicide.intelligence.evidence_validator import (
    assess_evidence,
    classify_change_relevance,
    compute_finding_id,
    is_evidence_supported,
    normalize_ws,
    strip_evidence_line_prefixes,
    validate_and_assess_candidates,
)
from deicide.intelligence.models import (
    ChangeRelevance,
    DeduplicationGroup,
    EvidenceAssessment,
    EvidenceQuality,
    FindingIntelligenceResult,
    FindingIntelligenceSummary,
    NormalizationAdjustment,
)
from deicide.intelligence.pipeline import FindingIntelligenceEngine
from deicide.intelligence.prioritizer import (
    compute_finding_priority_key,
    prioritize_findings,
)
from deicide.intelligence.severity_normalizer import (
    normalize_finding_severity,
    normalize_findings,
)

__all__ = [
    "ChangeRelevance",
    "DeduplicationGroup",
    "EvidenceAssessment",
    "EvidenceQuality",
    "FindingIntelligenceEngine",
    "FindingIntelligenceResult",
    "FindingIntelligenceSummary",
    "NormalizationAdjustment",
    "are_categories_compatible",
    "are_findings_duplicates",
    "assess_evidence",
    "classify_change_relevance",
    "compute_finding_id",
    "compute_finding_priority_key",
    "deduplicate_findings_semantic",
    "extract_code_tokens",
    "extract_defect_concepts",
    "extract_semantic_tokens",
    "is_evidence_supported",
    "normalize_code_evidence",
    "normalize_finding_severity",
    "normalize_findings",
    "normalize_ws",
    "prioritize_findings",
    "strip_evidence_line_prefixes",
    "token_jaccard",
    "validate_and_assess_candidates",
]
