"""Data models for the DEICIDE MVP 1.3 Finding Intelligence Engine."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from deicide.core.models import AnalysisWarning, Confidence, Finding, Severity


class EvidenceQuality(StrEnum):
    """Deterministic classification of how strongly a finding's evidence is grounded in the diff."""

    EXACT = "exact"
    STRONG = "strong"
    PARTIAL = "partial"
    WEAK = "weak"
    UNSUPPORTED = "unsupported"

    @property
    def rank(self) -> int:
        ranks = {
            EvidenceQuality.EXACT: 4,
            EvidenceQuality.STRONG: 3,
            EvidenceQuality.PARTIAL: 2,
            EvidenceQuality.WEAK: 1,
            EvidenceQuality.UNSUPPORTED: 0,
        }
        return ranks[self]


class ChangeRelevance(StrEnum):
    """How directly a finding's location maps to modified lines in the Git diff."""

    DIRECT_ADDED_OR_MODIFIED = "direct_added_or_modified"
    DELETION_ADJACENT = "deletion_adjacent"
    HUNK_CONTEXT = "hunk_context"

    @property
    def rank(self) -> int:
        ranks = {
            ChangeRelevance.DIRECT_ADDED_OR_MODIFIED: 3,
            ChangeRelevance.DELETION_ADJACENT: 2,
            ChangeRelevance.HUNK_CONTEXT: 1,
        }
        return ranks[self]


class EvidenceAssessment(BaseModel):
    """Detailed result of validating a candidate finding's location and evidence."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    is_valid: bool
    quality: EvidenceQuality
    change_relevance: ChangeRelevance = ChangeRelevance.DIRECT_ADDED_OR_MODIFIED
    matched_lines: list[int] = Field(default_factory=list)
    invented_symbols: list[str] = Field(default_factory=list)
    from_supplementary_only: bool = False
    rejection_code: str | None = None
    rejection_reason: str | None = None
    downgrade_reason: str | None = None


class NormalizationAdjustment(BaseModel):
    """Explainable record of a deterministic severity or confidence normalization."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    finding_id: str
    original_severity: Severity
    normalized_severity: Severity
    original_confidence: Confidence
    normalized_confidence: Confidence
    rule_id: str
    reason: str


class DeduplicationGroup(BaseModel):
    """Traceable record of duplicate findings merged into a primary finding."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kept_finding_id: str
    merged_finding_ids: list[str] = Field(default_factory=list)
    reason: str
    similarity_score: float = Field(default=1.0, ge=0.0, le=1.0)


class FindingIntelligenceSummary(BaseModel):
    """Summary metrics for the Finding Intelligence Engine post-processing pipeline."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    candidates_received: int = Field(default=0, ge=0)
    validated_count: int = Field(default=0, ge=0)
    rejected_count: int = Field(default=0, ge=0)
    downgraded_count: int = Field(default=0, ge=0)
    severity_normalized_count: int = Field(default=0, ge=0)
    deduplicated_count: int = Field(default=0, ge=0)
    final_findings_count: int = Field(default=0, ge=0)


class FindingIntelligenceResult(BaseModel):
    """Complete output of the Finding Intelligence Engine pipeline."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    findings: list[Finding] = Field(default_factory=list)
    rejected_count: int = Field(default=0, ge=0)
    deduplicated_count: int = Field(default=0, ge=0)
    downgraded_count: int = Field(default=0, ge=0)
    severity_normalized_count: int = Field(default=0, ge=0)
    warnings: list[AnalysisWarning] = Field(default_factory=list)
    adjustments: list[NormalizationAdjustment] = Field(default_factory=list)
    deduplication_groups: list[DeduplicationGroup] = Field(default_factory=list)
    assessments: dict[str, EvidenceAssessment] = Field(default_factory=dict)
    summary: FindingIntelligenceSummary = Field(default_factory=FindingIntelligenceSummary)
