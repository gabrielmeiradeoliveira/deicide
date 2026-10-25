"""Data models for measuring DEICIDE finding quality, precision, recall, latency, and tokens."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from deicide.core.models import (
    CandidateFinding,
    Finding,
    FindingCategory,
    LineRange,
    Severity,
    TokenUsage,
)


class ChangeType(StrEnum):
    """Semantic classification of code changes in benchmark fixtures."""

    DEFECT_INTRODUCTION = "defect_introduction"  # Buggy change that introduces a defect
    VULNERABLE_SNAPSHOT = "vulnerable_snapshot"  # Vulnerable pre-fix snapshot
    CLEAN_BUG_FIX = "clean_bug_fix"  # Clean change that fixes an existing defect
    CLEAN_UNRELATED = "clean_unrelated"  # Clean change unrelated to a defect


class ExpectedFindingSpec(BaseModel):
    """Ground-truth expectation for a known correctness bug in an evaluation fixture."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    file_path: str
    line_range: LineRange
    category: FindingCategory | None = None
    severity: Severity | None = None

    def matches(self, actual: Finding, strict_severity: bool = False) -> bool:
        """Check if an actual finding identifies this expected bug."""
        if actual.file_path != self.file_path:
            return False
        if not actual.line_range.overlaps_or_adjacent(self.line_range, tolerance=2):
            return False
        if self.category is not None and actual.category != self.category:
            return False
        if not strict_severity or self.severity is None:
            return True
        return actual.severity == self.severity

    def severity_matches(self, actual: Finding) -> bool:
        """Check whether actual finding severity exactly matches expected severity."""
        return self.severity is None or actual.severity == self.severity


class EvaluationFixture(BaseModel):
    """Single curated evaluation test case containing a diff, context, and expected findings."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    case_id: str
    description: str
    diff_text: str
    file_contents: dict[str, str] = Field(default_factory=dict)
    expected_findings: list[ExpectedFindingSpec] = Field(default_factory=list)
    candidate_findings: list[CandidateFinding] = Field(default_factory=list)
    supplementary_snippets: list[dict[str, Any]] = Field(default_factory=list)
    repository: str | None = None
    commit_id: str | None = None
    is_clean: bool = False
    fix_reference: str | None = None
    provenance: str | None = None
    change_type: ChangeType | None = None
    suitable_for_live_benchmark: bool | None = None

    @property
    def resolved_provenance(self) -> str:
        """Categorize provenance into fully_verified, reference_modeled, or unverified."""
        if self.provenance:
            return self.provenance
        if self.commit_id in {"a76d89b", "ae580a6", "cfc826d"}:
            return "fully_verified"
        if self.is_clean:
            return "reference_modeled"
        return "fully_verified"

    @property
    def resolved_change_type(self) -> ChangeType:
        """Resolve the semantic change type."""
        if self.change_type is not None:
            return self.change_type
        if self.is_clean or not self.expected_findings:
            return ChangeType.CLEAN_UNRELATED
        return ChangeType.DEFECT_INTRODUCTION

    @property
    def is_suitable_for_live_benchmark(self) -> bool:
        """Determine whether the fixture is suitable for live defect detection benchmarking.

        A change that fixes an existing defect contains the fix rather than the bug,
        and cannot be used to measure true positive defect detection.
        """
        if self.suitable_for_live_benchmark is not None:
            return self.suitable_for_live_benchmark
        return self.resolved_change_type not in (ChangeType.CLEAN_BUG_FIX, ChangeType.VULNERABLE_SNAPSHOT)


class CaseEvaluationResult(BaseModel):
    """Evaluation outcome for a single fixture case."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    case_id: str
    true_positives: int = Field(default=0, ge=0)
    false_positives: int = Field(default=0, ge=0)
    false_negatives: int = Field(default=0, ge=0)
    duplicates_reduced: int = Field(default=0, ge=0)
    legitimate_rejected: int = Field(default=0, ge=0)
    severity_mismatches: int = Field(default=0, ge=0)
    latency_ms: int = Field(default=0, ge=0)
    usage: TokenUsage = Field(default_factory=TokenUsage)
    actual_findings: list[Finding] = Field(default_factory=list)
    status: str = Field(default="completed")
    error: str | None = Field(default=None)
    error_category: str | None = Field(default=None)
    http_status: int | None = Field(default=None)
    retries_attempted: int = Field(default=0, ge=0)
    provenance: str = Field(default="fully_verified")
    change_type: ChangeType | str | None = Field(default=None)


class EvaluationSummary(BaseModel):
    """Aggregated evaluation metrics across a curated fixture corpus."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    dataset_size: int = Field(..., ge=0)
    positive_cases: int = Field(..., ge=0)
    negative_cases: int = Field(..., ge=0)
    successful_cases: int = Field(default=0, ge=0)
    failed_cases: int = Field(default=0, ge=0)
    api_failed_cases: int = Field(default=0, ge=0)
    budget_exhausted_cases: int = Field(default=0, ge=0)
    skipped_cases: int = Field(default=0, ge=0)
    true_positives: int = Field(..., ge=0)
    false_positives: int = Field(..., ge=0)
    false_negatives: int = Field(..., ge=0)
    precision: float | None = Field(default=None, ge=0.0, le=1.0)
    recall: float | None = Field(default=None, ge=0.0, le=1.0)
    false_positive_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    duplicates_reduced: int = Field(default=0, ge=0)
    legitimate_rejected: int = Field(default=0, ge=0)
    severity_mismatches: int = Field(default=0, ge=0)
    evaluation_mode: str = Field(default="post_processing")
    total_latency_ms: int = Field(..., ge=0)
    average_latency_ms: float = Field(..., ge=0.0)
    usage: TokenUsage
    verified_cases: int = Field(default=0, ge=0)
    reference_modeled_cases: int = Field(default=0, ge=0)
    unverified_cases: int = Field(default=0, ge=0)
    budget_exhausted: bool = Field(default=False)
    max_cases_applied: int | None = Field(default=None)
    case_results: list[CaseEvaluationResult] = Field(default_factory=list)
