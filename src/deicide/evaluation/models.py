"""Data models for measuring DEICIDE finding quality, precision, recall, latency, and tokens."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from deicide.core.models import Finding, FindingCategory, LineRange, TokenUsage


class ExpectedFindingSpec(BaseModel):
    """Ground-truth expectation for a known correctness bug in an evaluation fixture."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    file_path: str
    line_range: LineRange
    category: FindingCategory | None = None

    def matches(self, actual: Finding) -> bool:
        if actual.file_path != self.file_path:
            return False
        if not actual.line_range.overlaps_or_adjacent(self.line_range, tolerance=2):
            return False
        return not (self.category is not None and actual.category != self.category)


class EvaluationFixture(BaseModel):
    """Single curated evaluation test case containing a diff, context, and expected findings."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    case_id: str
    description: str
    diff_text: str
    file_contents: dict[str, str] = Field(default_factory=dict)
    expected_findings: list[ExpectedFindingSpec] = Field(default_factory=list)


class CaseEvaluationResult(BaseModel):
    """Evaluation outcome for a single fixture case."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    case_id: str
    true_positives: int = Field(default=0, ge=0)
    false_positives: int = Field(default=0, ge=0)
    false_negatives: int = Field(default=0, ge=0)
    latency_ms: int = Field(default=0, ge=0)
    usage: TokenUsage = Field(default_factory=TokenUsage)
    actual_findings: list[Finding] = Field(default_factory=list)


class EvaluationSummary(BaseModel):
    """Aggregated evaluation metrics across a curated fixture corpus."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    dataset_size: int = Field(..., ge=0)
    positive_cases: int = Field(..., ge=0)
    negative_cases: int = Field(..., ge=0)
    true_positives: int = Field(..., ge=0)
    false_positives: int = Field(..., ge=0)
    false_negatives: int = Field(..., ge=0)
    precision: float = Field(..., ge=0.0, le=1.0)
    recall: float = Field(..., ge=0.0, le=1.0)
    false_positive_rate: float = Field(..., ge=0.0, le=1.0)
    total_latency_ms: int = Field(..., ge=0)
    average_latency_ms: float = Field(..., ge=0.0)
    usage: TokenUsage
    case_results: list[CaseEvaluationResult] = Field(default_factory=list)
