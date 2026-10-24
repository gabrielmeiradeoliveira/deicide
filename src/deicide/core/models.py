"""Shared data models, schemas, and execution budget trackers for DEICIDE."""

from __future__ import annotations

import time
from enum import StrEnum
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from deicide.core.errors import BudgetExceededError
from deicide.quota.store import RequestStatus

if TYPE_CHECKING:
    from deicide.quota.store import UsageStore


class Severity(StrEnum):
    """Severity levels for findings."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

    @property
    def rank(self) -> int:
        ranks = {
            Severity.CRITICAL: 5,
            Severity.HIGH: 4,
            Severity.MEDIUM: 3,
            Severity.LOW: 2,
            Severity.INFO: 1,
        }
        return ranks[self]


class Confidence(StrEnum):
    """Categorical confidence rating (not a calibrated probability without evaluation)."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

    @property
    def rank(self) -> int:
        ranks = {
            Confidence.HIGH: 3,
            Confidence.MEDIUM: 2,
            Confidence.LOW: 1,
        }
        return ranks[self]


class FindingCategory(StrEnum):
    """Supported correctness bug categories."""

    CORRECTNESS = "correctness"
    LOGIC_ERROR = "logic_error"
    NULL_HANDLING = "null_handling"
    BOUNDARY_CONDITION = "boundary_condition"
    RESOURCE_MANAGEMENT = "resource_management"
    CONCURRENCY = "concurrency"
    ERROR_HANDLING = "error_handling"
    STATE_MUTATION = "state_mutation"
    TYPE_MISMATCH = "type_mismatch"


class LineRange(BaseModel):
    """1-based inclusive line range within a source file."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    start_line: int = Field(..., ge=1, description="1-based start line number (inclusive).")
    end_line: int = Field(..., ge=1, description="1-based end line number (inclusive).")

    @model_validator(mode="after")
    def validate_order(self) -> LineRange:
        if self.end_line < self.start_line:
            msg = f"end_line ({self.end_line}) must be >= start_line ({self.start_line})"
            raise ValueError(msg)
        return self

    def overlaps_or_adjacent(self, other: LineRange, tolerance: int = 1) -> bool:
        """Return True if two line ranges overlap or are within `tolerance` lines."""
        return self.start_line <= other.end_line + tolerance and other.start_line <= self.end_line + tolerance

    def contains_line(self, line: int) -> bool:
        """Return True if `line` falls inside this inclusive range."""
        return self.start_line <= line <= self.end_line


class CandidateFinding(BaseModel):
    """Unvalidated candidate finding returned by an analysis module or LLM provider."""

    model_config = ConfigDict(extra="ignore")

    category: FindingCategory
    severity: Severity
    confidence: Confidence
    file_path: str = Field(..., min_length=1)
    start_line: int = Field(..., ge=1)
    end_line: int | None = Field(default=None, ge=1)
    title: str = Field(..., min_length=3, max_length=200)
    description: str = Field(..., min_length=10, max_length=4000)
    evidence: str = Field(..., min_length=1, max_length=2000)
    suggested_remediation: str | None = Field(default=None, max_length=2000)

    @field_validator("title", "description", "evidence")
    @classmethod
    def strip_non_empty(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Field cannot be blank or whitespace-only.")
        return cleaned

    @model_validator(mode="after")
    def normalize_end_line(self) -> CandidateFinding:
        if self.end_line is None:
            self.end_line = self.start_line
        if self.end_line < self.start_line:
            msg = f"end_line ({self.end_line}) cannot be less than start_line ({self.start_line})."
            raise ValueError(msg)
        return self


class Finding(BaseModel):
    """Validated, evidence-backed finding with a deterministic stable identifier."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(..., min_length=5, description="Deterministic stable finding identifier.")
    module_id: str = Field(..., min_length=1, description="Technical identifier of the analyzer module.")
    category: FindingCategory
    severity: Severity
    confidence: Confidence
    file_path: str = Field(..., min_length=1, description="Repository-relative POSIX file path.")
    line_range: LineRange
    title: str = Field(..., min_length=3, max_length=200)
    description: str = Field(..., min_length=10, max_length=4000)
    evidence: str = Field(..., min_length=1, max_length=2000)
    suggested_remediation: str | None = Field(default=None, max_length=2000)


def estimate_token_count(text: str) -> int:
    """Conservative heuristic token count (~4 characters per token)."""
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


class TokenUsage(BaseModel):
    """Token consumption and optional cost estimate for an analysis run."""

    model_config = ConfigDict(extra="forbid")

    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)
    is_estimated: bool = Field(
        default=False,
        description="True when token counts are estimated locally rather than reported by the provider.",
    )
    estimated_cost_usd: float | None = Field(
        default=None,
        ge=0.0,
        description="Estimated cost in USD, if known for the model.",
    )

    def add(self, other: TokenUsage) -> TokenUsage:
        """Accumulate another TokenUsage instance into a new TokenUsage."""
        estimated_cost: float | None = None
        if self.estimated_cost_usd is not None or other.estimated_cost_usd is not None:
            estimated_cost = (self.estimated_cost_usd or 0.0) + (other.estimated_cost_usd or 0.0)
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            is_estimated=self.is_estimated or other.is_estimated,
            estimated_cost_usd=estimated_cost,
        )


class ExecutionBudget(BaseModel):
    """Hard resource, context, token, timing, and persistent quota ceilings."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_files: int = Field(default=30, ge=1)
    max_file_bytes: int = Field(default=256_000, ge=1024)
    max_context_lines_per_hunk: int = Field(default=20, ge=0, le=200)
    max_total_context_bytes: int = Field(default=200_000, ge=1024)
    max_diff_bytes: int = Field(default=300_000, ge=1024)
    max_requests: int = Field(default=3, ge=1)
    max_total_tokens: int = Field(default=8_000, ge=1)
    request_timeout_seconds: float = Field(default=60.0, gt=0.0, le=600.0)
    max_elapsed_seconds: float = Field(default=180.0, gt=0.0, le=1800.0)
    max_retries: int = Field(default=0, ge=0, le=10)
    quota_enabled: bool = Field(default=True)
    rate_limit_rpm: int | None = Field(default=5, ge=1)
    rate_limit_tpm: int | None = Field(default=250_000, ge=1)
    rate_limit_rpd: int | None = Field(default=20, ge=1)
    reservation_ttl_seconds: float = Field(default=120.0, gt=0.0, le=3600.0)


class AnalysisWarning(BaseModel):
    """Structured warning when part of an analysis is skipped, truncated, or rejected."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    code: str = Field(..., min_length=1)
    message: str = Field(..., min_length=1)
    file_path: str | None = None


class ContextSummary(BaseModel):
    """Observable summary of context engine execution during code review."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    status: str = "ready"
    indexed_files: int = Field(default=0, ge=0)
    selected_snippets: int = Field(default=0, ge=0)
    related_files: int = Field(default=0, ge=0)
    estimated_tokens: int = Field(default=0, ge=0)
    omitted_candidates: int = Field(default=0, ge=0)
    fallback: bool = False
    fallback_reason: str | None = None


class ProviderInfo(BaseModel):
    """Metadata describing the LLM provider used during analysis."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    provider_name: str
    model_name: str
    requests_made: int = Field(default=0, ge=0)
    retries_performed: int = Field(default=0, ge=0)


class AnalysisReport(BaseModel):
    """Top-level report returned by the DEICIDE analysis orchestrator."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: str = Field(default="1.0")
    platform: str = Field(default="DEICIDE")
    repository_root: str
    base_revision: str | None = None
    target_description: str
    modules_executed: list[str]
    provider: ProviderInfo
    usage: TokenUsage
    duration_ms: int = Field(default=0, ge=0)
    files_analyzed: list[str] = Field(default_factory=list)
    findings: list[Finding] = Field(default_factory=list)
    rejected_findings_count: int = Field(default=0, ge=0)
    deduplicated_findings_count: int = Field(default=0, ge=0)
    warnings: list[AnalysisWarning] = Field(default_factory=list)
    status: Literal["completed", "empty_diff", "incomplete"] = "completed"
    context_summary: ContextSummary | None = Field(default=None)


class BudgetTracker:
    """Mutable runtime tracker enforcing per-analysis ExecutionBudget and persistent SQLite quotas."""

    def __init__(
        self,
        budget: ExecutionBudget,
        *,
        usage_store: UsageStore | None = None,
        provider_name: str = "fake",
        model_name: str = "fake-deterministic-v1",
        quota_scope: str = "default",
    ) -> None:
        self.budget = budget
        self.usage_store = usage_store
        self.provider_name = provider_name
        self.model_name = model_name
        self.quota_scope = quota_scope
        self._start_monotonic = time.monotonic()
        self.files_processed: int = 0
        self.context_bytes_used: int = 0
        self.requests_made: int = 0
        self.retries_performed: int = 0
        self.usage = TokenUsage()
        self.warnings: list[AnalysisWarning] = []

    @property
    def elapsed_seconds(self) -> float:
        return time.monotonic() - self._start_monotonic

    def check_elapsed_time(self) -> None:
        if self.elapsed_seconds > self.budget.max_elapsed_seconds:
            msg = (
                f"Execution time budget exceeded: {self.elapsed_seconds:.2f}s > {self.budget.max_elapsed_seconds:.2f}s"
            )
            raise BudgetExceededError(msg)

    def check_can_make_request(self, estimated_prompt_tokens: int = 0) -> None:
        self.check_elapsed_time()
        if self.requests_made >= self.budget.max_requests:
            msg = f"Per-analysis request budget exceeded: {self.requests_made} >= {self.budget.max_requests}"
            raise BudgetExceededError(msg)
        projected_tokens = self.usage.total_tokens + max(0, estimated_prompt_tokens)
        if projected_tokens > self.budget.max_total_tokens:
            msg = (
                f"Token budget exceeded before request: projected {projected_tokens} tokens > "
                f"budget of {self.budget.max_total_tokens} tokens"
            )
            raise BudgetExceededError(msg)

    def reserve_attempt(
        self,
        estimated_prompt_tokens: int = 0,
        *,
        provider_name: str | None = None,
        model_name: str | None = None,
        is_retry: bool = False,
    ) -> str | None:
        """Check per-analysis and persistent quotas and atomically reserve 1 request attempt."""
        self.check_can_make_request(estimated_prompt_tokens)
        resolved_provider = (provider_name or self.provider_name).strip().lower()
        resolved_model = (model_name or self.model_name).strip()

        reservation_id: str | None = None
        if self.usage_store is not None and self.budget.quota_enabled and resolved_provider != "fake":
            reservation_id = self.usage_store.reserve(
                provider=resolved_provider,
                model=resolved_model,
                quota_scope=self.quota_scope,
                estimated_input_tokens=estimated_prompt_tokens,
                attempt_number=self.requests_made + 1,
                rate_limit_rpm=self.budget.rate_limit_rpm,
                rate_limit_tpm=self.budget.rate_limit_tpm,
                rate_limit_rpd=self.budget.rate_limit_rpd,
            )

        self.requests_made += 1
        if is_retry:
            self.retries_performed += 1
        return reservation_id

    def complete_attempt(
        self,
        reservation_id: str | None,
        usage: TokenUsage,
    ) -> None:
        """Reconcile a reserved attempt with actual or estimated usage upon completion."""
        if self.usage_store is not None and reservation_id is not None:
            self.usage_store.reconcile(
                reservation_id,
                status=RequestStatus.SUCCEEDED,
                input_tokens=usage.prompt_tokens,
                output_tokens=usage.completion_tokens,
                total_tokens=usage.total_tokens,
                is_estimated=usage.is_estimated,
            )

        self.usage = self.usage.add(usage)
        self.check_elapsed_time()
        if self.usage.total_tokens > self.budget.max_total_tokens:
            msg = (
                f"Token budget exceeded: used {self.usage.total_tokens} tokens > "
                f"budget of {self.budget.max_total_tokens} tokens"
            )
            raise BudgetExceededError(msg)

    def fail_attempt(
        self,
        reservation_id: str | None,
        *,
        status: RequestStatus,
        estimated_prompt_tokens: int,
    ) -> None:
        """Record a failed, rate-limited, or uncertain attempt conservatively without refunding."""
        safe_in = max(0, int(estimated_prompt_tokens))
        conservative_usage = TokenUsage(
            prompt_tokens=safe_in,
            completion_tokens=0,
            total_tokens=safe_in,
            is_estimated=True,
        )
        if self.usage_store is not None and reservation_id is not None:
            self.usage_store.reconcile(
                reservation_id,
                status=status,
                input_tokens=safe_in,
                output_tokens=0,
                total_tokens=safe_in,
                is_estimated=True,
            )
        self.usage = self.usage.add(conservative_usage)

    def record_request(self, usage: TokenUsage, retries: int = 0) -> None:
        """Record a completed request directly when not using per-attempt reservations."""
        self.requests_made += 1
        self.retries_performed += retries
        self.usage = self.usage.add(usage)
        self.check_elapsed_time()
        if self.usage.total_tokens > self.budget.max_total_tokens:
            msg = (
                f"Token budget exceeded: used {self.usage.total_tokens} tokens > "
                f"budget of {self.budget.max_total_tokens} tokens"
            )
            raise BudgetExceededError(msg)

    def add_warning(self, code: str, message: str, file_path: str | None = None) -> None:
        self.warnings.append(AnalysisWarning(code=code, message=message, file_path=file_path))
