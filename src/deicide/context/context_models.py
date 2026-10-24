"""Data models for context candidate generation, snippet selection, and token budgeting."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from deicide.context.dependency_graph import EdgeResolutionStatus
from deicide.context.models import SourceLocation


class ContextRelationshipCategory(StrEnum):
    """Classification of relationship between a context candidate and changed code."""

    CHANGED_SYMBOL = "changed_symbol"
    DIRECT_CALLER = "direct_caller"
    DIRECT_CALLEE = "direct_callee"
    IMPORTED_DEFINITION = "imported_definition"
    RELATED_CLASS_METHOD = "related_class_method"
    MODULE_LEVEL_CONTEXT = "module_level_context"
    TRANSITIVE_CALLER = "transitive_caller"
    TRANSITIVE_CALLEE = "transitive_callee"
    SUPPORTING_REFERENCE = "supporting_reference"


class OmissionReason(StrEnum):
    """Reason why a candidate context snippet was rejected or omitted."""

    TOKEN_BUDGET_EXCEEDED = "token_budget_exceeded"  # noqa: S105
    FILE_LIMIT_EXCEEDED = "file_limit_exceeded"
    SNIPPET_LIMIT_EXCEEDED = "snippet_limit_exceeded"
    DUPLICATE = "duplicate"
    AMBIGUOUS_RELATIONSHIP = "ambiguous_relationship"
    UNSAFE_PATH = "unsafe_path"
    SOURCE_UNAVAILABLE = "source_unavailable"
    INVALID_SOURCE_SPAN = "invalid_source_span"
    FILE_SIZE_EXCEEDED = "file_size_exceeded"
    INDEX_INCOMPLETE = "index_incomplete"


class ProvenanceRecord(BaseModel):
    """Traceable evidence linking a context candidate to changed code."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_id: str
    target_id: str
    relationship: ContextRelationshipCategory
    call_site: SourceLocation | None = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    resolution_status: EdgeResolutionStatus = EdgeResolutionStatus.RESOLVED
    explanation: str = ""


class ContextCandidate(BaseModel):
    """Rankable context candidate derived from changed symbols and dependency graph."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    candidate_id: str
    file_path: str
    start_line: int = Field(..., ge=1)
    end_line: int = Field(..., ge=1)
    symbol_id: str | None = None
    qualified_name: str | None = None
    relationship: ContextRelationshipCategory
    relationship_depth: int = Field(default=0, ge=0)
    resolution_status: EdgeResolutionStatus = EdgeResolutionStatus.RESOLVED
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    priority: int = Field(default=0, ge=0)
    priority_explanation: str = ""
    provenance: list[ProvenanceRecord] = Field(default_factory=list)
    estimated_tokens: int = Field(default=0, ge=0)
    focus_lines: list[int] = Field(default_factory=list)


class SelectedSnippet(BaseModel):
    """Bounded, numbered slice of exact repository source text selected for AI review."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    snippet_id: str
    file_path: str
    start_line: int = Field(..., ge=1)
    end_line: int = Field(..., ge=1)
    symbol_name: str | None = None
    qualified_name: str | None = None
    content: str
    relationship: ContextRelationshipCategory
    provenance: list[ProvenanceRecord] = Field(default_factory=list)
    estimated_tokens: int = Field(default=0, ge=0)
    is_truncated: bool = False
    truncated_original_lines: int | None = None
    relevant_call_lines: list[int] = Field(default_factory=list)


class OmittedCandidate(BaseModel):
    """Record of a context candidate that was discarded with an explicit reason."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    candidate_id: str
    file_path: str
    symbol_name: str | None = None
    reason: OmissionReason
    detail: str = ""
    priority: int = Field(default=0, ge=0)
    estimated_tokens: int = Field(default=0, ge=0)


class ContextSelectionLimits(BaseModel):
    """Configurable safety limits for context selection."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_context_files: int = Field(default=8, ge=0)
    max_snippets: int = Field(default=16, ge=0)
    max_snippet_lines: int = Field(default=60, ge=1)
    max_graph_depth: int = Field(default=2, ge=0)
    max_visited_nodes: int = Field(default=100, ge=1)
    max_context_tokens: int = Field(default=2000, ge=0)


class ContextSelectionResult(BaseModel):
    """Result of context candidate discovery, ranking, snippet slicing, and budgeting."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    selected_snippets: list[SelectedSnippet] = Field(default_factory=list)
    omitted_candidates: list[OmittedCandidate] = Field(default_factory=list)
    total_estimated_tokens: int = Field(default=0, ge=0)
    max_allowed_tokens: int = Field(default=2000, ge=0)
    selected_files_count: int = Field(default=0, ge=0)
    selected_snippets_count: int = Field(default=0, ge=0)
    omitted_snippets_count: int = Field(default=0, ge=0)
    is_truncated: bool = False
    serialized_text: str = ""
    diagnostics: list[str] = Field(default_factory=list)
    limits: ContextSelectionLimits = Field(default_factory=ContextSelectionLimits)
