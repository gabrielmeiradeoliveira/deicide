"""Domain models for AST extraction, symbol tables, references, and changed symbol identification."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class SourceLocation(BaseModel):
    """1-based source location coordinates within a repository file."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    file_path: str = Field(..., min_length=1, description="Repository-relative POSIX file path.")
    start_line: int = Field(..., ge=1, description="1-based start line (inclusive).")
    end_line: int = Field(..., ge=1, description="1-based end line (inclusive).")
    start_col: int | None = Field(default=None, ge=0, description="0-based or 1-based start column.")
    end_col: int | None = Field(default=None, ge=0, description="0-based or 1-based end column.")

    @model_validator(mode="after")
    def validate_lines(self) -> SourceLocation:
        if self.end_line < self.start_line:
            msg = f"end_line ({self.end_line}) cannot be less than start_line ({self.start_line})."
            raise ValueError(msg)
        return self

    def contains_line(self, line: int) -> bool:
        """Return True if line is inside [start_line, end_line]."""
        return self.start_line <= line <= self.end_line


class SymbolKind(StrEnum):
    """Classification of an extracted code symbol."""

    MODULE = "module"
    FUNCTION = "function"
    ASYNC_FUNCTION = "async_function"
    CLASS = "class"
    METHOD = "method"
    ASYNC_METHOD = "async_method"
    CLASS_METHOD = "class_method"
    STATIC_METHOD = "static_method"
    NESTED_FUNCTION = "nested_function"
    NESTED_ASYNC_FUNCTION = "nested_async_function"
    NESTED_CLASS = "nested_class"


class Symbol(BaseModel):
    """Extracted code definition with stable identity, lexical scope, and location."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(..., min_length=1, description="Stable deterministic symbol identifier.")
    name: str = Field(..., min_length=1)
    qualified_name: str = Field(..., min_length=1)
    kind: SymbolKind
    location: SourceLocation
    parent_id: str | None = None
    scope_path: tuple[str, ...] = Field(default_factory=tuple)
    decorators: list[str] = Field(default_factory=list)
    docstring: str | None = None
    local_bindings: frozenset[str] = Field(default_factory=frozenset)


class ImportKind(StrEnum):
    """Classification of an import statement."""

    MODULE = "module"
    FROM_MODULE = "from_module"
    WILDCARD = "wildcard"


class ImportReference(BaseModel):
    """Syntactic reference to an imported module, class, function, or wildcard."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_file: str = Field(..., min_length=1)
    location: SourceLocation
    imported_module: str = Field(..., description="Module path being imported (e.g. 'os', 'pkg.mod').")
    imported_name: str | None = Field(default=None, description="Symbol name if imported from a module.")
    local_name: str = Field(..., min_length=1, description="Name bound in the local namespace.")
    relative_level: int = Field(default=0, ge=0, description="Relative import level (0 for absolute, 1 for .).")
    is_wildcard: bool = False
    import_kind: ImportKind


class CallReference(BaseModel):
    """Syntactic function or method invocation with static name resolution."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_file: str = Field(..., min_length=1)
    location: SourceLocation
    scope_path: tuple[str, ...] = Field(default_factory=tuple)
    caller_symbol_id: str | None = None
    called_expression: str = Field(..., min_length=1)
    simple_name: str | None = None
    attribute_chain: list[str] = Field(default_factory=list)
    is_dynamic: bool = False


class FileAnalysisStatus(StrEnum):
    """Outcome of parsing and extracting AST data from a source file."""

    PARSED = "parsed"
    SYNTAX_ERROR = "syntax_error"
    DECODING_ERROR = "decoding_error"
    FILE_NOT_FOUND = "file_not_found"
    SECURITY_VIOLATION = "security_violation"
    EMPTY_FILE = "empty_file"
    UNSUPPORTED = "unsupported"


class FileAnalysis(BaseModel):
    """Complete static AST extraction result for a single repository source file."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_file: str = Field(..., min_length=1)
    status: FileAnalysisStatus
    symbols: list[Symbol] = Field(default_factory=list)
    imports: list[ImportReference] = Field(default_factory=list)
    calls: list[CallReference] = Field(default_factory=list)
    error_message: str | None = None
    total_lines: int = Field(default=0, ge=0)


class OverlapClassification(StrEnum):
    """Classification of how changed diff lines overlap a code symbol."""

    EXACT = "exact"
    SIGNATURE = "signature"
    BODY = "body"
    PARTIAL = "partial"
    MODULE_LEVEL = "module_level"


class ChangedSymbol(BaseModel):
    """Symbol that directly intersects modified Git diff lines."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    symbol_id: str = Field(..., min_length=1)
    symbol_name: str = Field(..., min_length=1)
    qualified_name: str = Field(..., min_length=1)
    kind: SymbolKind
    location: SourceLocation
    parent_id: str | None = None
    overlapping_lines: list[int] = Field(default_factory=list)
    classification: OverlapClassification
    is_innermost: bool = True


class ModuleLevelChange(BaseModel):
    """Diff change that occurs outside any class, method, or function definition."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    file_path: str = Field(..., min_length=1)
    changed_lines: list[int] = Field(default_factory=list)
    description: str = "Module-level change outside symbol definitions"


class ChangedSymbolsResult(BaseModel):
    """Complete set of changed symbols and module-level changes for a file."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    file_path: str = Field(..., min_length=1)
    changed_symbols: list[ChangedSymbol] = Field(default_factory=list)
    module_level_changes: list[ModuleLevelChange] = Field(default_factory=list)
    deleted_lines_count: int = Field(default=0, ge=0)
    deleted_lines_note: str | None = None
