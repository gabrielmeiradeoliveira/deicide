"""Git models for diff parsing, changed-line mapping, and bounded source context."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from deicide.core.models import LineRange


class FileChangeStatus(StrEnum):
    """Change classification for a file in a Git diff."""

    ADDED = "added"
    MODIFIED = "modified"
    DELETED = "deleted"
    RENAMED = "renamed"
    BINARY = "binary"


class DiffHunk(BaseModel):
    """Single unified diff hunk with changed-line tracking."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    old_start: int = Field(..., ge=0)
    old_count: int = Field(..., ge=0)
    new_start: int = Field(..., ge=0)
    new_count: int = Field(..., ge=0)
    header: str = ""
    lines: list[str] = Field(default_factory=list)
    added_line_numbers: frozenset[int] = Field(default_factory=frozenset)
    context_line_numbers: frozenset[int] = Field(default_factory=frozenset)

    @property
    def new_line_range(self) -> LineRange | None:
        if self.new_start <= 0 or self.new_count <= 0:
            return None
        return LineRange(
            start_line=self.new_start,
            end_line=self.new_start + self.new_count - 1,
        )


class FileDiff(BaseModel):
    """Parsed diff information for a single changed file."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    old_path: str | None = None
    new_path: str
    status: FileChangeStatus
    is_binary: bool = False
    hunks: list[DiffHunk] = Field(default_factory=list)
    raw_diff: str = ""

    @property
    def added_lines(self) -> frozenset[int]:
        lines: set[int] = set()
        for hunk in self.hunks:
            lines.update(hunk.added_line_numbers)
        return frozenset(lines)

    @property
    def hunk_lines(self) -> frozenset[int]:
        lines: set[int] = set()
        for hunk in self.hunks:
            lines.update(hunk.added_line_numbers)
            lines.update(hunk.context_line_numbers)
        return frozenset(lines)

    def has_changed_overlap(self, line_range: LineRange) -> bool:
        """Return True if `line_range` intersects changed lines or active diff hunks in this file."""
        if self.status == FileChangeStatus.DELETED or self.is_binary:
            return False
        for line in self.added_lines:
            if line_range.contains_line(line):
                return True
        for hunk in self.hunks:
            hunk_range = hunk.new_line_range
            if (
                hunk_range is not None
                and line_range.overlaps_or_adjacent(hunk_range, tolerance=2)
                and (hunk.added_line_numbers or hunk.old_count > 0)
            ):
                return True
        return False


class ContextSnippet(BaseModel):
    """Bounded slice of numbered source lines around a diff hunk."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    start_line: int = Field(..., ge=1)
    end_line: int = Field(..., ge=1)
    lines: dict[int, str] = Field(default_factory=dict)

    def format_numbered(self) -> str:
        return "\n".join(f"{line_no}: {self.lines[line_no]}" for line_no in sorted(self.lines))


class FileContext(BaseModel):
    """Bounded source context retrieved from a changed file in the working tree."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    file_path: str
    total_lines: int = Field(default=0, ge=0)
    snippets: list[ContextSnippet] = Field(default_factory=list)
    truncated: bool = False

    def get_line_text(self, line_number: int) -> str | None:
        for snippet in self.snippets:
            if line_number in snippet.lines:
                return snippet.lines[line_number]
        return None

    def get_range_text(self, line_range: LineRange) -> list[str]:
        collected: list[str] = []
        for line_no in range(line_range.start_line, line_range.end_line + 1):
            text = self.get_line_text(line_no)
            if text is not None:
                collected.append(text)
        return collected


class DiffBundle(BaseModel):
    """Complete parsed Git diff across the repository."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    repository_root: str
    base_revision: str | None = None
    target_description: str
    files: list[FileDiff] = Field(default_factory=list)
    truncated: bool = False

    @property
    def is_empty(self) -> bool:
        return len(self.files) == 0

    @property
    def reviewable_files(self) -> list[FileDiff]:
        return [f for f in self.files if not f.is_binary and f.status != FileChangeStatus.DELETED and len(f.hunks) > 0]

    def get_file_diff(self, file_path: str) -> FileDiff | None:
        normalized = file_path.replace("\\", "/").lstrip("./")
        for f in self.files:
            if f.new_path == normalized:
                return f
        return None
