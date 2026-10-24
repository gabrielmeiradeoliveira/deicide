"""Git integration for change detection, diff extraction, and bounded context retrieval."""

from deicide.git.analyzer import GitAnalyzer, normalize_repo_relative_path, parse_unified_diff
from deicide.git.models import (
    ContextSnippet,
    DiffBundle,
    DiffHunk,
    FileChangeStatus,
    FileContext,
    FileDiff,
)

__all__ = [
    "ContextSnippet",
    "DiffBundle",
    "DiffHunk",
    "FileChangeStatus",
    "FileContext",
    "FileDiff",
    "GitAnalyzer",
    "normalize_repo_relative_path",
    "parse_unified_diff",
]
