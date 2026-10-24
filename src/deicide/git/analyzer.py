"""Git repository detection, diff parsing, and safe bounded source context extraction."""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path, PurePosixPath

from deicide.core.errors import (
    InvalidRevisionError,
    RepositoryError,
    SecurityViolationError,
)
from deicide.core.models import BudgetTracker, ExecutionBudget
from deicide.git.models import (
    ContextSnippet,
    DiffBundle,
    DiffHunk,
    FileChangeStatus,
    FileContext,
    FileDiff,
)

_HUNK_HEADER_RE = re.compile(
    r"^@@ -(?P<old_start>\d+)(?:,(?P<old_count>\d+))? "
    r"\+(?P<new_start>\d+)(?:,(?P<new_count>\d+))? @@(?P<section>.*)$"
)
_DIFF_HEADER_RE = re.compile(r"^diff --git a/(?P<old>.+?) b/(?P<new>.+?)$")


def normalize_repo_relative_path(raw_path: str) -> str:
    """Validate and normalize a repository-relative path, rejecting traversal or absolute paths."""
    cleaned = raw_path.strip().replace("\\", "/")
    if cleaned.startswith(('"', "'")) and cleaned.endswith(('"', "'")) and len(cleaned) >= 2:
        cleaned = cleaned[1:-1]
    if not cleaned:
        raise SecurityViolationError("File path cannot be empty.")
    if cleaned.startswith("/") or (len(cleaned) >= 2 and cleaned[1] == ":"):
        raise SecurityViolationError(f"Absolute file paths are not allowed: '{raw_path}'")
    posix = PurePosixPath(cleaned)
    parts = posix.parts
    if not parts or any(part in {"..", ""} for part in parts if part != "."):
        raise SecurityViolationError(f"Path traversal detected in file path: '{raw_path}'")
    normalized = str(posix)
    if normalized == "." or normalized.startswith("../") or "/../" in f"/{normalized}/":
        raise SecurityViolationError(f"Path traversal detected in file path: '{raw_path}'")
    return normalized


def ensure_safe_repo_file(repo_root: Path, relative_path: str) -> Path:
    """Resolve a repository-relative path and ensure it stays strictly inside `repo_root`.

    Rejects both `..` path traversal and symlinks that resolve outside the repository root.
    """
    norm_rel = normalize_repo_relative_path(relative_path)
    resolved_root = repo_root.resolve()
    candidate = repo_root.joinpath(*PurePosixPath(norm_rel).parts)

    # Walk parent components to detect symlink escapes even on intermediate directories
    current = repo_root
    for part in PurePosixPath(norm_rel).parts:
        current = current / part
        if current.is_symlink():
            target_resolved = current.resolve()
            if not target_resolved.is_relative_to(resolved_root):
                msg = f"Unsafe symlink escape detected for '{relative_path}': resolves outside repository root."
                raise SecurityViolationError(msg)

    resolved_candidate = candidate.resolve()
    if not resolved_candidate.is_relative_to(resolved_root):
        msg = f"Path traversal detected for '{relative_path}': resolves outside repository root."
        raise SecurityViolationError(msg)

    return resolved_candidate


def _unquote_git_path(path_str: str) -> str:
    """Strip Git quotes and prefix (`a/` or `b/`) from diff paths."""
    s = path_str.strip()
    if s.startswith('"') and s.endswith('"') and len(s) >= 2:
        s = s[1:-1].encode("utf-8").decode("unicode_escape", errors="replace")
    if s == "/dev/null":
        return s
    if s.startswith(("a/", "b/")):
        return s[2:]
    return s


def parse_unified_diff(diff_text: str) -> list[FileDiff]:
    """Parse unified Git diff output into structured FileDiff objects with line mappings."""
    if not diff_text.strip():
        return []

    file_diffs: list[FileDiff] = []
    lines = diff_text.splitlines()

    idx = 0
    total = len(lines)

    while idx < total:
        line = lines[idx]
        if not line.startswith("diff --git "):
            idx += 1
            continue

        start_idx = idx
        header_match = _DIFF_HEADER_RE.match(line)
        old_path: str | None = _unquote_git_path(header_match.group("old")) if header_match else None
        new_path: str = _unquote_git_path(header_match.group("new")) if header_match else ""

        status = FileChangeStatus.MODIFIED
        is_binary = False
        hunks: list[DiffHunk] = []
        idx += 1

        while idx < total and not lines[idx].startswith("diff --git "):
            cur = lines[idx]
            if cur.startswith("new file mode "):
                status = FileChangeStatus.ADDED
                old_path = None
                idx += 1
            elif cur.startswith("deleted file mode "):
                status = FileChangeStatus.DELETED
                idx += 1
            elif cur.startswith("rename from "):
                status = FileChangeStatus.RENAMED
                old_path = _unquote_git_path(cur[len("rename from ") :])
                idx += 1
            elif cur.startswith("rename to "):
                status = FileChangeStatus.RENAMED
                new_path = _unquote_git_path(cur[len("rename to ") :])
                idx += 1
            elif cur.startswith("--- "):
                raw_old = _unquote_git_path(cur[4:])
                if raw_old == "/dev/null":
                    status = FileChangeStatus.ADDED
                    old_path = None
                elif old_path is None:
                    old_path = raw_old
                idx += 1
            elif cur.startswith("+++ "):
                raw_new = _unquote_git_path(cur[4:])
                if raw_new == "/dev/null":
                    status = FileChangeStatus.DELETED
                else:
                    new_path = raw_new
                idx += 1
            elif cur.startswith("Binary files ") or cur.startswith("GIT binary patch"):
                is_binary = True
                if status == FileChangeStatus.MODIFIED:
                    status = FileChangeStatus.BINARY
                idx += 1
            elif cur.startswith("@@ "):
                hunk_match = _HUNK_HEADER_RE.match(cur)
                if not hunk_match:
                    idx += 1
                    continue

                old_start = int(hunk_match.group("old_start"))
                old_count = int(hunk_match.group("old_count") or "1")
                new_start = int(hunk_match.group("new_start"))
                new_count = int(hunk_match.group("new_count") or "1")
                section = (hunk_match.group("section") or "").strip()

                hunk_lines: list[str] = []
                added_nums: set[int] = set()
                context_nums: set[int] = set()
                current_new_line = new_start
                idx += 1

                while idx < total:
                    hline = lines[idx]
                    if hline.startswith(("diff --git ", "@@ ")):
                        break
                    if hline.startswith("+") and not hline.startswith("+++"):
                        hunk_lines.append(hline)
                        added_nums.add(current_new_line)
                        current_new_line += 1
                    elif hline.startswith("-") and not hline.startswith("---"):
                        hunk_lines.append(hline)
                    elif hline.startswith(" "):
                        hunk_lines.append(hline)
                        context_nums.add(current_new_line)
                        current_new_line += 1
                    elif hline.startswith("\\ No newline at end of file"):
                        hunk_lines.append(hline)
                    else:
                        break
                    idx += 1

                hunks.append(
                    DiffHunk(
                        old_start=old_start,
                        old_count=old_count,
                        new_start=new_start,
                        new_count=new_count,
                        header=section,
                        lines=hunk_lines,
                        added_line_numbers=frozenset(added_nums),
                        context_line_numbers=frozenset(context_nums),
                    )
                )
            else:
                idx += 1

        if not new_path and old_path:
            new_path = old_path

        if new_path:
            norm_new = normalize_repo_relative_path(new_path)
            norm_old = normalize_repo_relative_path(old_path) if old_path else None
            raw_block = "\n".join(lines[start_idx:idx])
            file_diffs.append(
                FileDiff(
                    old_path=norm_old,
                    new_path=norm_new,
                    status=status,
                    is_binary=is_binary,
                    hunks=hunks,
                    raw_diff=raw_block,
                )
            )

    return file_diffs


class GitAnalyzer:
    """Safe, read-only Git repository inspector and bounded context extractor."""

    def __init__(self, repo_path: Path, budget: ExecutionBudget | None = None) -> None:
        self.requested_path = repo_path.resolve()
        self.budget = budget or ExecutionBudget()
        self.repo_root = self._detect_repository_root(self.requested_path)

    def _run_git(self, args: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
        """Run a read-only git command safely without shell interpolation."""
        cmd = ["git", "-C", str(self.requested_path), *args]
        env = os.environ.copy()
        env["GIT_TERMINAL_PROMPT"] = "0"
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=15.0,
                check=False,
                env=env,
            )
        except FileNotFoundError as exc:
            raise RepositoryError("Git executable was not found on PATH.") from exc
        except subprocess.TimeoutExpired as exc:
            raise RepositoryError("Git command timed out.") from exc

        if check and result.returncode != 0:
            stderr = result.stderr.strip() or result.stdout.strip()
            raise RepositoryError(f"Git operation failed: {stderr}")
        return result

    def _detect_repository_root(self, target: Path) -> Path:
        if not target.exists() or not target.is_dir():
            raise RepositoryError(f"Repository path does not exist or is not a directory: '{target}'")
        res = self._run_git(["rev-parse", "--show-toplevel"], check=False)
        if res.returncode != 0:
            raise RepositoryError(f"Not a valid Git repository: '{target}'")
        return Path(res.stdout.strip()).resolve()

    def has_head_commit(self) -> bool:
        """Return True if the repository has at least one commit at HEAD."""
        res = self._run_git(["rev-parse", "--verify", "HEAD^{commit}"], check=False)
        return res.returncode == 0

    def validate_revision(self, revision: str) -> str:
        """Verify that `revision` resolves to a valid commit in the repository."""
        cleaned = revision.strip()
        if not cleaned or cleaned.startswith("-"):
            raise InvalidRevisionError(f"Invalid Git revision: '{revision}'")
        res = self._run_git(["rev-parse", "--verify", f"{cleaned}^{{commit}}"], check=False)
        if res.returncode != 0:
            raise InvalidRevisionError(
                f"Git revision '{revision}' could not be resolved in repository '{self.repo_root}'."
            )
        return res.stdout.strip()

    def extract_diff(
        self,
        base_revision: str | None = None,
        tracker: BudgetTracker | None = None,
    ) -> DiffBundle:
        """Extract diffs for uncommitted changes (staged + unstaged) or against `base_revision`.

        Default behavior (`base_revision=None`):
          - If `HEAD` exists, compares `HEAD` against the current working tree (including both
            staged and unstaged modifications). If `git diff HEAD` misses staged-only changes
            on an unborn branch, falls back to merging `git diff --cached` and `git diff`.
        Custom base behavior (`base_revision=<rev>`):
          - Validates `<rev>` and compares `<rev>` against the current working tree.
        """
        if base_revision is not None:
            self.validate_revision(base_revision)
            target_desc = f"working tree vs {base_revision}"
            res = self._run_git(
                [
                    "diff",
                    "--unified=3",
                    "--find-renames",
                    "--no-color",
                    "--no-ext-diff",
                    base_revision,
                    "--",
                ]
            )
            raw_diff = res.stdout
        else:
            target_desc = "local uncommitted changes (staged and unstaged)"
            if self.has_head_commit():
                res = self._run_git(
                    [
                        "diff",
                        "--unified=3",
                        "--find-renames",
                        "--no-color",
                        "--no-ext-diff",
                        "HEAD",
                        "--",
                    ]
                )
                raw_diff = res.stdout
            else:
                staged = self._run_git(
                    [
                        "diff",
                        "--cached",
                        "--unified=3",
                        "--find-renames",
                        "--no-color",
                        "--no-ext-diff",
                        "--",
                    ]
                ).stdout
                unstaged = self._run_git(
                    ["diff", "--unified=3", "--find-renames", "--no-color", "--no-ext-diff", "--"]
                ).stdout
                raw_diff = f"{staged}\n{unstaged}".strip()

        truncated = False
        encoded_len = len(raw_diff.encode("utf-8", errors="replace"))
        if encoded_len > self.budget.max_diff_bytes:
            truncated = True
            if tracker is not None:
                tracker.add_warning(
                    code="DIFF_TRUNCATED",
                    message=(
                        f"Git diff size ({encoded_len} bytes) exceeded budget "
                        f"({self.budget.max_diff_bytes} bytes); analyzing bounded subset."
                    ),
                )

        parsed_files = parse_unified_diff(raw_diff)
        if len(parsed_files) > self.budget.max_files:
            truncated = True
            if tracker is not None:
                tracker.add_warning(
                    code="MAX_FILES_EXCEEDED",
                    message=(
                        f"Changed file count ({len(parsed_files)}) exceeded budget "
                        f"({self.budget.max_files}); analyzing first {self.budget.max_files} files."
                    ),
                )
            parsed_files = parsed_files[: self.budget.max_files]

        return DiffBundle(
            repository_root=self.repo_root.as_posix(),
            base_revision=base_revision,
            target_description=target_desc,
            files=parsed_files,
            truncated=truncated,
        )

    def retrieve_context(
        self,
        diff_bundle: DiffBundle,
        tracker: BudgetTracker | None = None,
    ) -> dict[str, FileContext]:
        """Retrieve bounded surrounding source lines for each reviewable changed file."""
        contexts: dict[str, FileContext] = {}
        padding = self.budget.max_context_lines_per_hunk

        for file_diff in diff_bundle.reviewable_files:
            if tracker is not None:
                tracker.check_elapsed_time()
                if tracker.context_bytes_used >= self.budget.max_total_context_bytes:
                    tracker.add_warning(
                        code="CONTEXT_BUDGET_REACHED",
                        message="Total context byte budget reached; skipping remaining file context.",
                        file_path=file_diff.new_path,
                    )
                    break

            safe_path = ensure_safe_repo_file(self.repo_root, file_diff.new_path)
            if not safe_path.exists() or not safe_path.is_file():
                continue

            try:
                file_size = safe_path.stat().st_size
            except OSError:
                continue

            file_truncated = False
            if file_size > self.budget.max_file_bytes:
                file_truncated = True
                if tracker is not None:
                    tracker.add_warning(
                        code="FILE_TOO_LARGE",
                        message=(
                            f"File '{file_diff.new_path}' ({file_size} bytes) exceeds "
                            f"per-file budget ({self.budget.max_file_bytes} bytes); reading bounded prefix."
                        ),
                        file_path=file_diff.new_path,
                    )

            with safe_path.open("rb") as fh:
                raw_bytes = fh.read(self.budget.max_file_bytes)
            if b"\x00" in raw_bytes[:8192]:
                # Binary file detected via null bytes
                continue

            text = raw_bytes.decode("utf-8", errors="replace")
            all_lines = text.splitlines()
            total_lines = len(all_lines)
            if total_lines == 0:
                continue

            # Merge overlapping context windows around diff hunks
            windows: list[tuple[int, int]] = []
            for hunk in file_diff.hunks:
                if hunk.new_start <= 0:
                    continue
                span = max(hunk.new_count, 1)
                w_start = max(1, hunk.new_start - padding)
                w_end = min(total_lines, hunk.new_start + span - 1 + padding)
                if w_end < w_start:
                    w_end = w_start
                if windows and w_start <= windows[-1][1] + 2:
                    prev_start, prev_end = windows[-1]
                    windows[-1] = (prev_start, max(prev_end, w_end))
                else:
                    windows.append((w_start, w_end))

            snippets: list[ContextSnippet] = []
            for w_start, w_end in windows:
                line_map: dict[int, str] = {}
                for line_no in range(w_start, w_end + 1):
                    line_text = all_lines[line_no - 1]
                    encoded_line_len = len(line_text.encode("utf-8", errors="replace"))
                    if tracker is not None:
                        if tracker.context_bytes_used + encoded_line_len > self.budget.max_total_context_bytes:
                            file_truncated = True
                            break
                        tracker.context_bytes_used += encoded_line_len
                    line_map[line_no] = line_text

                if line_map:
                    actual_nums = sorted(line_map)
                    snippets.append(
                        ContextSnippet(
                            start_line=actual_nums[0],
                            end_line=actual_nums[-1],
                            lines=line_map,
                        )
                    )

            contexts[file_diff.new_path] = FileContext(
                file_path=file_diff.new_path,
                total_lines=total_lines,
                snippets=snippets,
                truncated=file_truncated,
            )

        return contexts
