"""Tests for Git repository detection, diff extraction, line mapping, renames, deletions, and revisions."""

from __future__ import annotations

from pathlib import Path

import pytest

from deicide.core.errors import InvalidRevisionError, RepositoryError, SecurityViolationError
from deicide.core.models import ExecutionBudget, LineRange
from deicide.git.analyzer import (
    GitAnalyzer,
    ensure_safe_repo_file,
    normalize_repo_relative_path,
    parse_unified_diff,
)
from deicide.git.models import FileChangeStatus
from tests.conftest import run_git_cmd


def test_detect_valid_and_invalid_repository(temp_git_repo: Path, tmp_path: Path) -> None:
    analyzer = GitAnalyzer(temp_git_repo)
    assert analyzer.repo_root == temp_git_repo.resolve()

    non_repo = tmp_path / "not_a_repo"
    non_repo.mkdir()
    with pytest.raises(RepositoryError, match="Not a valid Git repository"):
        GitAnalyzer(non_repo)

    missing_dir = tmp_path / "does_not_exist"
    with pytest.raises(RepositoryError, match="does not exist"):
        GitAnalyzer(missing_dir)


def test_staged_and_unstaged_changes_detected(temp_git_repo: Path) -> None:
    # Modify existing file (unstaged) and stage a new file
    calc_file = temp_git_repo / "src" / "calc.py"
    calc_file.write_text(
        "def add(a: int, b: int) -> int:\n    return a + b\n\ndef divide(x: int) -> float:\n    return x / 0\n",
        encoding="utf-8",
    )

    staged_file = temp_git_repo / "src" / "staged_mod.py"
    staged_file.write_text(
        "def record(item: str, bucket=[]) -> list[str]:\n    bucket.append(item)\n    return bucket\n",
        encoding="utf-8",
    )
    run_git_cmd(temp_git_repo, "add", "src/staged_mod.py")

    analyzer = GitAnalyzer(temp_git_repo)
    bundle = analyzer.extract_diff()

    assert not bundle.is_empty
    paths = {f.new_path for f in bundle.files}
    assert "src/calc.py" in paths
    assert "src/staged_mod.py" in paths

    calc_diff = bundle.get_file_diff("src/calc.py")
    assert calc_diff is not None
    assert 4 in calc_diff.added_lines
    assert 5 in calc_diff.added_lines
    assert calc_diff.has_changed_overlap(LineRange(start_line=5, end_line=5))
    assert not calc_diff.has_changed_overlap(LineRange(start_line=100, end_line=105))


def test_compare_against_base_revision_and_invalid_revision(temp_git_repo: Path) -> None:
    calc_file = temp_git_repo / "src" / "calc.py"
    calc_file.write_text(
        "def add(a: int, b: int) -> int:\n    return a + b\n\ndef avg(items: list[int]) -> float:\n    return sum(items) / len(items)\n",
        encoding="utf-8",
    )
    run_git_cmd(temp_git_repo, "add", ".")
    run_git_cmd(temp_git_repo, "commit", "-m", "Second commit")

    analyzer = GitAnalyzer(temp_git_repo)
    # Working tree is clean vs HEAD
    clean_bundle = analyzer.extract_diff()
    assert clean_bundle.is_empty

    # Comparing against HEAD~1 detects the commit's changes
    base_bundle = analyzer.extract_diff(base_revision="HEAD~1")
    assert not base_bundle.is_empty
    calc_diff = base_bundle.get_file_diff("src/calc.py")
    assert calc_diff is not None
    assert 5 in calc_diff.added_lines

    with pytest.raises(InvalidRevisionError):
        analyzer.extract_diff(base_revision="nonexistent-branch-xyz")

    with pytest.raises(InvalidRevisionError):
        analyzer.extract_diff(base_revision="--output=/tmp/pwn")


def test_deleted_renamed_and_binary_files_handled_safely(temp_git_repo: Path) -> None:
    # Create second file and commit
    old_file = temp_git_repo / "src" / "old_name.py"
    old_file.write_text("def helper() -> int:\n    return 42\n", encoding="utf-8")
    to_delete = temp_git_repo / "src" / "obsolete.py"
    to_delete.write_text("x = 1\n", encoding="utf-8")
    run_git_cmd(temp_git_repo, "add", ".")
    run_git_cmd(temp_git_repo, "commit", "-m", "Add files for rename and delete")

    # Rename old_name.py -> new_name.py, delete obsolete.py, add binary file
    run_git_cmd(temp_git_repo, "mv", "src/old_name.py", "src/new_name.py")
    new_file = temp_git_repo / "src" / "new_name.py"
    new_file.write_text("def helper() -> int:\n    return 42 / 0\n", encoding="utf-8")
    run_git_cmd(temp_git_repo, "rm", "src/obsolete.py")

    bin_file = temp_git_repo / "src" / "blob.bin"
    bin_file.write_bytes(b"\x00\x01\x02\x03\xff\xfe")
    run_git_cmd(temp_git_repo, "add", "src/blob.bin")

    analyzer = GitAnalyzer(temp_git_repo)
    bundle = analyzer.extract_diff()

    obsolete_diff = bundle.get_file_diff("src/obsolete.py")
    assert obsolete_diff is not None
    assert obsolete_diff.status == FileChangeStatus.DELETED

    renamed_diff = bundle.get_file_diff("src/new_name.py")
    assert renamed_diff is not None
    assert renamed_diff.status == FileChangeStatus.RENAMED
    assert renamed_diff.old_path == "src/old_name.py"

    # Only reviewable text files are returned for context retrieval
    contexts = analyzer.retrieve_context(bundle)
    assert "src/new_name.py" in contexts
    assert "src/obsolete.py" not in contexts
    assert "src/blob.bin" not in contexts


def test_path_traversal_and_symlink_protection(temp_git_repo: Path, tmp_path: Path) -> None:
    with pytest.raises(SecurityViolationError):
        normalize_repo_relative_path("../outside.py")

    with pytest.raises(SecurityViolationError):
        normalize_repo_relative_path("src/../../outside.py")

    with pytest.raises(SecurityViolationError):
        normalize_repo_relative_path("/etc/passwd")

    with pytest.raises(SecurityViolationError):
        normalize_repo_relative_path("C:\\Windows\\System32\\cmd.exe")

    with pytest.raises(SecurityViolationError):
        ensure_safe_repo_file(temp_git_repo, "../secret.txt")

    outside_file = tmp_path / "outside_secret.py"
    outside_file.write_text("SECRET = 'top-secret'\n", encoding="utf-8")
    symlink_path = temp_git_repo / "src" / "escape_link.py"
    try:
        symlink_path.symlink_to(outside_file)
        symlink_created = True
    except OSError:
        # Windows without developer mode / elevated privileges may disallow symlink creation
        symlink_created = False

    if symlink_created:
        with pytest.raises(SecurityViolationError, match="symlink escape"):
            ensure_safe_repo_file(temp_git_repo, "src/escape_link.py")


def test_parse_unified_diff_rejects_traversal_in_diff_headers() -> None:
    malicious_diff = (
        "diff --git a/../../etc/passwd b/../../etc/passwd\n"
        "--- a/../../etc/passwd\n"
        "+++ b/../../etc/passwd\n"
        "@@ -1,1 +1,1 @@\n"
        "-root\n"
        "+hacked\n"
    )
    with pytest.raises(SecurityViolationError):
        parse_unified_diff(malicious_diff)


def test_context_retrieval_respects_budgets(temp_git_repo: Path) -> None:
    calc_file = temp_git_repo / "src" / "calc.py"
    lines = [f"x_{i} = {i}" for i in range(1, 100)]
    calc_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    small_budget = ExecutionBudget(max_context_lines_per_hunk=3, max_total_context_bytes=1024)
    analyzer = GitAnalyzer(temp_git_repo, budget=small_budget)
    bundle = analyzer.extract_diff()
    contexts = analyzer.retrieve_context(bundle)
    assert "src/calc.py" in contexts
