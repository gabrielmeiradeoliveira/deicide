"""Shared pytest fixtures for creating temporary Git repositories."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def isolate_live_dotenv(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Prevent unit tests from reading the developer's live .env unless explicitly tested."""
    monkeypatch.setenv("DEICIDE_ENV_FILE", str(tmp_path / ".isolated_empty_dotenv"))


def run_git_cmd(repo: Path, *args: str) -> str:
    res = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    return res.stdout


@pytest.fixture
def temp_git_repo(tmp_path: Path) -> Path:
    """Create an initialized Git repository with one initial commit."""
    repo = tmp_path / "sample_repo"
    repo.mkdir(parents=True, exist_ok=True)
    run_git_cmd(repo, "init")
    run_git_cmd(repo, "config", "user.name", "Test User")
    run_git_cmd(repo, "config", "user.email", "test@example.com")

    calc_file = repo / "src" / "calc.py"
    calc_file.parent.mkdir(parents=True, exist_ok=True)
    calc_file.write_text(
        "def add(a: int, b: int) -> int:\n    return a + b\n",
        encoding="utf-8",
    )
    run_git_cmd(repo, "add", ".")
    run_git_cmd(repo, "commit", "-m", "Initial commit")
    return repo
