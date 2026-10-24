"""Tests for CLI commands, exit codes, terminal output, and strict JSON output."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from deicide.cli import app
from deicide.core.errors import ExitCode
from tests.conftest import run_git_cmd

runner = CliRunner()


def test_cli_help_and_version() -> None:
    res_help = runner.invoke(app, ["--help"])
    assert res_help.exit_code == 0
    assert "DEICIDE" in res_help.stdout
    assert "review" in res_help.stdout

    res_ver = runner.invoke(app, ["--version"])
    assert res_ver.exit_code == 0
    assert "deicide 0.1.0" in res_ver.stdout

    # Verify subcommands do not advertise Gemini
    for cmd in [["review", "--help"], ["usage", "--help"], ["evaluate", "--help"]]:
        res = runner.invoke(app, cmd)
        assert res.exit_code == 0
        assert "gemini" not in res.stdout.lower()
        assert "nvidia" in res.stdout.lower()


def test_cli_review_fake_terminal_and_json(temp_git_repo: Path) -> None:
    calc_file = temp_git_repo / "src" / "calc.py"
    calc_file.write_text(
        "def add(a: int, b: int) -> float:\n    return (a + b) / 0\n",
        encoding="utf-8",
    )

    res_term = runner.invoke(
        app,
        ["review", "--provider", "fake", "--repo", str(temp_git_repo)],
    )
    assert res_term.exit_code == int(ExitCode.SUCCESS)
    assert "DEICIDE Code Review Report" in res_term.stdout
    assert "Division or modulo by zero" in res_term.stdout

    res_json = runner.invoke(
        app,
        ["review", "--provider", "fake", "--format", "json", "--repo", str(temp_git_repo)],
    )
    assert res_json.exit_code == int(ExitCode.SUCCESS)
    payload = json.loads(res_json.stdout)
    assert payload["platform"] == "DEICIDE"
    assert payload["status"] == "completed"
    assert payload["provider"]["provider_name"] == "fake"
    assert len(payload["findings"]) == 1
    assert payload["findings"][0]["file_path"] == "src/calc.py"
    assert payload["findings"][0]["line_range"]["start_line"] == 2


def test_cli_review_with_base_revision(temp_git_repo: Path) -> None:
    calc_file = temp_git_repo / "src" / "calc.py"
    calc_file.write_text(
        "def compute_ratio(items: list[int]) -> float:\n    return sum(items) / len(items)\n",
        encoding="utf-8",
    )
    run_git_cmd(temp_git_repo, "add", ".")
    run_git_cmd(temp_git_repo, "commit", "-m", "Add buggy compute_ratio")

    res = runner.invoke(
        app,
        [
            "review",
            "--base",
            "HEAD~1",
            "--provider",
            "fake",
            "--format",
            "json",
            "--repo",
            str(temp_git_repo),
        ],
    )
    assert res.exit_code == int(ExitCode.SUCCESS)
    data = json.loads(res.stdout)
    assert data["base_revision"] == "HEAD~1"
    assert len(data["findings"]) == 1
    assert data["findings"][0]["category"] == "boundary_condition"


def test_cli_error_exit_codes(temp_git_repo: Path, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.delenv("DEICIDE_API_KEY", raising=False)

    # Missing API credentials -> CONFIGURATION_ERROR (2)
    res_creds = runner.invoke(
        app,
        ["review", "--provider", "nvidia", "--repo", str(temp_git_repo)],
    )
    assert res_creds.exit_code == int(ExitCode.CONFIGURATION_ERROR)

    # Unsupported provider (gemini is no longer supported) -> CONFIGURATION_ERROR (2)
    res_gemini = runner.invoke(
        app,
        ["review", "--provider", "gemini", "--repo", str(temp_git_repo)],
    )
    assert res_gemini.exit_code == int(ExitCode.CONFIGURATION_ERROR)
    assert "Unsupported provider 'gemini'" in res_gemini.stderr

    # Unsupported provider -> CONFIGURATION_ERROR (2)
    res_prov = runner.invoke(
        app,
        ["review", "--provider", "invalid_provider", "--repo", str(temp_git_repo)],
    )
    assert res_prov.exit_code == int(ExitCode.CONFIGURATION_ERROR)

    # Invalid repository -> REPOSITORY_ERROR (3)
    non_repo = tmp_path / "plain_dir"
    non_repo.mkdir()
    res_repo = runner.invoke(
        app,
        ["review", "--provider", "fake", "--repo", str(non_repo)],
    )
    assert res_repo.exit_code == int(ExitCode.REPOSITORY_ERROR)

    # Invalid revision -> REPOSITORY_ERROR (3)
    res_rev = runner.invoke(
        app,
        ["review", "--base", "bad_rev_999", "--provider", "fake", "--repo", str(temp_git_repo)],
    )
    assert res_rev.exit_code == int(ExitCode.REPOSITORY_ERROR)

    # Exceeded token budget -> BUDGET_EXCEEDED (5)
    calc_file = temp_git_repo / "src" / "calc.py"
    calc_file.write_text("x = 1 / 0\n" * 40, encoding="utf-8")
    res_budget = runner.invoke(
        app,
        [
            "review",
            "--provider",
            "fake",
            "--max-tokens",
            "100",
            "--repo",
            str(temp_git_repo),
        ],
    )
    assert res_budget.exit_code == int(ExitCode.BUDGET_EXCEEDED)
