"""Tests for deletion-only diff detection in the DEICIDE Context Engine."""

from __future__ import annotations

from pathlib import Path

from deicide.context.changed_symbols import identify_changed_symbols
from deicide.context.models import FileAnalysisStatus, OverlapClassification
from deicide.context.python_ast import PythonAstAnalyzer
from deicide.git.analyzer import parse_unified_diff
from deicide.git.models import FileChangeStatus


def test_deletion_only_symbol_detection() -> None:
    """Test that deleting lines inside a function identifies the function as a changed symbol."""
    code = "def calculate_average(values):\n    return sum(values) / len(values)\n"
    analysis = PythonAstAnalyzer().analyze_source(code, "calculator.py")
    assert analysis.status == FileAnalysisStatus.PARSED

    raw_diff = (
        "diff --git a/calculator.py b/calculator.py\n"
        "--- a/calculator.py\n"
        "+++ b/calculator.py\n"
        "@@ -1,4 +1,2 @@\n"
        " def calculate_average(values):\n"
        "-    if not values:\n"
        "-        return 0\n"
        "     return sum(values) / len(values)\n"
    )
    diffs = parse_unified_diff(raw_diff)
    assert len(diffs) == 1
    diff = diffs[0]
    assert diff.status == FileChangeStatus.MODIFIED
    assert len(diff.added_lines) == 0

    res = identify_changed_symbols(diff, analysis)
    assert len(res.changed_symbols) == 1
    cs = res.changed_symbols[0]
    assert cs.symbol_name == "calculate_average"
    assert cs.classification == OverlapClassification.BODY
    assert cs.is_innermost is True
    assert 2 in cs.overlapping_lines


def test_deletion_only_cross_file_pipeline_integration(tmp_path: Path) -> None:
    """Test full cross-file pipeline integration when diff is deletion-only.

    Verifies:
    1. calculator.py has guard lines deleted.
    2. reports.py is unchanged caller.
    3. Context Engine identifies calculate_average as changed symbol.
    4. Dependency graph resolves reports.py as direct caller.
    5. Supplementary context includes snippet from reports.py.
    6. CodeReviewModule runs with FakeProvider and validates findings.
    """
    from deicide.context.pipeline import prepare_review_context
    from deicide.core.config import DeicideConfig
    from deicide.core.models import ExecutionBudget
    from deicide.core.orchestrator import AnalysisOrchestrator
    from deicide.core.registry import ModuleRegistry
    from deicide.git.analyzer import GitAnalyzer
    from deicide.modules.code_review import CodeReviewModule
    from deicide.providers.fake import FakeProvider
    from tests.conftest import run_git_cmd

    repo = tmp_path / "repo_deletion_pipeline"
    repo.mkdir(parents=True, exist_ok=True)
    run_git_cmd(repo, "init")
    run_git_cmd(repo, "config", "user.name", "Deicide Tester")
    run_git_cmd(repo, "config", "user.email", "tester@deicide.local")

    calc_file = repo / "calculator.py"
    calc_file.write_text(
        "def calculate_average(values):\n    if not values:\n        return 0\n    return sum(values) / len(values)\n",
        encoding="utf-8",
    )

    reports_file = repo / "reports.py"
    reports_file.write_text(
        "from calculator import calculate_average\n\ndef generate_report():\n    return calculate_average([])\n",
        encoding="utf-8",
    )

    run_git_cmd(repo, "add", ".")
    run_git_cmd(repo, "commit", "-m", "Baseline commit")

    # Perform deletion-only edit in calculator.py
    calc_file.write_text(
        "def calculate_average(values):\n    return sum(values) / len(values)\n",
        encoding="utf-8",
    )

    diff_bundle = GitAnalyzer(repo).extract_diff()
    assert len(diff_bundle.reviewable_files) == 1
    calc_diff = diff_bundle.reviewable_files[0]
    assert calc_diff.new_path == "calculator.py"
    assert len(calc_diff.added_lines) == 0

    budget = ExecutionBudget(max_total_tokens=10000)
    outcome = prepare_review_context(
        repo_root=repo,
        diff_bundle=diff_bundle,
        budget=budget,
        context_enabled=True,
    )

    # Verify context engine didn't fail with no_changed_symbols
    assert outcome.summary.status != "no_changed_symbols"
    assert outcome.result is not None
    assert outcome.summary.indexed_files >= 2
    assert outcome.summary.selected_snippets >= 1
    assert outcome.summary.related_files >= 1

    # Verify snippets contain reports.py and caller
    found_caller_snippet = any(
        s.file_path == "reports.py" and "generate_report" in (s.symbol_name or "")
        for s in outcome.result.selected_snippets
    )
    assert found_caller_snippet, "Expected reports.py caller snippet in selected context"

    # Now verify end-to-end code review orchestration
    config = DeicideConfig(
        repo_path=repo,
        provider="fake",
        context_enabled=True,
    )
    provider = FakeProvider(model_name="fake-deterministic-v1")
    registry = ModuleRegistry()
    registry.register(CodeReviewModule())

    orchestrator = AnalysisOrchestrator(
        config=config,
        provider=provider,
        registry=registry,
    )

    report = orchestrator.run()
    # Confirm finding on calculator.py (ZeroDivisionError) is validated
    assert len(report.findings) >= 1
    finding = report.findings[0]
    assert finding.file_path == "calculator.py"
    assert "ZeroDivisionError" in finding.title
    # Confirm no findings leaked onto unchanged reports.py
    assert all(f.file_path != "reports.py" for f in report.findings)


def test_deletion_at_end_of_function() -> None:
    """Test that deleting trailing lines of a function maps to that function."""
    code = "def compute(a, b):\n    res = a + b\n\ndef other():\n    return 1\n"
    analysis = PythonAstAnalyzer().analyze_source(code, "math_ops.py")
    assert analysis.status == FileAnalysisStatus.PARSED

    raw_diff = (
        "diff --git a/math_ops.py b/math_ops.py\n"
        "--- a/math_ops.py\n"
        "+++ b/math_ops.py\n"
        "@@ -1,4 +1,3 @@\n"
        " def compute(a, b):\n"
        "     res = a + b\n"
        "-    return res\n"
        " \n"
        " def other():\n"
    )
    diffs = parse_unified_diff(raw_diff)
    assert len(diffs) == 1
    diff = diffs[0]

    res = identify_changed_symbols(diff, analysis)
    assert len(res.changed_symbols) == 1
    cs = res.changed_symbols[0]
    assert cs.symbol_name == "compute"
    assert cs.classification == OverlapClassification.BODY
    assert cs.is_innermost is True
    assert 2 in cs.overlapping_lines


def test_deletion_of_decorator() -> None:
    """Test that deleting a decorator on a function maps to SIGNATURE classification."""
    code = "def greet(name):\n    return f'Hello, {name}'\n"
    analysis = PythonAstAnalyzer().analyze_source(code, "greet.py")
    assert analysis.status == FileAnalysisStatus.PARSED

    raw_diff = (
        "diff --git a/greet.py b/greet.py\n"
        "--- a/greet.py\n"
        "+++ b/greet.py\n"
        "@@ -1,3 +1,2 @@\n"
        "-@lru_cache\n"
        " def greet(name):\n"
        "     return f'Hello, {name}'\n"
    )
    diffs = parse_unified_diff(raw_diff)
    assert len(diffs) == 1
    diff = diffs[0]

    res = identify_changed_symbols(diff, analysis)
    assert len(res.changed_symbols) == 1
    cs = res.changed_symbols[0]
    assert cs.symbol_name == "greet"
    assert cs.classification == OverlapClassification.SIGNATURE
    assert 1 in cs.overlapping_lines


def test_deletion_of_entire_symbol_not_mapped_to_adjacent() -> None:
    """Test that deleting an entire symbol does NOT map to adjacent intact symbol."""
    code = "def intact():\n    return 42\n"
    analysis = PythonAstAnalyzer().analyze_source(code, "app.py")
    assert analysis.status == FileAnalysisStatus.PARSED

    raw_diff = (
        "diff --git a/app.py b/app.py\n"
        "--- a/app.py\n"
        "+++ b/app.py\n"
        "@@ -1,4 +1,2 @@\n"
        "-def removed():\n"
        "-    return 0\n"
        "-\n"
        " def intact():\n"
        "     return 42\n"
    )
    diffs = parse_unified_diff(raw_diff)
    assert len(diffs) == 1
    diff = diffs[0]

    res = identify_changed_symbols(diff, analysis)
    # The intact function was not changed; removed() is gone
    assert len(res.changed_symbols) == 0
    assert res.deleted_lines_count == 3


def test_deletion_inside_class_method_innermost() -> None:
    """Test that deleting lines inside a class method marks method as innermost."""
    code = "class Service:\n    def run(self):\n        return 1\n"
    analysis = PythonAstAnalyzer().analyze_source(code, "svc.py")
    assert analysis.status == FileAnalysisStatus.PARSED

    raw_diff = (
        "diff --git a/svc.py b/svc.py\n"
        "--- a/svc.py\n"
        "+++ b/svc.py\n"
        "@@ -1,4 +1,3 @@\n"
        " class Service:\n"
        "     def run(self):\n"
        "-        log_start()\n"
        "         return 1\n"
    )
    diffs = parse_unified_diff(raw_diff)
    assert len(diffs) == 1
    diff = diffs[0]

    res = identify_changed_symbols(diff, analysis)
    assert len(res.changed_symbols) == 2
    # First is the innermost (run), second is the parent class (Service)
    assert res.changed_symbols[0].symbol_name == "run"
    assert res.changed_symbols[0].is_innermost is True
    assert res.changed_symbols[1].symbol_name == "Service"
    assert res.changed_symbols[1].is_innermost is False
