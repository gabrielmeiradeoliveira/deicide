"""Deterministic offline tests for DEICIDE Context Engine Phase 1: Python AST Analyzer."""

from __future__ import annotations

import os
from pathlib import Path

from deicide.context.changed_symbols import identify_changed_symbols
from deicide.context.models import (
    FileAnalysisStatus,
    ImportKind,
    OverlapClassification,
    SymbolKind,
)
from deicide.context.python_ast import PythonAstAnalyzer
from deicide.git.analyzer import GitAnalyzer
from deicide.git.models import DiffHunk, FileChangeStatus, FileDiff


# TEST 01: Top-level function extraction
def test_01_top_level_function_extraction() -> None:
    code = "def calculate_total(items: list[int]) -> int:\n    return sum(items)\n"
    analyzer = PythonAstAnalyzer()
    res = analyzer.analyze_source(code, "services/calc.py")
    assert res.status == FileAnalysisStatus.PARSED
    assert len(res.symbols) == 1
    sym = res.symbols[0]
    assert sym.name == "calculate_total"
    assert sym.qualified_name == "calculate_total"
    assert sym.kind == SymbolKind.FUNCTION
    assert sym.location.start_line == 1
    assert sym.location.end_line == 2
    assert sym.parent_id is None
    assert sym.scope_path == ()


# TEST 02: Async function extraction
def test_02_async_function_extraction() -> None:
    code = "async def fetch_user(uid: str) -> dict:\n    return {'uid': uid}\n"
    analyzer = PythonAstAnalyzer()
    res = analyzer.analyze_source(code, "handlers.py")
    assert res.status == FileAnalysisStatus.PARSED
    assert len(res.symbols) == 1
    sym = res.symbols[0]
    assert sym.name == "fetch_user"
    assert sym.kind == SymbolKind.ASYNC_FUNCTION


# TEST 03: Class and instance method extraction
def test_03_class_and_instance_method_extraction() -> None:
    code = "class OrderProcessor:\n    def process(self, order_id: str) -> bool:\n        return True\n"
    analyzer = PythonAstAnalyzer()
    res = analyzer.analyze_source(code, "orders.py")
    assert res.status == FileAnalysisStatus.PARSED
    assert len(res.symbols) == 2

    cls_sym = res.symbols[0]
    assert cls_sym.name == "OrderProcessor"
    assert cls_sym.qualified_name == "OrderProcessor"
    assert cls_sym.kind == SymbolKind.CLASS

    method_sym = res.symbols[1]
    assert method_sym.name == "process"
    assert method_sym.qualified_name == "OrderProcessor.process"
    assert method_sym.kind == SymbolKind.METHOD
    assert method_sym.parent_id == cls_sym.id
    assert method_sym.scope_path == ("OrderProcessor",)


# TEST 04: Static and class method extraction
def test_04_static_and_class_method_extraction() -> None:
    code = (
        "class ConfigFactory:\n"
        "    @classmethod\n"
        "    def from_env(cls) -> ConfigFactory:\n"
        "        return cls()\n"
        "\n"
        "    @staticmethod\n"
        "    def validate(name: str) -> bool:\n"
        "        return len(name) > 0\n"
    )
    analyzer = PythonAstAnalyzer()
    res = analyzer.analyze_source(code, "factory.py")
    assert res.status == FileAnalysisStatus.PARSED
    kinds = {s.name: s.kind for s in res.symbols}
    assert kinds["ConfigFactory"] == SymbolKind.CLASS
    assert kinds["from_env"] == SymbolKind.CLASS_METHOD
    assert kinds["validate"] == SymbolKind.STATIC_METHOD


# TEST 05: Nested function extraction
def test_05_nested_function_extraction() -> None:
    code = (
        "class PaymentService:\n"
        "    def process(self, amount: float) -> float:\n"
        "        def normalize(value: float) -> float:\n"
        "            return abs(value)\n"
        "        return normalize(amount)\n"
    )
    analyzer = PythonAstAnalyzer()
    res = analyzer.analyze_source(code, "payment.py")
    assert res.status == FileAnalysisStatus.PARSED
    assert len(res.symbols) == 3

    norm_sym = next(s for s in res.symbols if s.name == "normalize")
    assert norm_sym.qualified_name == "PaymentService.process.normalize"
    assert norm_sym.kind == SymbolKind.NESTED_FUNCTION
    assert norm_sym.scope_path == ("PaymentService", "process")


# TEST 06: Nested class extraction
def test_06_nested_class_extraction() -> None:
    code = "class Outer:\n    class Inner:\n        pass\n"
    analyzer = PythonAstAnalyzer()
    res = analyzer.analyze_source(code, "nested.py")
    assert res.status == FileAnalysisStatus.PARSED
    assert len(res.symbols) == 2
    inner = res.symbols[1]
    assert inner.name == "Inner"
    assert inner.qualified_name == "Outer.Inner"
    assert inner.kind == SymbolKind.NESTED_CLASS
    assert inner.parent_id == res.symbols[0].id


# TEST 07: Repeated symbol names in different scopes
def test_07_repeated_symbol_names_in_different_scopes() -> None:
    code = "def helper() -> int:\n    return 1\n\nclass Worker:\n    def helper(self) -> int:\n        return 2\n"
    analyzer = PythonAstAnalyzer()
    res = analyzer.analyze_source(code, "scopes.py")
    assert res.status == FileAnalysisStatus.PARSED
    helpers = [s for s in res.symbols if s.name == "helper"]
    assert len(helpers) == 2
    assert helpers[0].qualified_name == "helper"
    assert helpers[0].kind == SymbolKind.FUNCTION
    assert helpers[1].qualified_name == "Worker.helper"
    assert helpers[1].kind == SymbolKind.METHOD
    assert helpers[0].id != helpers[1].id


# TEST 08: Repeated definitions in the same scope
def test_08_repeated_definitions_in_the_same_scope() -> None:
    code = (
        "if True:\n"
        "    def fallback() -> str:\n"
        "        return 'first'\n"
        "else:\n"
        "    def fallback() -> str:\n"
        "        return 'second'\n"
    )
    analyzer = PythonAstAnalyzer()
    res = analyzer.analyze_source(code, "cond.py")
    assert res.status == FileAnalysisStatus.PARSED
    assert len(res.symbols) == 2
    assert res.symbols[0].id != res.symbols[1].id
    assert res.symbols[0].location.start_line != res.symbols[1].location.start_line


# TEST 09: Import alias extraction
def test_09_import_alias_extraction() -> None:
    code = "import os\nimport pathlib as pl\nfrom typing import Optional as Opt, List\n"
    analyzer = PythonAstAnalyzer()
    res = analyzer.analyze_source(code, "imports.py")
    assert res.status == FileAnalysisStatus.PARSED
    assert len(res.imports) == 4

    assert res.imports[0].imported_module == "os"
    assert res.imports[0].local_name == "os"
    assert res.imports[0].import_kind == ImportKind.MODULE

    assert res.imports[1].imported_module == "pathlib"
    assert res.imports[1].local_name == "pl"

    assert res.imports[2].imported_module == "typing"
    assert res.imports[2].imported_name == "Optional"
    assert res.imports[2].local_name == "Opt"
    assert res.imports[2].import_kind == ImportKind.FROM_MODULE

    assert res.imports[3].imported_name == "List"
    assert res.imports[3].local_name == "List"


# TEST 10: Relative import extraction
def test_10_relative_import_extraction() -> None:
    code = "from .utils import helper\nfrom ..core import Service\nfrom . import local_mod\n"
    analyzer = PythonAstAnalyzer()
    res = analyzer.analyze_source(code, "pkg/sub/module.py")
    assert res.status == FileAnalysisStatus.PARSED
    assert len(res.imports) == 3

    assert res.imports[0].imported_module == "utils"
    assert res.imports[0].relative_level == 1
    assert res.imports[0].imported_name == "helper"

    assert res.imports[1].imported_module == "core"
    assert res.imports[1].relative_level == 2
    assert res.imports[1].imported_name == "Service"

    assert res.imports[2].imported_module == ""
    assert res.imports[2].relative_level == 1
    assert res.imports[2].imported_name == "local_mod"


# TEST 11: Wildcard import extraction
def test_11_wildcard_import_extraction() -> None:
    code = "from math import *\n"
    analyzer = PythonAstAnalyzer()
    res = analyzer.analyze_source(code, "calc.py")
    assert res.status == FileAnalysisStatus.PARSED
    assert len(res.imports) == 1
    imp = res.imports[0]
    assert imp.is_wildcard is True
    assert imp.import_kind == ImportKind.WILDCARD
    assert imp.imported_module == "math"
    assert imp.local_name == "*"


# TEST 12: Simple function call extraction
def test_12_simple_function_call_extraction() -> None:
    code = "def compute() -> int:\n    return calculate_total([1, 2, 3])\n"
    analyzer = PythonAstAnalyzer()
    res = analyzer.analyze_source(code, "run.py")
    assert res.status == FileAnalysisStatus.PARSED
    assert len(res.calls) == 1
    call = res.calls[0]
    assert call.simple_name == "calculate_total"
    assert call.attribute_chain == ["calculate_total"]
    assert call.scope_path == ("compute",)
    assert call.is_dynamic is False


# TEST 13: Attribute call extraction
def test_13_attribute_call_extraction() -> None:
    code = "class Handler:\n    def execute(self) -> None:\n        self.validate()\n        pkg.service.process()\n"
    analyzer = PythonAstAnalyzer()
    res = analyzer.analyze_source(code, "handler.py")
    assert res.status == FileAnalysisStatus.PARSED
    assert len(res.calls) == 2

    assert res.calls[0].simple_name == "validate"
    assert res.calls[0].attribute_chain == ["self", "validate"]

    assert res.calls[1].simple_name == "process"
    assert res.calls[1].attribute_chain == ["pkg", "service", "process"]


# TEST 14: Dynamic call representation without invented targets
def test_14_dynamic_call_representation_without_invented_targets() -> None:
    code = "def run_dynamic(factory, registry, name):\n    factory()()\n    getattr(registry, name)()\n"
    analyzer = PythonAstAnalyzer()
    res = analyzer.analyze_source(code, "dynamic.py")
    assert res.status == FileAnalysisStatus.PARSED
    assert len(res.calls) >= 2
    dynamic_calls = [c for c in res.calls if c.is_dynamic]
    assert len(dynamic_calls) >= 1
    # Check that dynamic call does not invent a fake concrete target
    nested_call = next(c for c in res.calls if "factory()" in c.called_expression)
    assert nested_call.is_dynamic is True


# TEST 15: Correct 1-based source locations
def test_15_correct_1_based_source_locations() -> None:
    code = "# Line 1 comment\ndef add(a, b):\n    return a + b\n"
    analyzer = PythonAstAnalyzer()
    res = analyzer.analyze_source(code, "math_ops.py")
    assert res.status == FileAnalysisStatus.PARSED
    sym = res.symbols[0]
    assert sym.location.start_line == 2
    assert sym.location.end_line == 3
    assert sym.location.file_path == "math_ops.py"


# TEST 16: Multiline function source spans
def test_16_multiline_function_source_spans() -> None:
    code = (
        "@decorator_one\n"
        "@decorator_two(arg='value')\n"
        "def multiline_func(\n"
        "    first_arg: int,\n"
        "    second_arg: str,\n"
        ") -> bool:\n"
        "    result = True\n"
        "    return result\n"
    )
    analyzer = PythonAstAnalyzer()
    res = analyzer.analyze_source(code, "multiline.py")
    assert res.status == FileAnalysisStatus.PARSED
    sym = res.symbols[0]
    # Covered from earliest decorator line (1) to body end (8)
    assert sym.location.start_line == 1
    assert sym.location.end_line == 8
    assert len(sym.decorators) == 2


# TEST 17: Changed line inside a function
def test_17_changed_line_inside_a_function() -> None:
    code = "def calculate_average(values):\n    total = sum(values)\n    return total / len(values)\n"
    analysis = PythonAstAnalyzer().analyze_source(code, "stats.py")
    hunk = DiffHunk(
        old_start=2,
        old_count=1,
        new_start=2,
        new_count=1,
        added_line_numbers=frozenset({2}),
    )
    diff = FileDiff(
        old_path="stats.py",
        new_path="stats.py",
        status=FileChangeStatus.MODIFIED,
        hunks=[hunk],
    )
    changed = identify_changed_symbols(diff, analysis)
    assert len(changed.changed_symbols) == 1
    cs = changed.changed_symbols[0]
    assert cs.symbol_name == "calculate_average"
    assert cs.overlapping_lines == [2]
    assert cs.classification == OverlapClassification.BODY
    assert cs.is_innermost is True


# TEST 18: Changed line inside a method
def test_18_changed_line_inside_a_method() -> None:
    code = "class BankAccount:\n    def deposit(self, amount):\n        self.balance += amount\n"
    analysis = PythonAstAnalyzer().analyze_source(code, "account.py")
    hunk = DiffHunk(
        old_start=3,
        old_count=1,
        new_start=3,
        new_count=1,
        added_line_numbers=frozenset({3}),
    )
    diff = FileDiff(
        old_path="account.py",
        new_path="account.py",
        status=FileChangeStatus.MODIFIED,
        hunks=[hunk],
    )
    changed = identify_changed_symbols(diff, analysis)
    innermost = [s for s in changed.changed_symbols if s.is_innermost]
    assert len(innermost) == 1
    assert innermost[0].qualified_name == "BankAccount.deposit"
    assert innermost[0].kind == SymbolKind.METHOD


# TEST 19: Changed line inside a nested function
def test_19_changed_line_inside_a_nested_function() -> None:
    code = (
        "class PaymentService:\n"
        "    def process(self, amount):\n"
        "        def normalize(value):\n"
        "            return abs(value)\n"
        "        return normalize(amount)\n"
    )
    analysis = PythonAstAnalyzer().analyze_source(code, "payment.py")
    # Change line 4 (inside normalize)
    hunk = DiffHunk(
        old_start=4,
        old_count=1,
        new_start=4,
        new_count=1,
        added_line_numbers=frozenset({4}),
    )
    diff = FileDiff(
        old_path="payment.py",
        new_path="payment.py",
        status=FileChangeStatus.MODIFIED,
        hunks=[hunk],
    )
    changed = identify_changed_symbols(diff, analysis)
    # The innermost changed symbol must be 'normalize'
    assert changed.changed_symbols[0].qualified_name == "PaymentService.process.normalize"
    assert changed.changed_symbols[0].is_innermost is True
    # The outer symbols process and PaymentService are also reported with is_innermost=False
    outer = [s for s in changed.changed_symbols if not s.is_innermost]
    outer_names = {s.qualified_name for s in outer}
    assert "PaymentService.process" in outer_names
    assert "PaymentService" in outer_names


# TEST 20: Module-level changes
def test_20_module_level_changes() -> None:
    code = "DEFAULT_TIMEOUT = 30\n\ndef run():\n    pass\n"
    analysis = PythonAstAnalyzer().analyze_source(code, "settings.py")
    hunk = DiffHunk(
        old_start=1,
        old_count=1,
        new_start=1,
        new_count=1,
        added_line_numbers=frozenset({1}),
    )
    diff = FileDiff(
        old_path="settings.py",
        new_path="settings.py",
        status=FileChangeStatus.MODIFIED,
        hunks=[hunk],
    )
    changed = identify_changed_symbols(diff, analysis)
    assert len(changed.changed_symbols) == 0
    assert len(changed.module_level_changes) == 1
    assert changed.module_level_changes[0].changed_lines == [1]


# TEST 21: New Python file
def test_21_new_python_file() -> None:
    code = "def new_feature():\n    return 42\n"
    analysis = PythonAstAnalyzer().analyze_source(code, "feature.py")
    diff = FileDiff(
        old_path=None,
        new_path="feature.py",
        status=FileChangeStatus.ADDED,
        hunks=[DiffHunk(old_start=0, old_count=0, new_start=1, new_count=2, added_line_numbers=frozenset({1, 2}))],
    )
    changed = identify_changed_symbols(diff, analysis)
    assert len(changed.changed_symbols) == 1
    assert changed.changed_symbols[0].symbol_name == "new_feature"
    assert changed.changed_symbols[0].classification == OverlapClassification.EXACT


# TEST 22: Deleted Python file
def test_22_deleted_python_file() -> None:
    analysis = FileAnalysisStatus.FILE_NOT_FOUND
    diff = FileDiff(
        old_path="legacy.py",
        new_path="legacy.py",
        status=FileChangeStatus.DELETED,
        hunks=[
            DiffHunk(
                old_start=1,
                old_count=5,
                new_start=0,
                new_count=0,
                lines=["-def old():", "-    pass"],
            )
        ],
    )
    from deicide.context.models import FileAnalysis

    dummy_analysis = FileAnalysis(
        source_file="legacy.py",
        status=analysis,
    )
    changed = identify_changed_symbols(diff, dummy_analysis)
    assert len(changed.changed_symbols) == 0
    assert changed.deleted_lines_count == 2
    assert "File was deleted" in (changed.deleted_lines_note or "")


# TEST 23: Deleted-line mapping limitations
def test_23_deleted_line_mapping_limitations() -> None:
    code = "def intact_func():\n    return 1\n"
    analysis = PythonAstAnalyzer().analyze_source(code, "app.py")
    # Diff only has deleted lines from the old version
    hunk = DiffHunk(
        old_start=5,
        old_count=2,
        new_start=1,
        new_count=0,
        lines=["-def removed_func():", "-    return 0"],
    )
    diff = FileDiff(
        old_path="app.py",
        new_path="app.py",
        status=FileChangeStatus.MODIFIED,
        hunks=[hunk],
    )
    changed = identify_changed_symbols(diff, analysis)
    assert len(changed.changed_symbols) == 0
    assert changed.deleted_lines_count == 2
    assert "Deleted lines cannot be mapped" in (changed.deleted_lines_note or "")


# TEST 24: Syntax error isolation
def test_24_syntax_error_isolation() -> None:
    bad_code = "def broken(:\n    pass\n"
    analyzer = PythonAstAnalyzer()
    res = analyzer.analyze_source(bad_code, "broken.py")
    assert res.status == FileAnalysisStatus.SYNTAX_ERROR
    assert "SyntaxError" in (res.error_message or "")
    assert len(res.symbols) == 0


# TEST 25: Empty file
def test_25_empty_file() -> None:
    analyzer = PythonAstAnalyzer()
    res = analyzer.analyze_source("", "empty.py")
    assert res.status == FileAnalysisStatus.EMPTY_FILE
    assert len(res.symbols) == 0
    assert res.total_lines == 0


# TEST 26: Missing file
def test_26_missing_file(tmp_path: Path) -> None:
    analyzer = PythonAstAnalyzer()
    res = analyzer.analyze_file(tmp_path, "does_not_exist.py")
    assert res.status == FileAnalysisStatus.FILE_NOT_FOUND
    assert "File not found" in (res.error_message or "")


# TEST 27: Symlink escaping repository root
def test_27_symlink_escaping_repository_root(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    outside = tmp_path / "outside.py"
    outside.write_text("def secret(): pass\n", encoding="utf-8")

    link = repo / "escaped_link.py"
    try:
        os.symlink(outside, link)
    except (OSError, NotImplementedError):
        # On Windows without developer mode/admin rights, symlinks might not be creatable
        return

    analyzer = PythonAstAnalyzer()
    res = analyzer.analyze_file(repo, "escaped_link.py")
    assert res.status == FileAnalysisStatus.SECURITY_VIOLATION
    assert "resolves outside repository root" in (res.error_message or "")


# TEST 28: Stable identifiers across repeated runs
def test_28_stable_identifiers_across_repeated_runs() -> None:
    code = "class Service:\n    def execute(self):\n        pass\n"
    analyzer = PythonAstAnalyzer()
    run1 = analyzer.analyze_source(code, "stable.py")
    run2 = analyzer.analyze_source(code, "stable.py")
    assert [s.id for s in run1.symbols] == [s.id for s in run2.symbols]
    assert run1.symbols[1].id == "stable.py::Service.execute#method@2:3"


# TEST 29: Deterministic ordering
def test_29_deterministic_ordering() -> None:
    code = "import z\nimport a\n\ndef beta():\n    gamma()\n    alpha()\n\ndef alpha():\n    pass\n"
    analyzer = PythonAstAnalyzer()
    res = analyzer.analyze_source(code, "order.py")
    # Symbols sorted by line
    assert [s.name for s in res.symbols] == ["beta", "alpha"]
    # Imports sorted by line
    assert [i.imported_module for i in res.imports] == ["z", "a"]
    # Calls sorted by line
    assert [c.called_expression for c in res.calls] == ["gamma", "alpha"]


# TEST 30: Existing GitAnalyzer compatibility
def test_30_existing_git_analyzer_compatibility(temp_git_repo: Path) -> None:
    # Modify calc.py inside the fixture git repo
    calc_file = temp_git_repo / "src" / "calc.py"
    calc_file.write_text(
        "def add(a: int, b: int) -> int:\n    result = a + b\n    return result\n",
        encoding="utf-8",
    )
    git_analyzer = GitAnalyzer(temp_git_repo)
    diff_bundle = git_analyzer.extract_diff()
    assert not diff_bundle.is_empty
    file_diff = diff_bundle.get_file_diff("src/calc.py")
    assert file_diff is not None

    ast_analyzer = PythonAstAnalyzer()
    analysis = ast_analyzer.analyze_file(temp_git_repo, "src/calc.py")
    assert analysis.status == FileAnalysisStatus.PARSED

    changed = identify_changed_symbols(file_diff, analysis)
    assert len(changed.changed_symbols) == 1
    assert changed.changed_symbols[0].symbol_name == "add"
    assert changed.changed_symbols[0].is_innermost is True
