"""Deterministic offline tests for DEICIDE Context Engine Phase 2: Cross-File Dependency Graph."""

from __future__ import annotations

from pathlib import Path

from deicide.context.dependency_graph import (
    build_dependency_graph,
)
from deicide.context.import_resolver import ImportResolver, resolve_relative_module
from deicide.context.project_index import (
    IndexLimits,
    build_project_index,
    file_path_to_module_name,
)


def test_01_module_name_resolution_layouts() -> None:
    # Root flat layout
    assert file_path_to_module_name("calculator.py") == "calculator"
    assert file_path_to_module_name("billing.py") == "billing"

    # src layout
    assert file_path_to_module_name("src/mypackage/calculator.py") == "mypackage.calculator"
    assert file_path_to_module_name("src/mypackage/__init__.py") == "mypackage"

    # Nested subpackage
    assert file_path_to_module_name("package/services/payments.py") == "package.services.payments"
    assert file_path_to_module_name("package/services/__init__.py") == "package.services"


def test_02_bounded_file_discovery_exclusions_and_limits(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()

    # Valid Python files
    (repo / "calc.py").write_text("def add(a, b): return a + b\n", encoding="utf-8")
    src_dir = repo / "src" / "pkg"
    src_dir.mkdir(parents=True)
    (src_dir / "__init__.py").write_text("", encoding="utf-8")
    (src_dir / "service.py").write_text("class Service: pass\n", encoding="utf-8")

    # Excluded directories that MUST be ignored
    (repo / ".git").mkdir()
    (repo / ".git" / "hook.py").write_text("malicious()", encoding="utf-8")

    (repo / ".venv").mkdir()
    (repo / ".venv" / "lib.py").write_text("venv_code()", encoding="utf-8")

    (repo / "__pycache__").mkdir()
    (repo / "__pycache__" / "cached.py").write_text("cache()", encoding="utf-8")

    (repo / "node_modules").mkdir()
    (repo / "node_modules" / "script.py").write_text("node()", encoding="utf-8")

    index = build_project_index(repo)
    indexed_files = sorted(index.file_analyses.keys())
    assert "calc.py" in indexed_files
    assert "src/pkg/service.py" in indexed_files
    assert "src/pkg/__init__.py" in indexed_files

    # Excluded files must never be indexed
    assert not any(".git" in f for f in indexed_files)
    assert not any(".venv" in f for f in indexed_files)
    assert not any("__pycache__" in f for f in indexed_files)
    assert not any("node_modules" in f for f in indexed_files)


def test_03_discovery_enforces_max_file_count_and_size_limits(tmp_path: Path) -> None:
    repo = tmp_path / "repo_limits"
    repo.mkdir()

    for i in range(5):
        (repo / f"f{i}.py").write_text(f"x_{i} = {i}\n", encoding="utf-8")

    # Limit to max 3 files
    limits = IndexLimits(max_files=3, max_file_size_bytes=1000, max_total_bytes=10000)
    index = build_project_index(repo, limits=limits)
    assert len(index.file_analyses) == 3
    assert index.limits_reached is True
    assert any("Maximum file count limit reached" in d for d in index.diagnostics)


def test_04_discovery_skips_oversized_files(tmp_path: Path) -> None:
    repo = tmp_path / "repo_oversized"
    repo.mkdir()

    (repo / "small.py").write_text("x = 1\n", encoding="utf-8")
    (repo / "giant.py").write_text("#" * 5000 + "\n", encoding="utf-8")

    limits = IndexLimits(max_files=10, max_file_size_bytes=200, max_total_bytes=10000)
    index = build_project_index(repo, limits=limits)
    assert "small.py" in index.file_analyses
    assert "giant.py" not in index.file_analyses
    assert any("exceeds max file size" in d for d in index.diagnostics)


def test_05_relative_import_resolution_levels() -> None:
    # Level 1 from module in subpackage
    res1 = resolve_relative_module(
        current_module="pkg.services.payment",
        is_init_file=False,
        relative_level=1,
        imported_module="utils",
    )
    assert res1 == "pkg.services.utils"

    # Level 2 from module in subpackage
    res2 = resolve_relative_module(
        current_module="pkg.services.payment",
        is_init_file=False,
        relative_level=2,
        imported_module="core",
    )
    assert res2 == "pkg.core"

    # Level 1 from __init__.py
    res3 = resolve_relative_module(
        current_module="pkg.services",
        is_init_file=True,
        relative_level=1,
        imported_module="payment",
    )
    assert res3 == "pkg.services.payment"


def test_06_import_resolver_submodule_and_symbol_resolution(tmp_path: Path) -> None:
    repo = tmp_path / "import_repo"
    repo.mkdir()

    (repo / "calc.py").write_text(
        "def calculate_average(items):\n    return sum(items) / len(items)\n",
        encoding="utf-8",
    )
    (repo / "billing.py").write_text(
        "from calc import calculate_average as avg\nimport os\nfrom math import *\n",
        encoding="utf-8",
    )

    index = build_project_index(repo)
    resolver = ImportResolver(index)
    import_map = resolver.resolve_file_imports("billing.py")

    # Local symbol import resolved
    avg_binding = import_map.get_binding("avg")
    assert avg_binding is not None
    assert avg_binding.target_module == "calc"
    assert avg_binding.target_symbol_name == "calculate_average"
    assert avg_binding.target_symbol is not None
    assert avg_binding.target_symbol.name == "calculate_average"
    assert avg_binding.is_external is False

    # External import identified
    os_binding = import_map.get_binding("os")
    assert os_binding is not None
    assert os_binding.target_module == "os"
    assert os_binding.is_external is True

    # Wildcard tracked
    assert "math" in import_map.wildcard_modules


def test_07_cross_file_callers_and_callees(tmp_path: Path) -> None:
    repo = tmp_path / "call_repo"
    repo.mkdir()

    (repo / "service.py").write_text(
        "def compute_interest(principal: float) -> float:\n    return principal * 0.05\n",
        encoding="utf-8",
    )
    (repo / "client.py").write_text(
        "from service import compute_interest\n"
        "\n"
        "def run_workflow():\n"
        "    val = compute_interest(100.0)\n"
        "    return val\n",
        encoding="utf-8",
    )

    index = build_project_index(repo)
    graph = build_dependency_graph(index)

    # Find the compute_interest function symbol node
    compute_sym = next(s for s in index.symbols_by_id.values() if s.name == "compute_interest")
    run_sym = next(s for s in index.symbols_by_id.values() if s.name == "run_workflow")

    # Callers of compute_interest must include run_workflow
    callers = graph.get_callers(compute_sym.id)
    assert len(callers) == 1
    assert callers[0].id == run_sym.id

    # Callees of run_workflow must include compute_interest
    callees = graph.get_callees(run_sym.id)
    assert len(callees) == 1
    assert callees[0].id == compute_sym.id


def test_08_module_alias_calls_resolved(tmp_path: Path) -> None:
    repo = tmp_path / "alias_repo"
    repo.mkdir()

    (repo / "math_utils.py").write_text(
        "def square(x: int) -> int:\n    return x * x\n",
        encoding="utf-8",
    )
    (repo / "main.py").write_text(
        "import math_utils as mu\n\ndef run():\n    return mu.square(5)\n",
        encoding="utf-8",
    )

    index = build_project_index(repo)
    graph = build_dependency_graph(index)

    square_sym = next(s for s in index.symbols_by_id.values() if s.name == "square")
    run_sym = next(s for s in index.symbols_by_id.values() if s.name == "run")

    callers = graph.get_callers(square_sym.id)
    assert len(callers) == 1
    assert callers[0].id == run_sym.id


def test_09_self_method_call_resolution(tmp_path: Path) -> None:
    repo = tmp_path / "self_repo"
    repo.mkdir()

    (repo / "processor.py").write_text(
        "class OrderProcessor:\n"
        "    def validate(self) -> bool:\n"
        "        return True\n"
        "\n"
        "    def process(self) -> None:\n"
        "        self.validate()\n",
        encoding="utf-8",
    )

    index = build_project_index(repo)
    graph = build_dependency_graph(index)

    validate_sym = next(s for s in index.symbols_by_id.values() if s.qualified_name == "OrderProcessor.validate")
    process_sym = next(s for s in index.symbols_by_id.values() if s.qualified_name == "OrderProcessor.process")

    callers = graph.get_callers(validate_sym.id)
    assert len(callers) == 1
    assert callers[0].id == process_sym.id


def test_10_parameter_shadowing_prevents_false_call_edge(tmp_path: Path) -> None:
    repo = tmp_path / "shadow_repo"
    repo.mkdir()

    (repo / "payments.py").write_text(
        "def validate():\n    return True\n",
        encoding="utf-8",
    )
    (repo / "client.py").write_text(
        "from payments import validate\n"
        "\n"
        "def process(validate):\n"
        "    # Parameter 'validate' shadows the imported payments.validate\n"
        "    validate()\n",
        encoding="utf-8",
    )

    index = build_project_index(repo)
    graph = build_dependency_graph(index)

    payments_validate = next(s for s in index.symbols_by_id.values() if s.name == "validate")
    callers = graph.get_callers(payments_validate.id)
    # Must NOT contain process() because parameter 'validate' shadowed the import!
    assert len(callers) == 0


def test_11_dynamic_and_unknown_object_calls_unresolved(tmp_path: Path) -> None:
    repo = tmp_path / "dynamic_repo"
    repo.mkdir()

    (repo / "app.py").write_text(
        "def run(factory, service):\n    factory()()\n    service.process()\n",
        encoding="utf-8",
    )

    index = build_project_index(repo)
    graph = build_dependency_graph(index)

    run_sym = next(s for s in index.symbols_by_id.values() if s.name == "run")
    callees = graph.get_callees(run_sym.id)
    # Must not invent fake concrete targets
    assert len(callees) == 0


def test_12_circular_dependency_detection(tmp_path: Path) -> None:
    repo = tmp_path / "cycle_repo"
    repo.mkdir()

    (repo / "mod_a.py").write_text("import mod_b\n", encoding="utf-8")
    (repo / "mod_b.py").write_text("import mod_a\n", encoding="utf-8")

    index = build_project_index(repo)
    graph = build_dependency_graph(index)

    cycles = graph.find_cycles()
    assert len(cycles) >= 1
    cycle_nodes = set(cycles[0])
    assert "module::mod_a" in cycle_nodes
    assert "module::mod_b" in cycle_nodes


def test_13_impacted_symbols_traversal(tmp_path: Path) -> None:
    repo = tmp_path / "impact_repo"
    repo.mkdir()

    (repo / "core.py").write_text(
        "def leaf(): return 1\n",
        encoding="utf-8",
    )
    (repo / "mid.py").write_text(
        "from core import leaf\ndef middle(): return leaf()\n",
        encoding="utf-8",
    )
    (repo / "top.py").write_text(
        "from mid import middle\ndef root(): return middle()\n",
        encoding="utf-8",
    )

    index = build_project_index(repo)
    graph = build_dependency_graph(index)

    leaf_sym = next(s for s in index.symbols_by_id.values() if s.name == "leaf")
    mid_sym = next(s for s in index.symbols_by_id.values() if s.name == "middle")
    top_sym = next(s for s in index.symbols_by_id.values() if s.name == "root")

    # If 'leaf' is modified:
    # Depth 1: middle
    impacted_1 = graph.get_impacted_symbols({leaf_sym.id}, max_depth=1)
    assert mid_sym.id in impacted_1
    assert top_sym.id not in impacted_1

    # Depth 2: middle and root
    impacted_2 = graph.get_impacted_symbols({leaf_sym.id}, max_depth=2)
    assert mid_sym.id in impacted_2
    assert top_sym.id in impacted_2


def test_14_deterministic_serialization_and_stability(tmp_path: Path) -> None:
    repo = tmp_path / "stability_repo"
    repo.mkdir()

    (repo / "a.py").write_text("def fa(): pass\n", encoding="utf-8")
    (repo / "b.py").write_text("from a import fa\ndef fb(): fa()\n", encoding="utf-8")

    index1 = build_project_index(repo)
    graph1 = build_dependency_graph(index1)

    index2 = build_project_index(repo)
    graph2 = build_dependency_graph(index2)

    assert list(graph1.nodes.keys()) == list(graph2.nodes.keys())
    assert [(e.source_id, e.target_id, e.kind) for e in graph1.edges] == [
        (e.source_id, e.target_id, e.kind) for e in graph2.edges
    ]


def test_15_same_file_function_call(tmp_path: Path) -> None:
    repo = tmp_path / "same_file_repo"
    repo.mkdir()
    (repo / "math_ops.py").write_text(
        "def helper():\n    return 42\n\ndef compute():\n    return helper()\n",
        encoding="utf-8",
    )
    index = build_project_index(repo)
    graph = build_dependency_graph(index)

    helper_sym = next(s for s in index.symbols_by_id.values() if s.name == "helper")
    compute_sym = next(s for s in index.symbols_by_id.values() if s.name == "compute")

    callers = graph.get_callers(helper_sym.id)
    assert len(callers) == 1
    assert callers[0].id == compute_sym.id


def test_16_cls_method_call_resolution(tmp_path: Path) -> None:
    repo = tmp_path / "cls_repo"
    repo.mkdir()
    (repo / "factory.py").write_text(
        "class WidgetFactory:\n"
        "    @classmethod\n"
        "    def create_default(cls):\n"
        "        return cls.build(0)\n"
        "\n"
        "    @classmethod\n"
        "    def build(cls, count):\n"
        "        return count\n",
        encoding="utf-8",
    )
    index = build_project_index(repo)
    graph = build_dependency_graph(index)

    build_sym = next(s for s in index.symbols_by_id.values() if s.qualified_name == "WidgetFactory.build")
    create_sym = next(s for s in index.symbols_by_id.values() if s.qualified_name == "WidgetFactory.create_default")

    callers = graph.get_callers(build_sym.id)
    assert len(callers) == 1
    assert callers[0].id == create_sym.id


def test_17_local_assignment_shadowing_prevents_false_edge(tmp_path: Path) -> None:
    repo = tmp_path / "assign_shadow_repo"
    repo.mkdir()
    (repo / "utils.py").write_text("def run(): pass\n", encoding="utf-8")
    (repo / "main.py").write_text(
        "from utils import run\n\ndef execute():\n    run = lambda: None\n    run()\n",
        encoding="utf-8",
    )
    index = build_project_index(repo)
    graph = build_dependency_graph(index)

    utils_run = next(s for s in index.symbols_by_id.values() if s.name == "run")
    callers = graph.get_callers(utils_run.id)
    assert len(callers) == 0


def test_18_duplicate_edge_prevention_and_distinct_call_site_preservation(tmp_path: Path) -> None:
    repo = tmp_path / "call_sites_repo"
    repo.mkdir()
    (repo / "lib.py").write_text("def ping(): pass\n", encoding="utf-8")
    (repo / "caller.py").write_text(
        "from lib import ping\n\ndef work():\n    ping()\n    x = 1\n    ping()\n",
        encoding="utf-8",
    )
    index = build_project_index(repo)
    graph = build_dependency_graph(index)

    ping_sym = next(s for s in index.symbols_by_id.values() if s.name == "ping")
    evidence = graph.get_callers_with_evidence(ping_sym.id)

    # Both call sites (lines 4 and 6) are preserved with distinct line numbers
    assert len(evidence) == 2
    assert evidence[0].call_site.start_line == 4
    assert evidence[1].call_site.start_line == 6

    # Adding an exact identical edge twice does not duplicate it
    initial_edge_count = len(graph.edges)
    graph.add_edge(graph.edges[0])
    assert len(graph.edges) == initial_edge_count


def test_19_changed_symbol_impact_and_unchanged_caller_discovery(tmp_path: Path) -> None:
    repo = tmp_path / "impact_demo_repo"
    repo.mkdir()
    (repo / "calculator.py").write_text(
        "def calculate_average(values):\n    return sum(values) / len(values)\n",
        encoding="utf-8",
    )
    (repo / "billing.py").write_text(
        "from calculator import calculate_average\n\ndef generate_report():\n    return calculate_average([])\n",
        encoding="utf-8",
    )
    index = build_project_index(repo)
    graph = build_dependency_graph(index)

    calc_sym = next(s for s in index.symbols_by_id.values() if s.name == "calculate_average")

    from deicide.context.models import ChangedSymbol, OverlapClassification

    changed_sym = ChangedSymbol(
        symbol_id=calc_sym.id,
        symbol_name=calc_sym.name,
        qualified_name=calc_sym.qualified_name,
        kind=calc_sym.kind,
        location=calc_sym.location,
        overlapping_lines=[2],
        classification=OverlapClassification.BODY,
        is_innermost=True,
    )

    impact = graph.get_impact_for_changed_symbol(changed_sym)
    assert impact.changed_symbol.symbol_name == "calculate_average"
    assert len(impact.direct_callers) == 1
    assert impact.direct_callers[0].caller_node.name == "generate_report"
    assert impact.direct_callers[0].caller_node.file_path == "billing.py"
    assert impact.direct_callers[0].call_site.start_line == 4
    assert impact.direct_callers[0].confidence == 1.0
    assert "module::billing" in impact.imported_by_modules


def test_20_missing_imports_and_unrelated_syntax_errors(tmp_path: Path) -> None:
    repo = tmp_path / "errors_repo"
    repo.mkdir()
    (repo / "good.py").write_text(
        "from nonexistent_external_package import helper\ndef ok(): return 1\n",
        encoding="utf-8",
    )
    (repo / "broken.py").write_text(
        "def broken(:\n    pass\n",
        encoding="utf-8",
    )
    index = build_project_index(repo)
    graph = build_dependency_graph(index)

    assert "good.py" in index.file_analyses
    assert "broken.py" in index.file_analyses
    assert "module::good" in graph.nodes
    ok_sym = next(s for s in index.symbols_by_id.values() if s.name == "ok")
    assert ok_sym.id in graph.nodes


def test_21_max_aggregate_source_bytes_limit(tmp_path: Path) -> None:
    repo = tmp_path / "byte_limit_repo"
    repo.mkdir()
    for i in range(10):
        (repo / f"f{i}.py").write_text("# padding\n" + "x = 1\n" * 10, encoding="utf-8")

    limits = IndexLimits(max_files=100, max_file_size_bytes=1000, max_total_bytes=100)
    index = build_project_index(repo, limits=limits)
    assert index.limits_reached is True
    assert any("Aggregate source byte limit reached" in d for d in index.diagnostics)


def test_22_bounded_traversal_visited_limit(tmp_path: Path) -> None:
    repo = tmp_path / "traversal_repo"
    repo.mkdir()
    (repo / "chain.py").write_text(
        "def f1(): f2()\ndef f2(): f3()\ndef f3(): f4()\ndef f4(): f5()\ndef f5(): pass\n",
        encoding="utf-8",
    )
    index = build_project_index(repo)
    graph = build_dependency_graph(index)

    f1_sym = next(s for s in index.symbols_by_id.values() if s.name == "f1")

    # Limit to max 2 visited nodes
    deps = graph.get_dependencies(f1_sym.id, max_depth=5, max_visited_nodes=2)
    assert len(deps) <= 2
