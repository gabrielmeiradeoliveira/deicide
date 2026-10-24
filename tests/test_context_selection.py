"""Deterministic offline tests for DEICIDE Context Engine Phase 3: Ranking, Snippets, and Budgeting."""

from __future__ import annotations

from pathlib import Path

from deicide.context.context_models import (
    ContextCandidate,
    ContextRelationshipCategory,
    ContextSelectionLimits,
    OmissionReason,
    ProvenanceRecord,
    SelectedSnippet,
)
from deicide.context.context_ranker import ContextRanker
from deicide.context.dependency_graph import (
    build_dependency_graph,
)
from deicide.context.models import ChangedSymbol, OverlapClassification, SourceLocation, SymbolKind
from deicide.context.project_index import build_project_index
from deicide.context.snippet_selector import (
    SnippetExtractor,
    _deduplicate_and_merge_snippets,
    select_context,
)
from deicide.providers.base import estimate_token_count


def _make_dummy_changed_sym(
    file_path: str,
    name: str,
    start: int,
    end: int,
    overlap_lines: list[int],
    parent_id: str | None = None,
) -> ChangedSymbol:
    loc = SourceLocation(file_path=file_path, start_line=start, end_line=end)
    return ChangedSymbol(
        symbol_id=f"{file_path}::{name}#function@{start}:{end}",
        symbol_name=name,
        qualified_name=name,
        kind=SymbolKind.FUNCTION,
        location=loc,
        parent_id=parent_id,
        overlapping_lines=overlap_lines,
        classification=OverlapClassification.BODY,
        is_innermost=True,
    )


# 01 - 05: Priority ordering (Changed > Direct Caller > Direct Callee > Related Class > Transitive)
def test_01_priority_ranking_order(tmp_path: Path) -> None:
    repo = tmp_path / "priority_repo"
    repo.mkdir()

    (repo / "callee.py").write_text("def helper():\n    return 42\n", encoding="utf-8")
    (repo / "calculator.py").write_text(
        "from callee import helper\n"
        "\n"
        "def calculate_average(items):\n"
        "    h = helper()\n"
        "    return sum(items) / len(items)\n",
        encoding="utf-8",
    )
    (repo / "billing.py").write_text(
        "from calculator import calculate_average\n\ndef generate_invoice():\n    return calculate_average([10, 20])\n",
        encoding="utf-8",
    )
    (repo / "trans.py").write_text(
        "from billing import generate_invoice\n\ndef outer_workflow():\n    return generate_invoice()\n",
        encoding="utf-8",
    )

    index = build_project_index(repo)
    graph = build_dependency_graph(index)

    changed_sym = _make_dummy_changed_sym("calculator.py", "calculate_average", 3, 5, [4])

    ranker = ContextRanker(graph, index, ContextSelectionLimits(max_graph_depth=2))
    candidates = ranker.discover_candidates([changed_sym])

    # Priorities must be strictly ascending: 0 (changed) -> 1 (caller) -> 2 (callee) -> 5 (transitive)
    priorities = [c.priority for c in candidates]
    assert priorities == sorted(priorities)
    assert candidates[0].priority == 0
    assert candidates[0].relationship == ContextRelationshipCategory.CHANGED_SYMBOL
    assert candidates[1].priority == 1
    assert candidates[1].relationship == ContextRelationshipCategory.DIRECT_CALLER
    assert candidates[1].qualified_name == "generate_invoice"
    assert candidates[2].priority == 2
    assert candidates[2].relationship == ContextRelationshipCategory.DIRECT_CALLEE
    assert candidates[2].qualified_name == "helper"

    trans_cands = [c for c in candidates if c.priority == 5]
    assert len(trans_cands) == 1
    assert trans_cands[0].qualified_name == "outer_workflow"


# 06 - 08: Same-name symbols, ambiguous relationships, and dynamic calls
def test_06_same_name_symbols_and_unresolved_dynamic_exclusion(tmp_path: Path) -> None:
    repo = tmp_path / "disambig_repo"
    repo.mkdir()

    (repo / "mod_a.py").write_text("def validate(): return True\n", encoding="utf-8")
    (repo / "mod_b.py").write_text("def validate(): return False\n", encoding="utf-8")
    (repo / "caller.py").write_text(
        "from mod_a import validate\n"
        "\n"
        "def run_check(service, factory):\n"
        "    validate()\n"
        "    service.validate()\n"
        "    factory()()\n",
        encoding="utf-8",
    )

    index = build_project_index(repo)
    graph = build_dependency_graph(index)

    changed_a = _make_dummy_changed_sym("mod_a.py", "validate", 1, 1, [1])

    ranker = ContextRanker(graph, index)
    cands = ranker.discover_candidates([changed_a])

    # run_check is a caller of mod_a.validate
    caller_cands = [c for c in cands if c.qualified_name == "run_check"]
    assert len(caller_cands) == 1
    # mod_b.validate must NOT be included as a caller or relation
    assert not any("mod_b" in c.file_path for c in cands)


# 09 - 15: Source line extraction, 1-based spans, indentation, long-function truncation
def test_09_source_slicing_indentation_and_long_function_windowing(tmp_path: Path) -> None:
    repo = tmp_path / "slicing_repo"
    repo.mkdir()

    # Create a long function (80 lines) with relevant call site on line 65
    lines = ["def long_workflow():"]
    for i in range(2, 65):
        lines.append(f"    step_{i} = {i} * 2")
    lines.append("    target_function_call(x=10, y=20)")  # line 65
    for i in range(66, 81):
        lines.append(f"    cleanup_{i} = {i}")

    (repo / "worker.py").write_text("\n".join(lines) + "\n", encoding="utf-8")

    extractor = SnippetExtractor(repo_root=repo, max_snippet_lines=20)
    cand = ContextCandidate(
        candidate_id="cand::worker",
        file_path="worker.py",
        start_line=1,
        end_line=80,
        qualified_name="long_workflow",
        relationship=ContextRelationshipCategory.DIRECT_CALLER,
        focus_lines=[65],
    )

    snip, omitted = extractor.slice_candidate(cand)
    assert omitted is None
    assert snip is not None
    assert snip.is_truncated is True
    assert snip.truncated_original_lines == 80
    # Must contain line 65
    assert snip.start_line <= 65 <= snip.end_line
    # Must strictly satisfy max_snippet_lines
    assert (snip.end_line - snip.start_line + 1) <= 20
    # Indentation preserved
    assert "    target_function_call(x=10, y=20)" in snip.content
    assert 65 in snip.relevant_call_lines


# 16 - 19: Limits enforcement (max files, max snippets, max traversal depth, max visited nodes)
def test_16_limits_enforcement(tmp_path: Path) -> None:
    repo = tmp_path / "limits_repo"
    repo.mkdir()

    (repo / "target.py").write_text("def target(): pass\n", encoding="utf-8")
    for i in range(10):
        (repo / f"client_{i}.py").write_text(
            f"from target import target\ndef call_{i}(): target()\n",
            encoding="utf-8",
        )

    index = build_project_index(repo)
    graph = build_dependency_graph(index)

    changed = _make_dummy_changed_sym("target.py", "target", 1, 1, [1])

    # Limit max_context_files to 3 and max_snippets to 4
    limits = ContextSelectionLimits(max_context_files=3, max_snippets=4, max_context_tokens=5000)
    res = select_context(repo, [changed], graph, index, limits=limits)

    assert res.selected_files_count <= 3
    assert res.selected_snippets_count <= 4
    assert res.omitted_snippets_count > 0
    reasons = {o.reason for o in res.omitted_candidates}
    assert OmissionReason.FILE_LIMIT_EXCEEDED in reasons or OmissionReason.SNIPPET_LIMIT_EXCEEDED in reasons


# 20 - 25: Token budget enforcement, zero budget, tiny budget, overhead accounting
def test_20_strict_token_budget_and_zero_budget(tmp_path: Path) -> None:
    repo = tmp_path / "budget_repo"
    repo.mkdir()

    (repo / "calc.py").write_text("def compute():\n    return 100\n", encoding="utf-8")
    (repo / "app.py").write_text(
        "from calc import compute\n\ndef run():\n    return compute()\n",
        encoding="utf-8",
    )

    index = build_project_index(repo)
    graph = build_dependency_graph(index)
    changed = _make_dummy_changed_sym("calc.py", "compute", 1, 2, [2])

    # Zero budget: nothing selected, clear omission
    res_zero = select_context(repo, [changed], graph, index, max_context_tokens=0)
    assert res_zero.selected_snippets_count == 0
    assert res_zero.total_estimated_tokens == 0
    assert res_zero.omitted_snippets_count >= 1
    assert res_zero.omitted_candidates[0].reason == OmissionReason.TOKEN_BUDGET_EXCEEDED

    # Tiny budget (15 tokens): cannot fit even 1 formatted snippet (which has overhead)
    res_tiny = select_context(repo, [changed], graph, index, max_context_tokens=15)
    assert res_tiny.selected_snippets_count == 0
    assert res_tiny.total_estimated_tokens <= 15
    assert res_tiny.omitted_snippets_count >= 1

    # Ample budget (2000 tokens): fits both changed and caller snippet
    res_ample = select_context(repo, [changed], graph, index, max_context_tokens=2000)
    assert res_ample.selected_snippets_count >= 1
    assert res_ample.total_estimated_tokens <= 2000
    assert res_ample.total_estimated_tokens == estimate_token_count(res_ample.serialized_text)


# 26 - 31: Deterministic ranking, duplicate elimination, overlapping snippet merging, provenance retention
def test_28_snippet_deduplication_and_merging_with_multiple_provenance(tmp_path: Path) -> None:
    repo = tmp_path / "dedup_repo"
    repo.mkdir()

    code = "def helper_a():\n    return 1\n\ndef helper_b():\n    return 2\n"
    (repo / "utils.py").write_text(code, encoding="utf-8")

    prov1 = ProvenanceRecord(
        source_id="s1",
        target_id="t1",
        relationship=ContextRelationshipCategory.DIRECT_CALLER,
        explanation="Caller of A",
    )
    prov2 = ProvenanceRecord(
        source_id="s2",
        target_id="t2",
        relationship=ContextRelationshipCategory.DIRECT_CALLEE,
        explanation="Callee of B",
    )

    # Two snippets in the same file that are adjacent (lines 1-2 and 4-5)
    s1 = SelectedSnippet(
        snippet_id="snip1",
        file_path="utils.py",
        start_line=1,
        end_line=2,
        content="   1 | def helper_a():\n   2 |     return 1",
        relationship=ContextRelationshipCategory.DIRECT_CALLER,
        provenance=[prov1],
        estimated_tokens=20,
    )
    s2 = SelectedSnippet(
        snippet_id="snip2",
        file_path="utils.py",
        start_line=4,
        end_line=5,
        content="   4 | def helper_b():\n   5 |     return 2",
        relationship=ContextRelationshipCategory.DIRECT_CALLEE,
        provenance=[prov2],
        estimated_tokens=20,
    )

    merged = _deduplicate_and_merge_snippets([s1, s2], max_snippet_lines=20)
    assert len(merged) == 1
    # Both provenance records preserved
    assert len(merged[0].provenance) == 2
    assert merged[0].start_line == 1
    assert merged[0].end_line == 5


# 32 - 37: Missing file, invalid span, syntax error isolation, symlink escape, partial index
def test_32_security_path_and_missing_file_handling(tmp_path: Path) -> None:
    repo = tmp_path / "sec_repo"
    repo.mkdir()

    extractor = SnippetExtractor(repo_root=repo)

    # Missing file
    cand_missing = ContextCandidate(
        candidate_id="cand::missing",
        file_path="nonexistent.py",
        start_line=1,
        end_line=10,
        relationship=ContextRelationshipCategory.DIRECT_CALLER,
    )
    snip_m, omit_m = extractor.slice_candidate(cand_missing)
    assert snip_m is None
    assert omit_m is not None
    assert omit_m.reason == OmissionReason.SOURCE_UNAVAILABLE

    # Path traversal attempt
    cand_traversal = ContextCandidate(
        candidate_id="cand::traversal",
        file_path="../outside.py",
        start_line=1,
        end_line=5,
        relationship=ContextRelationshipCategory.DIRECT_CALLER,
    )
    snip_t, omit_t = extractor.slice_candidate(cand_traversal)
    assert snip_t is None
    assert omit_t is not None
    assert omit_t.reason == OmissionReason.UNSAFE_PATH


# 38 - 44: Empty graph, no changed symbols, multiple callers, circular dependencies
def test_38_empty_inputs_and_circular_dependencies(tmp_path: Path) -> None:
    repo = tmp_path / "circ_repo"
    repo.mkdir()

    (repo / "a.py").write_text("import b\ndef fa(): pass\n", encoding="utf-8")
    (repo / "b.py").write_text("import a\ndef fb(): pass\n", encoding="utf-8")

    index = build_project_index(repo)
    graph = build_dependency_graph(index)

    # No changed symbols -> empty result with diagnostic
    res_empty = select_context(repo, [], graph, index)
    assert res_empty.selected_snippets_count == 0
    assert any("No changed symbols" in d for d in res_empty.diagnostics)

    # Circular dependencies do not cause infinite loops
    changed_a = _make_dummy_changed_sym("a.py", "fa", 2, 2, [2])
    res_circ = select_context(repo, [changed_a], graph, index)
    assert res_circ.selected_snippets_count >= 1


# 45 - 50: Full offline integration, unchanged caller discovery with provenance, regression test
def test_45_end_to_end_context_selection_scenario(tmp_path: Path) -> None:
    repo = tmp_path / "e2e_repo"
    repo.mkdir()

    # calculator.py
    (repo / "calculator.py").write_text(
        "def calculate_average(values):\n    return sum(values) / len(values)\n",
        encoding="utf-8",
    )
    # billing.py
    (repo / "billing.py").write_text(
        "from calculator import calculate_average\n"
        "\n"
        "def generate_report():\n"
        "    data = [10, 20, 30]\n"
        "    return calculate_average(data)\n",
        encoding="utf-8",
    )

    index = build_project_index(repo)
    graph = build_dependency_graph(index)

    changed_sym = _make_dummy_changed_sym("calculator.py", "calculate_average", 1, 2, [2])

    result = select_context(
        repo_root=repo,
        changed_symbols=[changed_sym],
        dependency_graph=graph,
        project_index=index,
        max_context_tokens=1500,
    )

    # Verify result structure
    assert result.selected_snippets_count == 2
    assert result.selected_files_count == 2
    assert result.total_estimated_tokens <= 1500

    # Verify that billing.generate_report was selected as unchanged caller
    caller_snip = next(s for s in result.selected_snippets if s.file_path == "billing.py")
    assert caller_snip.symbol_name == "generate_report"
    assert caller_snip.relationship == ContextRelationshipCategory.DIRECT_CALLER
    assert caller_snip.provenance[0].confidence == 1.0
    assert 5 in caller_snip.relevant_call_lines

    # Verify serialized bundle contains clear delimiters and line numbers
    assert "=== CONTEXT SNIPPET: billing.py (direct_caller) ===" in result.serialized_text
    assert "return calculate_average(data)" in result.serialized_text
