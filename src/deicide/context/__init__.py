"""Context Engine package for DEICIDE: AST analysis, indexing, dependency graphs, and context selection."""

from __future__ import annotations

from deicide.context.changed_symbols import identify_changed_symbols
from deicide.context.context_budget import (
    enforce_context_budget,
    estimate_snippet_tokens,
    serialize_context_bundle,
    serialize_snippet_block,
)
from deicide.context.context_models import (
    ContextCandidate,
    ContextRelationshipCategory,
    ContextSelectionLimits,
    ContextSelectionResult,
    OmissionReason,
    OmittedCandidate,
    ProvenanceRecord,
    SelectedSnippet,
)
from deicide.context.context_ranker import ContextRanker
from deicide.context.dependency_graph import (
    CallerEvidence,
    ChangedSymbolImpact,
    DependencyGraph,
    EdgeResolutionStatus,
    GraphEdge,
    GraphEdgeKind,
    GraphNode,
    GraphNodeKind,
    build_dependency_graph,
)
from deicide.context.import_resolver import (
    FileImportMap,
    ImportResolver,
    ResolvedImport,
    resolve_relative_module,
)
from deicide.context.models import (
    CallReference,
    ChangedSymbol,
    ChangedSymbolsResult,
    FileAnalysis,
    FileAnalysisStatus,
    ImportKind,
    ImportReference,
    ModuleLevelChange,
    OverlapClassification,
    SourceLocation,
    Symbol,
    SymbolKind,
)
from deicide.context.project_index import (
    IndexLimits,
    ProjectIndex,
    build_project_index,
    file_path_to_module_name,
)
from deicide.context.python_ast import PythonAstAnalyzer
from deicide.context.snippet_selector import (
    SnippetExtractor,
    select_context,
)

__all__ = [
    "CallReference",
    "CallerEvidence",
    "ChangedSymbol",
    "ChangedSymbolImpact",
    "ChangedSymbolsResult",
    "ContextCandidate",
    "ContextRanker",
    "ContextRelationshipCategory",
    "ContextSelectionLimits",
    "ContextSelectionResult",
    "DependencyGraph",
    "EdgeResolutionStatus",
    "FileAnalysis",
    "FileAnalysisStatus",
    "FileImportMap",
    "GraphEdge",
    "GraphEdgeKind",
    "GraphNode",
    "GraphNodeKind",
    "ImportKind",
    "ImportReference",
    "ImportResolver",
    "IndexLimits",
    "ModuleLevelChange",
    "OmissionReason",
    "OmittedCandidate",
    "OverlapClassification",
    "ProjectIndex",
    "ProvenanceRecord",
    "PythonAstAnalyzer",
    "ResolvedImport",
    "SelectedSnippet",
    "SnippetExtractor",
    "SourceLocation",
    "Symbol",
    "SymbolKind",
    "build_dependency_graph",
    "build_project_index",
    "enforce_context_budget",
    "estimate_snippet_tokens",
    "file_path_to_module_name",
    "identify_changed_symbols",
    "resolve_relative_module",
    "select_context",
    "serialize_context_bundle",
    "serialize_snippet_block",
]
