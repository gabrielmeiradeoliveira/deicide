"""Deterministic cross-file dependency graph, evidence tracking, and bounded traversal for DEICIDE."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import StrEnum

from deicide.context.import_resolver import ImportResolver
from deicide.context.models import ChangedSymbol, SourceLocation, Symbol, SymbolKind
from deicide.context.project_index import ProjectIndex


class GraphNodeKind(StrEnum):
    """Classification of a dependency graph node."""

    MODULE = "module"
    FUNCTION = "function"
    METHOD = "method"
    CLASS = "class"
    EXTERNAL = "external"


class GraphEdgeKind(StrEnum):
    """Relationship classification between graph nodes."""

    IMPORTS_MODULE = "imports_module"
    IMPORTS_SYMBOL = "imports_symbol"
    CALLS = "calls"
    REFERENCES = "references"
    CONTAINS = "contains"


class EdgeResolutionStatus(StrEnum):
    """Resolution confidence status for a graph edge."""

    RESOLVED = "resolved"
    AMBIGUOUS = "ambiguous"
    UNRESOLVED = "unresolved"
    EXTERNAL = "external"


@dataclass(frozen=True)
class GraphNode:
    """Node in the cross-file dependency graph."""

    id: str
    name: str
    qualified_name: str
    kind: GraphNodeKind
    file_path: str
    location: SourceLocation | None = None
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class GraphEdge:
    """Directed relationship between two nodes in the dependency graph."""

    source_id: str
    target_id: str
    kind: GraphEdgeKind
    location: SourceLocation
    status: EdgeResolutionStatus
    confidence: float = 1.0
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class CallerEvidence:
    """Call-site provenance for a caller of a symbol."""

    caller_node: GraphNode
    call_site: SourceLocation
    confidence: float
    status: EdgeResolutionStatus


@dataclass(frozen=True)
class ChangedSymbolImpact:
    """Cross-file relationships and provenance for a modified symbol."""

    changed_symbol: ChangedSymbol
    direct_callers: list[CallerEvidence]
    direct_callees: list[GraphNode]
    imported_by_modules: list[str]
    related_classes: list[GraphNode]


def _symbol_kind_to_node_kind(kind: SymbolKind) -> GraphNodeKind:
    if kind in (
        SymbolKind.METHOD,
        SymbolKind.ASYNC_METHOD,
        SymbolKind.CLASS_METHOD,
        SymbolKind.STATIC_METHOD,
    ):
        return GraphNodeKind.METHOD
    if kind in (SymbolKind.CLASS, SymbolKind.NESTED_CLASS):
        return GraphNodeKind.CLASS
    if kind == SymbolKind.MODULE:
        return GraphNodeKind.MODULE
    return GraphNodeKind.FUNCTION


class DependencyGraph:
    """Queryable cross-file dependency graph with caller/callee and cycle analysis."""

    def __init__(self) -> None:
        self.nodes: dict[str, GraphNode] = {}
        self.edges: list[GraphEdge] = []
        self.out_edges: dict[str, list[GraphEdge]] = {}
        self.in_edges: dict[str, list[GraphEdge]] = {}
        self._edge_keys: set[tuple[str, str, GraphEdgeKind, str, int, int | None]] = set()

    def add_node(self, node: GraphNode) -> None:
        if node.id not in self.nodes:
            self.nodes[node.id] = node
            self.out_edges[node.id] = []
            self.in_edges[node.id] = []

    def add_edge(self, edge: GraphEdge) -> None:
        # Deduplicate identical edges while preserving distinct call sites as distinct evidence
        key = (
            edge.source_id,
            edge.target_id,
            edge.kind,
            edge.location.file_path,
            edge.location.start_line,
            edge.location.start_col,
        )
        if key in self._edge_keys:
            return

        self._edge_keys.add(key)
        self.edges.append(edge)
        self.out_edges.setdefault(edge.source_id, []).append(edge)
        self.in_edges.setdefault(edge.target_id, []).append(edge)

    def get_node(self, node_id: str) -> GraphNode | None:
        return self.nodes.get(node_id)

    def get_callers(self, target_id: str) -> list[GraphNode]:
        """Find all caller nodes that directly call `target_id` with resolved status."""
        incoming = self.in_edges.get(target_id, [])
        caller_ids = {
            e.source_id for e in incoming if e.kind == GraphEdgeKind.CALLS and e.status == EdgeResolutionStatus.RESOLVED
        }
        return [self.nodes[cid] for cid in sorted(caller_ids) if cid in self.nodes]

    def get_callers_with_evidence(self, target_id: str) -> list[CallerEvidence]:
        """Return callers of target_id with call-site source locations and confidence."""
        incoming = self.in_edges.get(target_id, [])
        evidence_list: list[CallerEvidence] = []
        for e in incoming:
            if e.kind == GraphEdgeKind.CALLS and e.status == EdgeResolutionStatus.RESOLVED:
                caller_node = self.nodes.get(e.source_id)
                if caller_node:
                    evidence_list.append(
                        CallerEvidence(
                            caller_node=caller_node,
                            call_site=e.location,
                            confidence=e.confidence,
                            status=e.status,
                        )
                    )
        return sorted(
            evidence_list,
            key=lambda ev: (ev.caller_node.file_path, ev.call_site.start_line, ev.caller_node.id),
        )

    def get_callees(self, source_id: str) -> list[GraphNode]:
        """Find all callee nodes directly called by `source_id` with resolved status."""
        outgoing = self.out_edges.get(source_id, [])
        callee_ids = {
            e.target_id for e in outgoing if e.kind == GraphEdgeKind.CALLS and e.status == EdgeResolutionStatus.RESOLVED
        }
        return [self.nodes[cid] for cid in sorted(callee_ids) if cid in self.nodes]

    def get_imported_by(self, target_id: str) -> list[GraphNode]:
        """Find all nodes that import `target_id`."""
        incoming = self.in_edges.get(target_id, [])
        importer_ids = {
            e.source_id
            for e in incoming
            if e.kind in (GraphEdgeKind.IMPORTS_MODULE, GraphEdgeKind.IMPORTS_SYMBOL)
            and e.status == EdgeResolutionStatus.RESOLVED
        }
        return [self.nodes[iid] for iid in sorted(importer_ids) if iid in self.nodes]

    def get_dependencies(
        self,
        node_id: str,
        max_depth: int = 1,
        max_visited_nodes: int = 100,
    ) -> list[GraphNode]:
        """Traverse forward outgoing calls and imports up to `max_depth` and `max_visited_nodes`."""
        visited: set[str] = set()
        queue: deque[tuple[str, int]] = deque([(node_id, 0)])

        while queue and len(visited) < max_visited_nodes:
            curr_id, depth = queue.popleft()
            if depth >= max_depth:
                continue
            for edge in self.out_edges.get(curr_id, []):
                if edge.status == EdgeResolutionStatus.RESOLVED and edge.target_id not in visited:
                    visited.add(edge.target_id)
                    queue.append((edge.target_id, depth + 1))
                    if len(visited) >= max_visited_nodes:
                        break

        return [self.nodes[nid] for nid in sorted(visited) if nid in self.nodes]

    def get_impacted_symbols(
        self,
        changed_symbol_ids: set[str],
        max_depth: int = 2,
        max_visited_nodes: int = 100,
    ) -> set[str]:
        """Traverse backward callers and references to identify impacted symbols."""
        impacted: set[str] = set()
        queue: deque[tuple[str, int]] = deque((sid, 0) for sid in sorted(changed_symbol_ids))

        while queue and len(impacted) < max_visited_nodes:
            curr_id, depth = queue.popleft()
            if depth >= max_depth:
                continue
            for edge in self.in_edges.get(curr_id, []):
                if (
                    edge.kind in (GraphEdgeKind.CALLS, GraphEdgeKind.REFERENCES)
                    and edge.status == EdgeResolutionStatus.RESOLVED
                    and edge.source_id not in impacted
                ):
                    impacted.add(edge.source_id)
                    queue.append((edge.source_id, depth + 1))
                    if len(impacted) >= max_visited_nodes:
                        break

        return impacted

    def get_impact_for_changed_symbol(self, changed_symbol: ChangedSymbol) -> ChangedSymbolImpact:
        """Retrieve cross-file callers, callees, and module relationships for a modified symbol."""
        symbol_id = changed_symbol.symbol_id
        callers = self.get_callers_with_evidence(symbol_id)
        callees = self.get_callees(symbol_id)

        importers: list[str] = []
        for e in self.in_edges.get(symbol_id, []):
            if e.kind in (GraphEdgeKind.IMPORTS_SYMBOL, GraphEdgeKind.IMPORTS_MODULE):
                importers.append(e.source_id)

        related_classes: list[GraphNode] = []
        if changed_symbol.parent_id and changed_symbol.parent_id in self.nodes:
            parent = self.nodes[changed_symbol.parent_id]
            if parent.kind == GraphNodeKind.CLASS:
                related_classes.append(parent)

        return ChangedSymbolImpact(
            changed_symbol=changed_symbol,
            direct_callers=callers,
            direct_callees=callees,
            imported_by_modules=sorted(set(importers)),
            related_classes=related_classes,
        )

    def find_cycles(self) -> list[list[str]]:
        """Detect directed cycles (e.g. circular imports or recursive calls) deterministically."""
        cycles: list[list[str]] = []
        visited: set[str] = set()
        rec_stack: list[str] = []
        rec_set: set[str] = set()

        def dfs(node_id: str) -> None:
            visited.add(node_id)
            rec_stack.append(node_id)
            rec_set.add(node_id)

            for edge in self.out_edges.get(node_id, []):
                if edge.status != EdgeResolutionStatus.RESOLVED:
                    continue
                neighbor = edge.target_id
                if neighbor not in visited:
                    dfs(neighbor)
                elif neighbor in rec_set:
                    # Found a cycle
                    idx = rec_stack.index(neighbor)
                    cycle = rec_stack[idx:] + [neighbor]
                    cycles.append(cycle)

            rec_stack.pop()
            rec_set.remove(node_id)

        for node_id in sorted(self.nodes.keys()):
            if node_id not in visited:
                dfs(node_id)

        # Normalize and deduplicate cycles
        normalized_cycles: list[list[str]] = []
        seen_cycle_keys: set[tuple[str, ...]] = set()

        for c in cycles:
            body = c[:-1]
            if not body:
                continue
            min_node = min(body)
            min_idx = body.index(min_node)
            rotated = body[min_idx:] + body[:min_idx]
            key = tuple(rotated)
            if key not in seen_cycle_keys:
                seen_cycle_keys.add(key)
                normalized_cycles.append(rotated + [min_node])

        return sorted(normalized_cycles, key=lambda c: (len(c), c))


def _resolve_call_target(
    call_expr: str,
    simple_name: str | None,
    attribute_chain: list[str],
    is_dynamic: bool,
    caller_symbol: Symbol | None,
    current_module: str,
    index: ProjectIndex,
    import_map,
) -> tuple[str | None, EdgeResolutionStatus, float]:
    """Statically resolve a call expression to a target symbol or module ID."""
    if is_dynamic or not simple_name:
        return None, EdgeResolutionStatus.UNRESOLVED, 0.0

    # 1. Parameter / local variable / assignment shadowing
    if caller_symbol and simple_name in caller_symbol.local_bindings:
        # Call target is a local parameter or variable, so it does not resolve to an external/module symbol
        return None, EdgeResolutionStatus.UNRESOLVED, 0.0

    # 2. `self.method()` or `cls.method()` inside a class
    if attribute_chain and attribute_chain[0] in ("self", "cls") and len(attribute_chain) == 2:
        method_name = attribute_chain[1]
        if caller_symbol and caller_symbol.scope_path:
            containing_class_name = caller_symbol.scope_path[0]
            target_sym = index.resolve_symbol(current_module, f"{containing_class_name}.{method_name}")
            if target_sym:
                return target_sym.id, EdgeResolutionStatus.RESOLVED, 1.0

    # 3. Direct local/module-level symbol in the same file
    if len(attribute_chain) == 1:
        # Check if caller is inside a scope and simple_name is defined in that scope
        if caller_symbol and caller_symbol.scope_path:
            scoped_target = index.resolve_symbol(current_module, f"{'.'.join(caller_symbol.scope_path)}.{simple_name}")
            if scoped_target:
                return scoped_target.id, EdgeResolutionStatus.RESOLVED, 1.0

        # Check if defined at module-level in current_module
        local_sym = index.resolve_symbol(current_module, simple_name)
        if local_sym:
            return local_sym.id, EdgeResolutionStatus.RESOLVED, 1.0

        # Check if simple_name is an explicit imported symbol
        binding = import_map.get_binding(simple_name)
        if binding and binding.target_symbol:
            return binding.target_symbol.id, EdgeResolutionStatus.RESOLVED, 1.0

    # 4. Attribute call via imported module alias (e.g. `calc.calculate_total()`)
    if len(attribute_chain) == 2:
        mod_alias, func_name = attribute_chain
        binding = import_map.get_binding(mod_alias)
        if binding and binding.target_module:
            target_sym = index.resolve_symbol(binding.target_module, func_name)
            if target_sym:
                return target_sym.id, EdgeResolutionStatus.RESOLVED, 1.0

    # Dynamic or unresolvable instance method call (e.g. `service.validate()`)
    return None, EdgeResolutionStatus.UNRESOLVED, 0.0


def build_dependency_graph(index: ProjectIndex) -> DependencyGraph:
    """Build the cross-file dependency graph from an indexed repository."""
    graph = DependencyGraph()
    resolver = ImportResolver(index)

    # 1. Add module nodes
    for file_path, module_name in sorted(index.file_to_module.items()):
        mod_id = f"module::{module_name}"
        graph.add_node(
            GraphNode(
                id=mod_id,
                name=module_name,
                qualified_name=module_name,
                kind=GraphNodeKind.MODULE,
                file_path=file_path,
            )
        )

    # 2. Add symbol nodes and CONTAINS edges
    for symbol_id, sym in sorted(index.symbols_by_id.items()):
        mod_name = index.get_module_for_file(sym.location.file_path) or ""
        mod_id = f"module::{mod_name}"

        node = GraphNode(
            id=symbol_id,
            name=sym.name,
            qualified_name=sym.qualified_name,
            kind=_symbol_kind_to_node_kind(sym.kind),
            file_path=sym.location.file_path,
            location=sym.location,
        )
        graph.add_node(node)

        # CONTAINS edge from parent symbol or containing module
        parent_id = sym.parent_id if sym.parent_id else mod_id
        graph.add_edge(
            GraphEdge(
                source_id=parent_id,
                target_id=symbol_id,
                kind=GraphEdgeKind.CONTAINS,
                location=sym.location,
                status=EdgeResolutionStatus.RESOLVED,
            )
        )

    # 3. Add import edges
    for file_path, module_name in sorted(index.file_to_module.items()):
        mod_id = f"module::{module_name}"
        import_map = resolver.resolve_file_imports(file_path)

        for binding in import_map.bindings.values():
            if binding.is_submodule or (binding.target_module in index.module_to_file):
                target_mod_id = f"module::{binding.target_module}"
                if target_mod_id in graph.nodes:
                    graph.add_edge(
                        GraphEdge(
                            source_id=mod_id,
                            target_id=target_mod_id,
                            kind=GraphEdgeKind.IMPORTS_MODULE,
                            location=binding.location,
                            status=EdgeResolutionStatus.RESOLVED,
                        )
                    )
            if binding.target_symbol:
                graph.add_edge(
                    GraphEdge(
                        source_id=mod_id,
                        target_id=binding.target_symbol.id,
                        kind=GraphEdgeKind.IMPORTS_SYMBOL,
                        location=binding.location,
                        status=EdgeResolutionStatus.RESOLVED,
                    )
                )

        # 4. Add call edges
        calls = index.get_calls_for_file(file_path)
        for call in calls:
            caller_sym = index.get_symbol_by_id(call.caller_symbol_id) if call.caller_symbol_id else None
            source_id = caller_sym.id if caller_sym else mod_id

            target_id, status, confidence = _resolve_call_target(
                call_expr=call.called_expression,
                simple_name=call.simple_name,
                attribute_chain=call.attribute_chain,
                is_dynamic=call.is_dynamic,
                caller_symbol=caller_sym,
                current_module=module_name,
                index=index,
                import_map=import_map,
            )

            if target_id and status == EdgeResolutionStatus.RESOLVED:
                graph.add_edge(
                    GraphEdge(
                        source_id=source_id,
                        target_id=target_id,
                        kind=GraphEdgeKind.CALLS,
                        location=call.location,
                        status=status,
                        confidence=confidence,
                    )
                )

    return graph
