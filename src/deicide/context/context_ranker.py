"""Deterministic context candidate discovery and relationship-based ranking."""

from __future__ import annotations

from deicide.context.context_models import (
    ContextCandidate,
    ContextRelationshipCategory,
    ContextSelectionLimits,
    ProvenanceRecord,
)
from deicide.context.dependency_graph import DependencyGraph, EdgeResolutionStatus
from deicide.context.models import ChangedSymbol
from deicide.context.project_index import ProjectIndex
from deicide.core.models import estimate_token_count


def _approximate_tokens_for_span(end_line: int, start_line: int) -> int:
    """Rough estimate based on average 35 chars per line."""
    line_count = max(1, end_line - start_line + 1)
    return estimate_token_count(" " * (line_count * 35))


class ContextRanker:
    """Discovers and ranks context candidates from changed symbols and dependency graph."""

    def __init__(
        self,
        dependency_graph: DependencyGraph,
        project_index: ProjectIndex,
        limits: ContextSelectionLimits | None = None,
    ) -> None:
        self.graph = dependency_graph
        self.index = project_index
        self.limits = limits or ContextSelectionLimits()

    def discover_candidates(
        self,
        changed_symbols: list[ChangedSymbol],
    ) -> list[ContextCandidate]:
        """Discover candidate context snippets across 0-6 priority categories."""
        candidates_by_id: dict[str, ContextCandidate] = {}

        # Priority 0: Changed code
        for cs in changed_symbols:
            cid = f"cand::{cs.symbol_id}"
            prov = ProvenanceRecord(
                source_id=cs.symbol_id,
                target_id=cs.symbol_id,
                relationship=ContextRelationshipCategory.CHANGED_SYMBOL,
                call_site=cs.location,
                confidence=1.0,
                resolution_status=EdgeResolutionStatus.RESOLVED,
                explanation=f"Directly modified symbol '{cs.qualified_name}'",
            )
            cand = ContextCandidate(
                candidate_id=cid,
                file_path=cs.location.file_path,
                start_line=cs.location.start_line,
                end_line=cs.location.end_line,
                symbol_id=cs.symbol_id,
                qualified_name=cs.qualified_name,
                relationship=ContextRelationshipCategory.CHANGED_SYMBOL,
                relationship_depth=0,
                resolution_status=EdgeResolutionStatus.RESOLVED,
                confidence=1.0,
                priority=0,
                priority_explanation="Directly modified code in Git diff",
                provenance=[prov],
                estimated_tokens=_approximate_tokens_for_span(cs.location.end_line, cs.location.start_line),
                focus_lines=cs.overlapping_lines,
            )
            candidates_by_id[cid] = cand

            # Priority 1: Direct callers of changed symbols
            callers_with_ev = self.graph.get_callers_with_evidence(cs.symbol_id)
            for ev in callers_with_ev:
                caller_sym = self.index.get_symbol_by_id(ev.caller_node.id)
                if not caller_sym:
                    continue

                caller_cid = f"cand::{caller_sym.id}"
                caller_prov = ProvenanceRecord(
                    source_id=caller_sym.id,
                    target_id=cs.symbol_id,
                    relationship=ContextRelationshipCategory.DIRECT_CALLER,
                    call_site=ev.call_site,
                    confidence=ev.confidence,
                    resolution_status=ev.status,
                    explanation=f"Directly calls modified symbol '{cs.qualified_name}' at line {ev.call_site.start_line}",
                )

                if caller_cid in candidates_by_id:
                    existing = candidates_by_id[caller_cid]
                    # Merge provenance
                    merged_prov = [*existing.provenance, caller_prov]
                    merged_focus = sorted(set(existing.focus_lines + [ev.call_site.start_line]))
                    candidates_by_id[caller_cid] = existing.model_copy(
                        update={"provenance": merged_prov, "focus_lines": merged_focus}
                    )
                else:
                    cand_caller = ContextCandidate(
                        candidate_id=caller_cid,
                        file_path=caller_sym.location.file_path,
                        start_line=caller_sym.location.start_line,
                        end_line=caller_sym.location.end_line,
                        symbol_id=caller_sym.id,
                        qualified_name=caller_sym.qualified_name,
                        relationship=ContextRelationshipCategory.DIRECT_CALLER,
                        relationship_depth=1,
                        resolution_status=ev.status,
                        confidence=ev.confidence,
                        priority=1,
                        priority_explanation=f"Unchanged direct caller of '{cs.qualified_name}'",
                        provenance=[caller_prov],
                        estimated_tokens=_approximate_tokens_for_span(
                            caller_sym.location.end_line, caller_sym.location.start_line
                        ),
                        focus_lines=[ev.call_site.start_line],
                    )
                    candidates_by_id[caller_cid] = cand_caller

            # Priority 2: Direct callees of changed symbols
            callees = self.graph.get_callees(cs.symbol_id)
            for callee_node in callees:
                callee_sym = self.index.get_symbol_by_id(callee_node.id)
                if not callee_sym:
                    continue

                callee_cid = f"cand::{callee_sym.id}"
                callee_prov = ProvenanceRecord(
                    source_id=cs.symbol_id,
                    target_id=callee_sym.id,
                    relationship=ContextRelationshipCategory.DIRECT_CALLEE,
                    confidence=1.0,
                    resolution_status=EdgeResolutionStatus.RESOLVED,
                    explanation=f"Directly called by modified symbol '{cs.qualified_name}'",
                )

                if callee_cid not in candidates_by_id:
                    cand_callee = ContextCandidate(
                        candidate_id=callee_cid,
                        file_path=callee_sym.location.file_path,
                        start_line=callee_sym.location.start_line,
                        end_line=callee_sym.location.end_line,
                        symbol_id=callee_sym.id,
                        qualified_name=callee_sym.qualified_name,
                        relationship=ContextRelationshipCategory.DIRECT_CALLEE,
                        relationship_depth=1,
                        resolution_status=EdgeResolutionStatus.RESOLVED,
                        confidence=1.0,
                        priority=2,
                        priority_explanation=f"Direct callee called by '{cs.qualified_name}'",
                        provenance=[callee_prov],
                        estimated_tokens=_approximate_tokens_for_span(
                            callee_sym.location.end_line, callee_sym.location.start_line
                        ),
                        focus_lines=[callee_sym.location.start_line],
                    )
                    candidates_by_id[callee_cid] = cand_callee

            # Priority 4: Related class or methods
            if cs.parent_id and cs.parent_id in self.index.symbols_by_id:
                parent_sym = self.index.symbols_by_id[cs.parent_id]
                parent_cid = f"cand::{parent_sym.id}"
                if parent_cid not in candidates_by_id:
                    candidates_by_id[parent_cid] = ContextCandidate(
                        candidate_id=parent_cid,
                        file_path=parent_sym.location.file_path,
                        start_line=parent_sym.location.start_line,
                        end_line=parent_sym.location.end_line,
                        symbol_id=parent_sym.id,
                        qualified_name=parent_sym.qualified_name,
                        relationship=ContextRelationshipCategory.RELATED_CLASS_METHOD,
                        relationship_depth=1,
                        resolution_status=EdgeResolutionStatus.RESOLVED,
                        confidence=1.0,
                        priority=4,
                        priority_explanation=f"Containing class for '{cs.qualified_name}'",
                        provenance=[
                            ProvenanceRecord(
                                source_id=parent_sym.id,
                                target_id=cs.symbol_id,
                                relationship=ContextRelationshipCategory.RELATED_CLASS_METHOD,
                                confidence=1.0,
                                explanation=f"Containing class of '{cs.qualified_name}'",
                            )
                        ],
                        estimated_tokens=_approximate_tokens_for_span(
                            parent_sym.location.end_line, parent_sym.location.start_line
                        ),
                        focus_lines=[cs.location.start_line],
                    )

            # Priority 5: Transitive callers (depth 2) if configured
            if self.limits.max_graph_depth >= 2:
                for ev in callers_with_ev:
                    trans_callers = self.graph.get_callers_with_evidence(ev.caller_node.id)
                    for t_ev in trans_callers:
                        t_sym = self.index.get_symbol_by_id(t_ev.caller_node.id)
                        if not t_sym or t_sym.id in candidates_by_id:
                            continue
                        t_cid = f"cand::{t_sym.id}"
                        candidates_by_id[t_cid] = ContextCandidate(
                            candidate_id=t_cid,
                            file_path=t_sym.location.file_path,
                            start_line=t_sym.location.start_line,
                            end_line=t_sym.location.end_line,
                            symbol_id=t_sym.id,
                            qualified_name=t_sym.qualified_name,
                            relationship=ContextRelationshipCategory.TRANSITIVE_CALLER,
                            relationship_depth=2,
                            resolution_status=t_ev.status,
                            confidence=t_ev.confidence * 0.8,
                            priority=5,
                            priority_explanation=f"Transitive caller of '{ev.caller_node.name}'",
                            provenance=[
                                ProvenanceRecord(
                                    source_id=t_sym.id,
                                    target_id=ev.caller_node.id,
                                    relationship=ContextRelationshipCategory.TRANSITIVE_CALLER,
                                    call_site=t_ev.call_site,
                                    confidence=t_ev.confidence,
                                    explanation=f"Transitive caller via '{ev.caller_node.name}'",
                                )
                            ],
                            estimated_tokens=_approximate_tokens_for_span(
                                t_sym.location.end_line, t_sym.location.start_line
                            ),
                            focus_lines=[t_ev.call_site.start_line],
                        )

        # Deterministic sorting
        sorted_candidates = sorted(
            candidates_by_id.values(),
            key=lambda c: (
                c.priority,
                c.relationship_depth,
                -c.confidence,
                c.estimated_tokens,
                c.file_path,
                c.start_line,
                c.candidate_id,
            ),
        )
        return sorted_candidates
