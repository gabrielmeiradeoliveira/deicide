"""Modular post-processing pipeline for the DEICIDE MVP 1.3 Finding Intelligence Engine."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from deicide.core.models import AnalysisWarning, CandidateFinding, Finding
from deicide.git.models import DiffBundle, FileContext
from deicide.intelligence.deduplicator import deduplicate_findings_semantic
from deicide.intelligence.evidence_validator import validate_and_assess_candidates
from deicide.intelligence.models import (
    EvidenceAssessment,
    FindingIntelligenceResult,
    FindingIntelligenceSummary,
)
from deicide.intelligence.prioritizer import prioritize_findings
from deicide.intelligence.severity_normalizer import normalize_findings


class FindingIntelligenceEngine:
    """Deterministic post-processing engine for validating, normalizing, deduplicating, and prioritizing findings."""

    def process_candidates(
        self,
        module_id: str,
        candidates: list[CandidateFinding],
        diff_bundle: DiffBundle,
        contexts: dict[str, FileContext],
        *,
        repo_root: Path | None = None,
        context_result: Any | None = None,
    ) -> FindingIntelligenceResult:
        """Run the full 4-stage Finding Intelligence pipeline on candidates from a single module."""
        return self.process_multi_module_candidates(
            candidates_by_module=[(module_id, candidates)],
            diff_bundle=diff_bundle,
            contexts=contexts,
            repo_root=repo_root,
            context_result=context_result,
        )

    def process_multi_module_candidates(
        self,
        candidates_by_module: list[tuple[str, list[CandidateFinding]]],
        diff_bundle: DiffBundle,
        contexts: dict[str, FileContext],
        *,
        repo_root: Path | None = None,
        context_result: Any | None = None,
    ) -> FindingIntelligenceResult:
        """Run the full 4-stage Finding Intelligence pipeline across one or more analysis modules."""
        total_received = 0
        all_validated: list[Finding] = []
        all_assessments: dict[str, EvidenceAssessment] = {}
        total_rejected = 0
        total_downgraded = 0
        all_warnings: list[AnalysisWarning] = []

        # Stage 1: Evidence validation and quality assessment
        for module_id, candidates in candidates_by_module:
            total_received += len(candidates)
            validated, assessments, rejected, downgraded, warnings = validate_and_assess_candidates(
                module_id=module_id,
                candidates=candidates,
                diff_bundle=diff_bundle,
                contexts=contexts,
                repo_root=repo_root,
                context_result=context_result,
            )
            all_validated.extend(validated)
            all_assessments.update(assessments)
            total_rejected += rejected
            total_downgraded += downgraded
            all_warnings.extend(warnings)

        # Stage 2: Severity normalization
        normalized_findings, adjustments = normalize_findings(
            all_validated,
            assessments=all_assessments,
        )
        sev_normalized_count = sum(1 for adj in adjustments if adj.normalized_severity != adj.original_severity)
        for adj in adjustments:
            if adj.normalized_severity != adj.original_severity:
                target_file = next(
                    (f.file_path for f in normalized_findings if f.id == adj.finding_id),
                    None,
                )
                all_warnings.append(
                    AnalysisWarning(
                        code="FINDING_SEVERITY_NORMALIZED",
                        message=(
                            f"Normalized finding '{adj.finding_id}' severity from "
                            f"'{adj.original_severity.value}' to '{adj.normalized_severity.value}' "
                            f"[{adj.rule_id}]: {adj.reason}"
                        ),
                        file_path=target_file,
                    )
                )

        # Stage 3: Semantic deduplication
        deduped_findings, deduped_count, dedup_groups = deduplicate_findings_semantic(
            normalized_findings,
            assessments=all_assessments,
        )

        # Stage 4: Deterministic prioritization
        final_findings = prioritize_findings(
            deduped_findings,
            assessments=all_assessments,
        )

        summary = FindingIntelligenceSummary(
            candidates_received=total_received,
            validated_count=len(all_validated),
            rejected_count=total_rejected,
            downgraded_count=total_downgraded,
            severity_normalized_count=sev_normalized_count,
            deduplicated_count=deduped_count,
            final_findings_count=len(final_findings),
        )

        return FindingIntelligenceResult(
            findings=final_findings,
            rejected_count=total_rejected,
            deduplicated_count=deduped_count,
            downgraded_count=total_downgraded,
            severity_normalized_count=sev_normalized_count,
            warnings=all_warnings,
            adjustments=adjustments,
            deduplication_groups=dedup_groups,
            assessments=all_assessments,
            summary=summary,
        )
