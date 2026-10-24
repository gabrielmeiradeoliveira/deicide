"""Deterministic, explainable severity normalization rules for DEICIDE findings."""

from __future__ import annotations

import re

from deicide.core.models import Confidence, Finding, FindingCategory, Severity
from deicide.intelligence.models import (
    EvidenceAssessment,
    EvidenceQuality,
    NormalizationAdjustment,
)

_UNCONDITIONAL_FATAL_RE = re.compile(r"(?<!/)/(?!/)\s*0(?![.0-9])|%\s*0(?![.0-9])")
_MINOR_FRAMING_RE = re.compile(
    r"\b(minor issue|nitpick|trivial|cosmetic|informational only)\b",
    re.IGNORECASE,
)
_SPECULATIVE_FRAMING_RE = re.compile(
    r"\b(might possibly|purely speculative|unlikely edge case|hypothetical)\b",
    re.IGNORECASE,
)

_CATEGORY_MAX_SEVERITY: dict[FindingCategory, Severity] = {
    FindingCategory.ERROR_HANDLING: Severity.HIGH,
    FindingCategory.STATE_MUTATION: Severity.HIGH,
    FindingCategory.RESOURCE_MANAGEMENT: Severity.HIGH,
}


def _cap_severity(current: Severity, ceiling: Severity) -> Severity:
    if current.rank > ceiling.rank:
        return ceiling
    return current


def normalize_finding_severity(
    finding: Finding,
    assessment: EvidenceAssessment | None = None,
) -> tuple[Finding, NormalizationAdjustment | None]:
    """Apply deterministic severity and confidence normalization rules to a validated Finding.

    Never arbitrarily upgrades severity. Caps inflated severities when confidence,
    evidence quality, category bounds, or speculative framing do not support them.
    """
    orig_sev = finding.severity
    orig_conf = finding.confidence
    sev = orig_sev
    conf = orig_conf

    quality = assessment.quality if assessment is not None else EvidenceQuality.EXACT
    reasons: list[tuple[str, str]] = []

    # 1. Evidence quality bounds
    if quality == EvidenceQuality.WEAK:
        if conf == Confidence.HIGH:
            conf = Confidence.MEDIUM
        if sev.rank > Severity.MEDIUM.rank:
            sev = _cap_severity(sev, Severity.MEDIUM)
            reasons.append(
                (
                    "SEV_WEAK_EVIDENCE_CAP",
                    "Severity capped at medium because supporting evidence is weak.",
                )
            )
    elif quality == EvidenceQuality.PARTIAL:
        if conf == Confidence.HIGH:
            conf = Confidence.MEDIUM
        if sev.rank > Severity.HIGH.rank:
            sev = _cap_severity(sev, Severity.HIGH)
            reasons.append(
                (
                    "SEV_PARTIAL_EVIDENCE_CAP",
                    "Severity capped at high because supporting evidence is only partially grounded.",
                )
            )

    # 2. Low-confidence severity ceiling
    if conf == Confidence.LOW and sev.rank > Severity.MEDIUM.rank:
        sev = _cap_severity(sev, Severity.MEDIUM)
        reasons.append(
            (
                "SEV_LOW_CONFIDENCE_CAP",
                "Low-confidence findings cannot exceed medium severity.",
            )
        )

    # 3. Speculative or minor framing in title/description
    combined_text = f"{finding.title} {finding.description}"
    if _MINOR_FRAMING_RE.search(combined_text) and sev.rank > Severity.LOW.rank:
        sev = _cap_severity(sev, Severity.LOW)
        reasons.append(
            (
                "SEV_MINOR_FRAMING_CAP",
                "Severity capped at low due to minor/informational framing in description.",
            )
        )
    elif _SPECULATIVE_FRAMING_RE.search(combined_text) and sev.rank > Severity.MEDIUM.rank:
        sev = _cap_severity(sev, Severity.MEDIUM)
        reasons.append(
            (
                "SEV_SPECULATIVE_FRAMING_CAP",
                "Severity capped at medium due to speculative framing in description.",
            )
        )

    # 4. Category-specific severity ceiling
    has_unconditional_fatal = bool(_UNCONDITIONAL_FATAL_RE.search(finding.evidence))
    category_ceiling = _CATEGORY_MAX_SEVERITY.get(finding.category)
    if category_ceiling is not None and not has_unconditional_fatal and sev.rank > category_ceiling.rank:
        sev = _cap_severity(sev, category_ceiling)
        reasons.append(
            (
                "SEV_CATEGORY_CEILING",
                f"Category '{finding.category.value}' is capped at '{category_ceiling.value}' severity without unconditional crash evidence.",
            )
        )

    # 5. Critical severity confidence gate
    if sev == Severity.CRITICAL and conf != Confidence.HIGH and not has_unconditional_fatal:
        sev = Severity.HIGH
        reasons.append(
            (
                "SEV_CRITICAL_CONFIDENCE_GATE",
                "Critical severity requires high confidence or unconditional fatal crash evidence.",
            )
        )

    if sev == orig_sev and conf == orig_conf:
        return finding, None

    rule_id = reasons[0][0] if reasons else "SEV_NORMALIZED"
    explanation = "; ".join(r[1] for r in reasons) if reasons else "Normalized severity/confidence."

    normalized_finding = finding.model_copy(
        update={
            "severity": sev,
            "confidence": conf,
        }
    )
    adjustment = NormalizationAdjustment(
        finding_id=finding.id,
        original_severity=orig_sev,
        normalized_severity=sev,
        original_confidence=orig_conf,
        normalized_confidence=conf,
        rule_id=rule_id,
        reason=explanation,
    )
    return normalized_finding, adjustment


def normalize_findings(
    findings: list[Finding],
    assessments: dict[str, EvidenceAssessment] | None = None,
) -> tuple[list[Finding], list[NormalizationAdjustment]]:
    """Normalize severity and confidence across a list of findings deterministically."""
    normalized: list[Finding] = []
    adjustments: list[NormalizationAdjustment] = []
    assessment_map = assessments or {}

    for finding in findings:
        assessment = assessment_map.get(finding.id)
        norm_finding, adj = normalize_finding_severity(finding, assessment=assessment)
        normalized.append(norm_finding)
        if adj is not None:
            adjustments.append(adj)

    return normalized, adjustments
