"""Semantic deduplication of findings referring to the same underlying defect."""

from __future__ import annotations

import re

from deicide.core.models import Finding, FindingCategory
from deicide.intelligence.evidence_validator import (
    normalize_ws,
    strip_evidence_line_prefixes,
)
from deicide.intelligence.models import (
    ChangeRelevance,
    DeduplicationGroup,
    EvidenceAssessment,
    EvidenceQuality,
)
from deicide.intelligence.prioritizer import prioritize_findings

_CODE_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+(?:\.\d+)?|==|!=|<=|>=|//|\*\*|[+\-*/%<>=]")
_WORD_RE = re.compile(r"[a-z0-9_]+")

_STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "because",
        "by",
        "can",
        "code",
        "could",
        "causes",
        "cause",
        "due",
        "error",
        "execution",
        "for",
        "from",
        "function",
        "has",
        "have",
        "if",
        "in",
        "into",
        "is",
        "issue",
        "it",
        "its",
        "lead",
        "leads",
        "line",
        "lines",
        "may",
        "method",
        "might",
        "not",
        "of",
        "on",
        "or",
        "potential",
        "program",
        "raises",
        "raise",
        "result",
        "results",
        "runtime",
        "should",
        "that",
        "the",
        "their",
        "this",
        "to",
        "unexpected",
        "used",
        "using",
        "value",
        "values",
        "when",
        "which",
        "will",
        "with",
        "without",
    }
)

_COMPATIBLE_CATEGORY_FAMILIES: tuple[frozenset[FindingCategory], ...] = (
    frozenset(
        {
            FindingCategory.CORRECTNESS,
            FindingCategory.LOGIC_ERROR,
            FindingCategory.BOUNDARY_CONDITION,
        }
    ),
    frozenset(
        {
            FindingCategory.NULL_HANDLING,
            FindingCategory.TYPE_MISMATCH,
            FindingCategory.CORRECTNESS,
        }
    ),
    frozenset(
        {
            FindingCategory.ERROR_HANDLING,
            FindingCategory.RESOURCE_MANAGEMENT,
        }
    ),
    frozenset(
        {
            FindingCategory.STATE_MUTATION,
            FindingCategory.CONCURRENCY,
        }
    ),
)

_DEFECT_CONCEPT_PATTERNS: dict[str, re.Pattern[str]] = {
    "zero_division": re.compile(
        r"\b(zerodivisionerror|divide[ds]?\s+by\s+zero|division\s+by\s+zero|modulo\s+by\s+zero|"
        r"dividing\s+by\s+len|empty\s+collection|empty\s+list|divisor\s+is\s+zero)\b|(?<!/)/(?!/)\s*0\b",
        re.IGNORECASE,
    ),
    "index_bounds": re.compile(
        r"\b(indexerror|out\s+of\s+bounds|off[\s\-]+by[\s\-]+one|upper\s+bound)\b|"
        r"range\(\s*(?:0\s*,\s*)?len\([^)]*\)\s*\+\s*1\s*\)",
        re.IGNORECASE,
    ),
    "null_deref": re.compile(
        r"\b(nonetype|null\s+dereference|none\s+dereference|unchecked\s+none|optional\s+none|"
        r"attributeerror\s+on\s+none|is\s+none\s+check)\b",
        re.IGNORECASE,
    ),
    "mutable_default": re.compile(
        r"\b(mutable\s+default|mutable\s+literal|default\s+argument|default\s+parameter|"
        r"shared\s+across\s+(?:function\s+)?calls|persist[s]?\s+across\s+calls)\b",
        re.IGNORECASE,
    ),
    "silent_except": re.compile(
        r"\b(silent\s+exception|exception\s+swallow\w*|swallow\w*\s+exception|bare\s+except|"
        r"ignoring\s+exception|masks\s+runtime)\b",
        re.IGNORECASE,
    ),
    "resource_leak": re.compile(
        r"\b(unclosed\s+file|context\s+manager|file\s+descriptor\w*|resource\s+leak|"
        r"without\s+close|with\s+open)\b",
        re.IGNORECASE,
    ),
    "identity_cmp": re.compile(
        r"\b(identity\s+comparison|object\s+identity|value\s+equality|used\s+instead\s+of\s+equality)\b",
        re.IGNORECASE,
    ),
    "race_deadlock": re.compile(
        r"\b(race\s+condition|deadlock|thread\s+safe\w*|shared\s+lock|unreleased\s+lock)\b",
        re.IGNORECASE,
    ),
}


def normalize_code_evidence(evidence: str) -> str:
    """Normalize evidence by stripping line prefixes, inline comments, and collapsing whitespace."""
    lines = strip_evidence_line_prefixes(evidence)
    cleaned_lines: list[str] = []
    for line in lines:
        code_only = line.split("#", 1)[0].strip()
        if code_only:
            cleaned_lines.append(normalize_ws(code_only))
        elif line.strip():
            cleaned_lines.append(normalize_ws(line))
    return " ".join(cleaned_lines)


def extract_code_tokens(normalized_evidence: str) -> set[str]:
    """Extract lexical code tokens (identifiers, numbers, operators) from normalized evidence."""
    return {tok.lower() for tok in _CODE_TOKEN_RE.findall(normalized_evidence) if tok.strip()}


def _stem_token(word: str) -> str:
    if len(word) > 5 and word.endswith("ing"):
        return word[:-3]
    if len(word) > 4 and word.endswith("ed"):
        return word[:-2]
    if len(word) > 4 and word.endswith("es"):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def extract_semantic_tokens(text: str) -> set[str]:
    """Extract normalized, stemmed content tokens from finding title/description text."""
    raw_words = _WORD_RE.findall(text.lower())
    tokens: set[str] = set()
    for word in raw_words:
        if word in _STOPWORDS or len(word) < 2:
            continue
        stemmed = _stem_token(word)
        if stemmed not in _STOPWORDS and len(stemmed) >= 2:
            tokens.add(stemmed)
    return tokens


def extract_defect_concepts(title: str, description: str, evidence: str = "") -> set[str]:
    """Identify canonical defect concepts from finding title and description (falling back to evidence)."""
    prose = f"{title}\n{description}"
    matched: set[str] = set()
    for concept_id, pattern in _DEFECT_CONCEPT_PATTERNS.items():
        if pattern.search(prose):
            matched.add(concept_id)
    if not matched and evidence:
        for concept_id, pattern in _DEFECT_CONCEPT_PATTERNS.items():
            if pattern.search(evidence):
                matched.add(concept_id)
    return matched


def token_jaccard(tokens_a: set[str], tokens_b: set[str]) -> float:
    """Compute deterministic Jaccard similarity between two token sets."""
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = len(tokens_a & tokens_b)
    union = len(tokens_a | tokens_b)
    return intersection / union if union > 0 else 0.0


def are_categories_compatible(cat_a: FindingCategory, cat_b: FindingCategory) -> bool:
    """Return True if two FindingCategory values are identical or belong to the same semantic family."""
    if cat_a == cat_b:
        return True
    return any(cat_a in family and cat_b in family for family in _COMPATIBLE_CATEGORY_FAMILIES)


def _line_gap(a: Finding, b: Finding) -> int:
    """Return 0 if line ranges overlap, 1 if adjacent, or the positive line distance."""
    return max(
        0,
        max(a.line_range.start_line, b.line_range.start_line) - min(a.line_range.end_line, b.line_range.end_line),
    )


def are_findings_duplicates(
    a: Finding,
    b: Finding,
    *,
    max_line_proximity: int = 3,
) -> tuple[bool, float, str]:
    """Determine if two findings refer to the same underlying issue.

    Returns (is_duplicate, similarity_score, reason).
    Avoids merging distinct bugs that happen to occur on nearby or identical lines.
    """
    if a.module_id != b.module_id or a.file_path != b.file_path:
        return False, 0.0, "different_file_or_module"

    if a.id == b.id:
        return True, 1.0, "identical_finding_id"

    gap = _line_gap(a, b)
    if gap > max_line_proximity:
        return False, 0.0, "beyond_line_proximity"

    ev_a = normalize_code_evidence(a.evidence)
    ev_b = normalize_code_evidence(b.evidence)
    exact_ev = bool(ev_a and ev_b and ev_a == ev_b)
    contained_ev = bool(
        ev_a and ev_b and not exact_ev and (ev_a in ev_b or ev_b in ev_a) and min(len(ev_a), len(ev_b)) >= 6
    )
    ev_jaccard = token_jaccard(extract_code_tokens(ev_a), extract_code_tokens(ev_b))
    ev_score = 1.0 if exact_ev else (0.85 if contained_ev else ev_jaccard)

    same_cat = a.category == b.category
    compat_cat = same_cat or are_categories_compatible(a.category, b.category)

    concepts_a = extract_defect_concepts(a.title, a.description, ev_a)
    concepts_b = extract_defect_concepts(b.title, b.description, ev_b)
    shared_concepts = concepts_a & concepts_b
    disjoint_concepts = bool(concepts_a and concepts_b and not shared_concepts)

    norm_title_a = normalize_ws(a.title.lower())
    norm_title_b = normalize_ws(b.title.lower())
    exact_title = bool(norm_title_a and norm_title_a == norm_title_b)

    title_sim = token_jaccard(extract_semantic_tokens(a.title), extract_semantic_tokens(b.title))
    desc_sim = token_jaccard(
        extract_semantic_tokens(f"{a.title} {a.description}"),
        extract_semantic_tokens(f"{b.title} {b.description}"),
    )

    # Anti-collision Guard 1: Disjoint canonical defect concepts (e.g., mutable default vs division by zero)
    if disjoint_concepts and not exact_title and desc_sim < 0.5:
        return False, 0.1, "disjoint_defect_concepts"

    # Anti-collision Guard 2: Non-overlapping lines with distinct evidence and no shared defect concept
    if gap > 0 and ev_score < 0.5 and not exact_title and desc_sim < 0.45 and not (same_cat and bool(shared_concepts)):
        return False, 0.15, "distinct_adjacent_lines_and_evidence"

    # Anti-collision Guard 3: Same line, same category, but completely different code evidence and low text similarity
    if gap == 0 and ev_score < 0.35 and not exact_title and desc_sim < 0.35 and not shared_concepts:
        return False, 0.2, "distinct_evidence_and_description"

    # Duplicate Rule 1: Identical normalized title within proximity
    if gap <= 2 and exact_title:
        return True, 0.95, "identical_title_in_proximity"

    # Duplicate Rule 2: Matching or contained evidence with compatible category or shared defect concept
    if gap <= 2 and (exact_ev or contained_ev) and (compat_cat or bool(shared_concepts) or desc_sim >= 0.3):
        return True, 0.9, "matching_evidence_and_issue"

    # Duplicate Rule 3: Overlapping lines with same category and strong evidence/semantic match
    if gap == 0 and same_cat and (ev_score >= 0.5 or desc_sim >= 0.35 or bool(shared_concepts)):
        return True, 0.85, "overlapping_line_same_category"

    # Duplicate Rule 4: Proximate lines sharing the same canonical defect concept and compatible category
    if gap <= max_line_proximity and compat_cat and bool(shared_concepts) and (ev_score >= 0.4 or desc_sim >= 0.3):
        return True, 0.8, "proximate_shared_defect_concept"

    # Duplicate Rule 5: High weighted multi-signal similarity
    proximity_score = 1.0 if gap == 0 else (0.8 if gap == 1 else (0.6 if gap == 2 else 0.4))
    cat_score = 1.0 if same_cat else (0.7 if compat_cat else 0.0)
    semantic_score = max(title_sim, desc_sim, 0.8 if shared_concepts else 0.0)
    composite = 0.35 * ev_score + 0.30 * semantic_score + 0.20 * cat_score + 0.15 * proximity_score

    if composite >= 0.72 and compat_cat:
        return True, round(composite, 4), "high_composite_similarity"

    return False, round(composite, 4), "distinct_findings"


def _survivor_sort_key(
    finding: Finding,
    assessment: EvidenceAssessment | None,
) -> tuple[int, int, int, int, int, int, int, str]:
    """Rank candidates for cluster survivorship (highest quality first)."""
    quality = assessment.quality if assessment is not None else EvidenceQuality.EXACT
    relevance = assessment.change_relevance if assessment is not None else ChangeRelevance.DIRECT_ADDED_OR_MODIFIED
    return (
        -finding.severity.rank,
        -finding.confidence.rank,
        -quality.rank,
        -relevance.rank,
        -(1 if finding.suggested_remediation else 0),
        -len(finding.description),
        finding.line_range.start_line,
        finding.id,
    )


def deduplicate_findings_semantic(
    findings: list[Finding],
    assessments: dict[str, EvidenceAssessment] | None = None,
) -> tuple[list[Finding], int, list[DeduplicationGroup]]:
    """Deduplicate identical or semantically equivalent findings, keeping the highest-quality entry.

    Returns (prioritized_kept_findings, deduplicated_count, deduplication_groups).
    """
    if not findings:
        return [], 0, []

    assessment_map = assessments or {}
    sorted_candidates = sorted(
        findings,
        key=lambda f: _survivor_sort_key(f, assessment_map.get(f.id)),
    )

    kept: list[Finding] = []
    groups_by_survivor: dict[str, DeduplicationGroup] = {}
    removed_count = 0

    for candidate in sorted_candidates:
        matched_idx: int | None = None
        match_score = 0.0
        match_reason = ""

        for idx, existing in enumerate(kept):
            is_dup, score, reason = are_findings_duplicates(existing, candidate)
            if is_dup:
                matched_idx = idx
                match_score = score
                match_reason = reason
                break

        if matched_idx is not None:
            removed_count += 1
            survivor = kept[matched_idx]
            # Enrich survivor with remediation if survivor lacked one and duplicate has one
            if not survivor.suggested_remediation and candidate.suggested_remediation:
                survivor = survivor.model_copy(update={"suggested_remediation": candidate.suggested_remediation})
                kept[matched_idx] = survivor

            if survivor.id in groups_by_survivor:
                prev = groups_by_survivor[survivor.id]
                groups_by_survivor[survivor.id] = DeduplicationGroup(
                    kept_finding_id=survivor.id,
                    merged_finding_ids=[*prev.merged_finding_ids, candidate.id],
                    reason=prev.reason,
                    similarity_score=max(prev.similarity_score, match_score),
                )
            else:
                groups_by_survivor[survivor.id] = DeduplicationGroup(
                    kept_finding_id=survivor.id,
                    merged_finding_ids=[candidate.id],
                    reason=match_reason,
                    similarity_score=match_score,
                )
        else:
            kept.append(candidate)

    prioritized = prioritize_findings(kept, assessments=assessment_map)
    groups = [groups_by_survivor[f.id] for f in prioritized if f.id in groups_by_survivor]
    return prioritized, removed_count, groups
