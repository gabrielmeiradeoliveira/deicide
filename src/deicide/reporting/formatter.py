"""Human-readable terminal and structured JSON report formatters."""

from __future__ import annotations

from deicide.core.models import AnalysisReport, Finding
from deicide.quota.store import QuotaSnapshot


def _format_line_span(finding: Finding) -> str:
    start = finding.line_range.start_line
    end = finding.line_range.end_line
    if start == end:
        return f"{finding.file_path}:{start}"
    return f"{finding.file_path}:{start}-{end}"


def format_terminal_report(report: AnalysisReport) -> str:
    """Format an AnalysisReport for human-readable terminal display."""
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append(f"DEICIDE Code Review Report ({report.status.upper()})")
    lines.append("=" * 72)
    if report.status == "incomplete":
        lines.append("NOTICE     : Analysis is INCOMPLETE due to context or execution budget limits.")
    lines.append(f"Repository : {report.repository_root}")
    lines.append(f"Target     : {report.target_description}")
    lines.append(f"Modules    : {', '.join(report.modules_executed) or 'none'}")
    lines.append(
        f"Provider   : {report.provider.provider_name} ({report.provider.model_name}) "
        f"[requests={report.provider.requests_made}, retries={report.provider.retries_performed}]"
    )
    est_label = " (estimated)" if report.usage.is_estimated else ""
    lines.append(
        f"Tokens     : {report.usage.total_tokens}{est_label} "
        f"(prompt={report.usage.prompt_tokens}, completion={report.usage.completion_tokens})"
    )
    lines.append(f"Duration   : {report.duration_ms} ms")
    lines.append(f"Files      : {len(report.files_analyzed)} analyzed")
    if report.context_summary:
        lines.append("-" * 72)
        lines.append("Context Engine:")
        lines.append(f"  Status                  : {report.context_summary.status}")
        lines.append(f"  Indexed Python Files    : {report.context_summary.indexed_files}")
        lines.append(f"  Selected Snippets       : {report.context_summary.selected_snippets}")
        lines.append(f"  Related Files           : {report.context_summary.related_files}")
        lines.append(f"  Estimated Context Tokens: {report.context_summary.estimated_tokens}")
        lines.append(f"  Omitted Candidates      : {report.context_summary.omitted_candidates}")
        lines.append(f"  Fallback Used           : {'yes' if report.context_summary.fallback else 'no'}")
    lines.append("-" * 72)

    if report.status == "empty_diff":
        lines.append("No reviewable Git changes detected.")
    elif not report.findings:
        lines.append("No validated correctness findings detected.")
    else:
        lines.append(f"Validated Findings ({len(report.findings)}):")
        lines.append("")
        for idx, finding in enumerate(report.findings, start=1):
            loc = _format_line_span(finding)
            sev = finding.severity.value.upper()
            conf = finding.confidence.value
            cat = finding.category.value
            lines.append(f"[{idx}] [{sev}] {finding.title}")
            lines.append(f"    ID          : {finding.id} (module: {finding.module_id})")
            lines.append(f"    Location    : {loc}")
            lines.append(f"    Category    : {cat} | Confidence: {conf}")
            lines.append(f"    Description : {finding.description}")
            lines.append(f"    Evidence    : {finding.evidence}")
            if finding.suggested_remediation:
                lines.append(f"    Remediation : {finding.suggested_remediation}")
            lines.append("")

    if report.rejected_findings_count > 0 or report.deduplicated_findings_count > 0:
        lines.append(
            f"Quality Gates: rejected={report.rejected_findings_count}, "
            f"deduplicated={report.deduplicated_findings_count}"
        )

    if report.warnings:
        lines.append("-" * 72)
        lines.append(f"Warnings ({len(report.warnings)}):")
        for warning in report.warnings:
            prefix = f"[{warning.code}]"
            if warning.file_path:
                prefix += f" ({warning.file_path})"
            lines.append(f"  - {prefix} {warning.message}")

    lines.append("=" * 72)
    return "\n".join(lines)


def format_json_report(report: AnalysisReport) -> str:
    """Serialize an AnalysisReport into strict, machine-readable JSON with no extra output."""
    return report.model_dump_json(indent=2)


def format_usage_terminal(snapshot: QuotaSnapshot) -> str:
    """Format a QuotaSnapshot for human-readable terminal display (`deicide usage`)."""
    rpd_str = str(snapshot.configured_rpd_limit) if snapshot.configured_rpd_limit is not None else "unlimited"
    rem_rpd_str = (
        f"{snapshot.estimated_remaining_daily_requests} (locally estimated)"
        if snapshot.estimated_remaining_daily_requests is not None
        else "unlimited"
    )
    rpm_str = str(snapshot.configured_rpm_limit) if snapshot.configured_rpm_limit is not None else "unlimited"
    tpm_str = str(snapshot.configured_tpm_limit) if snapshot.configured_tpm_limit is not None else "unlimited"

    lines: list[str] = [
        "=" * 72,
        "DEICIDE Persistent API Usage & Quota Report",
        "=" * 72,
        f"Provider & Model              : {snapshot.provider} ({snapshot.model})",
        f"Quota Scope                   : {snapshot.quota_scope}",
        f"Quota Enforcement             : {'enabled' if snapshot.quota_enabled else 'disabled'}",
        f"UTC Quota Day                 : {snapshot.utc_date} (resets at {snapshot.daily_reset_utc})",
        f"Database Path                 : {snapshot.database_path}",
        "-" * 72,
        f"Requests Recorded Today       : {snapshot.requests_today} "
        f"(succeeded={snapshot.succeeded_today}, failed={snapshot.failed_today}, "
        f"rate_limited={snapshot.rate_limited_today}, uncertain={snapshot.uncertain_today}, "
        f"active_reserved={snapshot.active_reservations_today})",
        f"Configured Daily Limit (RPD)  : {rpd_str}",
        f"Estimated Remaining Daily Req : {rem_rpd_str}",
        "-" * 72,
        f"Requests in Last 60 Seconds   : {snapshot.requests_last_60s}",
        f"Configured RPM Limit          : {rpm_str}",
        f"Input Tokens in Last 60s      : {snapshot.input_tokens_last_60s}",
        f"Configured TPM Limit          : {tpm_str}",
        "-" * 72,
        f"Input Tokens Today            : {snapshot.input_tokens_today}",
        f"Output Tokens Today           : {snapshot.output_tokens_today}",
        f"Total Tokens Today            : {snapshot.total_tokens_today}",
        f"Accounting Source             : {snapshot.token_accounting_source} "
        f"(locally_estimated={snapshot.is_locally_estimated})",
        "-" * 72,
        f"Note: {snapshot.disclaimer}",
        "=" * 72,
    ]
    return "\n".join(lines)


def format_usage_json(snapshot: QuotaSnapshot) -> str:
    """Serialize a QuotaSnapshot into valid machine-readable JSON."""
    return snapshot.model_dump_json(indent=2)
