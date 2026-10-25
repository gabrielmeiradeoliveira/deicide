"""Command-line interface for DEICIDE."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from deicide import __version__
from deicide.core.config import DeicideConfig, OutputFormat
from deicide.core.errors import DeicideError, ExitCode
from deicide.core.orchestrator import AnalysisOrchestrator
from deicide.core.registry import ModuleRegistry
from deicide.evaluation.runner import evaluate_corpus, load_fixtures_from_file
from deicide.modules.code_review import CodeReviewModule
from deicide.providers.factory import create_provider
from deicide.quota.store import UsageStore
from deicide.reporting.formatter import (
    format_json_report,
    format_terminal_report,
    format_usage_json,
    format_usage_terminal,
)

app = typer.Typer(
    name="deicide",
    help="DEICIDE: Open-source, modular, AI-powered software assurance and code review CLI.",
    add_completion=False,
    no_args_is_help=True,
)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"deicide {__version__}")
        raise typer.Exit(code=int(ExitCode.SUCCESS))


@app.callback()
def main_callback(
    version: Annotated[
        bool | None,
        typer.Option(
            "--version",
            "-V",
            help="Show the DEICIDE version and exit.",
            callback=_version_callback,
            is_eager=True,
        ),
    ] = None,
) -> None:
    """DEICIDE CLI root callback."""


def build_default_registry() -> ModuleRegistry:
    """Create a ModuleRegistry populated with default DEICIDE modules."""
    registry = ModuleRegistry()
    registry.register(CodeReviewModule())
    return registry


@app.command("review")
def review_command(
    base: Annotated[
        str | None,
        typer.Option(
            "--base",
            "-b",
            help="Compare working tree against a specific Git revision (e.g., HEAD~1, main).",
        ),
    ] = None,
    provider: Annotated[
        str | None,
        typer.Option(
            "--provider",
            "-p",
            help="AI provider to use ('nvidia' or 'fake'). Defaults to DEICIDE_PROVIDER or 'nvidia'.",
        ),
    ] = None,
    model: Annotated[
        str | None,
        typer.Option(
            "--model",
            "-m",
            help="Model identifier override for the selected provider.",
        ),
    ] = None,
    output_format: Annotated[
        str,
        typer.Option(
            "--format",
            "-f",
            help="Report output format: 'terminal' or 'json'.",
        ),
    ] = "terminal",
    repo: Annotated[
        Path | None,
        typer.Option(
            "--repo",
            "-r",
            help="Path to the target Git repository (defaults to current working directory).",
        ),
    ] = None,
    max_tokens: Annotated[
        int | None,
        typer.Option(
            "--max-tokens",
            help="Maximum total token budget for this analysis run.",
        ),
    ] = None,
    max_requests: Annotated[
        int | None,
        typer.Option(
            "--max-requests",
            help="Maximum API requests permitted during this analysis run.",
        ),
    ] = None,
    max_retries: Annotated[
        int | None,
        typer.Option(
            "--max-retries",
            help="Maximum automatic retries on transient HTTP 429/5xx or timeout (default: 0).",
        ),
    ] = None,
    timeout: Annotated[
        float | None,
        typer.Option(
            "--timeout",
            help="Per-request timeout in seconds.",
        ),
    ] = None,
    allow_external: Annotated[
        bool | None,
        typer.Option(
            "--allow-external/--no-allow-external",
            help="Explicitly authorize sending repository diffs to an external AI provider.",
        ),
    ] = None,
    db_path: Annotated[
        Path | None,
        typer.Option(
            "--db-path",
            help="Override path to the persistent SQLite usage database.",
        ),
    ] = None,
    context: Annotated[
        bool | None,
        typer.Option(
            "--context/--no-context",
            help="Enable or disable AST-based cross-file context engine (default: enabled).",
        ),
    ] = None,
    context_tokens: Annotated[
        int | None,
        typer.Option(
            "--context-tokens",
            help="Maximum token budget allocated for supplementary repository context.",
        ),
    ] = None,
    context_files: Annotated[
        int | None,
        typer.Option(
            "--context-files",
            help="Maximum number of supplementary context files to select (default: 8).",
        ),
    ] = None,
    context_snippets: Annotated[
        int | None,
        typer.Option(
            "--context-snippets",
            help="Maximum number of supplementary code snippets to select (default: 16).",
        ),
    ] = None,
) -> None:
    """Analyze local Git changes for correctness bugs."""
    try:
        config = DeicideConfig.from_env_and_args(
            repo_path=repo,
            base_revision=base,
            provider=provider,
            model=model,
            output_format=output_format,
            allow_external_code=allow_external,
            max_tokens=max_tokens,
            max_requests=max_requests,
            max_retries=max_retries,
            timeout_seconds=timeout,
            usage_db_path=db_path,
            context_enabled=context,
            context_max_tokens=context_tokens,
            context_max_files=context_files,
            context_max_snippets=context_snippets,
        )
        llm_provider = create_provider(config)
        registry = build_default_registry()
        orchestrator = AnalysisOrchestrator(
            config=config,
            provider=llm_provider,
            registry=registry,
        )
        report = orchestrator.run()

        if config.output_format == OutputFormat.JSON:
            typer.echo(format_json_report(report))
        else:
            typer.echo(format_terminal_report(report))

        raise typer.Exit(code=int(ExitCode.SUCCESS))

    except DeicideError as exc:
        typer.echo(f"Error: {exc.message}", err=True)
        raise typer.Exit(code=int(exc.exit_code)) from exc
    except typer.Exit:
        raise
    except Exception as exc:
        typer.echo(f"Unexpected error: {exc}", err=True)
        raise typer.Exit(code=int(ExitCode.GENERAL_ERROR)) from exc


@app.command("usage")
def usage_command(
    provider: Annotated[
        str | None,
        typer.Option(
            "--provider",
            "-p",
            help="AI provider to inspect ('nvidia' or 'fake'). Defaults to DEICIDE_PROVIDER or 'nvidia'.",
        ),
    ] = None,
    model: Annotated[
        str | None,
        typer.Option(
            "--model",
            "-m",
            help="Model identifier override.",
        ),
    ] = None,
    output_format: Annotated[
        str,
        typer.Option(
            "--format",
            "-f",
            help="Report output format: 'terminal' or 'json'.",
        ),
    ] = "terminal",
    scope: Annotated[
        str | None,
        typer.Option(
            "--scope",
            help="Quota scope override.",
        ),
    ] = None,
    db_path: Annotated[
        Path | None,
        typer.Option(
            "--db-path",
            help="Override path to the persistent SQLite usage database.",
        ),
    ] = None,
) -> None:
    """Display locally tracked persistent API usage and configured rate/daily limits."""
    try:
        config = DeicideConfig.from_env_and_args(
            provider=provider,
            model=model,
            output_format=output_format,
            quota_scope=scope,
            usage_db_path=db_path,
        )
        store = UsageStore(
            db_path=config.resolved_usage_db_path,
            reservation_ttl_seconds=config.budget.reservation_ttl_seconds,
        )
        snapshot = store.get_snapshot(
            provider=config.provider,
            model=config.resolved_model,
            quota_scope=config.resolved_quota_scope,
            quota_enabled=config.budget.quota_enabled,
            rate_limit_rpm=config.budget.rate_limit_rpm,
            rate_limit_tpm=config.budget.rate_limit_tpm,
            rate_limit_rpd=config.budget.rate_limit_rpd,
        )

        if config.output_format == OutputFormat.JSON:
            typer.echo(format_usage_json(snapshot))
        else:
            typer.echo(format_usage_terminal(snapshot))

        raise typer.Exit(code=int(ExitCode.SUCCESS))
    except DeicideError as exc:
        typer.echo(f"Error: {exc.message}", err=True)
        raise typer.Exit(code=int(exc.exit_code)) from exc
    except typer.Exit:
        raise
    except Exception as exc:
        typer.echo(f"Unexpected error: {exc}", err=True)
        raise typer.Exit(code=int(ExitCode.GENERAL_ERROR)) from exc


@app.command("evaluate")
def evaluate_command(
    fixtures_file: Annotated[
        Path,
        typer.Option(
            "--fixtures",
            help="Path to the evaluation fixtures JSON file.",
        ),
    ] = Path("evaluations/fixtures/code_review_corpus.json"),
    provider: Annotated[
        str,
        typer.Option(
            "--provider",
            "-p",
            help="Provider to evaluate ('fake' or 'nvidia').",
        ),
    ] = "fake",
    model: Annotated[
        str | None,
        typer.Option(
            "--model",
            "-m",
            help="Model identifier override for provider evaluation.",
        ),
    ] = None,
    allow_external: Annotated[
        bool,
        typer.Option(
            "--allow-external/--no-allow-external",
            help="Explicitly authorize sending evaluation fixtures to an external AI provider.",
        ),
    ] = False,
    baseline: Annotated[
        bool,
        typer.Option(
            "--baseline",
            help="Evaluate using pre-MVP 1.3 baseline validator instead of Finding Intelligence Engine.",
        ),
    ] = False,
    end_to_end: Annotated[
        bool,
        typer.Option(
            "--end-to-end",
            "--e2e",
            help="Run full end-to-end review pipeline with configured provider instead of post-processing candidates.",
        ),
    ] = False,
    strict_severity: Annotated[
        bool,
        typer.Option(
            "--strict-severity",
            help="Require finding severity to strictly match ground truth severity for a match.",
        ),
    ] = False,
    max_cases: Annotated[
        int | None,
        typer.Option(
            "--max-cases",
            help="Maximum number of evaluation cases to execute from the fixture dataset.",
        ),
    ] = None,
    max_requests: Annotated[
        int | None,
        typer.Option(
            "--max-requests",
            help="Maximum total API requests allowed across the evaluation run.",
        ),
    ] = None,
    max_tokens: Annotated[
        int | None,
        typer.Option(
            "--max-tokens",
            help="Maximum total tokens allowed across the evaluation run.",
        ),
    ] = None,
    timeout: Annotated[
        float | None,
        typer.Option(
            "--timeout",
            help="Per-request timeout in seconds for provider evaluation.",
        ),
    ] = None,
    max_retries: Annotated[
        int | None,
        typer.Option(
            "--max-retries",
            help="Maximum automatic retries on transient rate-limits or server errors.",
        ),
    ] = None,
    skip_unsuitable: Annotated[
        bool,
        typer.Option(
            "--skip-unsuitable",
            help="Skip fixtures marked unsuitable for live defect detection benchmarking.",
        ),
    ] = False,
    output_format: Annotated[
        str,
        typer.Option(
            "--format",
            "-f",
            help="Output format: 'terminal' or 'json'.",
        ),
    ] = "terminal",
) -> None:
    """Run the curated evaluation benchmark and report precision, recall, latency, and tokens."""
    try:
        config = DeicideConfig.from_env_and_args(
            provider=provider,
            model=model,
            allow_external_code=allow_external,
            max_tokens=max_tokens,
            max_requests=max_requests,
            max_retries=max_retries,
            timeout_seconds=timeout,
            output_format=output_format,
        )
        if config.provider != "fake":
            config.validate_provider_credentials()

        llm_provider = create_provider(config)
        if not fixtures_file.exists():
            typer.echo(f"Error: Evaluation fixtures file not found: '{fixtures_file}'", err=True)
            raise typer.Exit(code=int(ExitCode.CONFIGURATION_ERROR))

        fixtures = load_fixtures_from_file(fixtures_file)
        eval_budget = (
            config.budget
            if (max_requests is not None or max_tokens is not None or timeout is not None or max_retries is not None)
            else None
        )
        summary = evaluate_corpus(
            fixtures,
            provider=llm_provider,
            budget=eval_budget,
            use_baseline=baseline,
            force_end_to_end=end_to_end,
            strict_severity=strict_severity,
            max_cases=max_cases,
            skip_unsuitable=skip_unsuitable,
        )

        if config.output_format == OutputFormat.JSON:
            typer.echo(summary.model_dump_json(indent=2))
        else:
            engine_label = "Baseline Validator" if baseline else "Finding Intelligence Engine"
            if end_to_end:
                summary_title = f"DEICIDE Evaluation Summary (End-to-End {llm_provider.provider_name} + {engine_label})"
            else:
                summary_title = f"DEICIDE Evaluation Summary ({engine_label})"
            typer.echo("=" * 72)
            typer.echo(summary_title)
            typer.echo("=" * 72)
            typer.echo(
                f"Dataset Size        : {summary.dataset_size} "
                f"(positive={summary.positive_cases}, negative={summary.negative_cases})"
            )
            typer.echo(
                f"Provenance          : verified={summary.verified_cases}, "
                f"reference_modeled={summary.reference_modeled_cases}, "
                f"unverified={summary.unverified_cases}"
            )
            typer.echo(
                f"Counts              : TP={summary.true_positives}, "
                f"FP={summary.false_positives}, FN={summary.false_negatives}"
            )
            prec_str = f"{summary.precision:.2%}" if summary.precision is not None else "unavailable"
            rec_str = f"{summary.recall:.2%}" if summary.recall is not None else "unavailable"
            fpr_str = f"{summary.false_positive_rate:.2%}" if summary.false_positive_rate is not None else "unavailable"
            typer.echo(f"Precision           : {prec_str}")
            typer.echo(f"Recall              : {rec_str}")
            typer.echo(f"False Positive Rate : {fpr_str}")
            typer.echo(f"Duplicates Reduced  : {summary.duplicates_reduced}")
            typer.echo(f"Legitimate Rejected : {summary.legitimate_rejected}")
            if summary.severity_mismatches > 0:
                typer.echo(f"Severity Mismatches : {summary.severity_mismatches}")
            if summary.successful_cases > 0:
                typer.echo(f"Successful Cases    : {summary.successful_cases}")
            if summary.api_failed_cases > 0:
                typer.echo(f"API Failed Cases    : {summary.api_failed_cases}")
            if summary.budget_exhausted_cases > 0:
                typer.echo(f"Budget Exhausted    : {summary.budget_exhausted_cases}")
            if summary.skipped_cases > 0:
                typer.echo(f"Skipped Cases       : {summary.skipped_cases}")
            elif summary.failed_cases > 0:
                typer.echo(f"Failed Cases        : {summary.failed_cases}")
            if summary.budget_exhausted:
                typer.echo("Budget Status       : STOPPED (Budget / Quota Exhausted)")
            typer.echo(
                f"Latency             : total={summary.total_latency_ms} ms, avg={summary.average_latency_ms:.2f} ms"
            )
            typer.echo(
                f"Provider Tokens     : {summary.usage.total_tokens} "
                f"(prompt={summary.usage.prompt_tokens}, completion={summary.usage.completion_tokens})"
            )
            typer.echo("Local Engine Tokens : 0 (pure deterministic static analysis)")
            typer.echo("=" * 72)
        raise typer.Exit(code=int(ExitCode.SUCCESS))
    except DeicideError as exc:
        typer.echo(f"Error: {exc.message}", err=True)
        raise typer.Exit(code=int(exc.exit_code)) from exc


def main() -> None:
    """Entrypoint for the `deicide` console script."""
    app()
