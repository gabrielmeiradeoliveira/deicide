"""Tests for configuration validation, finding validation, evidence checks, and deduplication."""

from __future__ import annotations

from pathlib import Path

import pytest

from deicide.core.config import DeicideConfig, load_dotenv_file
from deicide.core.errors import (
    ConfigurationError,
    MissingCredentialsError,
    UnsupportedProviderError,
)
from deicide.core.models import (
    CandidateFinding,
    Confidence,
    FindingCategory,
    Severity,
)
from deicide.core.validation import (
    compute_finding_id,
    deduplicate_findings,
    validate_candidates,
)
from deicide.git.analyzer import parse_unified_diff
from deicide.git.models import ContextSnippet, DiffBundle, FileContext


def test_config_validation_and_credentials() -> None:
    # Fake provider needs no credentials
    cfg_fake = DeicideConfig.from_env_and_args(provider="fake", environ={})
    cfg_fake.validate_provider_credentials()
    assert cfg_fake.provider == "fake"

    # Gemini is unsupported and raises UnsupportedProviderError
    with pytest.raises(UnsupportedProviderError, match="Unsupported provider 'gemini'"):
        DeicideConfig.from_env_and_args(provider="gemini", environ={})

    # NVIDIA without API key fails validation
    cfg_no_key = DeicideConfig.from_env_and_args(provider="nvidia", environ={})
    with pytest.raises(MissingCredentialsError, match="Missing API credentials"):
        cfg_no_key.validate_provider_credentials()

    # NVIDIA with API key but explicit DEICIDE_ALLOW_EXTERNAL_CODE=0 fails validation
    cfg_disallowed = DeicideConfig.from_env_and_args(
        provider="nvidia",
        environ={"NVIDIA_API_KEY": "test-key", "DEICIDE_ALLOW_EXTERNAL_CODE": "0"},
    )
    with pytest.raises(MissingCredentialsError, match="External code transmission"):
        cfg_disallowed.validate_provider_credentials()

    # Unsupported provider raises UnsupportedProviderError
    with pytest.raises(UnsupportedProviderError):
        DeicideConfig.from_env_and_args(provider="unknown_llm", environ={})

    # Leading hyphen in base revision is rejected
    with pytest.raises(ConfigurationError):
        DeicideConfig.from_env_and_args(provider="fake", base_revision="--malicious-flag", environ={})


def test_provider_cleanup_and_nvidia_defaults() -> None:
    from deicide.core.config import SUPPORTED_PROVIDERS

    # 1. Default provider is NVIDIA
    cfg = DeicideConfig.from_env_and_args(environ={})
    assert cfg.provider == "nvidia"

    # 2. Default model is z-ai/glm-5.3-flash
    assert cfg.resolved_model == "z-ai/glm-5.3-flash"

    # 3. Supported providers are exactly NVIDIA and Fake
    assert frozenset({"fake", "nvidia"}) == SUPPORTED_PROVIDERS

    # 4. Explicit Gemini selection is rejected with UnsupportedProviderError
    with pytest.raises(UnsupportedProviderError, match="Unsupported provider 'gemini'"):
        DeicideConfig.from_env_and_args(provider="gemini", environ={})

    # 5. NVIDIA credential resolution works with NVIDIA_API_KEY and DEICIDE_API_KEY
    cfg_nv = DeicideConfig.from_env_and_args(
        environ={"NVIDIA_API_KEY": "nv-key-123", "DEICIDE_ALLOW_EXTERNAL_CODE": "1"}
    )
    assert cfg_nv.api_key is not None
    assert cfg_nv.api_key.get_secret_value() == "nv-key-123"

    cfg_deicide_key = DeicideConfig.from_env_and_args(
        environ={"DEICIDE_API_KEY": "deicide-fallback-key", "DEICIDE_ALLOW_EXTERNAL_CODE": "1"}
    )
    assert cfg_deicide_key.api_key is not None
    assert cfg_deicide_key.api_key.get_secret_value() == "deicide-fallback-key"

    # 6. FakeProvider requires no credentials
    cfg_fake = DeicideConfig.from_env_and_args(provider="fake", environ={})
    cfg_fake.validate_provider_credentials()
    assert cfg_fake.provider == "fake"
    assert cfg_fake.resolved_model == "fake-deterministic-v1"

    # 7. No GeminiProvider import exists in deicide.providers
    import deicide.providers as prov_pkg

    assert not hasattr(prov_pkg, "GeminiProvider")
    with pytest.raises(ImportError):
        from deicide.providers.gemini import GeminiProvider  # type: ignore[import-not-found] # noqa: F401


def test_dotenv_loading_and_env_precedence(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# Comment line\n"
        'NVIDIA_API_KEY="dotenv-secret-key"\n'
        "DEICIDE_MODEL='custom-glm-model'\n"
        "export DEICIDE_MAX_TOKENS=32000 # inline comment\n",
        encoding="utf-8",
    )

    parsed = load_dotenv_file(env_file)
    assert parsed["NVIDIA_API_KEY"] == "dotenv-secret-key"
    assert parsed["DEICIDE_MODEL"] == "custom-glm-model"
    assert parsed["DEICIDE_MAX_TOKENS"] == "32000"

    cfg_from_dotenv = DeicideConfig.from_env_and_args(
        dotenv_path=env_file,
        environ={"DEICIDE_ALLOW_EXTERNAL_CODE": "1"},
    )
    cfg_from_dotenv.validate_provider_credentials()
    assert cfg_from_dotenv.api_key is not None
    assert cfg_from_dotenv.api_key.get_secret_value() == "dotenv-secret-key"
    assert cfg_from_dotenv.resolved_model == "custom-glm-model"
    assert cfg_from_dotenv.budget.max_total_tokens == 32000

    # Explicit environment variable overrides .env value
    cfg_overridden = DeicideConfig.from_env_and_args(
        dotenv_path=env_file,
        environ={"NVIDIA_API_KEY": "os-env-override-key", "DEICIDE_ALLOW_EXTERNAL_CODE": "1"},
    )
    assert cfg_overridden.api_key is not None
    assert cfg_overridden.api_key.get_secret_value() == "os-env-override-key"


def test_finding_validation_rejects_hallucinations_and_accepts_valid() -> None:
    diff_text = (
        "diff --git a/src/service.py b/src/service.py\n"
        "--- a/src/service.py\n"
        "+++ b/src/service.py\n"
        "@@ -10,3 +10,4 @@\n"
        " def process(items: list[int]) -> float:\n"
        "+    return sum(items) / len(items)\n"
        "     # done\n"
    )
    parsed_files = parse_unified_diff(diff_text)
    bundle = DiffBundle(
        repository_root="/repo",
        base_revision="HEAD",
        target_description="test",
        files=parsed_files,
    )
    contexts = {
        "src/service.py": FileContext(
            file_path="src/service.py",
            total_lines=20,
            snippets=[
                ContextSnippet(
                    start_line=10,
                    end_line=12,
                    lines={
                        10: "def process(items: list[int]) -> float:",
                        11: "    return sum(items) / len(items)",
                        12: "    # done",
                    },
                )
            ],
        )
    }

    candidates = [
        # 1. Valid candidate
        CandidateFinding(
            category=FindingCategory.BOUNDARY_CONDITION,
            severity=Severity.HIGH,
            confidence=Confidence.HIGH,
            file_path="src/service.py",
            start_line=11,
            end_line=11,
            title="ZeroDivisionError on empty list",
            description="Dividing by len(items) raises ZeroDivisionError when items is empty.",
            evidence="return sum(items) / len(items)",
            suggested_remediation="Check if not items before dividing.",
        ),
        # 2. Invalid file (not in diff)
        CandidateFinding(
            category=FindingCategory.CORRECTNESS,
            severity=Severity.HIGH,
            confidence=Confidence.HIGH,
            file_path="src/other.py",
            start_line=1,
            end_line=1,
            title="Bug in untouched file",
            description="Should be rejected because file is not in diff.",
            evidence="return sum(items) / len(items)",
        ),
        # 3. Invalid line number (unchanged line 2 outside hunk)
        CandidateFinding(
            category=FindingCategory.CORRECTNESS,
            severity=Severity.MEDIUM,
            confidence=Confidence.MEDIUM,
            file_path="src/service.py",
            start_line=2,
            end_line=2,
            title="Unmapped line location",
            description="Should be rejected because line 2 was not changed.",
            evidence="return sum(items) / len(items)",
        ),
        # 4. Unsupported/hallucinated evidence
        CandidateFinding(
            category=FindingCategory.CORRECTNESS,
            severity=Severity.HIGH,
            confidence=Confidence.HIGH,
            file_path="src/service.py",
            start_line=11,
            end_line=11,
            title="Hallucinated code snippet",
            description="Should be rejected because evidence does not exist in source.",
            evidence="cursor.execute(DROP_TABLE_USERS)",
        ),
        # 5. Style noise
        CandidateFinding(
            category=FindingCategory.CORRECTNESS,
            severity=Severity.LOW,
            confidence=Confidence.LOW,
            file_path="src/service.py",
            start_line=11,
            end_line=11,
            title="PEP 8 line length and trailing whitespace",
            description="This line violates PEP 8 formatting conventions.",
            evidence="return sum(items) / len(items)",
        ),
        # 6. Path traversal attempt in candidate finding
        CandidateFinding(
            category=FindingCategory.CORRECTNESS,
            severity=Severity.HIGH,
            confidence=Confidence.HIGH,
            file_path="../../etc/shadow",
            start_line=1,
            end_line=1,
            title="Traversal finding",
            description="Should be rejected due to path traversal.",
            evidence="root:*",
        ),
    ]

    validated, rejected_count, warnings = validate_candidates(
        module_id="code_review",
        candidates=candidates,
        diff_bundle=bundle,
        contexts=contexts,
    )

    assert len(validated) == 1
    assert rejected_count == 5
    assert len(warnings) == 5
    assert validated[0].id.startswith("dcd-")
    assert validated[0].file_path == "src/service.py"
    assert validated[0].line_range.start_line == 11


def test_finding_deduplication_keeps_highest_severity() -> None:
    diff_text = (
        "diff --git a/src/app.py b/src/app.py\n"
        "--- a/src/app.py\n"
        "+++ b/src/app.py\n"
        "@@ -1,2 +1,3 @@\n"
        " def calc(x: int) -> float:\n"
        "+    return x / 0\n"
    )
    bundle = DiffBundle(
        repository_root="/repo",
        base_revision="HEAD",
        target_description="test",
        files=parse_unified_diff(diff_text),
    )
    contexts = {
        "src/app.py": FileContext(
            file_path="src/app.py",
            total_lines=5,
            snippets=[
                ContextSnippet(
                    start_line=1,
                    end_line=2,
                    lines={1: "def calc(x: int) -> float:", 2: "    return x / 0"},
                )
            ],
        )
    }

    c_low = CandidateFinding(
        category=FindingCategory.CORRECTNESS,
        severity=Severity.MEDIUM,
        confidence=Confidence.MEDIUM,
        file_path="src/app.py",
        start_line=2,
        end_line=2,
        title="Division by zero",
        description="Divides x by zero.",
        evidence="return x / 0",
    )
    c_high = CandidateFinding(
        category=FindingCategory.CORRECTNESS,
        severity=Severity.CRITICAL,
        confidence=Confidence.HIGH,
        file_path="src/app.py",
        start_line=2,
        end_line=2,
        title="Division by zero crash",
        description="Dividing x by literal 0 always raises ZeroDivisionError at runtime.",
        evidence="return x / 0",
    )

    validated, rejected, _ = validate_candidates("code_review", [c_low, c_high, c_high], bundle, contexts)
    assert rejected == 0
    assert len(validated) == 3

    deduped, removed = deduplicate_findings(validated)
    assert len(deduped) == 1
    assert removed == 2
    assert deduped[0].severity == Severity.CRITICAL
    assert deduped[0].id == compute_finding_id(
        "code_review",
        "src/app.py",
        "correctness",
        deduped[0].line_range,
        "return x / 0",
    )
