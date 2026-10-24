"""Offline tests for persistent SQLite quota management, reservations, rate limits, and CLI usage."""

from __future__ import annotations

import concurrent.futures
import json
import sqlite3
import urllib.error
import urllib.request
from pathlib import Path

import pytest
from pydantic import SecretStr
from typer.testing import CliRunner

from deicide.cli import app, build_default_registry
from deicide.core.config import DeicideConfig
from deicide.core.errors import (
    ConfigurationError,
    ExitCode,
    MissingCredentialsError,
    ProviderError,
    QuotaExhaustedError,
    RateLimitError,
)
from deicide.core.models import BudgetTracker, ExecutionBudget
from deicide.core.orchestrator import AnalysisOrchestrator
from deicide.providers.base import ProviderRequest
from deicide.providers.factory import create_provider
from deicide.providers.fake import FakeProvider
from deicide.providers.nvidia import NvidiaProvider
from deicide.quota.store import RequestStatus, UsageStore, derive_quota_scope

runner = CliRunner()


class MockClock:
    """Deterministic controllable UTC clock for testing rate and daily windows."""

    def __init__(self, initial_epoch: float = 1_761_271_200.0) -> None:
        # 2025-10-24T02:00:00Z
        self.now = initial_epoch

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _make_ok_transport(
    prompt_tokens: int = 100,
    completion_tokens: int = 25,
    include_usage_metadata: bool = True,
):
    def _transport(_req: urllib.request.Request, _timeout: float) -> tuple[int, dict[str, str], str]:
        payload: dict[str, object] = {
            "id": "chatcmpl-test",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": '{"findings": []}'},
                    "finish_reason": "stop",
                }
            ],
        }
        if include_usage_metadata:
            payload["usage"] = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            }
        return 200, {"content-type": "application/json"}, json.dumps(payload)

    return _transport


def test_persistent_usage_across_separate_executions(tmp_path: Path) -> None:
    db_file = tmp_path / "usage.sqlite3"
    clock = MockClock()
    budget = ExecutionBudget(
        max_requests=3,
        max_total_tokens=8000,
        rate_limit_rpm=5,
        rate_limit_tpm=250_000,
        rate_limit_rpd=20,
    )

    # Execution 1: new UsageStore instance
    store1 = UsageStore(db_path=db_file, clock_fn=clock)
    provider1 = NvidiaProvider(
        api_key=SecretStr("key-alpha"),
        model_name="z-ai/glm-5.3-flash",
        transport_fn=_make_ok_transport(prompt_tokens=150, completion_tokens=50),
        budget=budget,
        usage_store=store1,
        quota_scope="scope-1",
    )
    resp1 = provider1.generate(ProviderRequest(system_prompt="sys", user_prompt="hello"))
    assert resp1.usage.total_tokens == 200
    assert not resp1.usage.is_estimated

    # Execution 2: completely separate UsageStore instance pointing to same SQLite file
    clock.advance(5.0)
    store2 = UsageStore(db_path=db_file, clock_fn=clock)
    provider2 = NvidiaProvider(
        api_key=SecretStr("key-alpha"),
        model_name="z-ai/glm-5.3-flash",
        transport_fn=_make_ok_transport(prompt_tokens=200, completion_tokens=40),
        budget=budget,
        usage_store=store2,
        quota_scope="scope-1",
    )
    provider2.generate(ProviderRequest(system_prompt="sys", user_prompt="world"))

    snap = store2.get_snapshot(
        provider="nvidia",
        model="z-ai/glm-5.3-flash",
        quota_scope="scope-1",
        rate_limit_rpm=5,
        rate_limit_tpm=250_000,
        rate_limit_rpd=20,
    )
    assert snap.requests_today == 2
    assert snap.succeeded_today == 2
    assert snap.estimated_remaining_daily_requests == 18
    assert snap.requests_last_60s == 2
    assert snap.input_tokens_today == 350
    assert snap.output_tokens_today == 90
    assert snap.total_tokens_today == 440
    assert snap.token_accounting_source == "provider_confirmed"
    assert not snap.is_locally_estimated


def test_daily_quota_exhaustion_and_utc_day_reset(tmp_path: Path) -> None:
    db_file = tmp_path / "rpd.sqlite3"
    clock = MockClock()
    store = UsageStore(db_path=db_file, clock_fn=clock)
    budget = ExecutionBudget(
        max_requests=5,
        max_total_tokens=10_000,
        rate_limit_rpm=10,
        rate_limit_tpm=250_000,
        rate_limit_rpd=2,
    )

    transport_calls = 0

    def counting_transport(req: urllib.request.Request, timeout: float) -> tuple[int, dict[str, str], str]:
        nonlocal transport_calls
        transport_calls += 1
        return _make_ok_transport()(req, timeout)

    provider = NvidiaProvider(
        api_key=SecretStr("key-rpd"),
        model_name="z-ai/glm-5.3-flash",
        transport_fn=counting_transport,
        budget=budget,
        usage_store=store,
        quota_scope="rpd-scope",
    )

    provider.generate(ProviderRequest(system_prompt="s", user_prompt="1"))
    clock.advance(10.0)
    provider.generate(ProviderRequest(system_prompt="s", user_prompt="2"))
    assert transport_calls == 2

    # 3rd request on the same UTC day must be blocked BEFORE reaching transport_fn
    with pytest.raises(QuotaExhaustedError, match="Daily API request quota") as exc_info:
        provider.generate(ProviderRequest(system_prompt="s", user_prompt="3"))
    assert exc_info.value.limit_type == "rpd"
    assert transport_calls == 2

    # Advance clock to next UTC day (+24 hours): daily quota resets
    clock.advance(86_400.0)
    provider.generate(ProviderRequest(system_prompt="s", user_prompt="next-day"))
    assert transport_calls == 3


def test_rpm_and_tpm_exhaustion_and_rolling_window_recovery(tmp_path: Path) -> None:
    db_file = tmp_path / "rpm_tpm.sqlite3"
    clock = MockClock()
    store = UsageStore(db_path=db_file, clock_fn=clock)

    # Test RPM = 2
    res1 = store.reserve(
        provider="nvidia",
        model="z-ai/glm-5.3-flash",
        quota_scope="s1",
        estimated_input_tokens=100,
        rate_limit_rpm=2,
        rate_limit_tpm=1000,
        rate_limit_rpd=20,
    )
    store.reconcile(res1, status=RequestStatus.SUCCEEDED, input_tokens=100, output_tokens=20)

    clock.advance(20.0)
    res2 = store.reserve(
        provider="nvidia",
        model="z-ai/glm-5.3-flash",
        quota_scope="s1",
        estimated_input_tokens=100,
        rate_limit_rpm=2,
        rate_limit_tpm=1000,
        rate_limit_rpd=20,
    )
    store.reconcile(res2, status=RequestStatus.SUCCEEDED, input_tokens=100, output_tokens=20)

    # 3rd request within 60s exceeds RPM=2
    with pytest.raises(QuotaExhaustedError, match="RPM") as rpm_err:
        store.reserve(
            provider="nvidia",
            model="z-ai/glm-5.3-flash",
            quota_scope="s1",
            estimated_input_tokens=100,
            rate_limit_rpm=2,
            rate_limit_tpm=1000,
            rate_limit_rpd=20,
        )
    assert rpm_err.value.limit_type == "rpm"

    # Advance 45s (so first request at t=0 is now 65s old, outside 60s window)
    clock.advance(45.0)
    # Now 1 request (100 input tokens) is in the last 60s; test TPM=500 with 450 token request (100+450 > 500)
    with pytest.raises(QuotaExhaustedError, match="TPM") as tpm_err:
        store.reserve(
            provider="nvidia",
            model="z-ai/glm-5.3-flash",
            quota_scope="s1",
            estimated_input_tokens=450,
            rate_limit_rpm=5,
            rate_limit_tpm=500,
            rate_limit_rpd=20,
        )
    assert tpm_err.value.limit_type == "tpm"

    # A smaller 300-token request fits inside TPM=500 (100 + 300 <= 500)
    res3 = store.reserve(
        provider="nvidia",
        model="z-ai/glm-5.3-flash",
        quota_scope="s1",
        estimated_input_tokens=300,
        rate_limit_rpm=5,
        rate_limit_tpm=500,
        rate_limit_rpd=20,
    )
    store.reconcile(res3, status=RequestStatus.SUCCEEDED, input_tokens=280, output_tokens=30)


def test_concurrent_reservations_never_exceed_limit(tmp_path: Path) -> None:
    db_file = tmp_path / "concurrent.sqlite3"
    clock = MockClock()
    store = UsageStore(db_path=db_file, clock_fn=clock)

    def worker(_idx: int) -> bool:
        local_store = UsageStore(db_path=db_file, clock_fn=clock)
        try:
            rid = local_store.reserve(
                provider="nvidia",
                model="z-ai/glm-5.3-flash",
                quota_scope="concurrent-scope",
                estimated_input_tokens=50,
                rate_limit_rpm=4,
                rate_limit_tpm=250_000,
                rate_limit_rpd=20,
            )
            local_store.reconcile(
                rid,
                status=RequestStatus.SUCCEEDED,
                input_tokens=50,
                output_tokens=10,
            )
            return True
        except QuotaExhaustedError:
            return False

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as pool:
        outcomes = list(pool.map(worker, range(10)))

    assert outcomes.count(True) == 4
    assert outcomes.count(False) == 6

    snap = store.get_snapshot(
        provider="nvidia",
        model="z-ai/glm-5.3-flash",
        quota_scope="concurrent-scope",
        rate_limit_rpm=4,
    )
    assert snap.requests_today == 4
    assert snap.requests_last_60s == 4


def test_retry_accounting_http_429_retry_after_and_uncertain_failures(tmp_path: Path) -> None:
    db_file = tmp_path / "retries.sqlite3"
    clock = MockClock()
    store = UsageStore(db_path=db_file, clock_fn=clock)
    slept_durations: list[float] = []

    call_num = 0

    def flaky_transport(req: urllib.request.Request, timeout: float) -> tuple[int, dict[str, str], str]:
        nonlocal call_num
        call_num += 1
        if call_num == 1:
            return 429, {"Retry-After": "3.5"}, "Rate limited"
        return _make_ok_transport(prompt_tokens=80, completion_tokens=20)(req, timeout)

    budget = ExecutionBudget(
        max_requests=3,
        max_total_tokens=8000,
        max_retries=2,
        rate_limit_rpm=5,
        rate_limit_tpm=250_000,
        rate_limit_rpd=20,
    )
    tracker = BudgetTracker(
        budget,
        usage_store=store,
        provider_name="nvidia",
        model_name="z-ai/glm-5.3-flash",
        quota_scope="retry-scope",
    )
    provider = NvidiaProvider(
        api_key=SecretStr("key-retry"),
        model_name="z-ai/glm-5.3-flash",
        sleep_fn=lambda s: slept_durations.append(s),
        transport_fn=flaky_transport,
        budget=budget,
        usage_store=store,
        quota_scope="retry-scope",
    )
    provider.bind_tracker(tracker)
    resp = provider.generate(ProviderRequest(system_prompt="sys", user_prompt="retry test"))
    provider.bind_tracker(None)

    assert resp.retries_performed == 1
    assert slept_durations == [3.5]
    assert tracker.requests_made == 2
    assert tracker.retries_performed == 1

    snap = store.get_snapshot(
        provider="nvidia",
        model="z-ai/glm-5.3-flash",
        quota_scope="retry-scope",
    )
    # Attempt 1 (rate_limited) + Attempt 2 (succeeded) = 2 requests recorded today
    assert snap.requests_today == 2
    assert snap.rate_limited_today == 1
    assert snap.succeeded_today == 1

    # Now test network timeout -> recorded as 'uncertain' and NOT refunded
    def timeout_transport(_req: urllib.request.Request, _timeout: float) -> tuple[int, dict[str, str], str]:
        raise TimeoutError("Socket read timed out")

    provider_timeout = NvidiaProvider(
        api_key=SecretStr("key-retry"),
        model_name="z-ai/glm-5.3-flash",
        max_retries=0,
        sleep_fn=lambda _: None,
        transport_fn=timeout_transport,
        budget=budget,
        usage_store=store,
        quota_scope="retry-scope",
    )
    with pytest.raises(ProviderError, match="timeout"):
        provider_timeout.generate(ProviderRequest(system_prompt="sys", user_prompt="uncertain payload"))

    snap_after = store.get_snapshot(
        provider="nvidia",
        model="z-ai/glm-5.3-flash",
        quota_scope="retry-scope",
    )
    assert snap_after.requests_today == 3
    assert snap_after.uncertain_today == 1
    assert snap_after.input_tokens_today > snap.input_tokens_today


def test_excessive_retry_after_fails_immediately(tmp_path: Path) -> None:
    db_file = tmp_path / "retry_after_large.sqlite3"
    store = UsageStore(db_path=db_file)
    slept: list[float] = []

    def long_429_transport(_req: urllib.request.Request, _timeout: float) -> tuple[int, dict[str, str], str]:
        return 429, {"Retry-After": "3600"}, "Quota exhausted for an hour"

    provider = NvidiaProvider(
        api_key=SecretStr("key-long-429"),
        model_name="z-ai/glm-5.3-flash",
        max_retries=2,
        sleep_fn=lambda s: slept.append(s),
        transport_fn=long_429_transport,
        usage_store=store,
    )
    with pytest.raises(RateLimitError, match="exceeding maximum retry wait ceiling"):
        provider.generate(ProviderRequest(system_prompt="sys", user_prompt="test"))
    assert slept == []


def test_process_interruption_and_stale_reservation_expiration(tmp_path: Path) -> None:
    db_file = tmp_path / "stale.sqlite3"
    clock = MockClock()
    store = UsageStore(db_path=db_file, clock_fn=clock, reservation_ttl_seconds=60.0)

    # Simulate a process that reserved the only RPM slot and then crashed before reconciling
    _stuck_id = store.reserve(
        provider="nvidia",
        model="z-ai/glm-5.3-flash",
        quota_scope="crash-scope",
        estimated_input_tokens=100,
        rate_limit_rpm=1,
        rate_limit_rpd=1,
    )

    # Immediately (t + 5s), the reservation is still active and blocks new requests
    clock.advance(5.0)
    with pytest.raises(QuotaExhaustedError):
        store.reserve(
            provider="nvidia",
            model="z-ai/glm-5.3-flash",
            quota_scope="crash-scope",
            estimated_input_tokens=100,
            rate_limit_rpm=1,
            rate_limit_rpd=1,
        )

    # After TTL (t + 65s > 60s TTL), the stuck reservation is automatically expired and freed
    clock.advance(60.0)
    new_id = store.reserve(
        provider="nvidia",
        model="z-ai/glm-5.3-flash",
        quota_scope="crash-scope",
        estimated_input_tokens=100,
        rate_limit_rpm=1,
        rate_limit_rpd=1,
    )
    store.reconcile(new_id, status=RequestStatus.SUCCEEDED, input_tokens=90, output_tokens=10)

    snap = store.get_snapshot(
        provider="nvidia",
        model="z-ai/glm-5.3-flash",
        quota_scope="crash-scope",
    )
    assert snap.expired_reservations_today == 1
    assert snap.succeeded_today == 1
    assert snap.requests_today == 1


def test_malformed_configuration_and_untrusted_repo_env_ignored(
    temp_git_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Plant a malicious .env inside the analyzed repository
    malicious_env = temp_git_repo / ".env"
    malicious_env.write_text(
        "DEICIDE_PROVIDER=fake\n"
        "DEICIDE_QUOTA_ENABLED=false\n"
        "DEICIDE_MAX_REQUESTS=9999\n"
        "NVIDIA_API_KEY=malicious-repo-key\n",
        encoding="utf-8",
    )

    monkeypatch.chdir(temp_git_repo)
    cfg = DeicideConfig.from_env_and_args(repo_path=temp_git_repo, environ={})
    # Must NOT have loaded the analyzed repository's .env
    assert cfg.api_key is None
    assert cfg.budget.quota_enabled is True
    assert cfg.budget.max_requests == 3

    # Strict validation rejects negative, zero, or non-numeric limits and invalid booleans
    for bad_env in (
        {"DEICIDE_MAX_REQUESTS": "0"},
        {"DEICIDE_MAX_REQUESTS": "-2"},
        {"DEICIDE_MAX_REQUESTS": "abc"},
        {"DEICIDE_MAX_TOKENS": "0"},
        {"DEICIDE_RATE_LIMIT_RPM": "-1"},
        {"DEICIDE_RATE_LIMIT_TPM": "not_a_number"},
        {"DEICIDE_RATE_LIMIT_RPD": "0"},
        {"DEICIDE_QUOTA_ENABLED": "maybe"},
    ):
        with pytest.raises(ConfigurationError):
            DeicideConfig.from_env_and_args(environ=bad_env)


def test_fake_provider_isolation_and_api_key_scope_isolation(temp_git_repo: Path, tmp_path: Path) -> None:
    db_file = tmp_path / "isolation.sqlite3"
    clock = MockClock()
    store = UsageStore(db_path=db_file, clock_fn=clock)

    # Distinct API keys map to distinct quota scopes without storing the keys
    scope_a = derive_quota_scope(None, "AIzaKeyProjectA")
    scope_b = derive_quota_scope(None, "AIzaKeyProjectB")
    assert scope_a != scope_b
    assert "AIzaKeyProjectA" not in scope_a

    # Exhaust RPD=1 for NVIDIA on scope_a
    rid = store.reserve(
        provider="nvidia",
        model="z-ai/glm-5.3-flash",
        quota_scope=scope_a,
        estimated_input_tokens=100,
        rate_limit_rpd=1,
    )
    store.reconcile(rid, status=RequestStatus.SUCCEEDED, input_tokens=100)

    # FakeProvider review on temp_git_repo is completely isolated and succeeds
    calc_file = temp_git_repo / "src" / "calc.py"
    calc_file.write_text("def f(x: int) -> float:\n    return x / 0\n", encoding="utf-8")

    cfg_fake = DeicideConfig.from_env_and_args(
        repo_path=temp_git_repo,
        provider="fake",
        usage_db_path=db_file,
        rate_limit_rpd=1,
        environ={},
    )
    report = AnalysisOrchestrator(
        config=cfg_fake,
        provider=FakeProvider(),
        registry=build_default_registry(),
        usage_store=store,
    ).run()
    assert report.status == "completed"
    assert len(report.findings) == 1


def test_privacy_no_credentials_prompts_or_source_code_in_sqlite(temp_git_repo: Path, tmp_path: Path) -> None:
    db_file = tmp_path / "privacy.sqlite3"
    secret_api_key = "AIzaSySuperSensitiveCredential987654321"
    secret_source_marker = "SUPER_CONFIDENTIAL_PROPRIETARY_ALGORITHM_XYZ"

    calc_file = temp_git_repo / "src" / "calc.py"
    calc_file.write_text(
        f"def compute() -> float:\n    # {secret_source_marker}\n    return 10 / 0\n",
        encoding="utf-8",
    )

    store = UsageStore(db_path=db_file)
    cfg = DeicideConfig.from_env_and_args(
        repo_path=temp_git_repo,
        provider="nvidia",
        model="z-ai/glm-5.3-flash",
        usage_db_path=db_file,
        environ={"NVIDIA_API_KEY": secret_api_key, "DEICIDE_ALLOW_EXTERNAL_CODE": "1"},
    )
    provider = NvidiaProvider(
        api_key=SecretStr(secret_api_key),
        model_name="z-ai/glm-5.3-flash",
        transport_fn=_make_ok_transport(prompt_tokens=220, completion_tokens=45),
        budget=cfg.budget,
        usage_store=store,
        quota_scope=cfg.resolved_quota_scope,
    )
    orchestrator = AnalysisOrchestrator(
        config=cfg,
        provider=provider,
        registry=build_default_registry(),
        usage_store=store,
    )
    orchestrator.run()

    # Inspect raw SQLite database file bytes and row contents
    raw_db_bytes = db_file.read_bytes()
    assert secret_api_key.encode("utf-8") not in raw_db_bytes
    assert secret_source_marker.encode("utf-8") not in raw_db_bytes
    assert b"UNTRUSTED_REPOSITORY_DATA" not in raw_db_bytes

    conn = sqlite3.connect(str(db_file))
    rows = conn.execute("SELECT * FROM usage_records;").fetchall()
    conn.close()
    assert len(rows) == 1


def test_cli_usage_terminal_and_json_output(tmp_path: Path) -> None:
    db_file = tmp_path / "cli_usage.sqlite3"
    store = UsageStore(db_path=db_file)
    rid = store.reserve(
        provider="nvidia",
        model="z-ai/glm-5.3-flash",
        quota_scope="cli-scope",
        estimated_input_tokens=400,
    )
    store.reconcile(
        rid,
        status=RequestStatus.SUCCEEDED,
        input_tokens=420,
        output_tokens=80,
        total_tokens=500,
        is_estimated=False,
    )

    res_term = runner.invoke(
        app,
        [
            "usage",
            "--provider",
            "nvidia",
            "--model",
            "z-ai/glm-5.3-flash",
            "--scope",
            "cli-scope",
            "--db-path",
            str(db_file),
        ],
    )
    assert res_term.exit_code == int(ExitCode.SUCCESS)
    assert "DEICIDE Persistent API Usage & Quota Report" in res_term.stdout
    assert "nvidia (z-ai/glm-5.3-flash)" in res_term.stdout
    assert "Requests Recorded Today       : 1" in res_term.stdout
    assert "Input Tokens Today            : 420" in res_term.stdout
    assert "Output Tokens Today           : 80" in res_term.stdout

    res_json = runner.invoke(
        app,
        [
            "usage",
            "--provider",
            "nvidia",
            "--model",
            "z-ai/glm-5.3-flash",
            "--scope",
            "cli-scope",
            "--db-path",
            str(db_file),
            "--format",
            "json",
        ],
    )
    assert res_json.exit_code == int(ExitCode.SUCCESS)
    data = json.loads(res_json.stdout)
    assert data["provider"] == "nvidia"
    assert data["model"] == "z-ai/glm-5.3-flash"
    assert data["requests_today"] == 1
    assert data["configured_rpd_limit"] is None
    assert data["estimated_remaining_daily_requests"] is None
    assert data["requests_last_60s"] == 1
    assert data["configured_rpm_limit"] is None
    assert data["input_tokens_today"] == 420
    assert data["output_tokens_today"] == 80
    assert data["token_accounting_source"] == "provider_confirmed"
    assert "locally tracked estimates" in data["disclaimer"]


def test_trusted_dotenv_loaded_automatically_with_repo_arg_and_empty_deicide_key(
    temp_git_repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trusted_root = tmp_path / "trusted_deicide_home"
    trusted_root.mkdir()
    trusted_dotenv = trusted_root / ".env"
    trusted_dotenv.write_text(
        "DEICIDE_PROVIDER=nvidia\n"
        "DEICIDE_MODEL=z-ai/glm-5.3-flash\n"
        "NVIDIA_API_KEY=trusted-dotenv-nvidia-key\n"
        "DEICIDE_API_KEY=\n"
        "DEICIDE_ALLOW_EXTERNAL_CODE=1\n"
        "DEICIDE_TIMEOUT_SECONDS=75.0\n"
        "DEICIDE_MAX_RETRIES=0\n",
        encoding="utf-8",
    )

    # Plant an untrusted .env inside the target repository being analyzed
    (temp_git_repo / ".env").write_text(
        "NVIDIA_API_KEY=untrusted-repo-key\nDEICIDE_API_KEY=untrusted-deicide-key\nDEICIDE_TIMEOUT_SECONDS=5.0\n",
        encoding="utf-8",
    )

    monkeypatch.delenv("DEICIDE_ENV_FILE", raising=False)
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.delenv("DEICIDE_API_KEY", raising=False)
    monkeypatch.setattr("deicide.core.config._DEICIDE_INSTALL_ROOT", trusted_root)
    monkeypatch.setattr("deicide.core.config.get_user_data_dir", lambda *_a, **_kw: tmp_path / "empty_appdata")
    monkeypatch.chdir(temp_git_repo)

    # Even when running inside temp_git_repo and passing repo_path=temp_git_repo,
    # trusted_root/.env is loaded and temp_git_repo/.env is ignored.
    cfg = DeicideConfig.from_env_and_args(repo_path=temp_git_repo)
    cfg.validate_provider_credentials()
    assert cfg.api_key is not None
    assert cfg.api_key.get_secret_value() == "trusted-dotenv-nvidia-key"
    assert cfg.budget.request_timeout_seconds == 75.0
    assert cfg.budget.max_retries == 0


def test_credential_precedence_four_levels_and_empty_never_overrides(tmp_path: Path) -> None:
    dotenv_file = tmp_path / ".env"

    # Level 3 vs empty Level 1, 2, 4: NVIDIA_API_KEY in .env with empty DEICIDE_API_KEY in .env
    # and empty strings in process environment
    dotenv_file.write_text(
        "NVIDIA_API_KEY=level3-dotenv-nvidia\nDEICIDE_API_KEY=level4-dotenv-deicide\n",
        encoding="utf-8",
    )
    cfg3 = DeicideConfig.from_env_and_args(
        dotenv_path=dotenv_file,
        environ={"NVIDIA_API_KEY": "", "DEICIDE_API_KEY": "   "},
    )
    assert cfg3.api_key is not None
    assert cfg3.api_key.get_secret_value() == "level3-dotenv-nvidia"

    # Level 2 beats Level 3: Nonempty DEICIDE_API_KEY in process env beats NVIDIA_API_KEY in .env
    cfg2 = DeicideConfig.from_env_and_args(
        dotenv_path=dotenv_file,
        environ={"NVIDIA_API_KEY": "", "DEICIDE_API_KEY": "level2-env-deicide"},
    )
    assert cfg2.api_key is not None
    assert cfg2.api_key.get_secret_value() == "level2-env-deicide"

    # Level 1 beats Level 2, 3, 4: Nonempty NVIDIA_API_KEY in process env wins
    cfg1 = DeicideConfig.from_env_and_args(
        dotenv_path=dotenv_file,
        environ={
            "NVIDIA_API_KEY": "level1-env-nvidia",
            "DEICIDE_API_KEY": "level2-env-deicide",
        },
    )
    assert cfg1.api_key is not None
    assert cfg1.api_key.get_secret_value() == "level1-env-nvidia"

    # Level 4: Empty NVIDIA_API_KEY in .env falls back to nonempty DEICIDE_API_KEY in .env
    dotenv_file.write_text(
        "NVIDIA_API_KEY=\nDEICIDE_API_KEY=level4-dotenv-deicide\n",
        encoding="utf-8",
    )
    cfg4 = DeicideConfig.from_env_and_args(dotenv_path=dotenv_file, environ={})
    assert cfg4.api_key is not None
    assert cfg4.api_key.get_secret_value() == "level4-dotenv-deicide"


def test_timeout_propagation_to_http_transport(tmp_path: Path) -> None:
    dotenv_file = tmp_path / ".env"
    dotenv_file.write_text(
        "NVIDIA_API_KEY=timeout-test-key\nDEICIDE_ALLOW_EXTERNAL_CODE=1\nDEICIDE_TIMEOUT_SECONDS=92.5\n",
        encoding="utf-8",
    )
    cfg = DeicideConfig.from_env_and_args(
        dotenv_path=dotenv_file,
        usage_db_path=tmp_path / "timeout.sqlite3",
        environ={},
    )
    assert cfg.budget.request_timeout_seconds == 92.5

    provider = create_provider(cfg)
    assert isinstance(provider, NvidiaProvider)
    assert provider.timeout_seconds == 92.5

    observed_timeouts: list[float] = []

    def spy_transport(req: urllib.request.Request, timeout: float) -> tuple[int, dict[str, str], str]:
        observed_timeouts.append(timeout)
        return _make_ok_transport()(req, timeout)

    provider._transport_fn = spy_transport
    provider.generate(ProviderRequest(system_prompt="sys", user_prompt="check timeout"))
    assert observed_timeouts == [92.5]


def test_zero_retries_default_and_explicit_retries_with_uncertain_accounting(
    tmp_path: Path,
) -> None:
    db_file = tmp_path / "zero_retries.sqlite3"
    store = UsageStore(db_path=db_file)

    # 1. Default configuration has max_retries == 0 (1 initial attempt, 0 retries)
    cfg_default = DeicideConfig.from_env_and_args(
        usage_db_path=db_file,
        environ={"NVIDIA_API_KEY": "retry-key", "DEICIDE_ALLOW_EXTERNAL_CODE": "1"},
    )
    assert cfg_default.budget.max_retries == 0

    attempts_zero = 0

    def timeout_transport_zero(_req: urllib.request.Request, _timeout: float) -> tuple[int, dict[str, str], str]:
        nonlocal attempts_zero
        attempts_zero += 1
        raise TimeoutError("The read operation timed out")

    provider_zero = NvidiaProvider(
        api_key=SecretStr("retry-key"),
        model_name="z-ai/glm-5.3-flash",
        sleep_fn=lambda _: None,
        transport_fn=timeout_transport_zero,
        budget=cfg_default.budget,
        usage_store=store,
        quota_scope="zero-retry-scope",
    )
    req = ProviderRequest(system_prompt="sys", user_prompt="a" * 400)
    with pytest.raises(ProviderError, match="The read operation timed out"):
        provider_zero.generate(req)

    # Exactly 1 attempt was made and recorded as uncertain
    assert attempts_zero == 1
    snap_zero = store.get_snapshot(
        provider="nvidia",
        model="z-ai/glm-5.3-flash",
        quota_scope="zero-retry-scope",
    )
    assert snap_zero.requests_today == 1
    assert snap_zero.uncertain_today == 1
    assert snap_zero.input_tokens_today == req.estimated_prompt_tokens

    # 2. Explicitly configured DEICIDE_MAX_RETRIES=2 makes 3 attempts (1 initial + 2 retries)
    cfg_two = DeicideConfig.from_env_and_args(
        usage_db_path=db_file,
        environ={
            "NVIDIA_API_KEY": "retry-key",
            "DEICIDE_ALLOW_EXTERNAL_CODE": "1",
            "DEICIDE_MAX_RETRIES": "2",
        },
    )
    assert cfg_two.budget.max_retries == 2

    attempts_two = 0

    def timeout_transport_two(_req: urllib.request.Request, _timeout: float) -> tuple[int, dict[str, str], str]:
        nonlocal attempts_two
        attempts_two += 1
        raise TimeoutError("The read operation timed out")

    provider_two = NvidiaProvider(
        api_key=SecretStr("retry-key"),
        model_name="z-ai/glm-5.3-flash",
        sleep_fn=lambda _: None,
        transport_fn=timeout_transport_two,
        budget=cfg_two.budget,
        usage_store=store,
        quota_scope="two-retry-scope",
    )
    with pytest.raises(ProviderError, match="after 2 retries"):
        provider_two.generate(req)

    assert attempts_two == 3
    snap_two = store.get_snapshot(
        provider="nvidia",
        model="z-ai/glm-5.3-flash",
        quota_scope="two-retry-scope",
    )
    assert snap_two.requests_today == 3
    assert snap_two.uncertain_today == 3
    assert snap_two.input_tokens_today == req.estimated_prompt_tokens * 3

    # 3. Credential/auth errors (HTTP 401/403) are NEVER retried even when max_retries=2
    auth_attempts = 0

    def auth_error_transport(_req: urllib.request.Request, _timeout: float) -> tuple[int, dict[str, str], str]:
        nonlocal auth_attempts
        auth_attempts += 1
        return 403, {}, "PERMISSION_DENIED: Invalid API key"

    provider_auth = NvidiaProvider(
        api_key=SecretStr("retry-key"),
        model_name="z-ai/glm-5.3-flash",
        sleep_fn=lambda _: None,
        transport_fn=auth_error_transport,
        budget=cfg_two.budget,
        usage_store=store,
        quota_scope="auth-err-scope",
    )
    with pytest.raises(ProviderError, match="HTTP 403"):
        provider_auth.generate(req)
    assert auth_attempts == 1

    # Negative DEICIDE_MAX_RETRIES is rejected
    with pytest.raises(ConfigurationError):
        DeicideConfig.from_env_and_args(environ={"DEICIDE_MAX_RETRIES": "-1"})


def test_nvidia_dotenv_configuration_placeholder_and_precedence(tmp_path: Path) -> None:
    dotenv_file = tmp_path / ".env"
    dotenv_file.write_text(
        "# Provider\n"
        "DEICIDE_PROVIDER=nvidia\n"
        "DEICIDE_MODEL=z-ai/glm-5.3-flash\n"
        "\n"
        "# NVIDIA\n"
        "NVIDIA_API_KEY=SUA_CHAVE_NVIDIA\n"
        "NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1\n"
        "\n"
        "# Execution Budget\n"
        "DEICIDE_MAX_REQUESTS=3\n"
        "DEICIDE_MAX_TOKENS=8000\n"
        "\n"
        "# Quota Management\n"
        "DEICIDE_QUOTA_ENABLED=true\n"
        "DEICIDE_QUOTA_SCOPE=nvidia-glm\n"
        "\n"
        "# Configure these only after confirming NVIDIA's actual limits.\n"
        "# DEICIDE_RATE_LIMIT_RPM=\n"
        "# DEICIDE_RATE_LIMIT_TPM=\n"
        "# DEICIDE_RATE_LIMIT_RPD=\n"
        "\n"
        "# Security\n"
        "DEICIDE_ALLOW_EXTERNAL_CODE=0\n"
        "\n"
        "# Network\n"
        "DEICIDE_TIMEOUT_SECONDS=120\n"
        "DEICIDE_MAX_RETRIES=0\n",
        encoding="utf-8",
    )

    cfg_placeholder = DeicideConfig.from_env_and_args(dotenv_path=dotenv_file, environ={})
    assert cfg_placeholder.provider == "nvidia"
    assert cfg_placeholder.resolved_model == "z-ai/glm-5.3-flash"
    assert cfg_placeholder.nvidia_base_url == "https://integrate.api.nvidia.com/v1"
    assert cfg_placeholder.budget.max_requests == 3
    assert cfg_placeholder.budget.max_total_tokens == 8000
    assert cfg_placeholder.budget.quota_enabled is True
    assert cfg_placeholder.resolved_quota_scope == "nvidia-glm"
    # Rate limits are None when commented out for nvidia
    assert cfg_placeholder.budget.rate_limit_rpm is None
    assert cfg_placeholder.budget.rate_limit_tpm is None
    assert cfg_placeholder.budget.rate_limit_rpd is None
    assert cfg_placeholder.allow_external_code is False
    assert cfg_placeholder.budget.request_timeout_seconds == 120.0
    assert cfg_placeholder.budget.max_retries == 0

    # Placeholder SUA_CHAVE_NVIDIA is rejected as missing credentials
    assert cfg_placeholder.api_key is None
    with pytest.raises(MissingCredentialsError, match="NVIDIA_API_KEY"):
        cfg_placeholder.validate_provider_credentials()

    # With a real NVIDIA_API_KEY, DEICIDE_ALLOW_EXTERNAL_CODE=0 blocks external egress until enabled
    cfg_no_egress = DeicideConfig.from_env_and_args(
        dotenv_path=dotenv_file,
        environ={"NVIDIA_API_KEY": "nvapi-real-test-key"},
    )
    assert cfg_no_egress.api_key is not None
    assert cfg_no_egress.api_key.get_secret_value() == "nvapi-real-test-key"
    with pytest.raises(ConfigurationError, match="DEICIDE_ALLOW_EXTERNAL_CODE=1"):
        cfg_no_egress.validate_provider_credentials()


def test_nvidia_provider_end_to_end_review_and_sqlite_tracking(temp_git_repo: Path, tmp_path: Path) -> None:
    db_file = tmp_path / "nvidia_usage.sqlite3"
    store = UsageStore(db_path=db_file)

    calc_file = temp_git_repo / "src" / "calc.py"
    calc_file.write_text(
        "def divide(a: float, b: float) -> float:\n    return a / 0\n",
        encoding="utf-8",
    )

    cfg = DeicideConfig.from_env_and_args(
        repo_path=temp_git_repo,
        provider="nvidia",
        model="z-ai/glm-5.3-flash",
        quota_scope="nvidia-glm",
        usage_db_path=db_file,
        environ={
            "NVIDIA_API_KEY": "nvapi-secret-key-999",
            "NVIDIA_BASE_URL": "https://integrate.api.nvidia.com/v1",
            "DEICIDE_ALLOW_EXTERNAL_CODE": "1",
            "DEICIDE_TIMEOUT_SECONDS": "120",
            "DEICIDE_MAX_RETRIES": "0",
        },
    )
    provider = create_provider(cfg, usage_store=store)
    assert isinstance(provider, NvidiaProvider)
    assert provider.timeout_seconds == 120.0

    observed_urls: list[str] = []
    observed_timeouts: list[float] = []

    def fake_nvidia_transport(req: urllib.request.Request, timeout: float) -> tuple[int, dict[str, str], str]:
        observed_urls.append(req.full_url)
        observed_timeouts.append(timeout)
        # Simulate reasoning model output containing <think> block + JSON payload
        content = "<think>Checking divide function in src/calc.py line 2...</think>\n" + json.dumps(
            {
                "findings": [
                    {
                        "title": "Division by literal zero",
                        "category": "logic_error",
                        "severity": "critical",
                        "confidence": "high",
                        "file_path": "src/calc.py",
                        "start_line": 2,
                        "end_line": 2,
                        "evidence": "return a / 0",
                        "description": "Line 2 divides by literal 0, raising ZeroDivisionError.",
                        "suggested_remediation": "Divide by parameter b after checking b != 0.",
                    }
                ]
            }
        )
        body = json.dumps(
            {
                "id": "chatcmpl-nvidia-1",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": content}}],
                "usage": {
                    "prompt_tokens": 310,
                    "completion_tokens": 90,
                    "total_tokens": 400,
                },
            }
        )
        return 200, {}, body

    provider._transport_fn = fake_nvidia_transport

    report = AnalysisOrchestrator(
        config=cfg,
        provider=provider,
        registry=build_default_registry(),
        usage_store=store,
    ).run()

    assert report.status == "completed"
    assert len(report.findings) == 1
    assert report.findings[0].file_path == "src/calc.py"
    assert report.findings[0].line_range.start_line == 2
    assert observed_urls == ["https://integrate.api.nvidia.com/v1/chat/completions"]
    assert observed_timeouts == [120.0]

    snap = store.get_snapshot(
        provider="nvidia",
        model="z-ai/glm-5.3-flash",
        quota_scope="nvidia-glm",
    )
    assert snap.requests_today == 1
    assert snap.succeeded_today == 1
    assert snap.input_tokens_today == 310
    assert snap.output_tokens_today == 90
    assert snap.total_tokens_today == 400
    assert snap.token_accounting_source == "provider_confirmed"
