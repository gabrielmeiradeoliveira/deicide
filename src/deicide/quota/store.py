"""Provider-independent SQLite usage store and atomic request reservation."""

from __future__ import annotations

import hashlib
import os
import sqlite3
import sys
import time
import uuid
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from deicide.core.errors import ConfigurationError, QuotaExhaustedError


class RequestStatus(StrEnum):
    """Lifecycle status of an API request record in the usage ledger."""

    RESERVED = "reserved"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    RATE_LIMITED = "rate_limited"
    UNCERTAIN = "uncertain"
    EXPIRED = "expired"


ACTIVE_QUOTA_STATUSES: tuple[str, ...] = (
    RequestStatus.RESERVED.value,
    RequestStatus.SUCCEEDED.value,
    RequestStatus.FAILED.value,
    RequestStatus.RATE_LIMITED.value,
    RequestStatus.UNCERTAIN.value,
)

QUOTA_DISCLAIMER = (
    "Quota and usage metrics are locally tracked estimates for this DEICIDE installation "
    "and do not represent the provider's exact server-side account balance."
)


class QuotaSnapshot(BaseModel):
    """Point-in-time summary of persistent API usage and configured rate limits."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    provider: str
    model: str
    quota_scope: str
    quota_enabled: bool
    database_path: str
    utc_date: str
    daily_reset_utc: str
    requests_today: int = Field(default=0, ge=0)
    succeeded_today: int = Field(default=0, ge=0)
    failed_today: int = Field(default=0, ge=0)
    rate_limited_today: int = Field(default=0, ge=0)
    uncertain_today: int = Field(default=0, ge=0)
    active_reservations_today: int = Field(default=0, ge=0)
    expired_reservations_today: int = Field(default=0, ge=0)
    configured_rpd_limit: int | None = None
    estimated_remaining_daily_requests: int | None = None
    requests_last_60s: int = Field(default=0, ge=0)
    configured_rpm_limit: int | None = None
    estimated_remaining_rpm: int | None = None
    input_tokens_last_60s: int = Field(default=0, ge=0)
    configured_tpm_limit: int | None = None
    estimated_remaining_tpm: int | None = None
    input_tokens_today: int = Field(default=0, ge=0)
    output_tokens_today: int = Field(default=0, ge=0)
    total_tokens_today: int = Field(default=0, ge=0)
    token_accounting_source: Literal["locally_estimated", "provider_confirmed", "mixed_local_and_provider"] = (
        "locally_estimated"  # noqa: S105
    )
    is_locally_estimated: bool = True
    disclaimer: str = QUOTA_DISCLAIMER


def derive_quota_scope(explicit_scope: str | None, api_key_secret: str | None) -> str:
    """Derive a privacy-preserving quota bucket scope without storing raw credentials.

    Two different API keys automatically map to distinct SHA-256 fingerprint scopes
    unless the user explicitly configures a shared `DEICIDE_QUOTA_SCOPE`.
    """
    if explicit_scope and explicit_scope.strip():
        return explicit_scope.strip()
    if api_key_secret and api_key_secret.strip():
        digest = hashlib.sha256(f"deicide-quota-scope:{api_key_secret.strip()}".encode()).hexdigest()[:12]
        return f"key-{digest}"
    return "default"


def get_user_data_dir(environ: dict[str, str] | None = None) -> Path:
    """Return the platform-appropriate user data directory for DEICIDE.

    On Windows, prefers LOCALAPPDATA (e.g. C:\\Users\\<user>\\AppData\\Local\\deicide).
    """
    env = os.environ if environ is None else environ
    if os.name == "nt" or sys.platform == "win32":
        base = env.get("LOCALAPPDATA") or env.get("APPDATA")
        if base and base.strip():
            return Path(base.strip()) / "deicide"
        return Path.home() / "AppData" / "Local" / "deicide"

    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "deicide"

    xdg_data = env.get("XDG_DATA_HOME")
    if xdg_data and xdg_data.strip():
        return Path(xdg_data.strip()) / "deicide"
    return Path.home() / ".local" / "share" / "deicide"


def get_default_usage_db_path(environ: dict[str, str] | None = None) -> Path:
    """Return the resolved path to the persistent SQLite usage database."""
    env = os.environ if environ is None else environ
    override = env.get("DEICIDE_USAGE_DB_PATH")
    if override and override.strip():
        return Path(override.strip()).expanduser().resolve()
    return (get_user_data_dir(environ) / "usage.sqlite3").resolve()


def _utc_day_and_reset(epoch_seconds: float) -> tuple[str, str]:
    """Return (YYYY-MM-DD, next_midnight_iso_utc) for a UTC epoch timestamp.

    Daily quota windows are calculated per UTC calendar day (00:00:00Z to 23:59:59.999999Z).
    """
    dt = datetime.fromtimestamp(epoch_seconds, tz=UTC)
    day_str = dt.strftime("%Y-%m-%d")
    next_day = (dt.date() + timedelta(days=1)).strftime("%Y-%m-%dT00:00:00Z")
    return day_str, next_day


class UsageStore:
    """Thread-safe and multi-process-safe SQLite store for API quota reservations and usage."""

    def __init__(
        self,
        db_path: Path | None = None,
        clock_fn: Callable[[], float] = time.time,
        reservation_ttl_seconds: float = 120.0,
    ) -> None:
        if reservation_ttl_seconds <= 0:
            raise ConfigurationError("reservation_ttl_seconds must be positive.")
        self.db_path = (db_path or get_default_usage_db_path()).resolve()
        self._clock = clock_fn
        self.reservation_ttl_seconds = reservation_ttl_seconds
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(
            str(self.db_path),
            timeout=30.0,
            isolation_level=None,
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout = 30000;")
        return conn

    def _ensure_schema(self) -> None:
        conn = self._connect()
        try:
            conn.execute("PRAGMA journal_mode = WAL;")
            conn.execute("BEGIN IMMEDIATE;")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS usage_records (
                    reservation_id TEXT PRIMARY KEY,
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    quota_scope TEXT NOT NULL,
                    timestamp_utc REAL NOT NULL,
                    utc_date TEXT NOT NULL,
                    status TEXT NOT NULL,
                    input_tokens INTEGER NOT NULL CHECK (input_tokens >= 0),
                    output_tokens INTEGER NOT NULL CHECK (output_tokens >= 0),
                    total_tokens INTEGER NOT NULL CHECK (total_tokens >= 0),
                    is_estimated INTEGER NOT NULL CHECK (is_estimated IN (0, 1)),
                    attempt_number INTEGER NOT NULL CHECK (attempt_number >= 1),
                    updated_at_utc REAL NOT NULL
                );
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_usage_scope_date
                ON usage_records (provider, model, quota_scope, utc_date);
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_usage_scope_ts
                ON usage_records (provider, model, quota_scope, timestamp_utc);
                """
            )
            conn.execute("COMMIT;")
        except Exception:
            conn.execute("ROLLBACK;")
            raise
        finally:
            conn.close()

    def _expire_stale_in_tx(self, conn: sqlite3.Connection, now_utc: float) -> int:
        cutoff = now_utc - self.reservation_ttl_seconds
        cursor = conn.execute(
            """
            UPDATE usage_records
               SET status = ?, updated_at_utc = ?
             WHERE status = ? AND timestamp_utc <= ?;
            """,
            (
                RequestStatus.EXPIRED.value,
                now_utc,
                RequestStatus.RESERVED.value,
                cutoff,
            ),
        )
        return cursor.rowcount

    def expire_stale_reservations(self) -> int:
        """Mark reservations older than `reservation_ttl_seconds` as expired."""
        now_utc = float(self._clock())
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE;")
            expired = self._expire_stale_in_tx(conn, now_utc)
            conn.execute("COMMIT;")
            return expired
        except Exception:
            conn.execute("ROLLBACK;")
            raise
        finally:
            conn.close()

    def reserve(
        self,
        *,
        provider: str,
        model: str,
        quota_scope: str,
        estimated_input_tokens: int,
        attempt_number: int = 1,
        rate_limit_rpm: int | None = None,
        rate_limit_tpm: int | None = None,
        rate_limit_rpd: int | None = None,
    ) -> str:
        """Atomically check configured rate/daily limits and reserve capacity before an API call."""
        norm_provider = provider.strip().lower()
        norm_model = model.strip()
        norm_scope = quota_scope.strip() or "default"
        safe_input_tokens = max(0, int(estimated_input_tokens))
        safe_attempt = max(1, int(attempt_number))

        now_utc = float(self._clock())
        utc_date, next_reset = _utc_day_and_reset(now_utc)
        minute_window_start = now_utc - 60.0

        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE;")
            self._expire_stale_in_tx(conn, now_utc)

            # 1. Check daily request limit (RPD) for the current UTC day
            row_day = conn.execute(
                """
                SELECT COUNT(*) AS req_count
                  FROM usage_records
                 WHERE provider = ?
                   AND model = ?
                   AND quota_scope = ?
                   AND utc_date = ?
                   AND status IN (?, ?, ?, ?, ?);
                """,
                (norm_provider, norm_model, norm_scope, utc_date, *ACTIVE_QUOTA_STATUSES),
            ).fetchone()
            requests_today = int(row_day["req_count"]) if row_day else 0

            if rate_limit_rpd is not None and requests_today + 1 > rate_limit_rpd:
                conn.execute("ROLLBACK;")
                msg = (
                    f"Daily API request quota (RPD) exhausted for {norm_provider}/{norm_model} "
                    f"(scope='{norm_scope}'): {requests_today}/{rate_limit_rpd} requests used "
                    f"on UTC day {utc_date} (resets at {next_reset})."
                )
                raise QuotaExhaustedError(msg, limit_type="rpd")

            # 2. Check rolling 60-second window (RPM and TPM)
            row_min = conn.execute(
                """
                SELECT COUNT(*) AS req_count,
                       COALESCE(SUM(input_tokens), 0) AS input_sum
                  FROM usage_records
                 WHERE provider = ?
                   AND model = ?
                   AND quota_scope = ?
                   AND timestamp_utc > ?
                   AND status IN (?, ?, ?, ?, ?);
                """,
                (
                    norm_provider,
                    norm_model,
                    norm_scope,
                    minute_window_start,
                    *ACTIVE_QUOTA_STATUSES,
                ),
            ).fetchone()
            requests_60s = int(row_min["req_count"]) if row_min else 0
            input_tokens_60s = int(row_min["input_sum"]) if row_min else 0

            if rate_limit_rpm is not None and requests_60s + 1 > rate_limit_rpm:
                conn.execute("ROLLBACK;")
                msg = (
                    f"Per-minute request rate limit (RPM) reached for {norm_provider}/{norm_model} "
                    f"(scope='{norm_scope}'): {requests_60s}/{rate_limit_rpm} requests in the last 60s."
                )
                raise QuotaExhaustedError(msg, limit_type="rpm")

            if rate_limit_tpm is not None and input_tokens_60s + safe_input_tokens > rate_limit_tpm:
                conn.execute("ROLLBACK;")
                msg = (
                    f"Per-minute input token limit (TPM) reached for {norm_provider}/{norm_model} "
                    f"(scope='{norm_scope}'): projected {input_tokens_60s + safe_input_tokens} "
                    f"input tokens exceeds limit of {rate_limit_tpm} tokens/minute."
                )
                raise QuotaExhaustedError(msg, limit_type="tpm")

            reservation_id = str(uuid.uuid4())
            conn.execute(
                """
                INSERT INTO usage_records (
                    reservation_id,
                    provider,
                    model,
                    quota_scope,
                    timestamp_utc,
                    utc_date,
                    status,
                    input_tokens,
                    output_tokens,
                    total_tokens,
                    is_estimated,
                    attempt_number,
                    updated_at_utc
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    reservation_id,
                    norm_provider,
                    norm_model,
                    norm_scope,
                    now_utc,
                    utc_date,
                    RequestStatus.RESERVED.value,
                    safe_input_tokens,
                    0,
                    safe_input_tokens,
                    1,
                    safe_attempt,
                    now_utc,
                ),
            )
            conn.execute("COMMIT;")
            return reservation_id
        except QuotaExhaustedError:
            raise
        except Exception:
            conn.execute("ROLLBACK;")
            raise
        finally:
            conn.close()

    def reconcile(
        self,
        reservation_id: str,
        *,
        status: RequestStatus,
        input_tokens: int,
        output_tokens: int = 0,
        total_tokens: int | None = None,
        is_estimated: bool = True,
    ) -> None:
        """Reconcile an existing reservation with actual or conservative token usage and status."""
        now_utc = float(self._clock())
        safe_in = max(0, int(input_tokens))
        safe_out = max(0, int(output_tokens))
        safe_total = max(safe_in + safe_out, int(total_tokens or 0))
        est_flag = 1 if is_estimated else 0

        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE;")
            conn.execute(
                """
                UPDATE usage_records
                   SET status = ?,
                       input_tokens = ?,
                       output_tokens = ?,
                       total_tokens = ?,
                       is_estimated = ?,
                       updated_at_utc = ?
                 WHERE reservation_id = ?;
                """,
                (
                    status.value,
                    safe_in,
                    safe_out,
                    safe_total,
                    est_flag,
                    now_utc,
                    reservation_id,
                ),
            )
            conn.execute("COMMIT;")
        except Exception:
            conn.execute("ROLLBACK;")
            raise
        finally:
            conn.close()

    def get_snapshot(
        self,
        *,
        provider: str,
        model: str,
        quota_scope: str = "default",
        quota_enabled: bool = True,
        rate_limit_rpm: int | None = None,
        rate_limit_tpm: int | None = None,
        rate_limit_rpd: int | None = None,
    ) -> QuotaSnapshot:
        """Query current daily and 60-second usage statistics for `(provider, model, quota_scope)`."""
        norm_provider = provider.strip().lower()
        norm_model = model.strip()
        norm_scope = quota_scope.strip() or "default"

        now_utc = float(self._clock())
        utc_date, next_reset = _utc_day_and_reset(now_utc)
        minute_window_start = now_utc - 60.0

        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE;")
            self._expire_stale_in_tx(conn, now_utc)

            day_rows = conn.execute(
                """
                SELECT status,
                       COUNT(*) AS cnt,
                       COALESCE(SUM(input_tokens), 0) AS in_tok,
                       COALESCE(SUM(output_tokens), 0) AS out_tok,
                       COALESCE(SUM(total_tokens), 0) AS tot_tok,
                       COALESCE(SUM(is_estimated), 0) AS est_cnt
                  FROM usage_records
                 WHERE provider = ?
                   AND model = ?
                   AND quota_scope = ?
                   AND utc_date = ?
                 GROUP BY status;
                """,
                (norm_provider, norm_model, norm_scope, utc_date),
            ).fetchall()

            min_row = conn.execute(
                """
                SELECT COUNT(*) AS req_count,
                       COALESCE(SUM(input_tokens), 0) AS input_sum
                  FROM usage_records
                 WHERE provider = ?
                   AND model = ?
                   AND quota_scope = ?
                   AND timestamp_utc > ?
                   AND status IN (?, ?, ?, ?, ?);
                """,
                (
                    norm_provider,
                    norm_model,
                    norm_scope,
                    minute_window_start,
                    *ACTIVE_QUOTA_STATUSES,
                ),
            ).fetchone()
            conn.execute("COMMIT;")
        except Exception:
            conn.execute("ROLLBACK;")
            raise
        finally:
            conn.close()

        status_counts: dict[str, int] = {}
        input_today = 0
        output_today = 0
        total_today = 0
        active_records_today = 0
        estimated_records_today = 0

        for r in day_rows:
            st = str(r["status"])
            cnt = int(r["cnt"])
            status_counts[st] = cnt
            if st in ACTIVE_QUOTA_STATUSES:
                active_records_today += cnt
                input_today += int(r["in_tok"])
                output_today += int(r["out_tok"])
                total_today += int(r["tot_tok"])
                estimated_records_today += int(r["est_cnt"])

        requests_60s = int(min_row["req_count"]) if min_row else 0
        input_60s = int(min_row["input_sum"]) if min_row else 0

        rem_rpd = max(0, rate_limit_rpd - active_records_today) if rate_limit_rpd is not None else None
        rem_rpm = max(0, rate_limit_rpm - requests_60s) if rate_limit_rpm is not None else None
        rem_tpm = max(0, rate_limit_tpm - input_60s) if rate_limit_tpm is not None else None

        if active_records_today == 0 or estimated_records_today == active_records_today:
            source: Literal["locally_estimated", "provider_confirmed", "mixed_local_and_provider"] = "locally_estimated"
            is_est = True
        elif estimated_records_today == 0:
            source = "provider_confirmed"
            is_est = False
        else:
            source = "mixed_local_and_provider"
            is_est = True

        return QuotaSnapshot(
            provider=norm_provider,
            model=norm_model,
            quota_scope=norm_scope,
            quota_enabled=quota_enabled,
            database_path=self.db_path.as_posix(),
            utc_date=utc_date,
            daily_reset_utc=next_reset,
            requests_today=active_records_today,
            succeeded_today=status_counts.get(RequestStatus.SUCCEEDED.value, 0),
            failed_today=status_counts.get(RequestStatus.FAILED.value, 0),
            rate_limited_today=status_counts.get(RequestStatus.RATE_LIMITED.value, 0),
            uncertain_today=status_counts.get(RequestStatus.UNCERTAIN.value, 0),
            active_reservations_today=status_counts.get(RequestStatus.RESERVED.value, 0),
            expired_reservations_today=status_counts.get(RequestStatus.EXPIRED.value, 0),
            configured_rpd_limit=rate_limit_rpd,
            estimated_remaining_daily_requests=rem_rpd,
            requests_last_60s=requests_60s,
            configured_rpm_limit=rate_limit_rpm,
            estimated_remaining_rpm=rem_rpm,
            input_tokens_last_60s=input_60s,
            configured_tpm_limit=rate_limit_tpm,
            estimated_remaining_tpm=rem_tpm,
            input_tokens_today=input_today,
            output_tokens_today=output_today,
            total_tokens_today=total_today,
            token_accounting_source=source,
            is_locally_estimated=is_est,
        )
