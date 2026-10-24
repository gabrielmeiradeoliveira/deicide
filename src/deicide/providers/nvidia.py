"""NVIDIA NIM (OpenAI-compatible chat completions) REST API adapter using the standard library."""

from __future__ import annotations

import email.utils
import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from pydantic import SecretStr

from deicide.core.errors import (
    BudgetExceededError,
    ConfigurationError,
    MissingCredentialsError,
    ProviderError,
    QuotaExhaustedError,
    RateLimitError,
)
from deicide.core.models import BudgetTracker, ExecutionBudget, TokenUsage
from deicide.providers.base import (
    ProviderRequest,
    ProviderResponse,
    estimate_token_count,
)
from deicide.quota.store import RequestStatus, UsageStore

DEFAULT_NVIDIA_MODEL = "z-ai/glm-5.3-flash"
DEFAULT_NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
MAX_RETRY_WAIT_SECONDS = 60.0


def _redact_secret(text: str, secret: str) -> str:
    """Ensure API key never appears in error messages or logs."""
    if not secret or not text:
        return text
    return text.replace(secret, "[REDACTED]")


def _extract_header_case_insensitive(
    headers: dict[str, str] | Any | None,
    target_name: str,
) -> str | None:
    if headers is None:
        return None
    target_lower = target_name.lower()
    if hasattr(headers, "items"):
        for k, v in headers.items():
            if str(k).lower() == target_lower:
                return str(v)
    return None


class NvidiaProvider:
    """Provider adapter for NVIDIA NIM chat/completions API."""

    def __init__(
        self,
        api_key: SecretStr,
        model_name: str = DEFAULT_NVIDIA_MODEL,
        base_url: str = DEFAULT_NVIDIA_BASE_URL,
        timeout_seconds: float | None = None,
        max_retries: int | None = None,
        initial_backoff_seconds: float = 1.0,
        sleep_fn: Callable[[float], None] = time.sleep,
        transport_fn: Callable[[urllib.request.Request, float], tuple[int, dict[str, str], str]] | None = None,
        *,
        budget: ExecutionBudget | None = None,
        usage_store: UsageStore | None = None,
        quota_scope: str = "default",
    ) -> None:
        raw_key = api_key.get_secret_value().strip() if api_key else ""
        if not raw_key:
            raise MissingCredentialsError(
                "NvidiaProvider requires a non-empty API key (NVIDIA_API_KEY or DEICIDE_API_KEY)."
            )
        cleaned_base = base_url.strip().rstrip("/") if base_url else DEFAULT_NVIDIA_BASE_URL
        if not cleaned_base.startswith("https://"):
            raise ConfigurationError(f"Invalid NVIDIA_BASE_URL '{base_url}': URL must start with 'https://'.")

        self._api_key = api_key
        self._model_name = model_name.strip() or DEFAULT_NVIDIA_MODEL
        self.base_url = cleaned_base
        self.timeout_seconds = (
            float(timeout_seconds)
            if timeout_seconds is not None
            else (budget.request_timeout_seconds if budget is not None else 60.0)
        )
        self.max_retries = (
            int(max_retries) if max_retries is not None else (budget.max_retries if budget is not None else 0)
        )
        self.initial_backoff_seconds = initial_backoff_seconds
        self._sleep = sleep_fn
        self._transport_fn = transport_fn or self._default_transport
        self.budget = budget
        self.usage_store = usage_store
        self.quota_scope = quota_scope
        self._bound_tracker: BudgetTracker | None = None

    @property
    def provider_name(self) -> str:
        return "nvidia"

    @property
    def model_name(self) -> str:
        return self._model_name

    def bind_tracker(self, tracker: BudgetTracker | None) -> None:
        """Bind an active BudgetTracker for per-attempt reservation and reconciliation."""
        self._bound_tracker = tracker

    @staticmethod
    def _default_transport(
        req: urllib.request.Request,
        timeout: float,
    ) -> tuple[int, dict[str, str], str]:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            status = int(getattr(resp, "status", 200))
            headers = {k.lower(): v for k, v in resp.headers.items()}
            body = resp.read().decode("utf-8", errors="replace")
            return status, headers, body

    def _build_url(self) -> str:
        return f"{self.base_url}/chat/completions"

    def _build_payload(self, request: ProviderRequest) -> bytes:
        body: dict[str, Any] = {
            "model": self._model_name,
            "messages": [
                {"role": "system", "content": request.system_prompt},
                {"role": "user", "content": request.user_prompt},
            ],
            "temperature": request.temperature,
            "max_tokens": request.max_output_tokens,
        }
        return json.dumps(body).encode("utf-8")

    def _resolve_active_tracker(self) -> BudgetTracker | None:
        if self._bound_tracker is not None:
            return self._bound_tracker
        if self.usage_store is not None or self.budget is not None:
            effective_budget = self.budget or ExecutionBudget(
                request_timeout_seconds=self.timeout_seconds,
                max_retries=self.max_retries,
            )
            return BudgetTracker(
                effective_budget,
                usage_store=self.usage_store,
                provider_name="nvidia",
                model_name=self._model_name,
                quota_scope=self.quota_scope,
            )
        return None

    def _can_attempt_retry(self, attempt: int, tracker: BudgetTracker | None) -> bool:
        if attempt >= self.max_retries:
            return False
        return not (tracker is not None and tracker.requests_made >= tracker.budget.max_requests)

    def generate(self, request: ProviderRequest) -> ProviderResponse:
        secret = self._api_key.get_secret_value()
        url = self._build_url()
        payload_bytes = self._build_payload(request)
        estimated_in = request.estimated_prompt_tokens
        active_tracker = self._resolve_active_tracker()

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {secret}",
        }

        retries = 0
        last_error: Exception | None = None

        for attempt in range(self.max_retries + 1):
            if not url.startswith("https://"):
                raise ProviderError("NVIDIA API URL must use https:// scheme.")

            reservation_id: str | None = None
            if active_tracker is not None:
                reservation_id = active_tracker.reserve_attempt(
                    estimated_in,
                    provider_name="nvidia",
                    model_name=self._model_name,
                    is_retry=(attempt > 0),
                )

            http_req = urllib.request.Request(  # noqa: S310
                url=url,
                data=payload_bytes,
                headers=headers,
                method="POST",
            )
            try:
                status, resp_headers, resp_body = self._transport_fn(http_req, self.timeout_seconds)
                retry_after_val = _extract_header_case_insensitive(resp_headers, "Retry-After")

                if status == 429:
                    if active_tracker is not None:
                        active_tracker.fail_attempt(
                            reservation_id,
                            status=RequestStatus.RATE_LIMITED,
                            estimated_prompt_tokens=estimated_in,
                        )
                    if self._can_attempt_retry(attempt, active_tracker):
                        delay = self._compute_backoff(attempt, retry_after_val)
                        retries += 1
                        self._sleep(delay)
                        continue
                    raise RateLimitError(f"NVIDIA API rate limit exceeded (HTTP 429) after {retries} retries.")

                if status in {502, 503, 504}:
                    if active_tracker is not None:
                        active_tracker.fail_attempt(
                            reservation_id,
                            status=RequestStatus.FAILED,
                            estimated_prompt_tokens=estimated_in,
                        )
                    if self._can_attempt_retry(attempt, active_tracker):
                        delay = self._compute_backoff(attempt, retry_after_val)
                        retries += 1
                        self._sleep(delay)
                        continue
                    safe_msg = _redact_secret(resp_body[:300], secret)
                    raise ProviderError(f"NVIDIA API server error (HTTP {status}) after {retries} retries: {safe_msg}")

                if status >= 400:
                    if active_tracker is not None:
                        active_tracker.fail_attempt(
                            reservation_id,
                            status=RequestStatus.FAILED,
                            estimated_prompt_tokens=estimated_in,
                        )
                    safe_msg = _redact_secret(resp_body[:300], secret)
                    raise ProviderError(f"NVIDIA API returned HTTP {status}: {safe_msg}")

                try:
                    parsed_resp = self._parse_response(resp_body, request, retries)
                except Exception:
                    if active_tracker is not None:
                        active_tracker.fail_attempt(
                            reservation_id,
                            status=RequestStatus.FAILED,
                            estimated_prompt_tokens=estimated_in,
                        )
                    raise

                if active_tracker is not None:
                    active_tracker.complete_attempt(reservation_id, parsed_resp.usage)
                return parsed_resp

            except urllib.error.HTTPError as exc:
                code = exc.code
                retry_after_header = _extract_header_case_insensitive(exc.headers, "Retry-After")
                rec_status = RequestStatus.RATE_LIMITED if code == 429 else RequestStatus.FAILED
                if active_tracker is not None:
                    active_tracker.fail_attempt(
                        reservation_id,
                        status=rec_status,
                        estimated_prompt_tokens=estimated_in,
                    )

                if code in {429, 502, 503, 504} and self._can_attempt_retry(attempt, active_tracker):
                    delay = self._compute_backoff(attempt, retry_after_header)
                    retries += 1
                    self._sleep(delay)
                    last_error = exc
                    continue

                if code == 429:
                    raise RateLimitError(f"NVIDIA API rate limit exceeded (HTTP 429) after {retries} retries.") from exc

                err_detail = ""
                if exc.fp is not None:
                    try:
                        err_detail = exc.fp.read().decode("utf-8", errors="replace")[:300]
                    except Exception:
                        err_detail = ""
                safe_detail = _redact_secret(err_detail or str(exc.reason), secret)
                raise ProviderError(f"NVIDIA API request failed with HTTP {code}: {safe_detail}") from exc

            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                if active_tracker is not None:
                    active_tracker.fail_attempt(
                        reservation_id,
                        status=RequestStatus.UNCERTAIN,
                        estimated_prompt_tokens=estimated_in,
                    )
                if self._can_attempt_retry(attempt, active_tracker):
                    delay = self._compute_backoff(attempt, None)
                    retries += 1
                    self._sleep(delay)
                    last_error = exc
                    continue
                safe_reason = _redact_secret(str(exc), secret)
                raise ProviderError(f"NVIDIA API network/timeout error after {retries} retries: {safe_reason}") from exc
            except (ProviderError, QuotaExhaustedError, BudgetExceededError):
                raise
            except Exception as exc:
                if active_tracker is not None:
                    active_tracker.fail_attempt(
                        reservation_id,
                        status=RequestStatus.FAILED,
                        estimated_prompt_tokens=estimated_in,
                    )
                safe_msg = _redact_secret(str(exc), secret)
                raise ProviderError(f"Unexpected NVIDIA provider failure: {safe_msg}") from exc

        safe_last = _redact_secret(str(last_error), secret)
        raise ProviderError(f"NVIDIA API request failed after {retries} retries: {safe_last}")

    def _compute_backoff(self, attempt: int, retry_after: str | None) -> float:
        if retry_after and retry_after.strip():
            cleaned = retry_after.strip()
            try:
                parsed = float(cleaned)
                if parsed > MAX_RETRY_WAIT_SECONDS:
                    raise RateLimitError(
                        f"Provider requested Retry-After of {parsed:.1f}s, exceeding maximum "
                        f"retry wait ceiling ({MAX_RETRY_WAIT_SECONDS:.1f}s)."
                    )
                if parsed > 0:
                    return parsed
            except ValueError:
                try:
                    dt = email.utils.parsedate_to_datetime(cleaned)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=UTC)
                    delta = (dt - datetime.now(tz=UTC)).total_seconds()
                    if delta > MAX_RETRY_WAIT_SECONDS:
                        raise RateLimitError(
                            f"Provider requested Retry-After until {cleaned} ({delta:.1f}s), "
                            f"exceeding maximum retry wait ceiling ({MAX_RETRY_WAIT_SECONDS:.1f}s)."
                        )
                    if delta > 0:
                        return float(delta)
                except RateLimitError:
                    raise
                except (TypeError, ValueError, OverflowError):
                    return float(self.initial_backoff_seconds * (2**attempt))
        return float(self.initial_backoff_seconds * (2**attempt))

    def _parse_response(
        self,
        raw_body: str,
        request: ProviderRequest,
        retries: int,
    ) -> ProviderResponse:
        try:
            data = json.loads(raw_body)
        except json.JSONDecodeError as exc:
            raise ProviderError("NVIDIA API returned invalid JSON envelope.") from exc

        if not isinstance(data, dict):
            raise ProviderError("NVIDIA API response root must be a JSON object.")

        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ProviderError("NVIDIA API response contained no choices.")

        first = choices[0]
        message = first.get("message") if isinstance(first, dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str) or not content.strip():
            raise ProviderError("NVIDIA API choice message content was empty.")

        combined_text = content.strip()

        usage_meta = data.get("usage")
        if isinstance(usage_meta, dict) and "prompt_tokens" in usage_meta:
            prompt_tokens = int(usage_meta.get("prompt_tokens", 0))
            completion_tokens = int(usage_meta.get("completion_tokens", 0))
            total_tokens = int(usage_meta.get("total_tokens", prompt_tokens + completion_tokens))
            usage = TokenUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                is_estimated=False,
            )
        else:
            prompt_tokens = request.estimated_prompt_tokens
            completion_tokens = estimate_token_count(combined_text)
            usage = TokenUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
                is_estimated=True,
            )

        return ProviderResponse(
            content=combined_text,
            model_name=self._model_name,
            usage=usage,
            retries_performed=retries,
        )
