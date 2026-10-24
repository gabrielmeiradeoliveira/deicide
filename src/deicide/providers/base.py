"""Provider-independent contracts for LLM interactions."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from deicide.core.models import TokenUsage, estimate_token_count


class ProviderRequest(BaseModel):
    """Provider-neutral request payload."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    system_prompt: str
    user_prompt: str
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_output_tokens: int = Field(default=4096, ge=64, le=32768)
    response_json: bool = True

    @property
    def estimated_prompt_tokens(self) -> int:
        return estimate_token_count(self.system_prompt) + estimate_token_count(self.user_prompt)


class ProviderResponse(BaseModel):
    """Provider-neutral response payload with token usage and retry metadata."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    content: str
    model_name: str
    usage: TokenUsage
    retries_performed: int = Field(default=0, ge=0)


@runtime_checkable
class LLMProvider(Protocol):
    """Protocol implemented by all AI provider adapters."""

    @property
    def provider_name(self) -> str: ...

    @property
    def model_name(self) -> str: ...

    def generate(self, request: ProviderRequest) -> ProviderResponse: ...
