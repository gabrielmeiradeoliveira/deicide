---
name: llm-providers
description: Add or modify LLM providers, model selection, retries, and token or cost controls.
---
# Provider integration

- Implement provider-neutral request, response, usage, and error contracts.
- Support explicit model selection and optional fallback; never silently switch providers when code privacy policies differ.
- Support free-tier limits, rate-limit backoff, cancellation, timeouts, and bounded retries.
- Track reported token usage; label estimates as estimates.
- Do not assume subscription access provides API credits.
- Use fake providers in unit tests; keep real API calls opt-in.
