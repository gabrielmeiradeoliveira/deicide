"""Deterministic offline FakeProvider for local testing, CLI verification, and evaluation."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

from deicide.core.config import DEFAULT_FAKE_MODEL
from deicide.core.models import TokenUsage
from deicide.providers.base import (
    ProviderRequest,
    ProviderResponse,
    estimate_token_count,
)

_UNTRUSTED_BLOCK_RE = re.compile(
    r"<UNTRUSTED_REPOSITORY_DATA>\s*(?P<payload>\{.*\})\s*</UNTRUSTED_REPOSITORY_DATA>",
    re.DOTALL,
)

_MUTABLE_DEFAULT_RE = re.compile(
    r"^\s*def\s+[A-Za-z_][A-Za-z0-9_]*\s*\([^)]*=\s*(\[\]|\{\}|set\(\))[^)]*\)\s*(?:->[^:]+)?:"
)
_DIV_BY_ZERO_LITERAL_RE = re.compile(r"(?<!/)/(?!/)\s*0(?![.0-9])|%\s*0(?![.0-9])")
_DIV_BY_LEN_RE = re.compile(r"/\s*len\(\s*(?P<var>[A-Za-z_][A-Za-z0-9_.]*)\s*\)")
_OFF_BY_ONE_RANGE_RE = re.compile(r"range\(\s*(?:0\s*,\s*)?len\(\s*(?P<var>[A-Za-z_][A-Za-z0-9_]*)\s*\)\s*\+\s*1\s*\)")
_BARE_EXCEPT_PASS_RE = re.compile(r"^\s*except(?:\s+Exception|\s+BaseException)?\s*:\s*(?:pass|\.\.\.)\s*$")
_UNCLOSED_OPEN_RE = re.compile(r"^\s*(?!with\b)[A-Za-z_][A-Za-z0-9_]*\s*=\s*open\(")
_IS_LITERAL_COMPARISON_RE = re.compile(r"\b(?:is|is\s+not)\s+(?:['\"][^'\"]*['\"]|\d+(?:\.\d+)?)\b")


class FakeProvider:
    """Deterministic offline provider supporting scripted responses and heuristic bug detection."""

    def __init__(
        self,
        model_name: str = DEFAULT_FAKE_MODEL,
        scripted_responses: list[str | Exception | Callable[[ProviderRequest], str]] | None = None,
    ) -> None:
        self._model_name = model_name
        self._scripted_responses = list(scripted_responses) if scripted_responses else []
        self.requests: list[ProviderRequest] = []

    @property
    def provider_name(self) -> str:
        return "fake"

    @property
    def model_name(self) -> str:
        return self._model_name

    def generate(self, request: ProviderRequest) -> ProviderResponse:
        self.requests.append(request)
        prompt_tokens = request.estimated_prompt_tokens

        if self._scripted_responses:
            next_item = self._scripted_responses.pop(0)
            if isinstance(next_item, Exception):
                raise next_item
            content = next_item(request) if callable(next_item) else next_item
        else:
            content = self._analyze_deterministically(request.user_prompt)

        completion_tokens = estimate_token_count(content)
        usage = TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            is_estimated=True,
            estimated_cost_usd=0.0,
        )
        return ProviderResponse(
            content=content,
            model_name=self._model_name,
            usage=usage,
            retries_performed=0,
        )

    def _analyze_deterministically(self, user_prompt: str) -> str:
        """Inspect structured diff/context data and emit deterministic correctness findings."""
        match = _UNTRUSTED_BLOCK_RE.search(user_prompt)
        if not match:
            return json.dumps({"findings": []})

        try:
            payload: dict[str, Any] = json.loads(match.group("payload"))
        except json.JSONDecodeError:
            return json.dumps({"findings": []})

        findings: list[dict[str, Any]] = []
        files = payload.get("files", [])
        if not isinstance(files, list):
            return json.dumps({"findings": []})

        for file_entry in files:
            if not isinstance(file_entry, dict):
                continue
            file_path = str(file_entry.get("file_path", ""))
            added_lines_raw = file_entry.get("added_lines", [])
            added_set = {int(x) for x in added_lines_raw if isinstance(x, int | str) and str(x).isdigit()}
            context_lines: dict[int, str] = {}

            for snippet in file_entry.get("context_snippets", []):
                if isinstance(snippet, dict):
                    for k, v in snippet.get("lines", {}).items():
                        if str(k).isdigit() and isinstance(v, str):
                            context_lines[int(k)] = v

            # Also parse added lines directly from hunks if context_lines lacks any
            for hunk in file_entry.get("hunks", []):
                if not isinstance(hunk, dict):
                    continue
                cur_new = int(hunk.get("new_start", 1))
                for raw_line in hunk.get("lines", []):
                    if not isinstance(raw_line, str):
                        continue
                    if raw_line.startswith("+") and not raw_line.startswith("+++") or raw_line.startswith(" "):
                        context_lines.setdefault(cur_new, raw_line[1:])
                        cur_new += 1

            all_context_text = "\n".join(context_lines.values())

            for line_no in sorted(added_set):
                line_text = context_lines.get(line_no, "")
                stripped = line_text.strip()
                if not stripped or stripped.startswith("#"):
                    continue

                # 1. Explicit division or modulo by zero
                if _DIV_BY_ZERO_LITERAL_RE.search(stripped):
                    findings.append(
                        {
                            "category": "correctness",
                            "severity": "critical",
                            "confidence": "high",
                            "file_path": file_path,
                            "start_line": line_no,
                            "end_line": line_no,
                            "title": "Division or modulo by zero",
                            "description": (
                                "This expression divides or computes modulo by literal zero, "
                                "which raises a ZeroDivisionError at runtime."
                            ),
                            "evidence": stripped,
                            "suggested_remediation": "Ensure the divisor is non-zero before dividing.",
                        }
                    )
                    continue

                # 2. Unchecked division by len(collection)
                div_len_match = _DIV_BY_LEN_RE.search(stripped)
                if div_len_match:
                    var_name = div_len_match.group("var")
                    # Check nearby lines before line_no for an emptiness check on var_name
                    window_lines = [context_lines.get(n, "") for n in range(max(1, line_no - 6), line_no + 1)]
                    window_text = "\n".join(window_lines)
                    has_guard = bool(
                        re.search(
                            rf"\bif\s+(?:not\s+{re.escape(var_name)}|len\(\s*{re.escape(var_name)}\s*\)\s*[=!<>]=?\s*0|{re.escape(var_name)}\s*:\s*)",
                            window_text,
                        )
                    )
                    if not has_guard:
                        findings.append(
                            {
                                "category": "boundary_condition",
                                "severity": "high",
                                "confidence": "high",
                                "file_path": file_path,
                                "start_line": line_no,
                                "end_line": line_no,
                                "title": f"Potential ZeroDivisionError when '{var_name}' is empty",
                                "description": (
                                    f"Dividing by 'len({var_name})' without verifying that '{var_name}' "
                                    "is non-empty will raise ZeroDivisionError when given an empty collection."
                                ),
                                "evidence": stripped,
                                "suggested_remediation": (
                                    f"Guard against an empty '{var_name}' before dividing by 'len({var_name})'."
                                ),
                            }
                        )
                        continue

                # 3. Mutable default argument in function signature
                if _MUTABLE_DEFAULT_RE.search(stripped):
                    findings.append(
                        {
                            "category": "state_mutation",
                            "severity": "high",
                            "confidence": "high",
                            "file_path": file_path,
                            "start_line": line_no,
                            "end_line": line_no,
                            "title": "Mutable default argument shared across function calls",
                            "description": (
                                "Using a mutable literal ([], {}, or set()) as a default parameter value "
                                "causes state to persist and leak across subsequent invocations."
                            ),
                            "evidence": stripped,
                            "suggested_remediation": (
                                "Default the parameter to None and initialize a new container inside the function body."
                            ),
                        }
                    )
                    continue

                # 4. Off-by-one loop bound range(len(x) + 1)
                obo_match = _OFF_BY_ONE_RANGE_RE.search(stripped)
                if obo_match:
                    seq_var = obo_match.group("var")
                    findings.append(
                        {
                            "category": "boundary_condition",
                            "severity": "high",
                            "confidence": "high",
                            "file_path": file_path,
                            "start_line": line_no,
                            "end_line": line_no,
                            "title": f"Off-by-one upper bound in range(len({seq_var}) + 1)",
                            "description": (
                                f"Iterating through 'range(len({seq_var}) + 1)' includes index 'len({seq_var})', "
                                f"which is out of bounds when indexing '{seq_var}' and raises IndexError."
                            ),
                            "evidence": stripped,
                            "suggested_remediation": f"Use 'range(len({seq_var}))' or 'enumerate({seq_var})'.",
                        }
                    )
                    continue

                # 5. Silent exception swallowing
                if _BARE_EXCEPT_PASS_RE.search(stripped) or (
                    stripped in {"except:", "except Exception:"}
                    and context_lines.get(line_no + 1, "").strip() == "pass"
                ):
                    end_ln = line_no + 1 if context_lines.get(line_no + 1, "").strip() == "pass" else line_no
                    findings.append(
                        {
                            "category": "error_handling",
                            "severity": "medium",
                            "confidence": "high",
                            "file_path": file_path,
                            "start_line": line_no,
                            "end_line": end_ln,
                            "title": "Silent exception swallowing masks runtime failures",
                            "description": (
                                "Catching broad exceptions and ignoring them with 'pass' hides unexpected "
                                "errors and can leave the program in an inconsistent state."
                            ),
                            "evidence": stripped,
                            "suggested_remediation": (
                                "Catch only specific expected exceptions and handle or log the failure explicitly."
                            ),
                        }
                    )
                    continue

                # 6. Identity comparison with value literal (`is` with string/number literal)
                if _IS_LITERAL_COMPARISON_RE.search(stripped):
                    findings.append(
                        {
                            "category": "logic_error",
                            "severity": "medium",
                            "confidence": "high",
                            "file_path": file_path,
                            "start_line": line_no,
                            "end_line": line_no,
                            "title": "Identity comparison ('is') used instead of equality ('==') with literal",
                            "description": (
                                "Using 'is' or 'is not' to compare with a string or numeric literal checks "
                                "object identity rather than value equality, leading to unreliable results."
                            ),
                            "evidence": stripped,
                            "suggested_remediation": "Use '==' or '!=' when comparing values to literals.",
                        }
                    )
                    continue

                # 7. Unclosed file handle opened without context manager
                if _UNCLOSED_OPEN_RE.search(stripped) and ".close()" not in all_context_text:
                    findings.append(
                        {
                            "category": "resource_management",
                            "severity": "medium",
                            "confidence": "medium",
                            "file_path": file_path,
                            "start_line": line_no,
                            "end_line": line_no,
                            "title": "File opened without context manager or explicit close()",
                            "description": (
                                "Opening a file without a 'with' statement or guaranteed '.close()' "
                                "can leak file descriptors, especially if an exception occurs."
                            ),
                            "evidence": stripped,
                            "suggested_remediation": "Use a 'with open(...) as f:' context manager.",
                        }
                    )

        return json.dumps({"findings": findings})
