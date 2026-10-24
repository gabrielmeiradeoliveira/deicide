"""Module registration and typed contracts for DEICIDE analysis modules."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from deicide.core.errors import ConfigurationError
from deicide.core.models import BudgetTracker, CandidateFinding
from deicide.git.models import DiffBundle, FileContext
from deicide.providers.base import LLMProvider


@runtime_checkable
class AnalysisModule(Protocol):
    """Typed contract implemented by every DEICIDE analysis module."""

    @property
    def module_id(self) -> str:
        """Stable technical identifier (e.g., 'code_review')."""
        ...

    @property
    def description(self) -> str:
        """Human-readable summary of the module's responsibility."""
        ...

    def analyze(
        self,
        diff_bundle: DiffBundle,
        contexts: dict[str, FileContext],
        provider: LLMProvider,
        tracker: BudgetTracker,
    ) -> list[CandidateFinding]:
        """Execute analysis on the diff and bounded context, returning candidate findings."""
        ...


class ModuleRegistry:
    """Registry managing available DEICIDE analysis modules by technical `module_id`."""

    def __init__(self) -> None:
        self._modules: dict[str, AnalysisModule] = {}

    def register(self, module: AnalysisModule) -> None:
        module_id = module.module_id.strip()
        if not module_id:
            raise ConfigurationError("Module technical identifier cannot be empty.")
        if module_id in self._modules:
            raise ConfigurationError(f"Module '{module_id}' is already registered.")
        self._modules[module_id] = module

    def get(self, module_id: str) -> AnalysisModule:
        if module_id not in self._modules:
            available = ", ".join(sorted(self._modules)) or "none"
            raise ConfigurationError(f"Unknown analysis module '{module_id}'. Registered modules: {available}.")
        return self._modules[module_id]

    def list_modules(self) -> list[AnalysisModule]:
        return [self._modules[k] for k in sorted(self._modules)]
