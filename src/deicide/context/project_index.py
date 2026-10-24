"""Deterministic, security-bounded repository indexing for Python AST analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from deicide.context.models import CallReference, FileAnalysis, FileAnalysisStatus, ImportReference, Symbol
from deicide.context.python_ast import PythonAstAnalyzer
from deicide.git.analyzer import ensure_safe_repo_file, normalize_repo_relative_path

_EXCLUDED_DIR_NAMES = frozenset(
    {
        ".git",
        ".venv",
        "venv",
        "env",
        ".env",
        "__pycache__",
        "node_modules",
        "build",
        "dist",
        ".tox",
        ".nox",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "site-packages",
        ".idea",
        ".vscode",
    }
)


@dataclass(frozen=True)
class IndexLimits:
    """Configurable safety limits for scanning repository source files."""

    max_files: int = 500
    max_file_size_bytes: int = 500_000
    max_total_bytes: int = 5_000_000


def file_path_to_module_name(relative_path: str) -> str:
    """Convert a repository-relative Python file path to its canonical module name.

    Supports root modules, `src/` layout, package submodules, and `__init__.py`.
    """
    cleaned = relative_path.replace("\\", "/").lstrip("./")
    posix = PurePosixPath(cleaned)
    parts = list(posix.parts)

    if parts and parts[0] == "src":
        parts = parts[1:]

    if not parts:
        return ""

    if parts[-1].endswith(".py"):
        parts[-1] = parts[-1][:-3]

    if parts[-1] == "__init__":
        parts = parts[:-1]

    return ".".join(parts)


@dataclass
class ProjectIndex:
    """In-memory index of parsed repository Python files, symbols, imports, and calls."""

    repo_root: Path
    file_analyses: dict[str, FileAnalysis] = field(default_factory=dict)
    module_to_file: dict[str, str] = field(default_factory=dict)
    file_to_module: dict[str, str] = field(default_factory=dict)
    symbols_by_id: dict[str, Symbol] = field(default_factory=dict)
    symbols_by_module: dict[str, list[Symbol]] = field(default_factory=dict)
    qualified_to_symbol: dict[tuple[str, str], Symbol] = field(default_factory=dict)
    diagnostics: list[str] = field(default_factory=list)
    limits_reached: bool = False

    def get_analysis(self, file_path: str) -> FileAnalysis | None:
        norm = normalize_repo_relative_path(file_path)
        return self.file_analyses.get(norm)

    def get_module_for_file(self, file_path: str) -> str | None:
        norm = normalize_repo_relative_path(file_path)
        return self.file_to_module.get(norm)

    def get_file_for_module(self, module_name: str) -> str | None:
        return self.module_to_file.get(module_name)

    def get_symbols_in_module(self, module_name: str) -> list[Symbol]:
        return self.symbols_by_module.get(module_name, [])

    def resolve_symbol(self, module_name: str, qualified_name: str) -> Symbol | None:
        return self.qualified_to_symbol.get((module_name, qualified_name))

    def get_symbol_by_id(self, symbol_id: str) -> Symbol | None:
        return self.symbols_by_id.get(symbol_id)

    def get_imports_for_file(self, file_path: str) -> list[ImportReference]:
        analysis = self.get_analysis(file_path)
        return analysis.imports if analysis else []

    def get_calls_for_file(self, file_path: str) -> list[CallReference]:
        analysis = self.get_analysis(file_path)
        return analysis.calls if analysis else []


def _discover_python_files(repo_root: Path, limits: IndexLimits) -> tuple[list[str], list[str], bool]:
    """Deterministically discover Python files within `repo_root` respecting exclusions and limits."""
    discovered: list[str] = []
    diagnostics: list[str] = []
    total_bytes = 0
    limits_reached = False

    resolved_root = repo_root.resolve()

    # Collect candidate files using os.walk with explicit directory filtering
    for root, _dirnames, filenames in os_walk_bounded(resolved_root):
        rel_root = Path(root).relative_to(resolved_root)
        for fname in filenames:
            if not fname.endswith(".py"):
                continue

            rel_file = str((rel_root / fname).as_posix()) if str(rel_root) != "." else fname
            try:
                safe_path = ensure_safe_repo_file(repo_root, rel_file)
            except Exception as exc:
                diagnostics.append(f"Security/path exclusion for '{rel_file}': {exc}")
                continue

            try:
                file_size = safe_path.stat().st_size
            except OSError as exc:
                diagnostics.append(f"Unable to stat file '{rel_file}': {exc}")
                continue

            if file_size > limits.max_file_size_bytes:
                diagnostics.append(
                    f"File '{rel_file}' exceeds max file size ({file_size} > {limits.max_file_size_bytes} bytes); skipped."
                )
                continue

            if total_bytes + file_size > limits.max_total_bytes:
                diagnostics.append("Aggregate source byte limit reached; stopping file discovery.")
                limits_reached = True
                break

            if len(discovered) >= limits.max_files:
                diagnostics.append("Maximum file count limit reached; stopping file discovery.")
                limits_reached = True
                break

            discovered.append(rel_file)
            total_bytes += file_size

        if limits_reached:
            break

    return sorted(discovered), diagnostics, limits_reached


def os_walk_bounded(start_path: Path):
    """Walk directories while pruning excluded directories in-place."""
    import os

    for root, dirs, files in os.walk(start_path, topdown=True, followlinks=False):
        dirs[:] = [d for d in dirs if d not in _EXCLUDED_DIR_NAMES and not d.startswith(".")]
        yield root, dirs, files


def build_project_index(
    repo_root: Path,
    explicit_files: list[str] | None = None,
    limits: IndexLimits | None = None,
    analyzer: PythonAstAnalyzer | None = None,
) -> ProjectIndex:
    """Build a complete, deterministic ProjectIndex for an analyzed repository."""
    active_limits = limits or IndexLimits()
    active_analyzer = analyzer or PythonAstAnalyzer()

    if explicit_files is not None:
        file_list = sorted({normalize_repo_relative_path(f) for f in explicit_files if f.endswith(".py")})
        diagnostics: list[str] = []
        limits_reached = False
    else:
        file_list, diagnostics, limits_reached = _discover_python_files(repo_root, active_limits)

    index = ProjectIndex(
        repo_root=repo_root,
        diagnostics=diagnostics,
        limits_reached=limits_reached,
    )

    for rel_path in file_list:
        try:
            analysis = active_analyzer.analyze_file(repo_root, rel_path)
        except Exception as exc:
            analysis = FileAnalysis(
                source_file=rel_path,
                status=FileAnalysisStatus.UNSUPPORTED,
                error_message=f"Analyzer error: {exc}",
            )

        index.file_analyses[rel_path] = analysis
        mod_name = file_path_to_module_name(rel_path)
        if mod_name:
            if mod_name in index.module_to_file and index.module_to_file[mod_name] != rel_path:
                index.diagnostics.append(
                    f"Ambiguous module '{mod_name}': mapped to '{index.module_to_file[mod_name]}' and '{rel_path}'"
                )
            else:
                index.module_to_file[mod_name] = rel_path
                index.file_to_module[rel_path] = mod_name

        if analysis.status == FileAnalysisStatus.PARSED:
            index.symbols_by_module.setdefault(mod_name, [])
            for sym in analysis.symbols:
                index.symbols_by_id[sym.id] = sym
                index.symbols_by_module[mod_name].append(sym)
                index.qualified_to_symbol[(mod_name, sym.qualified_name)] = sym

    return index
