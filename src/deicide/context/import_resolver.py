"""Deterministic resolution of Python imports against a ProjectIndex."""

from __future__ import annotations

from dataclasses import dataclass, field

from deicide.context.models import ImportKind, SourceLocation, Symbol
from deicide.context.project_index import ProjectIndex


@dataclass(frozen=True)
class ResolvedImport:
    """Resolved binding for an imported name within a file."""

    local_name: str
    location: SourceLocation
    target_module: str
    target_symbol_name: str | None = None
    target_symbol: Symbol | None = None
    is_submodule: bool = False
    is_external: bool = False
    is_ambiguous: bool = False
    is_wildcard: bool = False


@dataclass
class FileImportMap:
    """Complete map of imported local bindings for a single source file."""

    file_path: str
    current_module: str
    bindings: dict[str, ResolvedImport] = field(default_factory=dict)
    wildcard_modules: list[str] = field(default_factory=list)

    def get_binding(self, name: str) -> ResolvedImport | None:
        return self.bindings.get(name)


def resolve_relative_module(
    current_module: str,
    is_init_file: bool,
    relative_level: int,
    imported_module: str,
) -> str | None:
    """Resolve a relative import level and module string to an absolute module name."""
    if relative_level <= 0:
        return imported_module

    parts = current_module.split(".") if current_module else []
    # If not __init__.py, relative level 1 starts from the containing package
    base_parts = parts if is_init_file else parts[:-1]

    # Level 1 is the package itself; each additional level goes up one package
    up_steps = relative_level - 1
    if up_steps > len(base_parts):
        return None  # Escapes root package

    target_pkg_parts = base_parts[: len(base_parts) - up_steps] if up_steps > 0 else base_parts
    base_name = ".".join(target_pkg_parts)

    if not imported_module:
        return base_name
    return f"{base_name}.{imported_module}" if base_name else imported_module


class ImportResolver:
    """Resolves ImportReferences in a file against an indexed repository."""

    def __init__(self, index: ProjectIndex) -> None:
        self.index = index

    def resolve_file_imports(self, file_path: str) -> FileImportMap:
        """Resolve all import statements in `file_path` into bound symbols/modules."""
        current_module = self.index.get_module_for_file(file_path) or ""
        is_init = file_path.replace("\\", "/").endswith("/__init__.py") or file_path == "__init__.py"
        import_refs = self.index.get_imports_for_file(file_path)

        result = FileImportMap(file_path=file_path, current_module=current_module)

        for ref in import_refs:
            # Handle wildcards
            if ref.import_kind == ImportKind.WILDCARD or ref.is_wildcard:
                target_mod = (
                    resolve_relative_module(current_module, is_init, ref.relative_level, ref.imported_module)
                    if ref.relative_level > 0
                    else ref.imported_module
                )
                if target_mod:
                    result.wildcard_modules.append(target_mod)
                continue

            # Resolve module name (relative or absolute)
            if ref.relative_level > 0:
                target_mod = resolve_relative_module(
                    current_module,
                    is_init,
                    ref.relative_level,
                    ref.imported_module,
                )
                if target_mod is None:
                    # Invalid relative import
                    continue
            else:
                target_mod = ref.imported_module

            local_name = ref.local_name

            # Form 1: `import X` or `import X as Y`
            if ref.import_kind == ImportKind.MODULE:
                is_local = target_mod in self.index.module_to_file
                resolved = ResolvedImport(
                    local_name=local_name,
                    location=ref.location,
                    target_module=target_mod,
                    target_symbol_name=None,
                    target_symbol=None,
                    is_submodule=is_local,
                    is_external=not is_local,
                )
                result.bindings[local_name] = resolved
                continue

            # Form 2: `from X import Y` or `from X import Y as Z`
            symbol_name = ref.imported_name
            submodule_candidate = f"{target_mod}.{symbol_name}" if target_mod else (symbol_name or "")

            # Check if `target_mod.symbol_name` is actually an indexed submodule
            if submodule_candidate in self.index.module_to_file:
                resolved = ResolvedImport(
                    local_name=local_name,
                    location=ref.location,
                    target_module=submodule_candidate,
                    target_symbol_name=None,
                    target_symbol=None,
                    is_submodule=True,
                    is_external=False,
                )
                result.bindings[local_name] = resolved
                continue

            # Check if `target_mod` is an indexed module and `symbol_name` is a symbol in it
            if target_mod in self.index.module_to_file:
                target_sym = self.index.resolve_symbol(target_mod, symbol_name or "") if symbol_name else None
                resolved = ResolvedImport(
                    local_name=local_name,
                    location=ref.location,
                    target_module=target_mod,
                    target_symbol_name=symbol_name,
                    target_symbol=target_sym,
                    is_submodule=False,
                    is_external=False,
                )
                result.bindings[local_name] = resolved
                continue

            # Otherwise external module / package
            resolved = ResolvedImport(
                local_name=local_name,
                location=ref.location,
                target_module=target_mod,
                target_symbol_name=symbol_name,
                target_symbol=None,
                is_submodule=False,
                is_external=True,
            )
            result.bindings[local_name] = resolved

        return result
