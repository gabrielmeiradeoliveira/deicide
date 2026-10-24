"""Deterministic Python AST extractor for symbols, lexical scopes, imports, and calls."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

from deicide.context.models import (
    CallReference,
    FileAnalysis,
    FileAnalysisStatus,
    ImportKind,
    ImportReference,
    SourceLocation,
    Symbol,
    SymbolKind,
)
from deicide.core.errors import SecurityViolationError
from deicide.git.analyzer import ensure_safe_repo_file, normalize_repo_relative_path


@dataclass(frozen=True)
class _ScopeEntry:
    name: str
    kind: SymbolKind
    symbol_id: str
    start_line: int
    end_line: int


def _extract_decorator_name(decorator_node: ast.AST) -> str:
    """Extract string representation of a decorator node."""
    try:
        return ast.unparse(decorator_node).strip()
    except Exception:
        if isinstance(decorator_node, ast.Name):
            return decorator_node.id
        if isinstance(decorator_node, ast.Attribute):
            return f"{_extract_decorator_name(decorator_node.value)}.{decorator_node.attr}"
        return "decorator"


def _build_symbol_id(
    file_path: str,
    qualified_name: str,
    kind: SymbolKind,
    start_line: int,
    end_line: int,
) -> str:
    """Build a deterministic, stable identifier for an extracted symbol."""
    norm_path = file_path.replace("\\", "/").lstrip("./")
    return f"{norm_path}::{qualified_name}#{kind.value}@{start_line}:{end_line}"


def _flatten_attribute_chain(node: ast.AST) -> tuple[list[str], bool]:
    """Recursively decompose an attribute expression into a list of name parts.

    Returns (chain, is_dynamic).
    """
    if isinstance(node, ast.Name):
        return [node.id], False
    if isinstance(node, ast.Attribute):
        base_chain, is_dyn = _flatten_attribute_chain(node.value)
        return [*base_chain, node.attr], is_dyn
    # Dynamic root (Call, Subscript, etc.)
    return ["<dynamic>"], True


def _extract_names_from_target(target: ast.AST) -> list[str]:
    names: list[str] = []
    if isinstance(target, ast.Name):
        names.append(target.id)
    elif isinstance(target, (ast.Tuple, ast.List)):
        for elt in target.elts:
            names.extend(_extract_names_from_target(elt))
    elif isinstance(target, ast.Starred):
        names.extend(_extract_names_from_target(target.value))
    return names


def _extract_function_local_bindings(node: ast.FunctionDef | ast.AsyncFunctionDef) -> frozenset[str]:
    names: set[str] = set()
    for arg in getattr(node.args, "posonlyargs", []) + node.args.args + node.args.kwonlyargs:
        names.add(arg.arg)
    if node.args.vararg:
        names.add(node.args.vararg.arg)
    if node.args.kwarg:
        names.add(node.args.kwarg.arg)

    for child in ast.walk(node):
        if child is node:
            continue
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(child.name)
        elif isinstance(child, ast.Assign):
            for t in child.targets:
                names.update(_extract_names_from_target(t))
        elif (
            isinstance(child, ast.AnnAssign)
            and isinstance(child.target, ast.Name)
            or isinstance(child, ast.AugAssign)
            and isinstance(child.target, ast.Name)
        ):
            names.add(child.target.id)
        elif isinstance(child, ast.For):
            names.update(_extract_names_from_target(child.target))
        elif isinstance(child, ast.With):
            for item in child.items:
                if item.optional_vars:
                    names.update(_extract_names_from_target(item.optional_vars))
        elif isinstance(child, ast.ExceptHandler) and child.name:
            names.add(child.name)
        elif isinstance(child, ast.NamedExpr) and isinstance(child.target, ast.Name):
            names.add(child.target.id)

    return frozenset(names)


class _PythonAstVisitor(ast.NodeVisitor):
    """AST visitor extracting symbols, imports, and calls with scope tracking."""

    def __init__(self, file_path: str, total_lines: int) -> None:
        self.file_path = file_path
        self.total_lines = total_lines
        self.scope_stack: list[_ScopeEntry] = []
        self.symbols: list[Symbol] = []
        self.imports: list[ImportReference] = []
        self.calls: list[CallReference] = []

    def _current_scope_path(self) -> tuple[str, ...]:
        return tuple(s.name for s in self.scope_stack)

    def _current_parent_id(self) -> str | None:
        return self.scope_stack[-1].symbol_id if self.scope_stack else None

    def _make_location(self, node: ast.AST, custom_start_line: int | None = None) -> SourceLocation:
        start_line = custom_start_line if custom_start_line is not None else getattr(node, "lineno", 1)
        end_line = getattr(node, "end_lineno", start_line)
        start_col = getattr(node, "col_offset", None)
        end_col = getattr(node, "end_col_offset", None)
        # Ensure start_line >= 1 and end_line >= start_line
        start_line = max(1, start_line)
        end_line = max(start_line, end_line)
        return SourceLocation(
            file_path=self.file_path,
            start_line=start_line,
            end_line=end_line,
            start_col=start_col,
            end_col=end_col,
        )

    def _compute_definition_start_line(self, node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> int:
        """Return the start line covering decorators if present, else node.lineno."""
        if node.decorator_list:
            dec_lines = [getattr(d, "lineno", node.lineno) for d in node.decorator_list]
            return min(min(dec_lines), node.lineno)
        return node.lineno

    def visit_Import(self, node: ast.Import) -> None:
        loc = self._make_location(node)
        for alias in node.names:
            ref = ImportReference(
                source_file=self.file_path,
                location=loc,
                imported_module=alias.name,
                imported_name=None,
                local_name=alias.asname or alias.name,
                relative_level=0,
                is_wildcard=False,
                import_kind=ImportKind.MODULE,
            )
            self.imports.append(ref)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        loc = self._make_location(node)
        mod = node.module or ""
        level = node.level or 0
        for alias in node.names:
            is_wild = alias.name == "*"
            kind = ImportKind.WILDCARD if is_wild else ImportKind.FROM_MODULE
            ref = ImportReference(
                source_file=self.file_path,
                location=loc,
                imported_module=mod,
                imported_name=None if is_wild else alias.name,
                local_name=alias.name if is_wild else (alias.asname or alias.name),
                relative_level=level,
                is_wildcard=is_wild,
                import_kind=kind,
            )
            self.imports.append(ref)
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        is_nested = any(s.kind in (SymbolKind.CLASS, SymbolKind.NESTED_CLASS) for s in self.scope_stack)
        kind = SymbolKind.NESTED_CLASS if is_nested else SymbolKind.CLASS

        qual_name = ".".join([*self._current_scope_path(), node.name])
        start_line = self._compute_definition_start_line(node)
        end_line = max(start_line, getattr(node, "end_lineno", node.lineno))
        loc = self._make_location(node, custom_start_line=start_line)

        sym_id = _build_symbol_id(self.file_path, qual_name, kind, start_line, end_line)
        decorators = [_extract_decorator_name(d) for d in node.decorator_list]
        docstring = ast.get_docstring(node)

        symbol = Symbol(
            id=sym_id,
            name=node.name,
            qualified_name=qual_name,
            kind=kind,
            location=loc,
            parent_id=self._current_parent_id(),
            scope_path=self._current_scope_path(),
            decorators=decorators,
            docstring=docstring,
        )
        self.symbols.append(symbol)

        entry = _ScopeEntry(
            name=node.name,
            kind=kind,
            symbol_id=sym_id,
            start_line=start_line,
            end_line=end_line,
        )
        self.scope_stack.append(entry)
        for child in node.body:
            self.visit(child)
        self.scope_stack.pop()

    def _classify_function(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        is_async: bool,
    ) -> SymbolKind:
        dec_names = [_extract_decorator_name(d) for d in node.decorator_list]
        parent = self.scope_stack[-1] if self.scope_stack else None

        if parent is not None and parent.kind in (SymbolKind.CLASS, SymbolKind.NESTED_CLASS):
            if "staticmethod" in dec_names:
                return SymbolKind.STATIC_METHOD
            if "classmethod" in dec_names:
                return SymbolKind.CLASS_METHOD
            return SymbolKind.ASYNC_METHOD if is_async else SymbolKind.METHOD

        if any(
            s.kind
            in (
                SymbolKind.FUNCTION,
                SymbolKind.ASYNC_FUNCTION,
                SymbolKind.METHOD,
                SymbolKind.ASYNC_METHOD,
                SymbolKind.CLASS_METHOD,
                SymbolKind.STATIC_METHOD,
                SymbolKind.NESTED_FUNCTION,
                SymbolKind.NESTED_ASYNC_FUNCTION,
            )
            for s in self.scope_stack
        ):
            return SymbolKind.NESTED_ASYNC_FUNCTION if is_async else SymbolKind.NESTED_FUNCTION

        return SymbolKind.ASYNC_FUNCTION if is_async else SymbolKind.FUNCTION

    def _process_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef, is_async: bool) -> None:
        kind = self._classify_function(node, is_async)
        qual_name = ".".join([*self._current_scope_path(), node.name])
        start_line = self._compute_definition_start_line(node)
        end_line = max(start_line, getattr(node, "end_lineno", node.lineno))
        loc = self._make_location(node, custom_start_line=start_line)

        sym_id = _build_symbol_id(self.file_path, qual_name, kind, start_line, end_line)
        decorators = [_extract_decorator_name(d) for d in node.decorator_list]
        docstring = ast.get_docstring(node)
        local_bindings = _extract_function_local_bindings(node)

        symbol = Symbol(
            id=sym_id,
            name=node.name,
            qualified_name=qual_name,
            kind=kind,
            location=loc,
            parent_id=self._current_parent_id(),
            scope_path=self._current_scope_path(),
            decorators=decorators,
            docstring=docstring,
            local_bindings=local_bindings,
        )
        self.symbols.append(symbol)

        entry = _ScopeEntry(
            name=node.name,
            kind=kind,
            symbol_id=sym_id,
            start_line=start_line,
            end_line=end_line,
        )
        self.scope_stack.append(entry)
        for child in node.body:
            self.visit(child)
        self.scope_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._process_function(node, is_async=False)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._process_function(node, is_async=True)

    def visit_Call(self, node: ast.Call) -> None:
        loc = self._make_location(node)
        try:
            expr_str = ast.unparse(node.func)
        except Exception:
            expr_str = "call()"

        chain, is_dyn = _flatten_attribute_chain(node.func)
        simple_name: str | None = None
        if isinstance(node.func, ast.Name):
            simple_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            simple_name = node.func.attr

        call_ref = CallReference(
            source_file=self.file_path,
            location=loc,
            scope_path=self._current_scope_path(),
            caller_symbol_id=self._current_parent_id(),
            called_expression=expr_str,
            simple_name=simple_name,
            attribute_chain=chain,
            is_dynamic=is_dyn,
        )
        self.calls.append(call_ref)
        self.generic_visit(node)


class PythonAstAnalyzer:
    """Safe, deterministic parser extracting symbols, imports, and calls from Python source."""

    def analyze_source(self, source_code: str, file_path: str = "source.py") -> FileAnalysis:
        """Parse in-memory Python source code string into structured FileAnalysis."""
        norm_path = normalize_repo_relative_path(file_path)
        lines = source_code.splitlines()
        total_lines = len(lines)

        if not source_code.strip():
            return FileAnalysis(
                source_file=norm_path,
                status=FileAnalysisStatus.EMPTY_FILE,
                symbols=[],
                imports=[],
                calls=[],
                total_lines=total_lines,
            )

        try:
            tree = ast.parse(source_code, filename=norm_path)
        except SyntaxError as exc:
            err_line = exc.lineno or 1
            err_col = exc.offset or 0
            msg = f"SyntaxError at line {err_line}, col {err_col}: {exc.msg}"
            return FileAnalysis(
                source_file=norm_path,
                status=FileAnalysisStatus.SYNTAX_ERROR,
                error_message=msg,
                total_lines=total_lines,
            )
        except Exception as exc:
            return FileAnalysis(
                source_file=norm_path,
                status=FileAnalysisStatus.UNSUPPORTED,
                error_message=f"Parse failure: {exc}",
                total_lines=total_lines,
            )

        visitor = _PythonAstVisitor(file_path=norm_path, total_lines=total_lines)
        visitor.visit(tree)

        # Deterministic sorting preserving source appearance
        symbols = sorted(
            visitor.symbols,
            key=lambda s: (s.location.start_line, s.location.start_col or 0, s.qualified_name, s.id),
        )
        # Preserve original source statement order for imports and calls on the same line
        imports = sorted(
            enumerate(visitor.imports),
            key=lambda pair: (pair[1].location.start_line, pair[1].location.start_col or 0, pair[0]),
        )
        calls = sorted(
            enumerate(visitor.calls),
            key=lambda pair: (pair[1].location.start_line, pair[1].location.start_col or 0, pair[0]),
        )

        return FileAnalysis(
            source_file=norm_path,
            status=FileAnalysisStatus.PARSED,
            symbols=symbols,
            imports=[p[1] for p in imports],
            calls=[p[1] for p in calls],
            total_lines=total_lines,
        )

    def analyze_file(self, repo_root: Path, relative_path: str) -> FileAnalysis:
        """Safely read and parse a Python file strictly contained within `repo_root`."""
        try:
            norm_rel = normalize_repo_relative_path(relative_path)
            safe_path = ensure_safe_repo_file(repo_root, norm_rel)
        except SecurityViolationError as exc:
            return FileAnalysis(
                source_file=relative_path.replace("\\", "/").lstrip("./"),
                status=FileAnalysisStatus.SECURITY_VIOLATION,
                error_message=exc.message,
            )

        if not safe_path.exists() or not safe_path.is_file():
            return FileAnalysis(
                source_file=norm_rel,
                status=FileAnalysisStatus.FILE_NOT_FOUND,
                error_message=f"File not found: '{norm_rel}'",
            )

        try:
            source_code = safe_path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            return FileAnalysis(
                source_file=norm_rel,
                status=FileAnalysisStatus.DECODING_ERROR,
                error_message=f"Unicode decode error: {exc}",
            )
        except Exception as exc:
            return FileAnalysis(
                source_file=norm_rel,
                status=FileAnalysisStatus.UNSUPPORTED,
                error_message=f"Unable to read file: {exc}",
            )

        return self.analyze_source(source_code, file_path=norm_rel)
