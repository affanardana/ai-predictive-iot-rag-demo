"""Layer boundaries, enforced by inspecting the source.

`import-linter` enforces the same rules and runs as its own CI step, but it only
runs when someone remembers to invoke it. These tests run as part of `pytest`,
so a violation fails the ordinary test run too.

The rules are checked by parsing the AST rather than by importing modules:
an import that would break the architecture should be caught whether or not the
module happens to import cleanly in this environment.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator
from pathlib import Path

import pytest

API_PACKAGE = Path(__file__).resolve().parents[3] / "src" / "api"

#: Packages the domain must never import. CODING_STANDARDS.md names the first
#: five explicitly; the rest are the same category of dependency.
FRAMEWORK_ROOTS = frozenset(
    {
        "alembic",
        "fastapi",
        "httpx",
        "numpy",
        "pandas",
        "psycopg",
        "pydantic",
        "pydantic_settings",
        "sqlalchemy",
        "starlette",
        "torch",
        "uvicorn",
    }
)

#: Layers the domain must not reach outward to.
OUTER_LAYERS = frozenset(
    {
        "api.application",
        "api.composition",
        "api.infrastructure",
        "api.presentation",
    }
)


def _python_files(package: str) -> Iterator[Path]:
    """Yield every Python file under a package directory."""
    root = API_PACKAGE / package
    if not root.exists():
        pytest.fail(f"Expected package '{package}' at {root}")
    yield from sorted(root.rglob("*.py"))


def _imported_roots(path: Path) -> Iterator[tuple[str, int]]:
    """Yield `(module_name, line_number)` for every import in a file."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name, node.lineno
        # `level > 0` marks a relative import, which cannot escape the package
        # it lives in, so it is not a cross-layer concern.
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            yield node.module, node.lineno


def _matches(module: str, prefixes: frozenset[str]) -> str | None:
    """Return the matching prefix, if `module` equals or descends from one."""
    for prefix in prefixes:
        if module == prefix or module.startswith(f"{prefix}."):
            return prefix
    return None


def test_domain_files_exist() -> None:
    """Guard against the checker silently passing on an empty directory."""
    assert len(list(_python_files("domain"))) >= 10


@pytest.mark.parametrize("path", list(_python_files("domain")), ids=lambda p: p.name)
def test_domain_imports_no_framework(path: Path) -> None:
    """No domain module imports a framework, driver, or web library."""
    violations = [
        f"{path.relative_to(API_PACKAGE)}:{line} imports '{module}'"
        for module, line in _imported_roots(path)
        if _matches(module, FRAMEWORK_ROOTS)
    ]

    assert not violations, "Domain must stay framework-free:\n" + "\n".join(violations)


@pytest.mark.parametrize("path", list(_python_files("domain")), ids=lambda p: p.name)
def test_domain_does_not_import_outer_layers(path: Path) -> None:
    """Dependencies point inward: the domain never imports an outer layer."""
    violations = [
        f"{path.relative_to(API_PACKAGE)}:{line} imports '{module}'"
        for module, line in _imported_roots(path)
        if _matches(module, OUTER_LAYERS)
    ]

    assert not violations, "Domain must not depend on outer layers:\n" + "\n".join(violations)


@pytest.mark.parametrize("path", list(_python_files("application")), ids=lambda p: p.name)
def test_application_does_not_import_infrastructure_or_presentation(path: Path) -> None:
    """Application orchestrates through ports; it never names an adapter."""
    forbidden = frozenset({"api.infrastructure", "api.presentation"})

    violations = [
        f"{path.relative_to(API_PACKAGE)}:{line} imports '{module}'"
        for module, line in _imported_roots(path)
        if _matches(module, forbidden)
    ]

    assert not violations, "Application must depend only on the domain:\n" + "\n".join(violations)


@pytest.mark.parametrize("path", list(_python_files("presentation")), ids=lambda p: p.name)
def test_presentation_does_not_import_infrastructure(path: Path) -> None:
    """The HTTP layer must not reach past the composition root into adapters."""
    violations = [
        f"{path.relative_to(API_PACKAGE)}:{line} imports '{module}'"
        for module, line in _imported_roots(path)
        if _matches(module, frozenset({"api.infrastructure"}))
    ]

    assert not violations, "Presentation must not import infrastructure:\n" + "\n".join(violations)
