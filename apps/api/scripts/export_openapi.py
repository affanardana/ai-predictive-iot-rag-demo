"""Emit the API's OpenAPI document.

Exists so the frontend's types can be regenerated without a running server.
`create_app` builds its routes from the container and the engine is lazy, so
nothing here connects to a database -- which is what lets CI regenerate the
document in a job that has no database.

    uv run python apps/api/scripts/export_openapi.py --out apps/web/openapi.json

**Prefer `--out` to a shell redirect.** `... > openapi.json` is fine in bash and
wrong in Windows PowerShell, which writes UTF-16LE -- every character separated
by a null byte. The result is a file that looks like JSON to a human and that
every parser rejects, and the error names a byte offset rather than the shell
that produced it. Writing the bytes here removes the question, on every shell.

Lives outside `apps/api/src` so it is not part of the installed package. It is
still linted, because `ruff check .` covers the repository, but it is not in
mypy's `files` -- which is why every call here is written to be obvious rather
than clever.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

#: A URL that is never dialled. `Settings` requires the variable to exist, and
#: the engine it configures opens no connection until a query runs.
PLACEHOLDER_DATABASE_URL = "sqlite+aiosqlite:///:memory:"


def _parse_args() -> argparse.Namespace:
    """Read the optional output path."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Write the document here as UTF-8. Defaults to stdout.",
    )
    return parser.parse_args()


def main() -> None:
    """Write the document, sorted so a diff means a real change."""
    # Set before `Settings` is constructed rather than at import time, so the
    # script has no import-order requirement on its caller.
    os.environ.setdefault("APP_ENV", "local")
    os.environ.setdefault("DATABASE_URL", PLACEHOLDER_DATABASE_URL)

    from api.composition import build_in_memory_container
    from api.infrastructure.config import Settings
    from api.presentation.app import create_app

    # `_env_file=None` so a developer's `.env` cannot change the published
    # contract. Only the environment decides, exactly as in CI.
    app = create_app(build_in_memory_container(Settings(_env_file=None)))

    # `sort_keys` and a trailing newline so the generated file is stable and
    # the drift check compares content rather than formatting.
    document = json.dumps(app.openapi(), indent=2, sort_keys=True) + "\n"

    out: Path | None = _parse_args().out
    if out is None:
        sys.stdout.write(document)
        return

    # `encoding="utf-8"` explicitly rather than relying on the platform default,
    # and `newline="\n"` so a Windows run does not produce CRLF that would show
    # up as a whole-file diff in CI.
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(document, encoding="utf-8", newline="\n")
    print(f"Wrote {out} ({len(document)} bytes).", file=sys.stderr)


if __name__ == "__main__":
    main()
