"""The knowledge commands.

These stop short of a server on purpose: what is tested here is the dispatch,
the flags, and the failure messages, all of which are reachable without one.
The measured run needs a stack and is not asserted anywhere.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ml.cli import main


@pytest.mark.parametrize(
    ("action", "flag"),
    [
        ("extract", "--corpus"),
        ("ingest", "--corpus"),
        ("evaluate", "--questions"),
    ],
)
def test_every_action_reaches_its_own_handler(
    action: str,
    flag: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Each subcommand runs its own code path rather than crashing on dispatch.

    Regression, and the shape of the bug matters: the dispatcher read the corpus
    manifest before it looked at which action had been asked for, and `evaluate`
    has no `--corpus` flag — so it died with `AttributeError: 'Namespace' object
    has no attribute 'corpus'` before it reached the API, on the server, in a
    container. A missing file is used here because it fails inside the handler,
    which is the point: dispatch itself has to survive the trip.
    """
    code = main(["knowledge", action, flag, "knowledge/does-not-exist.json"])

    assert code == 1
    # A clean failure with a message, not a traceback out of `main`.
    assert "does-not-exist" in capsys.readouterr().err


def test_evaluate_does_not_need_a_manifest(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`evaluate` measures a stack, so a missing manifest is not its problem.

    It reads the corpus size from the API it is measuring, which is also the
    honest number: what was measured is what that stack holds, not what the
    manifest says should be there.
    """
    absent = tmp_path / "absent.json"

    code = main(["knowledge", "evaluate", "--questions", str(absent)])

    assert code == 1
    assert str(absent) in capsys.readouterr().err


def test_extract_refuses_a_document_that_is_not_in_the_manifest(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A mistyped key is a usage error, not a traceback."""
    empty = tmp_path / "corpus.json"
    empty.write_text('{"manifest_version": 1, "documents": []}', encoding="utf-8")

    assert main(["knowledge", "extract", "--corpus", str(empty)]) == 1
    assert "no documents" in capsys.readouterr().err
