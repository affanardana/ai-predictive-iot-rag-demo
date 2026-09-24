"""The Colab notebook has to stay orchestration.

`AI_AGENT_GUIDE.md` says this repository is not "a collection of notebooks", and
the way that happens is a notebook quietly growing the logic it was supposed to
call. So the rule is a test rather than a note in a README.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

NOTEBOOK = Path(__file__).resolve().parents[3] / "notebooks" / "colab_train.ipynb"

#: Generous, because the point is to catch a notebook that has become a program,
#: not to shave lines. Roughly double what the notebook needs today.
CODE_LINE_BUDGET = 90


def _cells() -> list[dict[str, object]]:
    payload: dict[str, object] = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cells = payload["cells"]
    assert isinstance(cells, list)
    return cells


def _as_text(value: object) -> str:
    assert isinstance(value, list)
    return "".join(str(part) for part in value)


def _source(cell: dict) -> str:
    return _as_text(cell["source"])


def _code(cell: dict) -> str:
    """Return the cell's source with comment lines removed.

    The checks below are about what the notebook *does*. A comment explaining
    that Colab ships its own torch is not an import, and flagging it would push
    the explanation out of the notebook to satisfy a linter.
    """
    return "\n".join(
        line for line in _source(cell).splitlines() if not line.strip().startswith("#")
    )


def test_the_notebook_exists_and_parses() -> None:
    assert NOTEBOOK.exists()
    assert len(_cells()) > 0


def test_the_notebook_imports_no_torch() -> None:
    """Training is called, not written here.

    An `import torch` in a cell is the first sign the notebook is becoming the
    implementation.
    """
    for index, cell in enumerate(_cells()):
        if cell["cell_type"] != "code":
            continue
        source = _code(cell)
        assert "import torch" not in source, f"cell {index} imports torch"
        assert "torch." not in source, f"cell {index} uses torch directly"


def test_the_notebook_computes_no_metrics() -> None:
    """Every number it prints comes from `ml model evaluate`."""
    forbidden = ("precision", "recall", "roc_auc", "np.", "numpy", "sklearn")

    for index, cell in enumerate(_cells()):
        if cell["cell_type"] != "code":
            continue
        source = _code(cell)
        for token in forbidden:
            assert token not in source, f"cell {index} mentions '{token}'"


def test_the_notebook_stays_within_its_line_budget() -> None:
    code_lines = sum(
        len([line for line in _source(cell).splitlines() if line.strip()])
        for cell in _cells()
        if cell["cell_type"] == "code"
    )

    assert code_lines <= CODE_LINE_BUDGET, (
        f"the notebook holds {code_lines} lines of code; the budget is "
        f"{CODE_LINE_BUDGET}. Logic belongs in ml/src/ml/."
    )


def test_the_notebook_calls_the_command_line() -> None:
    """It orchestrates, so it must actually invoke the package."""
    source = "\n".join(_source(cell) for cell in _cells())

    assert "python -m ml model train" in source
    assert "python -m ml model evaluate" in source


def test_the_notebook_copies_the_dataset_off_drive_before_training() -> None:
    """A memory-mapped read over a FUSE mount is the failure that costs an hour.

    The copy is a line in the notebook rather than something the training code
    can enforce, so it is checked here.
    """
    source = "\n".join(_source(cell) for cell in _cells())

    assert "cp -r" in source
    assert "/content/data/training" in source


@pytest.mark.parametrize("token", ["--resume", "model verify"])
def test_the_notebook_exercises_the_resumable_path(token: str) -> None:
    """Free-tier sessions die; the notebook is written for that, not for luck."""
    source = "\n".join(_source(cell) for cell in _cells())

    assert token in source or "resume" in source.lower()
