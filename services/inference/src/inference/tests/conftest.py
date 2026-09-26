"""Fixtures for the inference service.

Most of these avoid torch entirely: the HTTP contract is exercised against a
fake scorer, so the API's behaviour is tested without loading a model. Only the
scorer's own tests need the real thing.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from ml.dataset.features import FEATURE_COLUMNS
from ml.dataset.windows import WINDOW_MINUTES


def a_reading(value: float = 1.0) -> dict[str, float]:
    """Return one reading with every signal set to `value`."""
    return dict.fromkeys(FEATURE_COLUMNS, value)


def a_window(count: int = WINDOW_MINUTES, value: float = 1.0) -> list[dict[str, float]]:
    """Return `count` readings."""
    return [a_reading(value) for _ in range(count)]


@pytest.fixture
def artifact_dir(tmp_path: Path) -> Path:
    """A directory that looks like a training artifact to `Settings`."""
    (tmp_path / "normalization.json").write_text(
        '{"mean": [0,0,0,0,0,0], "std": [1,1,1,1,1,1], "ddof": 0,'
        ' "computed_from": {"split": "train", "machines": 1, "rows": 1}}',
        encoding="utf-8",
    )
    return tmp_path


@pytest.fixture
def checkpoint_path(tmp_path: Path) -> Path:
    """A placeholder checkpoint file, for `Settings` to find."""
    path = tmp_path / "best.pt"
    path.write_bytes(b"not a real checkpoint")
    return path


@pytest.fixture
def environment(
    monkeypatch: pytest.MonkeyPatch, checkpoint_path: Path, artifact_dir: Path
) -> Iterator[None]:
    """Set the two variables the service reads at startup."""
    monkeypatch.setenv("INFERENCE_CHECKPOINT", str(checkpoint_path))
    monkeypatch.setenv("INFERENCE_ARTIFACT_DIR", str(artifact_dir))
    yield
