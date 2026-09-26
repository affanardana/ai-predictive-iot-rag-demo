"""Configuration read from the environment. No torch needed."""

from __future__ import annotations

from pathlib import Path

import pytest

from inference.settings import (
    ARTIFACT_VARIABLE,
    CHECKPOINT_VARIABLE,
    ConfigurationError,
    Settings,
)


def test_it_reads_both_paths(environment: None, checkpoint_path: Path, artifact_dir: Path) -> None:
    settings = Settings.from_environment()

    assert settings.checkpoint == checkpoint_path
    assert settings.artifact_dir == artifact_dir


def test_a_missing_variable_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(CHECKPOINT_VARIABLE, raising=False)
    monkeypatch.delenv(ARTIFACT_VARIABLE, raising=False)

    with pytest.raises(ConfigurationError, match="must be set"):
        Settings.from_environment()


def test_a_missing_checkpoint_stops_startup(
    monkeypatch: pytest.MonkeyPatch, artifact_dir: Path
) -> None:
    """A service that starts without a model and fails on the first request.

    Refusing to start is worse for availability and better for everything else:
    a deployment that is up but cannot answer looks healthy to a load balancer.
    """
    monkeypatch.setenv("INFERENCE_CHECKPOINT", str(artifact_dir / "absent.pt"))
    monkeypatch.setenv("INFERENCE_ARTIFACT_DIR", str(artifact_dir))

    with pytest.raises(ConfigurationError, match="No checkpoint"):
        Settings.from_environment()


def test_a_directory_that_is_not_an_artifact_is_refused(
    monkeypatch: pytest.MonkeyPatch, checkpoint_path: Path, tmp_path: Path
) -> None:
    monkeypatch.setenv("INFERENCE_CHECKPOINT", str(checkpoint_path))
    monkeypatch.setenv("INFERENCE_ARTIFACT_DIR", str(tmp_path / "empty"))

    with pytest.raises(ConfigurationError, match=r"normalization\.json"):
        Settings.from_environment()
