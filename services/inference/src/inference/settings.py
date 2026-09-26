"""Where the checkpoint and the artifact are, read from the environment."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

CHECKPOINT_VARIABLE = "INFERENCE_CHECKPOINT"
ARTIFACT_VARIABLE = "INFERENCE_ARTIFACT_DIR"


class ConfigurationError(RuntimeError):
    """Raised when the service cannot start."""


@dataclass(frozen=True, slots=True)
class Settings:
    """Everything the service needs to start."""

    checkpoint: Path
    artifact_dir: Path

    @classmethod
    def from_environment(cls) -> Settings:
        """Read both paths from the environment.

        Raises:
            ConfigurationError: if either is unset, or the checkpoint is not
                there. A service that starts without a model and fails on the
                first request is worse than one that refuses to start.
        """
        checkpoint = os.environ.get(CHECKPOINT_VARIABLE)
        artifact_dir = os.environ.get(ARTIFACT_VARIABLE)
        if not checkpoint or not artifact_dir:
            raise ConfigurationError(
                f"Both {CHECKPOINT_VARIABLE} and {ARTIFACT_VARIABLE} must be set."
            )
        if not Path(checkpoint).exists():
            raise ConfigurationError(f"No checkpoint at '{checkpoint}'.")
        if not (Path(artifact_dir) / "normalization.json").exists():
            raise ConfigurationError(
                f"'{artifact_dir}' does not look like a training artifact; "
                "normalization.json is missing."
            )
        return cls(checkpoint=Path(checkpoint), artifact_dir=Path(artifact_dir))
