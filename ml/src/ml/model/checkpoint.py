"""Saving and loading a checkpoint, under the safe default.

`torch.load` defaults to `weights_only=True` from torch 2.6, and that default
is deliberately hard to work around: a checkpoint holding a config dataclass, a
custom object, or a numpy array will raise rather than load, because unpickling
arbitrary objects from a file is how a checkpoint becomes an exploit.

Rather than reaching for `weights_only=False` or an `add_safe_globals`
allowlist, this module is built so it does not need them. Everything saved is
either a tensor or a **string** — the config and manifest travel as JSON text —
so the checkpoint loads under the safe default and stays loadable when that
default tightens again. Numpy arrays are avoided too: their support under
`weights_only` has moved between versions.

A golden batch is stored alongside the weights so that `ml model verify` can
re-score a fixed sample and show the model still produces the predictions it
did, without needing the training data or a GPU.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from ml.dataset.splits import Split
from ml.experiment.errors import ExperimentError

CHECKPOINT_VERSION = 1


@dataclass(frozen=True, slots=True)
class Checkpoint:
    """Everything needed to rebuild a model's predictions."""

    state_dict: dict[str, Any]
    config_json: str
    manifest_json: str
    epoch: int
    metrics: dict[str, float]
    trained_rate: float
    natural_rate: float
    split: Split
    golden_windows: torch.Tensor
    golden_probabilities: torch.Tensor
    # Optional, and declared last because they are: a `best.pt` carries neither,
    # and a resumable `last.pt` carries both.
    optimiser_state: dict[str, Any] | None = None
    scheduler_state: dict[str, Any] | None = None

    @property
    def run_id(self) -> str:
        """Return the run that produced this checkpoint.

        Read from the manifest it carries, so the value reported by a service
        and the value persisted against a prediction are the same string as the
        one in the run directory.
        """
        try:
            return str(json.loads(self.manifest_json).get("run_id") or "unknown")
        except (ValueError, AttributeError):
            return "unknown"

    def __post_init__(self) -> None:
        """Reject a checkpoint that cannot be verified or re-calibrated."""
        if not self.config_json or not self.manifest_json:
            raise ExperimentError(
                "A checkpoint must carry its config and manifest, or it cannot "
                "say what produced it."
            )
        if self.golden_windows.shape[0] != self.golden_probabilities.shape[0]:
            raise ExperimentError(
                "The golden batch's windows and probabilities disagree in length."
            )


def save(path: Path, checkpoint: Checkpoint) -> Path:
    """Write a checkpoint.

    Every value is a tensor or a string. That is a constraint, not a style: see
    the module docstring.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "version": CHECKPOINT_VERSION,
            "state_dict": checkpoint.state_dict,
            "config_json": checkpoint.config_json,
            "manifest_json": checkpoint.manifest_json,
            "epoch": int(checkpoint.epoch),
            # Present only on a resumable checkpoint. A `best.pt` does not need
            # them, and an empty dict would be indistinguishable from a
            # correctly restored zero state.
            "optimiser_state": checkpoint.optimiser_state,
            "scheduler_state": checkpoint.scheduler_state,
            "metrics": {key: float(value) for key, value in checkpoint.metrics.items()},
            "trained_rate": float(checkpoint.trained_rate),
            "natural_rate": float(checkpoint.natural_rate),
            "split": checkpoint.split.value,
            "golden_windows": checkpoint.golden_windows,
            "golden_probabilities": checkpoint.golden_probabilities,
        },
        path,
    )
    return path


def load(path: Path, *, map_location: str = "cpu") -> Checkpoint:
    """Read a checkpoint written by `save`.

    `map_location="cpu"` by default, so a GPU-trained checkpoint verifies on a
    machine without one — which is the whole point of the verification command.

    Raises:
        ExperimentError: if the file is missing or from another version.
    """
    if not path.exists():
        raise ExperimentError(f"{path} does not exist.")

    # `weights_only=True` is the default from torch 2.6 and is passed
    # explicitly so the intent survives a downgrade that would default it off.
    payload = torch.load(path, map_location=map_location, weights_only=True)
    version = int(payload.get("version", 0))
    if version != CHECKPOINT_VERSION:
        raise ExperimentError(
            f"{path} is checkpoint version {version}; this build writes and reads "
            f"{CHECKPOINT_VERSION}."
        )

    return Checkpoint(
        state_dict=dict(payload["state_dict"]),
        config_json=str(payload["config_json"]),
        manifest_json=str(payload["manifest_json"]),
        epoch=int(payload["epoch"]),
        optimiser_state=payload.get("optimiser_state"),
        scheduler_state=payload.get("scheduler_state"),
        metrics={key: float(value) for key, value in payload["metrics"].items()},
        trained_rate=float(payload["trained_rate"]),
        natural_rate=float(payload["natural_rate"]),
        split=Split(str(payload["split"])),
        golden_windows=payload["golden_windows"],
        golden_probabilities=payload["golden_probabilities"],
    )
