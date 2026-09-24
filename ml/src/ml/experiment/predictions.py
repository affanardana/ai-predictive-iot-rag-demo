"""The prediction table — the seam between training and evaluation.

The exit condition is "a versioned model produces reproducible predictions and
has documented evaluation results", and this is what makes both halves of that
checkable. Training writes a prediction table; evaluation reads one and knows
nothing else. The model is not an input to any reported figure.

Two consequences, and they are the reason the type exists at all. A reviewer
with no GPU re-derives every number by running a command over files, and a
regression in the evaluation half can be tested without a model, a checkpoint,
or torch — which is what lets most of this phase be verified on a laptop.

Parquet rather than `.npy`, matching the repository's idiom, and pyarrow is
already a core dependency of `ml`, so the table costs nothing in the environment
where torch is absent.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ml.dataset.splits import Split
from ml.experiment.errors import ExperimentError

SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class Predictions:
    """One scored window per row, with everything needed to evaluate it."""

    split: Split
    run_id: str
    model: str
    ends: np.ndarray
    life: np.ndarray
    machine: np.ndarray
    position: np.ndarray
    label: np.ndarray
    probability: np.ndarray
    calibrated: np.ndarray

    def __post_init__(self) -> None:
        """Reject a table whose columns do not line up."""
        lengths = {
            "ends": self.ends.shape[0],
            "life": self.life.shape[0],
            "machine": self.machine.shape[0],
            "position": self.position.shape[0],
            "label": self.label.shape[0],
            "probability": self.probability.shape[0],
            "calibrated": self.calibrated.shape[0],
        }
        if len(set(lengths.values())) != 1:
            raise ExperimentError(f"Prediction columns disagree in length: {lengths}.")
        if not self.run_id:
            raise ExperimentError("A prediction table must name the run that produced it.")

    def __len__(self) -> int:
        """Return how many windows were scored."""
        return int(self.label.shape[0])

    @property
    def positives(self) -> int:
        """Return how many windows are positive."""
        return int(np.count_nonzero(self.label == 1))

    @property
    def prevalence(self) -> float:
        """Return the positive rate."""
        return self.positives / len(self) if len(self) else 0.0


def write_predictions(path: Path, predictions: Predictions) -> Path:
    """Write a prediction table, with its provenance repeated on every row.

    `split`, `run_id` and `model` are stored per row rather than in a sidecar
    file. That is denormalised and deliberate: a table that gets separated from
    its metadata is a table whose numbers cannot be attributed, and the columns
    cost a few hundred kilobytes against a file that is already tens of
    megabytes.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = len(predictions)
    table = pa.table(
        {
            "split": pa.array([predictions.split.value] * rows, type=pa.string()),
            "run_id": pa.array([predictions.run_id] * rows, type=pa.string()),
            "model": pa.array([predictions.model] * rows, type=pa.string()),
            "schema": pa.array([SCHEMA_VERSION] * rows, type=pa.int32()),
            "ends": pa.array(predictions.ends, type=pa.int64()),
            "life": pa.array(predictions.life, type=pa.int32()),
            "machine": pa.array(predictions.machine, type=pa.int32()),
            "position": pa.array(predictions.position, type=pa.int32()),
            "label": pa.array(predictions.label, type=pa.int64()),
            "probability": pa.array(predictions.probability, type=pa.float64()),
            "calibrated": pa.array(predictions.calibrated, type=pa.float64()),
        }
    )
    pq.write_table(table, path, compression="snappy")
    return path


def read_predictions(path: Path) -> Predictions:
    """Read a prediction table.

    Raises:
        ExperimentError: if the file is missing or was written by a different
            schema version. A reader that guessed at an unfamiliar layout would
            report numbers from columns it had misread.
    """
    if not path.exists():
        raise ExperimentError(f"{path} does not exist.")
    table = pq.read_table(path)
    version = int(table.column("schema")[0].as_py()) if table.num_rows else 0
    if version != SCHEMA_VERSION:
        raise ExperimentError(
            f"{path} is prediction schema {version}; this build reads {SCHEMA_VERSION}."
        )
    if table.num_rows == 0:
        raise ExperimentError(f"{path} holds no predictions.")

    return Predictions(
        split=Split(table.column("split")[0].as_py()),
        run_id=table.column("run_id")[0].as_py(),
        model=table.column("model")[0].as_py(),
        ends=table.column("ends").to_numpy(),
        life=table.column("life").to_numpy(),
        machine=table.column("machine").to_numpy(),
        position=table.column("position").to_numpy(),
        label=table.column("label").to_numpy(),
        probability=table.column("probability").to_numpy(),
        calibrated=table.column("calibrated").to_numpy(),
    )
