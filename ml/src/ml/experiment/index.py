"""Which windows exist, what they contain, and what they are labelled.

This is the torch-free half of sequence generation, and it is the half where the
mistakes would be invisible. A window built from the wrong row produces a
plausible tensor; a label taken from the wrong timestep produces a model that
trains fine and measures nothing. So the index is a plain array of end
positions, built from `ml.dataset.windows` — which already encodes the two rules
— and every downstream consumer reads it rather than re-deriving anything.

`ml.model.sequences` wraps this in a `torch.utils.data.Dataset` and does nothing
else. That is the whole reason the torch layer is small enough to be worth
trusting: it has no opinion about what a window is.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ml.dataset.artifacts import Dataset
from ml.dataset.baseline import HazardBaseline
from ml.dataset.labelling import HORIZON_MINUTES, NEVER_FAILS
from ml.dataset.splits import Split
from ml.dataset.windows import window_ends
from ml.experiment.errors import ExperimentError

Float64Array = npt.NDArray[np.float64]
Int32Array = npt.NDArray[np.int32]
Int64Array = npt.NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class WindowIndex:
    """One entry per valid window, describing where it is and what it means."""

    ends: Int64Array
    life: Int32Array
    machine: Int32Array
    position: Int32Array
    label: Int64Array

    def __len__(self) -> int:
        """Return how many windows the index holds."""
        return int(self.ends.shape[0])

    @property
    def positives(self) -> int:
        """Return how many windows are positive."""
        return int(np.count_nonzero(self.label == 1))

    @property
    def prevalence(self) -> float:
        """Return the positive rate over windows."""
        return self.positives / len(self) if len(self) else 0.0

    def select(self, positions: Int64Array) -> WindowIndex:
        """Return a new index holding only the given positions."""
        return WindowIndex(
            ends=self.ends[positions],
            life=self.life[positions],
            machine=self.machine[positions],
            position=self.position[positions],
            label=self.label[positions],
        )

    def floor_scores(self, hazard: HazardBaseline) -> Float64Array:
        """Return the clock baseline's score for each window.

        The floor has to be recomputed on the window set the model is scored on.
        `ml.dataset.report` publishes it per *trainable row*, and a window set is
        a smaller, differently-balanced subset of the same lives — so comparing
        the model's average precision against the published figure would be
        comparing two different questions and calling the difference skill.
        """
        return hazard.score(self.position.astype(np.int64))

    def failed_lives(self) -> Int64Array:
        """Return a 0/1 flag per life, for the life-level view.

        Built from the windows actually present rather than from the artifact's
        life table, so a life with no usable windows cannot contribute a label
        that nothing scores.
        """
        count = int(self.life.max()) + 1 if len(self) else 0
        failed = np.zeros(count, dtype=np.int64)
        for identifier in np.unique(self.life):
            failed[int(identifier)] = int(self.label[self.life == identifier].max())
        return failed


def build_index(dataset: Dataset, split: Split, *, stride: int = 1) -> WindowIndex:
    """Return every valid window of every life a split holds.

    Built on `window_ends`, so the two rules it encodes are inherited rather
    than restated: a window never crosses a life boundary, and never includes a
    post-onset row.

    Raises:
        ExperimentError: if the split holds no machines, or no life in it is
            long enough to contain a complete window.
    """
    members = {index for index, value in enumerate(dataset.splits) if value is split}
    if not members:
        raise ExperimentError(f"No machines are assigned to the {split.value} split.")

    lengths = np.diff(dataset.life_offsets)
    wanted = [life for life in range(dataset.lives) if int(dataset.life_machine[life]) in members]

    ends: list[npt.NDArray[np.int64]] = []
    life_of: list[npt.NDArray[np.int32]] = []
    machine_of: list[npt.NDArray[np.int32]] = []
    position_of: list[npt.NDArray[np.int32]] = []

    for life in wanted:
        count = int(lengths[life])
        onset = int(dataset.life_onset[life])
        onsets = None if onset == NEVER_FAILS else onset
        # `window_ends` is a range over positions within the life; the stride
        # only ever thins it, so a caller cannot stride past a boundary.
        positions = np.asarray(list(window_ends(count, onsets))[::stride], dtype=np.int32)
        if positions.shape[0] == 0:
            continue
        base = int(dataset.life_offsets[life])
        ends.append(positions.astype(np.int64) + base)
        life_of.append(np.full(positions.shape[0], life, dtype=np.int32))
        machine_of.append(
            np.full(positions.shape[0], int(dataset.life_machine[life]), dtype=np.int32)
        )
        position_of.append(positions)

    if not ends:
        raise ExperimentError(
            f"The {split.value} split holds no complete {HORIZON_MINUTES}-minute windows."
        )

    ends_array = np.concatenate(ends)
    minutes = dataset.minutes_to_onset[ends_array].astype(np.int64)
    label = ((minutes > 0) & (minutes <= HORIZON_MINUTES)).astype(np.int64)

    return WindowIndex(
        ends=ends_array,
        life=np.concatenate(life_of),
        machine=np.concatenate(machine_of),
        position=np.concatenate(position_of),
        label=label,
    )


def machine_days(dataset: Dataset, split: Split) -> float:
    """Return how much observation time a split covers, in machine-days.

    The denominator for a false-alarm rate. Counted in time rather than in lives
    because a 6-hour healthy life and a 48-hour healthy life are not the same
    amount of opportunity to raise a false alarm.
    """
    return float(np.diff(dataset.life_offsets)[_lives_of(dataset, split)].sum()) / 1440.0


def _lives_of(dataset: Dataset, split: Split) -> Int64Array:
    """Return the life indices belonging to a split."""
    members = {index for index, value in enumerate(dataset.splits) if value is split}
    return np.asarray(
        [life for life in range(dataset.lives) if int(dataset.life_machine[life]) in members],
        dtype=np.int64,
    )
