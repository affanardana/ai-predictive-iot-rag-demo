"""Drawing a training epoch, weighted so no life is amplified.

The naive sampler takes `n` positives and `n` negatives uniformly from all
available windows. It gets the class ratio right and the *life* ratio badly
wrong: every failure contributes exactly sixty near-identical positive windows,
so a life with more surviving windows — a longer life — is handed more chances
to be drawn, and a machine with long lives dominates the epoch. The very
correlation the evaluation half works to undo would be baked into training.

## The weighting, and the mistake it is easy to make

Windows are drawn **uniformly within each class**. A life's contribution is then
proportional to how many windows it actually holds, which is what "not
amplified" means — a 48-hour life carries eight times the operating time of a
6-hour one and should count eight times as much.

The tempting alternative is to weight each window by the reciprocal of its
life's size, reading "life-aware" as "every life counts equally". That does the
opposite of what it sounds like: a short life's few windows get resampled until
they weigh as much as a long life's thousands, so the short life is amplified
and the training distribution stops resembling the data. It is worst in
`test_shift`, where lives differ in length by a factor of eight.

The near-duplication between the sixty windows before a failure is a real
problem, but it is an **evaluation** problem — it inflates window-level
confidence and per-window F1 — and `ml.evaluation.events` handles it there. It
is not fixed by distorting what the model trains on.

Drawing *with* replacement is what makes each epoch a fresh sample rather than
the same sixty neighbours replayed.

## The class ratio is exact, by construction

The counts come from the configuration rather than from the split, so the prior
the model trains at is a stated number and not an observed one. That is what
makes the correction in `ml.evaluation.prior` algebra rather than guesswork.

**Do not also weight the loss.** `pos_weight` on top of this sampler counts the
resampling twice and the calibration correction stops being true — a bug that
shows up as a mystery offset rather than as an error.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ml.experiment.errors import ExperimentError
from ml.experiment.index import WindowIndex

Float64Array = npt.NDArray[np.float64]
Int64Array = npt.NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class EpochDraw:
    """One epoch's worth of window positions, in presentation order."""

    positions: Int64Array
    positives: int
    negatives: int
    seed: int

    def __len__(self) -> int:
        """Return how many windows the epoch holds."""
        return int(self.positions.shape[0])

    @property
    def positive_rate(self) -> float:
        """Return the class balance actually drawn."""
        return self.positives / len(self) if len(self) else 0.0


def draw_epoch(
    index: WindowIndex,
    *,
    positives: int,
    negatives: int,
    seed: int,
) -> EpochDraw:
    """Draw one epoch, life-weighted within each class.

    Raises:
        ExperimentError: if an arm is empty, or the counts are not positive.
    """
    if positives < 1 or negatives < 1:
        raise ExperimentError(
            f"An epoch needs at least one window of each class, got {positives} "
            f"positives and {negatives} negatives."
        )

    positive_positions = np.flatnonzero(index.label == 1).astype(np.int64)
    negative_positions = np.flatnonzero(index.label == 0).astype(np.int64)
    if positive_positions.shape[0] == 0:
        raise ExperimentError("The index holds no positive windows to draw from.")
    if negative_positions.shape[0] == 0:
        raise ExperimentError("The index holds no negative windows to draw from.")

    generator = np.random.default_rng(seed)
    drawn_positive = _draw(generator, positive_positions, positives)
    drawn_negative = _draw(generator, negative_positions, negatives)

    positions = np.concatenate((drawn_positive, drawn_negative))
    # Shuffled so the loader never sees a run of one class. Without this the
    # first batches are all positive and the gradient oscillates.
    generator.shuffle(positions)

    return EpochDraw(
        positions=positions,
        positives=int(positives),
        negatives=int(negatives),
        seed=seed,
    )


def _draw(
    generator: np.random.Generator,
    candidates: Int64Array,
    count: int,
) -> Int64Array:
    """Draw `count` positions uniformly from `candidates`, with replacement.

    Uniform over *windows*, not over lives. A life's contribution is then
    proportional to how many windows it actually holds, which is precisely what
    "not amplified" means: a 48-hour life carries eight times the operating
    time of a 6-hour one and should count eight times as much.

    Weighting each window by the reciprocal of its life's size — the tempting
    reading of "life-aware" — does the opposite. It gives every life equal
    weight regardless of how much data it holds, so a short life's handful of
    windows are resampled until they count as much as a long life's thousands.
    That is amplification, and it is worst in `test_shift`, where the lives
    differ in length by a factor of eight.

    Replacement is what makes each epoch a fresh sample rather than the same
    sixty near-identical windows replayed.
    """
    chosen = generator.choice(candidates, size=count, replace=True)
    return np.asarray(chosen, dtype=np.int64)


def life_share(index: WindowIndex, positions: Int64Array) -> Float64Array:
    """Return each life's share of a draw, for checking the weighting.

    Exists so a test can assert the property the sampler claims — that a life's
    expected contribution is proportional to its own window count, not equal to
    every other life's — rather than asserting the implementation against
    itself. That property is the one the first version of this module got
    backwards, so it is worth an independent check.
    """
    lives = np.unique(index.life)
    counts = np.bincount(index.life[positions], minlength=int(lives.max()) + 1)
    total = counts.sum()
    return counts / total if total else counts.astype(np.float64)
