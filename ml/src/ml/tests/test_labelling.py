"""What the label means, and that it agrees with the ground truth."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from ml.dataset.artifacts import Dataset
from ml.dataset.generation import GROUND_TRUTH_FILENAME
from ml.dataset.labelling import (
    HORIZON_MINUTES,
    NEVER_FAILS,
    is_post_onset,
    is_trainable,
    label,
    minutes_to_onset,
    onset_index,
    positive_rate,
)


def test_onset_is_the_first_imminent_tick() -> None:
    assert onset_index([False, False, True, True]) == 2
    assert onset_index([True]) == 0
    assert onset_index([False, False]) is None
    assert onset_index([]) is None


def test_offsets_count_down_to_the_onset() -> None:
    """One signed integer per timestep: positive before, zero at, negative after."""
    assert minutes_to_onset(4, 2) == [2, 1, 0, -1]


def test_a_life_that_never_fails_carries_the_sentinel() -> None:
    assert minutes_to_onset(3, None) == [NEVER_FAILS] * 3


@pytest.mark.parametrize(
    ("minutes", "expected"),
    [
        (1, 1),
        (HORIZON_MINUTES, 1),
        (HORIZON_MINUTES + 1, 0),
        (0, 0),
        (-1, 0),
        (NEVER_FAILS, 0),
    ],
)
def test_the_label_covers_the_horizon_and_excludes_the_onset(minutes: int, expected: int) -> None:
    """At onset the machine is failing *now*, which is a different question."""
    assert label(minutes) == expected


def test_the_post_onset_tail_is_neither_class() -> None:
    """Excluded rather than assigned a class — see the module docstring."""
    assert not is_trainable(0)
    assert not is_trainable(-5)
    assert is_post_onset(0)
    assert is_post_onset(-5)
    assert not is_post_onset(NEVER_FAILS)

    assert is_trainable(1)
    assert is_trainable(NEVER_FAILS)


def test_prevalence_counts_only_trainable_rows() -> None:
    """The metric that decides whether the dataset is trainable at all."""
    values = [1, 1, NEVER_FAILS, -3, 500]

    # Four trainable rows, two of them positive.
    assert positive_rate(values) == pytest.approx(0.5)


def test_labels_agree_with_the_ground_truth_file(
    dataset: Dataset, fleet: tuple[Path, tuple]
) -> None:
    """Recomputed here from the Parquet file, by a different route.

    `build_dataset` finds each life's onset by scanning the truth column. This
    does it again with its own loop over the same file and compares the two, so
    a bug in the exporter cannot agree with itself.
    """
    directory, _ = fleet
    flags = (
        pq.read_table(directory / GROUND_TRUTH_FILENAME, columns=["failure_imminent"])
        .column("failure_imminent")
        .combine_chunks()
        .to_numpy(zero_copy_only=False)
    )

    for life_index in range(dataset.lives):
        start = int(dataset.life_offsets[life_index])
        stop = int(dataset.life_offsets[life_index + 1])

        expected = None
        for position, value in enumerate(flags[start:stop]):
            if value:
                expected = position
                break

        actual = int(dataset.life_onset[life_index])
        assert actual == (NEVER_FAILS if expected is None else expected)

        for offset in range(stop - start):
            if expected is None:
                assert int(dataset.minutes_to_onset[start + offset]) == NEVER_FAILS
            else:
                assert int(dataset.minutes_to_onset[start + offset]) == expected - offset


def test_enough_lives_fail_to_train_on(dataset: Dataset) -> None:
    """The composition has to produce failure events, not just timesteps.

    This is the number the whole lives design exists to raise. One session per
    machine would give roughly one event each; composing thirty days from
    twenty-odd lives gives twenty-odd. A quarter is a floor well below what the
    composition should deliver, so it catches a collapse rather than noise.
    """
    failing = int(np.sum(dataset.life_onset != NEVER_FAILS))

    assert failing > dataset.lives // 4
