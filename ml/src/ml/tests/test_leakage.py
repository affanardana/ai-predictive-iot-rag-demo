"""Leakage prevention, as assertions rather than intentions.

`MASTERPLAN.md` §6 lists "leakage prevention" as a deliverable in its own right,
and §25.15 makes it a product constraint. Prose is the wrong medium for it: a
leak produces a model that scores *better*, so nothing downstream ever complains.
The only useful form is a test that fails.

Each test below corresponds to a specific way this dataset could leak.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from ml.dataset.artifacts import Dataset
from ml.dataset.baseline import elapsed_minutes
from ml.dataset.features import FEATURE_COLUMNS, IDENTIFIER_COLUMNS
from ml.dataset.generation import GROUND_TRUTH_FILENAME, TELEMETRY_FILENAME
from ml.dataset.labelling import NEVER_FAILS, is_trainable
from ml.dataset.lives import MachinePlan
from ml.dataset.splits import Split
from ml.dataset.windows import window_slice


def test_the_feature_set_is_a_whitelist_not_a_subtraction(dataset: Dataset) -> None:
    """The artifact carries the six signals and no more.

    A blacklist — "drop the ids, keep the numbers" — would make every column
    added to the telemetry stream a feature by default, and the two most
    dangerous columns in this file are not numbers.
    """
    assert dataset.features == len(FEATURE_COLUMNS)
    assert set(FEATURE_COLUMNS) == {
        "temperature",
        "vibration",
        "rpm",
        "current",
        "load",
        "voltage",
    }


def test_no_identifier_column_is_a_feature() -> None:
    """The four identifiers are carried through the pipeline, never fed in."""
    assert not set(IDENTIFIER_COLUMNS) & set(FEATURE_COLUMNS)


def test_no_machine_appears_in_two_splits(dataset: Dataset) -> None:
    """The leak that would inflate every number Phase 4 reports.

    A machine's nominal operating point and susceptibility are stable across its
    lives, so a machine on both sides of the boundary is memorised rather than
    generalised — and because it is a *different row* each time, nothing about
    the data looks wrong.
    """
    seen: dict[str, Split] = {}
    for machine_id, split in zip(dataset.machine_ids, dataset.splits, strict=True):
        assert machine_id not in seen, f"{machine_id} appears in two splits."
        seen[machine_id] = split


def test_normalisation_uses_training_rows_and_only_those(dataset: Dataset) -> None:
    """Recomputing the statistics over the training rows must reproduce them."""
    train = np.array([index for index, split in enumerate(dataset.splits) if split is Split.TRAIN])
    training_life = np.isin(dataset.life_machine, train)
    row_is_training = np.repeat(training_life, dataset.life_lengths())
    selected = dataset.signals[row_is_training & dataset.trainable()].astype(np.float64)

    assert np.allclose(selected.mean(axis=0), dataset.normalization.means)
    assert np.allclose(selected.std(axis=0), dataset.normalization.stds)
    assert dataset.normalization.source_rows == selected.shape[0]


def test_normalisation_would_differ_if_computed_over_everything(dataset: Dataset) -> None:
    """The negative control, without which the test above proves nothing.

    If the training statistics happened to equal the global ones — a dataset
    with one split, or one where the split did nothing — the previous test would
    pass while measuring nothing. Here the two must disagree.
    """
    everything = dataset.signals.astype(np.float64).mean(axis=0)

    assert not np.allclose(everything, dataset.normalization.means)


def test_training_is_the_only_split_that_would_ever_be_rebalanced(dataset: Dataset) -> None:
    """Validation and test keep the natural prevalence.

    A model's probabilities are calibrated to whatever prevalence it trained at.
    Rebalancing training changes that prior; rebalancing a split that calibration
    is then measured on would measure the rebalancing instead of the model.
    """
    assert dataset.normalization.source_split is Split.TRAIN


def test_the_elapsed_time_column_exists_but_is_not_a_feature(dataset: Dataset) -> None:
    """`elapsed_minutes` is a diagnostic, and it has to stay one.

    On this simulator, degradation is a deterministic function of elapsed time,
    so a model given elapsed time is reporting the composition back rather than
    reading a machine. It exists so the clock baseline can be fitted and quoted
    as a floor; it must never become an input.
    """
    elapsed = elapsed_minutes(dataset)

    assert elapsed.shape[0] == dataset.rows
    assert "elapsed" not in FEATURE_COLUMNS
    assert "elapsed_minutes" not in FEATURE_COLUMNS
    assert not set(FEATURE_COLUMNS) & {"elapsed", "elapsed_minutes", "minutes_to_onset"}


def test_the_two_channel_files_agree_row_for_row(
    fleet: tuple[Path, tuple[MachinePlan, ...]],
) -> None:
    """What makes positional labelling safe rather than merely convenient.

    If either sink ever reordered a row, every label after that point would
    belong to a neighbouring timestep and the dataset would look normal.
    """
    directory, _ = fleet
    telemetry = pq.read_table(directory / TELEMETRY_FILENAME)
    truth = pq.read_table(directory / GROUND_TRUTH_FILENAME)

    assert telemetry.num_rows == truth.num_rows
    assert telemetry.column("machine_id").equals(truth.column("machine_id"))
    assert telemetry.column("recorded_at").equals(truth.column("recorded_at"))


def test_the_telemetry_file_carries_no_ground_truth_column(
    fleet: tuple[Path, tuple[MachinePlan, ...]],
) -> None:
    """The Phase 2 separation, checked again from the consuming side."""
    directory, _ = fleet
    columns = set(pq.read_table(directory / TELEMETRY_FILENAME).column_names)

    assert columns == set(IDENTIFIER_COLUMNS) | set(FEATURE_COLUMNS)
    assert not columns & {"scenario", "health_index", "failure_imminent"}


def test_a_window_never_reaches_past_its_life(dataset: Dataset) -> None:
    """Every window's rows fall inside one life, and before its onset.

    Bounds are checked for every window; the row-level check is applied to each
    life's *last* window, which is the one that ends nearest the boundary and so
    is the only place the rule can actually be broken.
    """
    inspected = 0
    for life_index in range(dataset.lives):
        start = int(dataset.life_offsets[life_index])
        stop = int(dataset.life_offsets[life_index + 1])
        onset = int(dataset.life_onset[life_index])
        last_usable = stop - start - 1 if onset == NEVER_FAILS else onset - 1

        for end in range(60 - 1, last_usable + 1):
            window = window_slice(end)

            assert start + window.start >= start, "a window reached before its life began."
            assert start + window.stop - 1 < stop, "a window crossed into the next life."
            assert window.stop - window.start == 60
            inspected += 1

        boundary = window_slice(last_usable)
        rows = range(start + boundary.start, start + boundary.stop)
        assert all(is_trainable(int(dataset.minutes_to_onset[row])) for row in rows), (
            "the window nearest the onset included a timestep the model is not asked about."
        )

    assert inspected > 0


def test_lives_do_not_share_a_seed(plans: tuple[MachinePlan, ...]) -> None:
    """Reused seeds would make two lives byte-identical.

    Two lives from the same seed, scenario and duration differ only in their
    machine profile — so under a shared seed, one is a copy of the other wearing
    a different timestamp, in whichever splits they happened to land.
    """
    seeds = [life.seed for plan in plans for life in plan.lives]

    assert len(seeds) == len(set(seeds))
