"""The artifact, and the checks that refuse a dataset that does not add up."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from ml.dataset.artifacts import (
    Dataset,
    _verify_blocks,
    build_dataset,
    life_table,
    read_artifact,
    write_artifact,
)
from ml.dataset.errors import DatasetError
from ml.dataset.generation import GROUND_TRUTH_FILENAME, TELEMETRY_FILENAME
from ml.dataset.labelling import NEVER_FAILS
from ml.dataset.lives import MachinePlan

_TIMESTAMP = pa.timestamp("us", tz="UTC")


def test_the_artifact_round_trips(dataset: Dataset, tmp_path: Path) -> None:
    write_artifact(tmp_path, dataset)
    restored = read_artifact(tmp_path)

    assert np.array_equal(restored.signals, dataset.signals)
    assert np.array_equal(restored.minutes_to_onset, dataset.minutes_to_onset)
    assert np.array_equal(restored.life_offsets, dataset.life_offsets)
    assert np.array_equal(restored.life_machine, dataset.life_machine)
    assert np.array_equal(restored.life_onset, dataset.life_onset)
    assert restored.machine_ids == dataset.machine_ids
    assert restored.splits == dataset.splits
    assert np.allclose(restored.normalization.means, dataset.normalization.means)
    assert np.allclose(restored.normalization.stds, dataset.normalization.stds)


def test_normalising_leaves_the_training_rows_centred(
    dataset: Dataset, plans: tuple[MachinePlan, ...]
) -> None:
    """Applying the stored statistics really does standardise the stored data."""
    from ml.dataset.splits import Split

    train = np.array([index for index, split in enumerate(dataset.splits) if split is Split.TRAIN])
    training_life = np.isin(dataset.life_machine, train)
    rows = np.repeat(training_life, dataset.life_lengths()) & dataset.trainable()
    scaled = dataset.normalization.apply(dataset.signals[rows].astype(np.float64))

    assert np.allclose(scaled.mean(axis=0), 0.0, atol=1e-6)
    assert np.allclose(scaled.std(axis=0), 1.0, atol=1e-6)


def test_the_artifact_carries_no_ground_truth_column(tmp_path: Path, dataset: Dataset) -> None:
    """`machines.json` names machines and splits, and nothing else."""
    write_artifact(tmp_path, dataset)
    written = {path.name for path in tmp_path.iterdir()}

    assert "scenario.json" not in written
    assert "lives.parquet" not in written
    assert "ground_truth.parquet" not in written


def test_the_life_table_records_the_ground_truth(
    dataset: Dataset, plans: tuple[MachinePlan, ...]
) -> None:
    table = life_table(dataset, plans)
    scenarios = set(table.column("scenario").to_pylist())

    assert table.num_rows == dataset.lives
    assert "NORMAL" in scenarios
    assert len(scenarios) > 1
    assert table.column("ticks").to_pylist() == dataset.life_lengths().tolist()
    onsets = table.column("onset").to_pylist()
    assert any(value is None for value in onsets)
    assert any(value is not None for value in onsets)


def test_a_truncated_channel_is_refused(fresh_fleet: tuple[Path, tuple]) -> None:
    """Fewer rows on one side means the positional correspondence is broken."""
    directory, _ = fresh_fleet
    path = directory / TELEMETRY_FILENAME
    table = pq.read_table(path)
    pq.write_table(table.slice(0, table.num_rows - 1), path)

    with pytest.raises(DatasetError, match="rows"):
        build_dataset(directory)


def test_a_disagreeing_channel_is_refused(fresh_fleet: tuple[Path, tuple]) -> None:
    """The silent failure this whole check exists to prevent.

    One row's machine id moved makes every subsequent label belong to the wrong
    timestep, and nothing else about the dataset would look wrong.
    """
    directory, _ = fresh_fleet
    path = directory / GROUND_TRUTH_FILENAME
    table = pq.read_table(path)
    column = table.column("machine_id").to_pylist()
    column[5] = "M999"
    index = table.schema.get_field_index("machine_id")
    pq.write_table(table.set_column(index, "machine_id", pa.array(column, type=pa.string())), path)

    with pytest.raises(DatasetError, match="disagree"):
        build_dataset(directory)


def test_a_timestamp_anomaly_is_refused(fresh_fleet: tuple[Path, tuple]) -> None:
    """Rows reordered *within* a life keep every block the right length.

    Which is why the row counts and the session boundaries cannot catch it, and
    why the interval is checked separately. Both files are shifted together, so
    the correspondence check passes and only this one fires.
    """
    directory, _ = fresh_fleet
    for name in (TELEMETRY_FILENAME, GROUND_TRUTH_FILENAME):
        path = directory / name
        table = pq.read_table(path)
        # `.copy()` because pyarrow hands back a read-only view when it can
        # avoid a copy, and this test is about to write through it.
        micros = table.column("recorded_at").combine_chunks().cast(pa.int64()).to_numpy().copy()
        micros[5] += 1_000_000
        index = table.schema.get_field_index("recorded_at")
        pq.write_table(
            table.set_column(index, "recorded_at", pa.array(micros).cast(_TIMESTAMP)),
            path,
        )

    with pytest.raises(DatasetError, match="jump"):
        build_dataset(directory)


def test_misaligned_life_boundaries_are_refused(fleet: tuple[Path, tuple]) -> None:
    """Checked directly, since a shifted boundary keeps the total row count.

    Moving one interior boundary leaves the file the right length and every
    session id present — only the *positions* are wrong, and every label after
    that point would be attributed to the wrong life.
    """
    directory, _ = fleet
    telemetry = pq.read_table(directory / TELEMETRY_FILENAME)

    from ml.dataset.artifacts import _life_offsets

    plans = _plans(directory)
    offsets = _life_offsets([life for plan in plans for life in plan.lives])
    shifted = offsets.copy()
    shifted[1] += 1

    assert int(shifted[-1]) == telemetry.num_rows
    with pytest.raises(DatasetError, match="line up"):
        _verify_blocks(telemetry, shifted)


def test_a_plan_from_another_run_is_refused(fresh_fleet: tuple[Path, tuple]) -> None:
    """A plan that describes a different fleet cannot label this one."""
    directory, _ = fresh_fleet
    plan_path = directory / "plan.json"
    payload = plan_path.read_text(encoding="utf-8").replace('"seed": 7', '"seed": 8')
    plan_path.write_text(payload, encoding="utf-8")

    with pytest.raises(DatasetError):
        build_dataset(directory)


def test_the_never_fails_sentinel_cannot_collide_with_a_real_value(dataset: Dataset) -> None:
    """The sentinel has to be unmistakable, not merely unlikely.

    The deepest a real offset goes is the length of the longest life, since a
    life's onset can be at most that far back. The sentinel sits far below that,
    so it can never be confused with "already failed a while ago".

    The sentinel's own rows are excluded from the comparison rather than
    asserted away: a never-failing life is *entirely* sentinel, so a rule that
    no row may equal it would fail on exactly the rows it exists for.
    """
    longest = int(dataset.life_lengths().max())
    offsets = dataset.minutes_to_onset
    real = offsets[offsets != NEVER_FAILS]

    assert -longest > NEVER_FAILS
    assert real.size > 0
    assert bool((real >= -longest).all())


def _plans(directory: Path) -> tuple[MachinePlan, ...]:
    from ml.dataset.plan import PLAN_FILENAME, read_plan

    return read_plan(directory / PLAN_FILENAME).build()
