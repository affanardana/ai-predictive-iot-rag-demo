"""Reading the generated files back, and writing what a model trains on.

This module is where ground truth and observations meet, so it is worth being
precise about how. They meet in exactly two ways:

1. **As labels.** `ground_truth.parquet` says when each life entered failure;
   that becomes `minutes_to_onset`, and nothing else about the truth survives
   into the artifact.
2. **As a row correspondence.** Row *i* of the telemetry file is matched to row
   *i* of the ground-truth file — and that is *verified*, not assumed.

The second point is not pedantry. Joining the two files positionally is the
obvious thing to do and it is a silent hazard: if either sink ever reordered a
row, every label from that point on would belong to a neighbouring timestep, the
dataset would look completely normal, and every metric computed from it would be
subtly wrong. `_verify_correspondence` compares the two files' key columns
element-wise and refuses to continue if they differ, which costs one comparison
over two columns and removes the failure mode entirely.

## Why the artifact is flat

One row per `(life, timestep)`, not one row per window. Consecutive windows
overlap by 59 of their 60 timesteps, so a pre-expanded tensor stores the same
reading sixty times: 3.5 GB at `float16` against 58 MB flat. `ml.dataset.windows`
slices sequences out of the flat arrays instead, which is index arithmetic and
costs nothing. On Colab's free tier that difference is the entire memory budget.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from ml.dataset.errors import DatasetError
from ml.dataset.features import FEATURE_COLUMNS
from ml.dataset.generation import GROUND_TRUTH_FILENAME, TELEMETRY_FILENAME
from ml.dataset.labelling import (
    HORIZON_MINUTES,
    NEVER_FAILS,
    minutes_to_onset,
    onset_index,
)
from ml.dataset.lives import Life, MachinePlan
from ml.dataset.plan import PLAN_FILENAME, read_plan
from ml.dataset.splits import Split, assign_splits
from ml.dataset.windows import WINDOW_MINUTES
from simulator.domain.session import DEFAULT_SAMPLE_INTERVAL

ARTIFACT_VERSION = 1

SIGNALS_FILENAME = "signals.npy"
MINUTES_FILENAME = "minutes_to_onset.npy"
OFFSETS_FILENAME = "life_offsets.npy"
LIFE_MACHINE_FILENAME = "life_machine.npy"
LIFE_ONSET_FILENAME = "life_onset.npy"
MACHINES_FILENAME = "machines.json"
NORMALIZATION_FILENAME = "normalization.json"
METADATA_FILENAME = "metadata.json"

LIVES_TABLE_FILENAME = "lives.parquet"

Float16Array = npt.NDArray[np.float16]
Float64Array = npt.NDArray[np.float64]
Int16Array = npt.NDArray[np.int16]
Int32Array = npt.NDArray[np.int32]
Int64Array = npt.NDArray[np.int64]
BoolArray = npt.NDArray[np.bool_]

#: One sample interval in microseconds, matching the Parquet timestamp unit.
_SAMPLE_MICROS = int(DEFAULT_SAMPLE_INTERVAL.total_seconds() * 1_000_000)


@dataclass(frozen=True, slots=True)
class Normalization:
    """Per-signal statistics, and where they came from.

    Computed over **training rows only**. Recomputing them across the whole
    dataset would carry information about the validation and test machines into
    training — the classic silent leak, and one that leaves no trace in the code
    because nothing about a global mean looks wrong.
    """

    means: Float64Array
    stds: Float64Array
    source_split: Split
    source_machines: int
    source_rows: int

    def apply(self, signals: Float64Array) -> Float64Array:
        """Return `signals` standardised to zero mean and unit variance."""
        return (signals - self.means) / self.stds


@dataclass(frozen=True, slots=True)
class Dataset:
    """The flat training artifact and the tables that describe it."""

    signals: Float16Array
    minutes_to_onset: Int16Array
    life_offsets: Int64Array
    life_machine: Int32Array
    life_onset: Int32Array
    machine_ids: tuple[str, ...]
    splits: tuple[Split, ...]
    normalization: Normalization

    @property
    def rows(self) -> int:
        """Return the number of timesteps."""
        return int(self.signals.shape[0])

    @property
    def lives(self) -> int:
        """Return the number of lives."""
        return int(self.life_machine.shape[0])

    @property
    def features(self) -> int:
        """Return how many signals each timestep carries."""
        return int(self.signals.shape[1])

    def life_lengths(self) -> Int64Array:
        """Return how many timesteps each life holds."""
        return np.diff(self.life_offsets)

    def trainable(self) -> BoolArray:
        """Return which timesteps pose the question the model answers.

        Everything before onset, plus lives that never fail. The post-onset tail
        is excluded — see `ml.dataset.labelling` for why it is neither class.
        """
        # Bound to a local so the element-wise `|` is checked against the
        # return type here rather than inferred as `Any` at the boundary.
        trainable: BoolArray = (self.minutes_to_onset == NEVER_FAILS) | (self.minutes_to_onset > 0)
        return trainable

    def split_of(self, machine_id: str) -> Split:
        """Return which split a machine belongs to."""
        return self.splits[self.machine_ids.index(machine_id)]


def build_dataset(dataset_dir: Path) -> Dataset:
    """Read a generated dataset and produce the training artifact.

    Raises:
        DatasetError: if the files do not match the plan that produced them, or
            the two channels are not row-for-row aligned.
    """
    document = read_plan(dataset_dir / PLAN_FILENAME)
    plans = document.build()

    telemetry = pq.read_table(dataset_dir / TELEMETRY_FILENAME)
    truth = pq.read_table(dataset_dir / GROUND_TRUTH_FILENAME)
    _verify_correspondence(telemetry, truth)

    lives = [life for plan in plans for life in plan.lives]
    offsets = _life_offsets(lives)
    _verify_blocks(telemetry, offsets)

    life_machine = np.array(
        [index for index, plan in enumerate(plans) for _ in plan.lives], dtype=np.int32
    )
    minutes, onsets = _label_lives(truth, lives, offsets)

    assignments = assign_splits(
        [(plan.machine_id, plan.primary_scenario) for plan in plans],
        shift=frozenset(document.shift_machines),
    )
    split_by_machine = {item.machine_id: item.split for item in assignments}
    machine_position = {plan.machine_id: index for index, plan in enumerate(plans)}

    # Cast before computing the statistics, not after. The artifact stores
    # `float16`, and statistics taken from the `float64` values would describe a
    # tensor that is not the one on disk — the z-scored data would come out
    # slightly off centre for no reason anyone could later reconstruct.
    signals = _read_signals(telemetry, rows=telemetry.num_rows).astype(np.float16)
    normalization = _compute_normalization(
        signals,
        minutes,
        life_machine,
        offsets,
        training_lives=[
            machine_position[item.machine_id] for item in assignments if item.split is Split.TRAIN
        ],
    )

    return Dataset(
        signals=signals,
        minutes_to_onset=minutes,
        life_offsets=offsets,
        life_machine=life_machine,
        life_onset=onsets,
        machine_ids=tuple(plan.machine_id for plan in plans),
        splits=tuple(split_by_machine[plan.machine_id] for plan in plans),
        normalization=normalization,
    )


def life_table(dataset: Dataset, plans: Sequence[MachinePlan]) -> pa.Table:
    """Build the per-life analysis table.

    Carries the scenario and the onset, which are ground truth. It therefore
    belongs beside the data it describes and never in the training artifact —
    `machines.json` there holds machine ids and splits, and nothing else.
    """
    lives = [life for plan in plans for life in plan.lives]
    length = dataset.life_lengths()
    if len(lives) != dataset.lives:
        raise DatasetError(
            f"the plan holds {len(lives)} lives and the artifact {dataset.lives}. "
            "They were produced from different plans."
        )

    life_index: list[int] = []
    seen: dict[str, int] = {}
    for life in lives:
        life_index.append(seen.get(life.machine_id, 0))
        seen[life.machine_id] = life_index[-1] + 1

    return pa.table(
        {
            "machine_id": [life.machine_id for life in lives],
            "life_index": life_index,
            "session_id": [life.life_id for life in lives],
            "scenario": [life.scenario.value for life in lives],
            "primary_scenario": [
                plans[index].primary_scenario.value for index in dataset.life_machine
            ],
            "split": [dataset.splits[index].value for index in dataset.life_machine],
            "started_at": [life.started_at for life in lives],
            "duration_minutes": [life.duration.total_seconds() // 60 for life in lives],
            "row_offset": dataset.life_offsets[:-1].tolist(),
            "ticks": length.tolist(),
            "onset": [
                None if int(value) == NEVER_FAILS else int(value) for value in dataset.life_onset
            ],
        }
    )


def write_life_table(directory: Path, table: pa.Table) -> Path:
    """Write the life table beside the data it describes."""
    path = directory / LIVES_TABLE_FILENAME
    pq.write_table(table, path, compression="snappy")
    return path


def write_artifact(directory: Path, dataset: Dataset) -> None:
    """Write the artifact to `directory`."""
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / SIGNALS_FILENAME, dataset.signals)
    np.save(directory / MINUTES_FILENAME, dataset.minutes_to_onset)
    np.save(directory / OFFSETS_FILENAME, dataset.life_offsets)
    np.save(directory / LIFE_MACHINE_FILENAME, dataset.life_machine)
    np.save(directory / LIFE_ONSET_FILENAME, dataset.life_onset)

    _write_json(
        directory / MACHINES_FILENAME,
        {
            "machines": [
                {"machine_id": machine_id, "split": split.value}
                for machine_id, split in zip(dataset.machine_ids, dataset.splits, strict=True)
            ]
        },
    )
    _write_json(
        directory / NORMALIZATION_FILENAME,
        {
            "mean": dataset.normalization.means.tolist(),
            "std": dataset.normalization.stds.tolist(),
            "ddof": 0,
            "computed_from": {
                "split": dataset.normalization.source_split.value,
                "machines": dataset.normalization.source_machines,
                "rows": dataset.normalization.source_rows,
            },
        },
    )
    _write_json(
        directory / METADATA_FILENAME,
        {
            "version": ARTIFACT_VERSION,
            "rows": dataset.rows,
            "lives": dataset.lives,
            "machines": len(dataset.machine_ids),
            "features": list(FEATURE_COLUMNS),
            "horizon_minutes": HORIZON_MINUTES,
            "window_minutes": WINDOW_MINUTES,
            "never_fails": NEVER_FAILS,
        },
    )


def read_artifact(directory: Path) -> Dataset:
    """Load an artifact written by `write_artifact`."""
    machines = json.loads((directory / MACHINES_FILENAME).read_text(encoding="utf-8"))
    return Dataset(
        # Memory-mapped: the signals are read in windows, and a training run
        # should not have to hold the whole fleet in RAM to look at one life.
        signals=np.load(directory / SIGNALS_FILENAME, mmap_mode="r"),
        minutes_to_onset=np.load(directory / MINUTES_FILENAME),
        life_offsets=np.load(directory / OFFSETS_FILENAME),
        life_machine=np.load(directory / LIFE_MACHINE_FILENAME),
        life_onset=np.load(directory / LIFE_ONSET_FILENAME),
        machine_ids=tuple(item["machine_id"] for item in machines["machines"]),
        splits=tuple(Split(item["split"]) for item in machines["machines"]),
        normalization=read_normalization(directory),
    )


def read_normalization(directory: Path) -> Normalization:
    """Read only the standardisation statistics.

    The inference service needs these and nothing else from the artifact — the
    signals it standardises arrive in the request. Reading the whole 58 MB array
    to recover a six-element mean would be a strange thing to deploy.
    """
    payload = json.loads((directory / NORMALIZATION_FILENAME).read_text(encoding="utf-8"))
    computed_from = payload["computed_from"]
    return Normalization(
        means=np.asarray(payload["mean"], dtype=np.float64),
        stds=np.asarray(payload["std"], dtype=np.float64),
        source_split=Split(computed_from["split"]),
        source_machines=int(computed_from["machines"]),
        source_rows=int(computed_from["rows"]),
    )


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _verify_correspondence(telemetry: pa.Table, truth: pa.Table) -> None:
    """Prove row *i* of one file describes the same machine and instant as row *i* of the other."""
    if telemetry.num_rows != truth.num_rows:
        raise DatasetError(
            f"telemetry has {telemetry.num_rows:,} rows and ground truth has "
            f"{truth.num_rows:,}. They are written one tick at a time and must "
            "agree; regenerate the dataset."
        )
    for column in ("machine_id", "recorded_at"):
        agrees = pc.all(
            pc.equal(telemetry.column(column), truth.column(column)), min_count=1
        ).as_py()
        if not agrees:
            raise DatasetError(
                f"the two channel files disagree on '{column}'. Matching them "
                "positionally would mislabel every row from the first difference "
                "onwards, so the export stops here. Regenerate the dataset."
            )


def _life_offsets(lives: Sequence[Life]) -> Int64Array:
    """Return the start offset of every life in the flat arrays."""
    counts = np.array([life.duration // DEFAULT_SAMPLE_INTERVAL for life in lives], dtype=np.int64)
    offsets = np.zeros(len(lives) + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])
    return offsets


def _verify_blocks(telemetry: pa.Table, offsets: Int64Array) -> None:
    """Check the file is exactly the lives, in order, with nothing merged or split.

    Three facts together prove the row-to-life mapping everything below depends
    on:

    * the file has exactly as many rows as the plan calls for;
    * the session id changes at every life boundary and nowhere else;
    * the machine id changes only at a life boundary; and
    * timestamps advance by exactly one sample interval within a life.

    The last is what rules out duplicated or reordered rows *inside* a life,
    which would leave each block the right length while its labels belonged to
    the wrong timesteps — the failure that survives every length check.
    """
    if int(offsets[-1]) != telemetry.num_rows:
        raise DatasetError(
            f"the plan calls for {int(offsets[-1]):,} rows but the telemetry file "
            f"holds {telemetry.num_rows:,}. It was generated from a different "
            "plan, or generation was interrupted."
        )

    boundaries = offsets[1:-1]
    session_changes = _change_positions(telemetry.column("session_id"))
    if not np.array_equal(session_changes, boundaries):
        raise DatasetError(
            "the telemetry file's sessions do not line up with the plan's lives. "
            "Label attribution would be shifted for every row after the first "
            "mismatch."
        )

    machine_changes = _change_positions(telemetry.column("machine_id"))
    if not np.isin(machine_changes, boundaries, assume_unique=True).all():
        raise DatasetError("a machine changes part-way through a life.")

    _verify_timestamps(telemetry.column("recorded_at"), boundaries)


def _verify_timestamps(column: pa.ChunkedArray, boundaries: Int64Array) -> None:
    """Check timestamps advance by one interval inside each life."""
    micros = column.combine_chunks().cast(pa.int64()).to_numpy(zero_copy_only=False)
    if micros.size < 2:
        return
    deltas = np.diff(micros)

    # The step *between* lives is not constrained here, so those positions are
    # excluded rather than asserted: this check is about ordering within a life.
    interior = np.ones(deltas.shape, dtype=bool)
    interior[boundaries[boundaries > 0] - 1] = False
    wrong = np.flatnonzero(interior & (deltas != _SAMPLE_MICROS))
    if wrong.size:
        position = int(wrong[0])
        raise DatasetError(
            f"timestamps jump by {int(deltas[position])} microseconds at row "
            f"{position + 1}, not {_SAMPLE_MICROS}. Rows are missing, "
            "duplicated, or out of order within a life."
        )


def _change_positions(column: pa.ChunkedArray) -> Int64Array:
    """Return the positions at which `column` differs from its predecessor."""
    flat = column.combine_chunks()
    if len(flat) < 2:
        return np.empty(0, dtype=np.int64)
    differs = pc.not_equal(flat.slice(1), flat.slice(0, len(flat) - 1))
    return np.flatnonzero(differs.to_numpy(zero_copy_only=False)) + 1


def _label_lives(
    truth: pa.Table, lives: Sequence[Life], offsets: Int64Array
) -> tuple[Int16Array, Int32Array]:
    """Turn each life's `failure_imminent` column into signed offsets."""
    imminent = truth.column("failure_imminent").combine_chunks().to_numpy(zero_copy_only=False)
    minutes = np.empty(offsets[-1], dtype=np.int16)
    onsets = np.empty(len(lives), dtype=np.int32)

    for index, _ in enumerate(lives):
        start = int(offsets[index])
        stop = int(offsets[index + 1])
        onset = onset_index(imminent[start:stop])
        onsets[index] = NEVER_FAILS if onset is None else onset
        minutes[start:stop] = minutes_to_onset(stop - start, onset)

    return minutes, onsets


def _read_signals(telemetry: pa.Table, *, rows: int) -> Float64Array:
    """Read the six feature columns, and only those.

    Selecting by name from `FEATURE_COLUMNS` rather than dropping identifiers
    from the whole table is the difference between a whitelist and a blacklist:
    a column added to the telemetry stream later is then absent by default
    rather than a feature by accident.
    """
    signals = np.empty((rows, len(FEATURE_COLUMNS)), dtype=np.float64)
    for position, name in enumerate(FEATURE_COLUMNS):
        signals[:, position] = (
            telemetry.column(name).combine_chunks().to_numpy(zero_copy_only=False)
        )
    return signals


def _compute_normalization(
    signals: Float16Array,
    minutes: Int16Array,
    life_machine: Int32Array,
    offsets: Int64Array,
    *,
    training_lives: Sequence[int],
) -> Normalization:
    """Compute per-signal statistics from training machines only.

    Over training *rows the model will actually see*, too. Including the
    post-onset tail would skew every statistic towards the saturated end of the
    damage curve, so the normalisation would describe a mixture the model never
    trains on.

    Accumulated in `float64` regardless of the storage dtype: a mean taken in
    `float16` would lose precision to the running sum long before it finished.
    """
    # Sized and keyed by *machine*: `training_lives` holds machine positions and
    # `life_machine` holds machine indices, so both index this table directly.
    # Sizing it by lives instead happens to work while there are more lives than
    # machines, and stops working the moment there are not — the same conflation
    # that silently broke `baseline.elapsed_minutes`.
    training_machine = np.zeros(int(life_machine.max()) + 1, dtype=bool)
    training_machine[np.asarray(training_lives, dtype=np.int64)] = True

    row_is_training = np.repeat(training_machine[life_machine], np.diff(offsets))
    trainable = (minutes == NEVER_FAILS) | (minutes > 0)
    selected = signals[row_is_training & trainable].astype(np.float64)

    if selected.shape[0] == 0:
        raise DatasetError(
            "no training rows were selected, so normalisation statistics cannot "
            "be computed. The split is empty or the plan is degenerate."
        )

    stds = selected.std(axis=0)
    if not bool((stds > 0.0).all()):
        flat = [FEATURE_COLUMNS[position] for position, value in enumerate(stds) if not value > 0.0]
        raise DatasetError(f"these signals are constant across the training set: {flat}.")

    return Normalization(
        means=selected.mean(axis=0),
        stds=stds,
        source_split=Split.TRAIN,
        source_machines=len(set(training_lives)),
        source_rows=int(selected.shape[0]),
    )
