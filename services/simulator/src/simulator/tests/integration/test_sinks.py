"""Sinks: what comes back out matches what went in."""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from simulator.application import run_session
from simulator.domain.engine import MachineSimulator
from simulator.domain.readings import SIGNAL_NAMES
from simulator.domain.scenario import Scenario
from simulator.domain.session import SimulationSession
from simulator.infrastructure.sinks import (
    JsonLinesGroundTruthSink,
    JsonLinesTelemetrySink,
    ParquetGroundTruthSink,
    ParquetTelemetrySink,
)
from simulator.tests.conftest import FIXED_START

#: Columns the two files are allowed to share, and the only ones: the join keys.
SHARED_COLUMNS = {"machine_id", "recorded_at"}

#: Columns that must never appear in a telemetry file.
GROUND_TRUTH_ONLY_COLUMNS = {
    "scenario",
    "bearing_wear",
    "thermal_stress",
    "load_stress",
    "health_index",
    "failure_imminent",
}


@pytest.fixture
def short_session(session: SimulationSession) -> SimulationSession:
    """A five-tick-per-minute run, so a round trip stays fast."""
    return SimulationSession.create(
        machine_ids=["M003"],
        scenario=session.scenario,
        seed=session.seed,
        duration=timedelta(minutes=5),
        started_at=session.started_at,
    )


def test_parquet_round_trip_preserves_the_readings(
    tmp_path: Path,
    short_session: SimulationSession,
) -> None:
    """Values survive the columnar encoding, including timezone-aware instants."""
    path = tmp_path / "telemetry.parquet"
    run_session(
        short_session,
        ParquetTelemetrySink(path),
        ParquetGroundTruthSink(tmp_path / "ground_truth.parquet"),
    )

    rows = pq.read_table(path).to_pylist()
    expected = MachineSimulator(short_session, short_session.machines[0]).tick(0)

    assert len(rows) == short_session.tick_count
    assert rows[0]["event_id"] == expected.telemetry.event_id
    assert rows[0]["recorded_at"] == expected.telemetry.recorded_at
    assert rows[0]["recorded_at"].tzinfo is not None
    for name in SIGNAL_NAMES:
        assert rows[0][name] == pytest.approx(getattr(expected.telemetry.reading, name))


def test_parquet_round_trip_preserves_the_ground_truth(
    tmp_path: Path,
    short_session: SimulationSession,
) -> None:
    """The hidden channels and the scenario survive too."""
    path = tmp_path / "ground_truth.parquet"
    run_session(
        short_session,
        ParquetTelemetrySink(tmp_path / "telemetry.parquet"),
        ParquetGroundTruthSink(path),
    )

    rows = pq.read_table(path).to_pylist()
    expected = MachineSimulator(short_session, short_session.machines[0]).tick(0).ground_truth

    assert len(rows) == short_session.tick_count
    assert rows[0]["scenario"] == short_session.scenario.value
    assert rows[0]["health_index"] == pytest.approx(expected.health_index)
    assert rows[0]["bearing_wear"] == pytest.approx(expected.degradation.bearing_wear)


def test_jsonl_round_trip_preserves_the_readings(
    tmp_path: Path,
    short_session: SimulationSession,
) -> None:
    """The line-delimited format carries the same values as the columnar one."""
    path = tmp_path / "telemetry.jsonl"
    run_session(
        short_session,
        JsonLinesTelemetrySink(path),
        JsonLinesGroundTruthSink(tmp_path / "ground_truth.jsonl"),
    )

    lines = path.read_text(encoding="utf-8").splitlines()
    first = json.loads(lines[0])
    expected = MachineSimulator(short_session, short_session.machines[0]).tick(0)

    assert len(lines) == short_session.tick_count
    assert first["event_id"] == expected.telemetry.event_id
    assert first["temperature"] == pytest.approx(expected.telemetry.reading.temperature)


def test_the_two_files_share_only_their_join_keys(
    tmp_path: Path,
    short_session: SimulationSession,
) -> None:
    """The ground-truth separation, verified where it matters most.

    Phase 3 builds features from the telemetry file and labels from the
    ground-truth file. Disjoint columns mean a feature builder physically cannot
    reach a label — the guarantee is a property of the files, not of anyone
    remembering to be careful.
    """
    telemetry_path = tmp_path / "telemetry.parquet"
    ground_truth_path = tmp_path / "ground_truth.parquet"
    run_session(
        short_session,
        ParquetTelemetrySink(telemetry_path),
        ParquetGroundTruthSink(ground_truth_path),
    )

    telemetry_columns = set(pq.read_schema(telemetry_path).names)
    ground_truth_columns = set(pq.read_schema(ground_truth_path).names)

    assert telemetry_columns & ground_truth_columns == SHARED_COLUMNS
    assert not telemetry_columns & GROUND_TRUTH_ONLY_COLUMNS
    assert not ground_truth_columns & set(SIGNAL_NAMES)


def test_every_row_is_written_when_batches_flush_repeatedly(tmp_path: Path) -> None:
    """A flush threshold smaller than the run must not lose or duplicate rows.

    The threshold exists so a Phase 3 dataset does not hold millions of rows in
    memory; getting the drain wrong would silently truncate the output, and a
    short dataset would look perfectly healthy.
    """
    session = SimulationSession.create(
        machine_ids=["M001"],
        scenario=Scenario.BEARING_DEGRADATION,
        seed=1,
        duration=timedelta(minutes=50),
        started_at=FIXED_START,
    )
    telemetry_path = tmp_path / "telemetry.parquet"
    ground_truth_path = tmp_path / "ground_truth.parquet"

    run_session(
        session,
        ParquetTelemetrySink(telemetry_path, flush_threshold=7),
        ParquetGroundTruthSink(ground_truth_path, flush_threshold=7),
    )

    assert pq.read_table(telemetry_path).num_rows == session.tick_count
    assert pq.read_table(ground_truth_path).num_rows == session.tick_count


def test_parent_directories_are_created(tmp_path: Path, short_session: SimulationSession) -> None:
    """The output directory will not exist in a fresh clone.

    `data/` is gitignored and cannot hold a committed placeholder — git will not
    re-include a file whose parent directory is excluded — so the sink has to
    create the path itself.
    """
    nested = tmp_path / "does" / "not" / "exist" / "telemetry.parquet"

    run_session(
        short_session,
        ParquetTelemetrySink(nested),
        ParquetGroundTruthSink(tmp_path / "does" / "not" / "exist" / "ground_truth.parquet"),
    )

    assert nested.exists()


def test_a_sink_with_nothing_to_write_creates_no_file(tmp_path: Path) -> None:
    """Closing an untouched sink leaves no zero-byte artefact behind.

    The writers open lazily on the first flush, so this is the case where they
    never opened at all — and a stray empty file is worse than none, because it
    looks like a dataset.
    """
    path = tmp_path / "telemetry.parquet"

    ParquetTelemetrySink(path).close()

    assert not path.exists()
