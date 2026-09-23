"""The command line, end to end."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from simulator.cli import EXIT_USAGE_ERROR, main
from simulator.domain.scenario import Scenario


@pytest.mark.parametrize("scenario", list(Scenario))
def test_listing_scenarios_names_them_all(
    scenario: Scenario, capsys: pytest.CaptureFixture[str]
) -> None:
    """The command exists so a caller can discover the names to pass."""
    assert main(["scenarios"]) == 0

    assert scenario.value in capsys.readouterr().out


def test_dataset_writes_two_separate_files(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The separation, end to end: two files, each with only its own columns."""
    exit_code = main(
        ["dataset", "--machines", "2", "--minutes", "5", "--seed", "5", "--out", str(tmp_path)]
    )

    assert exit_code == 0
    telemetry = tmp_path / "telemetry.parquet"
    ground_truth = tmp_path / "ground_truth.parquet"
    assert telemetry.exists()
    assert ground_truth.exists()
    # Two machines, five instants each.
    assert pq.read_table(telemetry).num_rows == 10
    assert pq.read_table(ground_truth).num_rows == 10

    assert "observations" in capsys.readouterr().out


def test_dataset_can_write_jsonl(tmp_path: Path) -> None:
    """The inspectable format, for looking at a small run by eye."""
    assert (
        main(
            [
                "dataset",
                "--machines",
                "1",
                "--minutes",
                "3",
                "--out",
                str(tmp_path),
                "--format",
                "jsonl",
            ]
        )
        == 0
    )

    lines = (tmp_path / "telemetry.jsonl").read_text(encoding="utf-8").splitlines()

    assert len(lines) == 3
    assert (tmp_path / "ground_truth.jsonl").exists()


def test_the_demo_stream_is_reproducible(tmp_path: Path) -> None:
    """PRD section 12, exercised through the command a reviewer would run.

    Both halves are pinned: `--demo` fixes the machine, scenario, and seed, and
    `--started-at` fixes the clock. Without the second, two runs agree on every
    reading but disagree on every timestamp — reproducible enough to look right
    and not enough to diff.
    """
    first = tmp_path / "first"
    second = tmp_path / "second"
    arguments = [
        "realtime",
        "--demo",
        "--tick-seconds",
        "0",
        "--sink",
        "jsonl",
        "--started-at",
        "2026-09-23T12:00:00+00:00",
        "--out",
    ]

    assert main([*arguments, str(first)]) == 0
    assert main([*arguments, str(second)]) == 0

    assert (first / "telemetry.jsonl").read_text(encoding="utf-8") == (
        second / "telemetry.jsonl"
    ).read_text(encoding="utf-8")
    assert (first / "ground_truth.jsonl").read_text(encoding="utf-8") == (
        second / "ground_truth.jsonl"
    ).read_text(encoding="utf-8")


def test_an_unpinned_stream_uses_the_current_time(tmp_path: Path) -> None:
    """Without `--started-at` the run starts now, which is what a live demo wants."""
    out = tmp_path / "live"
    before = datetime.now(UTC)

    assert (
        main(
            [
                "realtime",
                "--minutes",
                "2",
                "--tick-seconds",
                "0",
                "--sink",
                "jsonl",
                "--out",
                str(out),
            ]
        )
        == 0
    )

    first = json.loads((out / "telemetry.jsonl").read_text(encoding="utf-8").splitlines()[0])
    recorded_at = datetime.fromisoformat(first["recorded_at"])

    assert recorded_at >= before


def test_a_naive_start_instant_is_rejected(capsys: pytest.CaptureFixture[str]) -> None:
    """Caught at the flag, with a message about the flag."""
    with pytest.raises(SystemExit):
        main(["realtime", "--started-at", "2026-09-23T12:00:00"])

    assert "timezone" in capsys.readouterr().err


def test_an_unknown_scenario_is_a_usage_error(capsys: pytest.CaptureFixture[str]) -> None:
    """A bad flag exits with a message, not a traceback.

    The scenario name comes from the command line, so getting it wrong is a
    usage problem and should read like one.
    """
    assert main(["dataset", "--scenario", "EXPLODING"]) == EXIT_USAGE_ERROR

    error = capsys.readouterr().err
    assert "EXPLODING" in error
    assert "BEARING_DEGRADATION" in error


def test_an_impossible_configuration_is_a_usage_error(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A run with no machines is rejected the same way."""
    assert main(["dataset", "--machines", "0"]) == EXIT_USAGE_ERROR

    assert "machine" in capsys.readouterr().err


def test_the_console_sink_prints_the_signals_and_the_truth(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """What a demonstration actually shows.

    Both channels appear because the point of watching a run is seeing vibration
    climb *and* the bearing wear behind it — while remaining two separate sinks
    that happen to share a stream.
    """
    assert main(["realtime", "--minutes", "3", "--tick-seconds", "0"]) == 0

    output = capsys.readouterr().out
    assert "M001" in output
    assert "temperature" not in output  # values are labelled compactly, not named
    assert "T " in output
    assert "V " in output
    assert "truth" in output
