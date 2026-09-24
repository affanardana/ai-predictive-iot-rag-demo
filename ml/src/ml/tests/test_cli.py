"""The pipeline, end to end, through the commands a reviewer would run."""

from __future__ import annotations

from pathlib import Path

import pytest

from ml.cli import main

STARTED_AT = "2026-01-01T00:00:00+00:00"
TINY = ["--machines", "9", "--shift-machines", "3", "--days", "1"]


@pytest.fixture(scope="module")
def generated(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """One small generation, shared by the tests that read it."""
    out = tmp_path_factory.mktemp("generated")
    assert (
        main(
            [
                "dataset",
                "generate",
                *TINY,
                "--seed",
                "11",
                "--started-at",
                STARTED_AT,
                "--out",
                str(out),
            ]
        )
        == 0
    )
    return out


def test_generate_writes_both_channels_and_the_plan(generated: Path) -> None:
    assert (generated / "telemetry.parquet").exists()
    assert (generated / "ground_truth.parquet").exists()
    assert (generated / "plan.json").exists()


def test_generation_is_reproducible(tmp_path: Path) -> None:
    """PRD §12's requirement, applied to the dataset.

    Both halves are pinned: the seed fixes the fleet and every reading, and
    `--started-at` fixes the clock. Without the second, two runs agree on every
    value and disagree on every timestamp.
    """
    first = tmp_path / "first"
    second = tmp_path / "second"
    arguments = ["dataset", "generate", *TINY, "--seed", "12", "--started-at", STARTED_AT]

    assert main([*arguments, "--out", str(first)]) == 0
    assert main([*arguments, "--out", str(second)]) == 0

    for name in ("telemetry.parquet", "ground_truth.parquet", "plan.json"):
        assert (first / name).read_bytes() == (second / name).read_bytes()


def test_export_writes_a_loadable_artifact(generated: Path, tmp_path: Path) -> None:
    out = tmp_path / "artifact"

    assert main(["dataset", "export", "--dataset", str(generated), "--out", str(out)]) == 0

    for name in (
        "signals.npy",
        "minutes_to_onset.npy",
        "life_offsets.npy",
        "life_machine.npy",
        "life_onset.npy",
        "machines.json",
        "normalization.json",
        "metadata.json",
    ):
        assert (out / name).exists(), f"{name} was not written."


def test_the_life_table_is_written_beside_the_data(generated: Path, tmp_path: Path) -> None:
    """With the data it describes, not with the artifact.

    It carries the scenario, which is ground truth, and the training artifact is
    the one directory a ground-truth column must never reach.
    """
    out = tmp_path / "artifact"

    assert main(["dataset", "export", "--dataset", str(generated), "--out", str(out)]) == 0
    assert (generated / "lives.parquet").exists()
    assert not (out / "lives.parquet").exists()


def test_report_prints_prevalence_and_the_clock_baseline(
    generated: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The exit condition, made checkable by eye."""
    out = tmp_path / "artifact"
    assert main(["dataset", "export", "--dataset", str(generated), "--out", str(out)]) == 0

    assert main(["dataset", "report", "--dataset", str(generated), "--artifact", str(out)]) == 0

    printed = capsys.readouterr().out
    assert "prevalence" in printed
    assert "clock baseline" in printed
    assert "train" in printed
    assert "average precision" in printed


def test_an_impossible_fleet_is_a_usage_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Caught at the flag, with a message about the flag."""
    assert main(["dataset", "generate", "--machines", "0", "--out", str(tmp_path)]) == 2

    assert "machines" in capsys.readouterr().err


def test_exporting_before_generating_fails_clearly(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A missing dataset reads as a missing dataset, not as a traceback."""
    assert (
        main(
            [
                "dataset",
                "export",
                "--dataset",
                str(tmp_path / "absent"),
                "--out",
                str(tmp_path / "out"),
            ]
        )
        == 1
    )

    assert "plan.json" in capsys.readouterr().err
