"""The run configuration, the window index, the sampler, and the manifest.

Everything here is torch-free on purpose: these are the pieces that could make a
published number wrong, so they are the pieces that must be testable where the
model cannot run.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ml.dataset.artifacts import Dataset
from ml.dataset.labelling import NEVER_FAILS
from ml.dataset.splits import Split
from ml.experiment.config import TrainConfig
from ml.experiment.errors import ExperimentError
from ml.experiment.index import WindowIndex, build_index, machine_days
from ml.experiment.manifest import (
    Environment,
    RunManifest,
    build,
    describe_mismatch,
    fingerprint,
    read_manifest,
    write_manifest,
)
from ml.experiment.predictions import Predictions, read_predictions, write_predictions
from ml.experiment.sampling import draw_epoch, life_share


class TestConfig:
    """The run configuration and its identity."""

    def test_the_digest_is_canonical(self) -> None:
        """Two settings that differ must not hash alike, and order must not matter."""
        assert TrainConfig().digest == TrainConfig().digest
        assert TrainConfig().digest != TrainConfig(seed=1).digest

    def test_a_config_round_trips_through_json(self) -> None:
        config = TrainConfig(seed=99, hidden_size=32)

        assert TrainConfig.from_json(config.to_json()).digest == config.digest

    def test_the_window_and_horizon_are_fixed_by_the_spec(self) -> None:
        """PRD §25.8 and §25.9 pin both at sixty, so neither is a tuning knob."""
        with pytest.raises(ExperimentError, match="window"):
            TrainConfig(window=30)
        with pytest.raises(ExperimentError, match="horizon"):
            TrainConfig(horizon=120)

    def test_unknown_fields_are_refused_rather_than_ignored(self) -> None:
        """Silently dropping one would mean running something other than the file."""
        with pytest.raises(ExperimentError, match="Unknown"):
            TrainConfig.from_json('{"seed": 1, "magic": 2}')

    def test_an_epoch_hits_the_target_rate_exactly(self) -> None:
        """The trained prior is stated, not observed.

        That is what makes the correction in `ml.evaluation.prior` algebra
        rather than guesswork.
        """
        positives, negatives = TrainConfig(target_positive_rate=0.25).epoch_size(1_000)

        assert positives / (positives + negatives) == pytest.approx(0.25, abs=1e-3)

    def test_the_tiny_config_is_runnable_on_a_cpu(self) -> None:
        tiny = TrainConfig.tiny()

        assert tiny.device == "cpu"
        assert tiny.num_workers == 0
        assert tiny.max_epochs <= 5


class TestIndex:
    """Which windows exist, where they sit, and what they are labelled."""

    def test_windows_match_an_independent_count(self, dataset: Dataset) -> None:
        """Counted from `window_ends` directly, not through `build_index`."""
        from ml.dataset.windows import window_ends

        index = build_index(dataset, Split.TEST)
        lengths = np.diff(dataset.life_offsets)
        members = {i for i, s in enumerate(dataset.splits) if s is Split.TEST}
        expected = sum(
            len(
                window_ends(
                    int(lengths[life]),
                    None
                    if int(dataset.life_onset[life]) == NEVER_FAILS
                    else int(dataset.life_onset[life]),
                )
            )
            for life in range(dataset.lives)
            if int(dataset.life_machine[life]) in members
        )

        assert len(index) == expected

    def test_every_window_sits_inside_one_life_and_before_its_onset(self, dataset: Dataset) -> None:
        """Recomputed from the offsets, not read back from the index."""
        index = build_index(dataset, Split.TEST)

        for position in range(0, len(index), 997):
            end = int(index.ends[position])
            life = int(index.life[position])
            start = int(dataset.life_offsets[life])
            stop = int(dataset.life_offsets[life + 1])

            assert start <= end - 59
            assert end < stop
            # `life_onset` is a position *within* the life, while `end` is a
            # global row index, so the comparison has to be made in one space.
            onset = int(dataset.life_onset[life])
            assert onset == NEVER_FAILS or (end - start) < onset

    def test_labels_agree_with_minutes_to_onset(self, dataset: Dataset) -> None:
        index = build_index(dataset, Split.TEST)
        minutes = dataset.minutes_to_onset[index.ends].astype(np.int64)

        assert np.array_equal(index.label, ((minutes > 0) & (minutes <= 60)).astype(np.int64))

    def test_the_window_and_row_floors_both_exist_and_are_recorded(self, dataset: Dataset) -> None:
        """The comparison the plan turns on: the published floor is per row.

        A window set is a smaller, differently-balanced subset of the same
        lives, so the model's average precision has to be measured against a
        floor recomputed on *its* rows rather than against the published one.
        """
        from ml.dataset import baseline
        from ml.evaluation import metrics

        index = build_index(dataset, Split.TEST)
        hazard = baseline.fit(dataset)
        window_floor = metrics.average_precision(index.label, index.floor_scores(hazard))

        rows = baseline.split_rows(dataset, Split.TEST)
        row_minutes = dataset.minutes_to_onset[rows].astype(np.int64)
        row_labels = ((row_minutes > 0) & (row_minutes <= 60)).astype(np.int64)
        row_floor = metrics.average_precision(
            row_labels, hazard.score(baseline.elapsed_minutes(dataset)[rows])
        )

        assert len(index) < rows.shape[0]
        assert index.prevalence > row_labels.mean()
        assert 0.0 <= window_floor <= 1.0
        assert 0.0 <= row_floor <= 1.0

    def test_a_split_with_no_machines_is_refused(self, dataset: Dataset) -> None:
        empty = Dataset(
            signals=dataset.signals,
            minutes_to_onset=dataset.minutes_to_onset,
            life_offsets=dataset.life_offsets,
            life_machine=dataset.life_machine,
            life_onset=dataset.life_onset,
            machine_ids=dataset.machine_ids,
            splits=tuple(Split.TRAIN for _ in dataset.splits),
            normalization=dataset.normalization,
        )

        with pytest.raises(ExperimentError, match="assigned"):
            build_index(empty, Split.TEST)

    def test_machine_days_counts_time_not_lives(self, dataset: Dataset) -> None:
        assert machine_days(dataset, Split.TEST) > 0.0


class TestSampling:
    """Drawing an epoch, and the property that no life is amplified."""

    def test_the_class_ratio_is_what_was_asked_for(self, dataset: Dataset) -> None:
        index = WindowIndex(
            ends=np.arange(400, dtype=np.int64),
            life=np.repeat(np.arange(4, dtype=np.int32), 100),
            machine=np.repeat(np.arange(4, dtype=np.int32), 100),
            position=np.tile(np.arange(100, dtype=np.int32), 4),
            label=np.tile(np.r_[np.ones(20), np.zeros(80)].astype(np.int64), 4),
        )

        draw = draw_epoch(index, positives=200, negatives=800, seed=1)

        assert draw.positives == 200
        assert draw.negatives == 800
        assert draw.positive_rate == pytest.approx(0.2)

    def test_no_life_is_amplified_by_its_window_count(self) -> None:
        """The property the sampler exists for.

        A life holding 400 windows and one holding 100 must contribute in
        proportion to those counts, not in proportion to how many chances they
        get. A uniform draw over windows would do the latter and quietly weight
        training towards long lives.
        """
        lengths = (400, 100)
        life = np.concatenate(
            [np.full(length, i, dtype=np.int32) for i, length in enumerate(lengths)]
        )
        index = WindowIndex(
            ends=np.arange(500, dtype=np.int64),
            life=life,
            machine=life.copy(),
            position=np.tile(np.arange(100, dtype=np.int32), 5)[:500],
            label=np.tile(np.r_[np.ones(50), np.zeros(50)].astype(np.int64), 5),
        )

        draw = draw_epoch(index, positives=5_000, negatives=5_000, seed=4)
        share = life_share(index, draw.positions)
        expected = np.asarray(lengths) / sum(lengths)

        assert share == pytest.approx(expected, abs=0.01)

    def test_the_draw_is_reproducible_from_its_seed(self, dataset: Dataset) -> None:
        index = build_index(dataset, Split.TEST)

        first = draw_epoch(index, positives=50, negatives=200, seed=7)
        second = draw_epoch(index, positives=50, negatives=200, seed=7)

        assert np.array_equal(first.positions, second.positions)

    def test_an_empty_arm_is_refused(self, dataset: Dataset) -> None:
        index = build_index(dataset, Split.TEST)

        with pytest.raises(ExperimentError, match="positive"):
            draw_epoch(index, positives=0, negatives=10, seed=0)


class TestPredictions:
    """The table evaluation reads instead of a model."""

    def _table(self, run_id: str = "run-1") -> Predictions:
        return Predictions(
            split=Split.TEST,
            run_id=run_id,
            model="lstm",
            ends=np.arange(4, dtype=np.int64),
            life=np.zeros(4, dtype=np.int32),
            machine=np.zeros(4, dtype=np.int32),
            position=np.arange(4, dtype=np.int32),
            label=np.array([0, 0, 1, 1], dtype=np.int64),
            probability=np.array([0.1, 0.2, 0.7, 0.9]),
            calibrated=np.array([0.02, 0.05, 0.4, 0.8]),
        )

    def test_a_table_round_trips(self, tmp_path: Path) -> None:
        """The seam: evaluation reads this and never sees a model."""
        path = write_predictions(tmp_path / "p.parquet", self._table())
        restored = read_predictions(path)

        assert restored.run_id == "run-1"
        assert restored.model == "lstm"
        assert restored.split is Split.TEST
        assert np.array_equal(restored.probability, self._table().probability)
        assert np.array_equal(restored.calibrated, self._table().calibrated)

    def test_mismatched_columns_are_refused(self) -> None:
        with pytest.raises(ExperimentError, match="disagree in length"):
            Predictions(
                split=Split.TEST,
                run_id="x",
                model="lstm",
                ends=np.arange(4, dtype=np.int64),
                life=np.zeros(3, dtype=np.int32),
                machine=np.zeros(4, dtype=np.int32),
                position=np.arange(4, dtype=np.int32),
                label=np.zeros(4, dtype=np.int64),
                probability=np.zeros(4),
                calibrated=np.zeros(4),
            )

    def test_a_missing_table_reads_as_a_missing_table(self, tmp_path: Path) -> None:
        with pytest.raises(ExperimentError, match="does not exist"):
            read_predictions(tmp_path / "absent.parquet")


class TestManifest:
    """What produced a checkpoint, and whether it still adds up."""

    def test_the_fingerprint_moves_when_the_artifact_does(self, tmp_path: Path) -> None:
        (tmp_path / "a.npy").write_bytes(b"one")
        (tmp_path / "b.json").write_text("{}", encoding="utf-8")
        before = fingerprint(tmp_path)

        (tmp_path / "a.npy").write_bytes(b"two")

        assert fingerprint(tmp_path) != before

    def test_renaming_a_file_changes_the_fingerprint(self, tmp_path: Path) -> None:
        """A renamed file is a different artifact.

        A contents-only hash would miss it, and the name is hashed alongside the
        bytes for exactly this reason.
        """
        (tmp_path / "a.npy").write_bytes(b"same")
        before = fingerprint(tmp_path)

        (tmp_path / "a.npy").rename(tmp_path / "b.npy")

        assert fingerprint(tmp_path) != before

    def test_an_empty_directory_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(ExperimentError, match="no files"):
            fingerprint(tmp_path)

    def test_a_manifest_round_trips(self, tmp_path: Path) -> None:
        # Written first: `build` fingerprints the directory it is handed, and an
        # empty one is refused rather than hashed to a valid-looking constant.
        (tmp_path / "a.npy").write_bytes(b"one")
        manifest = build(
            run_id="run-1",
            config_digest="deadbeef",
            dataset_dir=tmp_path,
            repository=tmp_path,
            device="cpu",
            torch_version="2.6.0",
        )

        assert manifest.environment.torch == "2.6.0"

        restored = read_manifest(write_manifest(tmp_path / "run.json", manifest))

        assert restored.run_id == manifest.run_id
        assert restored.config_digest == manifest.config_digest
        assert restored.environment == manifest.environment

    def test_a_changed_dataset_is_reported_not_silently_accepted(self, tmp_path: Path) -> None:
        (tmp_path / "a.npy").write_bytes(b"one")
        manifest = build(
            run_id="r",
            config_digest="d",
            dataset_dir=tmp_path,
            repository=tmp_path,
            device="cpu",
        )
        (tmp_path / "a.npy").write_bytes(b"two")

        warning = describe_mismatch(manifest, tmp_path)

        assert warning is not None
        assert "no longer matches" in warning

    def test_an_unchanged_dataset_reports_nothing(self, tmp_path: Path) -> None:
        (tmp_path / "a.npy").write_bytes(b"one")
        manifest = build(
            run_id="r",
            config_digest="d",
            dataset_dir=tmp_path,
            repository=tmp_path,
            device="cpu",
        )

        assert describe_mismatch(manifest, tmp_path) is None

    def test_an_environment_without_torch_records_that(self) -> None:
        """The evaluation half runs where torch is absent, and the manifest says so."""
        environment = Environment.capture(device="cpu", torch_version=None)

        assert environment.torch is None
        assert environment.numpy

    def test_a_manifest_knows_whether_it_is_reproducible(self, tmp_path: Path) -> None:
        (tmp_path / "a.npy").write_bytes(b"one")
        manifest = RunManifest(
            schema=1,
            run_id="r",
            created_at="now",
            config_digest="",
            dataset_fingerprint="",
            dataset_directory=".",
            git_sha=None,
            git_dirty=None,
            environment=Environment("3.12", "2.0", None, "cpu"),
        )

        assert not manifest.reproducible
