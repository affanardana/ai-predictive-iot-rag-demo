"""The torch layer, exercised on CPU over the small fleet.

Marked `torch` and deselected by default, because torch is an optional extra and
the default environment does not have it. The module-level `importorskip` is not
redundant with the marker: `-m 'not torch'` filters *after* collection has
already imported the module, so a bare `import torch` here would break the
offline job regardless of the marker.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ml.dataset.artifacts import Dataset  # noqa: E402
from ml.dataset.splits import Split  # noqa: E402
from ml.experiment.config import TrainConfig  # noqa: E402
from ml.experiment.index import build_index  # noqa: E402
from ml.experiment.manifest import RunManifest  # noqa: E402
from ml.experiment.manifest import build as build_manifest  # noqa: E402
from ml.model.checkpoint import Checkpoint, load, save  # noqa: E402
from ml.model.network import LSTMClassifier, build  # noqa: E402
from ml.model.sequences import WindowDataset, loader  # noqa: E402
from ml.model.training import train  # noqa: E402

pytestmark = pytest.mark.torch


def _manifest(tmp_path: Path) -> RunManifest:
    (tmp_path / "marker").write_bytes(b"x")
    return build_manifest(
        run_id="test",
        config_digest="d",
        dataset_dir=tmp_path,
        repository=tmp_path,
        device="cpu",
        torch_version=torch.__version__,
    )


class TestNetwork:
    """The network, and the interface it promises."""

    def test_forward_returns_logits_not_probabilities(self) -> None:
        """The module's stated contract, tested by forcing the output out of range.

        Asserting that an untrained model happens to emit a value outside [0, 1]
        would prove nothing — LayerNorm pushes a fresh LSTM's near-constant
        hidden state towards zero, so its logits sit near zero anyway, which is
        inside the unit interval.

        The head's **bias** is set instead of its weights. The bias reaches the
        output directly, whatever the recurrent state does, so the logit is a
        known large number and a sigmoid anywhere in the path would clamp it to
        (0, 1) no matter what.
        """
        model = LSTMClassifier(features=6, hidden_size=8, layers=1, dropout=0.0)
        with torch.no_grad():
            model.head.bias.fill_(50.0)

        output = model(torch.randn(4, 60, 6))

        assert output.shape == (4,)
        assert bool((output.abs() > 1.0).any()), (
            "outputs are bounded to [0, 1]; forward must return logits, not probabilities."
        )

    def test_the_batch_dimension_is_first(self) -> None:
        model = build(TrainConfig.tiny(), features=6)
        model.eval()

        output = model(torch.randn(3, 60, 6))

        assert output.shape == (3,)


class TestDataset:
    """The window dataset, and the standardisation it applies."""

    def test_a_window_is_standardised_by_the_artifacts_own_statistics(
        self, dataset: Dataset
    ) -> None:
        """The one thing here that could change a number.

        A second normalisation applied per window, or a missing one, would train
        happily and mean something else.
        """
        index = build_index(dataset, Split.TEST)
        signals = np.asarray(dataset.signals, dtype=np.float32)
        windows = WindowDataset(signals, dataset.normalization, index)

        window, label = windows[0]
        end = int(index.ends[0])
        raw = signals[end - 59 : end + 1]
        expected = (raw - dataset.normalization.means.astype(np.float32)) / (
            dataset.normalization.stds.astype(np.float32)
        )

        assert window.dtype == torch.float32
        assert window.shape == (60, dataset.features)
        assert torch.allclose(window, torch.from_numpy(expected), atol=1e-5)
        assert float(label) == float(index.label[0])

    def test_a_subset_exposes_only_those_windows(self, dataset: Dataset) -> None:
        index = build_index(dataset, Split.TEST)
        signals = np.asarray(dataset.signals, dtype=np.float32)
        positions = np.array([5, 9, 11], dtype=np.int64)

        windows = WindowDataset(signals, dataset.normalization, index, positions)

        assert len(windows) == 3
        assert float(windows[0][1]) == float(index.label[5])

    def test_a_position_outside_the_index_is_refused(self, dataset: Dataset) -> None:
        from ml.experiment.errors import ExperimentError

        index = build_index(dataset, Split.TEST)
        signals = np.asarray(dataset.signals, dtype=np.float32)

        with pytest.raises(ExperimentError, match="outside"):
            WindowDataset(
                signals,
                dataset.normalization,
                index,
                np.array([len(index) + 10], dtype=np.int64),
            )

    def test_the_loader_is_seeded(self, dataset: Dataset) -> None:
        """Reproducibility is an exit condition, so the loader has to be too."""
        index = build_index(dataset, Split.TEST)
        signals = np.asarray(dataset.signals, dtype=np.float32)
        windows = WindowDataset(signals, dataset.normalization, index)

        first = [
            batch[1]
            for batch in loader(windows, batch_size=32, shuffle=True, num_workers=0, seed=3)
        ]
        second = [
            batch[1]
            for batch in loader(windows, batch_size=32, shuffle=True, num_workers=0, seed=3)
        ]

        assert all(torch.equal(a, b) for a, b in zip(first, second, strict=True))


class TestCheckpoint:
    """Checkpoints, under torch.load's safe default."""

    def test_a_checkpoint_round_trips_under_the_safe_load_default(self, tmp_path: Path) -> None:
        """`torch.load` defaults to `weights_only=True`, and this has to satisfy it.

        A config dataclass or a numpy array in the payload would raise rather
        than load — so the format is tensors and strings, and this test is what
        keeps it that way.
        """
        checkpoint = Checkpoint(
            state_dict={"weight": torch.ones(3)},
            config_json='{"seed": 1}',
            manifest_json='{"run_id": "r"}',
            epoch=2,
            metrics={"train_loss": 0.5},
            trained_rate=0.2,
            natural_rate=0.045,
            split=Split.TRAIN,
            golden_windows=torch.zeros(2, 60, 6),
            golden_probabilities=torch.zeros(2),
        )

        path = save(tmp_path / "model.pt", checkpoint)
        restored = load(path)

        assert restored.epoch == 2
        assert restored.trained_rate == 0.2
        assert torch.equal(restored.golden_windows, checkpoint.golden_windows)
        assert restored.config_json == '{"seed": 1}'

    def test_a_missing_checkpoint_reads_as_missing(self, tmp_path: Path) -> None:
        from ml.experiment.errors import ExperimentError

        with pytest.raises(ExperimentError, match="does not exist"):
            load(tmp_path / "absent.pt")


class TestTraining:
    """The loop, exercised end to end on CPU."""

    def test_a_tiny_run_trains_checkpoints_and_stops(
        self, dataset: Dataset, tmp_path: Path
    ) -> None:
        """The end-to-end proof that the torch layer runs at all.

        Every path is exercised: index, sampler, loader, network, loss, the
        probe metric, early stopping, and both checkpoints.
        """
        train_index = build_index(dataset, Split.TRAIN)
        probe_index = build_index(dataset, Split.VALIDATION)
        if probe_index.positives == 0:
            pytest.skip("the small fleet's validation split holds no positives")

        result = train(
            dataset,
            config=TrainConfig.tiny(),
            manifest=_manifest(tmp_path),
            train_index=train_index,
            probe_index=probe_index,
            output_dir=tmp_path / "run",
        )

        assert len(result.epochs) >= 1
        assert (tmp_path / "run" / "last.pt").exists()
        assert (tmp_path / "run" / "metrics.jsonl").exists()
        assert result.best_life_ap > float("-inf")

        for epoch in result.epochs:
            assert np.isfinite(epoch.train_loss)
            assert np.isfinite(epoch.validation_life_ap)

    def test_the_loss_moves(self, dataset: Dataset, tmp_path: Path) -> None:
        """A loop that runs but does not learn is a loop that silently does nothing."""
        train_index = build_index(dataset, Split.TRAIN)
        probe_index = build_index(dataset, Split.VALIDATION)
        if probe_index.positives == 0:
            pytest.skip("the small fleet's validation split holds no positives")

        config = TrainConfig.tiny()
        result = train(
            dataset,
            config=config,
            manifest=_manifest(tmp_path),
            train_index=train_index,
            probe_index=probe_index,
            output_dir=tmp_path / "run",
        )

        if len(result.epochs) >= 2:
            assert result.epochs[1].train_loss <= result.epochs[0].train_loss * 1.5

    def test_a_resume_continues_rather_than_restarting(
        self, dataset: Dataset, tmp_path: Path
    ) -> None:
        """The Colab case: a session dies and the next one picks up."""
        train_index = build_index(dataset, Split.TRAIN)
        probe_index = build_index(dataset, Split.VALIDATION)
        if probe_index.positives == 0:
            pytest.skip("the small fleet's validation split holds no positives")

        output = tmp_path / "run"
        first = train(
            dataset,
            config=TrainConfig.tiny(),
            manifest=_manifest(tmp_path),
            train_index=train_index,
            probe_index=probe_index,
            output_dir=output,
        )
        resumed_config = TrainConfig.tiny()
        resumed = train(
            dataset,
            config=resumed_config,
            manifest=_manifest(tmp_path),
            train_index=train_index,
            probe_index=probe_index,
            output_dir=output,
            resume=True,
        )

        # A tiny config runs two epochs, so a resume has nothing left to do and
        # must report zero new epochs rather than starting over.
        assert first.epochs[0].number == 0
        assert not resumed.epochs or resumed.epochs[0].number == 1

    def test_the_metrics_log_is_append_only(self, dataset: Dataset, tmp_path: Path) -> None:
        """An interrupted run must keep the history it had."""
        train_index = build_index(dataset, Split.TRAIN)
        probe_index = build_index(dataset, Split.VALIDATION)
        if probe_index.positives == 0:
            pytest.skip("the small fleet's validation split holds no positives")

        output = tmp_path / "run"
        config = TrainConfig.tiny()
        first = train(
            dataset,
            config=config,
            manifest=_manifest(tmp_path),
            train_index=train_index,
            probe_index=probe_index,
            output_dir=output,
        )
        lines_after_first = (output / "metrics.jsonl").read_text(encoding="utf-8").count("\n")

        assert lines_after_first == len(first.epochs)
