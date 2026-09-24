"""Training, predicting and verifying, driven from the command line.

Imported lazily by the CLI so `ml dataset ...` keeps working without torch.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch

from ml.dataset.artifacts import Dataset, read_artifact
from ml.dataset.splits import Split
from ml.dataset.windows import WINDOW_MINUTES
from ml.evaluation.prior import PriorShift
from ml.experiment.config import TrainConfig
from ml.experiment.index import WindowIndex, build_index
from ml.experiment.manifest import build as build_manifest
from ml.experiment.predictions import Predictions, write_predictions
from ml.model.checkpoint import Checkpoint, load
from ml.model.network import build
from ml.model.sequences import WindowDataset, loader
from ml.model.training import TrainingResult, train

PREDICTIONS_TEMPLATE = "predictions_{split}.parquet"


@dataclass(frozen=True, slots=True)
class Verification:
    """Result of re-scoring a checkpoint's golden batch."""

    max_deviation: float
    rows: int
    passed: bool


def train_run(
    *,
    config_path: Path,
    artifact_dir: Path,
    output_dir: Path,
    repository: Path,
    resume: bool = False,
    device: str | None = None,
) -> TrainingResult:
    """Train from a config file and write checkpoints into `output_dir`."""
    config = TrainConfig.from_file(config_path)
    if device is not None:
        config = replace(config, device=device)
    dataset = read_artifact(artifact_dir)

    manifest = build_manifest(
        run_id=output_dir.name,
        config_digest=config.digest,
        dataset_dir=artifact_dir,
        repository=repository,
        device=config.device,
        torch_version=torch.__version__,
    )
    return train(
        dataset,
        config=config,
        manifest=manifest,
        train_index=build_index(dataset, Split.TRAIN),
        probe_index=_probe_index(dataset, config),
        output_dir=output_dir,
        resume=resume,
    )


def _probe_index(dataset: Dataset, config: TrainConfig) -> WindowIndex:
    """Validation windows, thinned to a fixed per-life subsample.

    Fixed so the metric is comparable between epochs, thinned so a long life
    cannot dominate it.
    """
    index = build_index(dataset, Split.VALIDATION)
    per_life = config.validation_probe_per_life
    if per_life >= WINDOW_MINUTES:
        return index

    keep: list[int] = []
    for life in np.unique(index.life):
        rows = np.flatnonzero(index.life == life)
        chosen = rows[
            np.linspace(0, rows.shape[0] - 1, min(per_life, rows.shape[0])).round().astype(int)
        ]
        keep.extend(int(value) for value in chosen)
    return index.select(np.asarray(sorted(keep), dtype=np.int64))


def predict_run(
    *,
    checkpoint_path: Path,
    artifact_dir: Path,
    split: Split,
    output_path: Path | None = None,
) -> Path:
    """Score a split and write a prediction table."""
    checkpoint = load(checkpoint_path)
    config = TrainConfig.from_json(checkpoint.config_json)
    dataset = read_artifact(artifact_dir)
    index = build_index(dataset, split)

    probabilities = _score(checkpoint, dataset, index, config)
    shift = PriorShift(trained_rate=checkpoint.trained_rate, natural_rate=checkpoint.natural_rate)

    predictions = Predictions(
        split=split,
        run_id=_run_id(checkpoint, checkpoint_path),
        model="lstm",
        ends=index.ends,
        life=index.life,
        machine=index.machine,
        position=index.position,
        label=index.label,
        probability=probabilities,
        calibrated=shift.apply(probabilities),
    )
    target = output_path or checkpoint_path.parent / PREDICTIONS_TEMPLATE.format(split=split.value)
    return write_predictions(target, predictions)


def _score(
    checkpoint: Checkpoint,
    dataset: Dataset,
    index: WindowIndex,
    config: TrainConfig,
) -> np.ndarray:
    """Return the model's probability for every window in an index."""
    model = build(config, features=dataset.features)
    model.load_state_dict(checkpoint.state_dict)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    windows = WindowDataset(
        np.asarray(dataset.signals, dtype=np.float32), dataset.normalization, index
    )
    data_loader = loader(
        windows, batch_size=config.batch_size, shuffle=False, num_workers=0, seed=config.seed
    )
    collected: list[np.ndarray] = []
    with torch.no_grad():
        for batch, _ in data_loader:
            collected.append(torch.sigmoid(model(batch.to(device))).cpu().numpy())
    return np.concatenate(collected).astype(np.float64)


def verify_checkpoint(*, checkpoint_path: Path, tolerance: float = 1e-5) -> Verification:
    """Re-score a checkpoint's stored golden batch and compare.

    The exit condition made mechanical: a versioned model reproduces its own
    predictions. Seconds, CPU, no retraining.
    """
    checkpoint = load(checkpoint_path)
    config = TrainConfig.from_json(checkpoint.config_json)

    model = build(config, features=checkpoint.golden_windows.shape[-1])
    model.load_state_dict(checkpoint.state_dict)
    model.eval()
    with torch.no_grad():
        probabilities = torch.sigmoid(model(checkpoint.golden_windows))

    deviation = float((probabilities - checkpoint.golden_probabilities).abs().max())
    return Verification(
        max_deviation=deviation,
        rows=int(checkpoint.golden_windows.shape[0]),
        passed=deviation <= tolerance,
    )


def _run_id(checkpoint: Checkpoint, path: Path) -> str:
    """Recover the run id from the checkpoint's manifest, or its directory."""
    try:
        return str(json.loads(checkpoint.manifest_json).get("run_id") or path.parent.name)
    except (ValueError, AttributeError):
        return path.parent.name
