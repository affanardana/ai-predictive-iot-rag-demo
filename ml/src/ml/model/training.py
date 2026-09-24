"""The training loop, and everything about it that has to be deliberate.

This is the largest and least testable module in the phase, which is the whole
argument for having pushed the index, the labels, the sampler and the metrics
out of it. What is left is a loop, and three decisions inside it that are easy
to get subtly wrong.

## Early stopping reads a group-averaged metric, not a per-window one

Sixty consecutive windows before a failure are near-duplicates, so a per-window
validation loss is dominated by whichever lives happen to be long. The probe
reduces each life to one number first, over a **fixed** subsample so the metric
is comparable across epochs and length-neutral across lives.

## The optimiser is restored last

The documented order is model, then optimiser, then scheduler, then load the
*scheduler's* state, then load the *optimiser's*. Doing it the other way round
lets the optimiser's own learning rates overwrite the restored ones, and the run
continues at the wrong rate with nothing to show for it.

## Reproduction is switched on together

Seeding, the cuDNN flags and `use_deterministic_algorithms` all have to be set;
setting some of them is how a run ends up reproducible on one machine and not
another. `warn_only=True` because some CUDA kernels have no deterministic
implementation, and a hard failure there would stop a run that is otherwise
fine — the flag is recorded in the manifest instead.

Mixed precision is deliberately not used: it breaks bitwise reproducibility and
this model does not need it.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from ml.dataset.artifacts import Dataset
from ml.dataset.labelling import HORIZON_MINUTES, NEVER_FAILS
from ml.dataset.splits import Split
from ml.dataset.windows import WINDOW_MINUTES
from ml.evaluation import metrics
from ml.evaluation.events import life_scores
from ml.experiment.config import TrainConfig
from ml.experiment.errors import ExperimentError
from ml.experiment.index import WindowIndex
from ml.experiment.manifest import RunManifest, write_manifest
from ml.experiment.sampling import draw_epoch
from ml.model.checkpoint import Checkpoint, load, save
from ml.model.network import build
from ml.model.sequences import WindowDataset, loader

LAST_CHECKPOINT = "last.pt"
BEST_CHECKPOINT = "best.pt"
METRICS_LOG = "metrics.jsonl"
MANIFEST_FILE = "manifest.json"
CONFIG_FILE = "config.json"


@dataclass(frozen=True, slots=True)
class Epoch:
    """One epoch's outcome."""

    number: int
    train_loss: float
    validation_loss: float
    validation_life_ap: float
    learning_rate: float


@dataclass(frozen=True, slots=True)
class TrainingResult:
    """What a training run produced."""

    epochs: tuple[Epoch, ...]
    best_epoch: int
    best_life_ap: float
    stopped_early: bool
    output_dir: Path


def seed_everything(seed: int, *, deterministic: bool = True) -> None:
    """Seed every generator that could affect a run.

    `torch.use_deterministic_algorithms` needs `CUBLAS_WORKSPACE_CONFIG` set
    before the first CUDA operation, which is why it happens here rather than
    lazily somewhere in the loop.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # warn_only: some CUDA kernels have no deterministic implementation, and
        # refusing to run at all would be worse than recording the caveat.
        torch.use_deterministic_algorithms(True, warn_only=True)


def train(
    dataset: Dataset,
    *,
    config: TrainConfig,
    manifest: RunManifest,
    train_index: WindowIndex,
    probe_index: WindowIndex,
    output_dir: Path,
    resume: bool = False,
) -> TrainingResult:
    """Train a model, checkpointing every epoch.

    Raises:
        ExperimentError: if the probe holds no positives, or a resume is asked
            for and there is nothing to resume from.
    """
    if probe_index.positives == 0:
        raise ExperimentError(
            "The validation probe holds no positive windows, so early stopping "
            "would be measuring noise."
        )

    seed_everything(config.seed)
    torch.manual_seed(config.seed)

    signals = np.asarray(dataset.signals, dtype=np.float32)
    # Only the probe is built once. A training Dataset is bound to a drawn epoch
    # and is rebuilt each pass, since the draw is what decides which windows it
    # exposes.
    probe_dataset = WindowDataset(signals, dataset.normalization, probe_index)

    model = build(config, features=dataset.features)
    device = _device(config)
    model.to(device)

    optimiser = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="max", factor=0.5, patience=max(1, config.patience // 2)
    )

    state = _resume_state(output_dir, model, optimiser, scheduler, config, resume, device)
    start_epoch = state

    # Plain BCE on logits. No `pos_weight`: the sampler has already fixed the
    # class ratio, and applying both double-counts the rebalancing, which
    # silently falsifies the prior correction in `ml.evaluation.prior`.
    loss_function = nn.BCEWithLogitsLoss()

    output_dir.mkdir(parents=True, exist_ok=True)
    config.write(output_dir / CONFIG_FILE)
    write_manifest(output_dir / MANIFEST_FILE, manifest)

    history: list[Epoch] = []
    best_life_ap = float("-inf")
    best_epoch = start_epoch
    since_improvement = 0

    for number in range(start_epoch, config.max_epochs):
        drawn = draw_epoch(
            train_index,
            positives=config.epoch_size(train_index.positives)[0],
            negatives=config.epoch_size(train_index.positives)[1],
            seed=config.seed + number,
        )
        train_loader = loader(
            WindowDataset(signals, dataset.normalization, train_index, drawn.positions),
            batch_size=config.batch_size,
            shuffle=False,  # the draw already fixed the order
            num_workers=config.num_workers,
            seed=config.seed + number,
        )
        train_loss = _train_one_epoch(model, train_loader, loss_function, optimiser, config, device)
        validation_loss = _validation_loss(model, probe_dataset, loss_function, config, device)
        life_ap = _probe_ap(model, probe_dataset, probe_index, config, device)

        scheduler.step(life_ap)
        learning_rate = float(optimiser.param_groups[0]["lr"])

        epoch = Epoch(
            number=number,
            train_loss=train_loss,
            validation_loss=validation_loss,
            validation_life_ap=life_ap,
            learning_rate=learning_rate,
        )
        history.append(epoch)
        _append_metrics(output_dir, epoch)

        if life_ap > best_life_ap:
            best_life_ap = life_ap
            best_epoch = number
            since_improvement = 0
            _write_checkpoint(
                output_dir / BEST_CHECKPOINT,
                model,
                config,
                manifest,
                epoch,
                dataset,
                device,
                resumable=False,
            )
        else:
            since_improvement += 1

        # Unconditional and every epoch. A run that spans Colab sessions is the
        # expected case here, not the exception, so the resumable state cannot
        # be written only when something improved.
        _write_checkpoint(
            output_dir / LAST_CHECKPOINT,
            model,
            config,
            manifest,
            epoch,
            dataset,
            device,
            resumable=True,
            optimiser=optimiser,
            scheduler=scheduler,
        )

        if since_improvement >= config.patience:
            break

    return TrainingResult(
        epochs=tuple(history),
        best_epoch=best_epoch,
        best_life_ap=best_life_ap,
        stopped_early=len(history) < config.max_epochs,
        output_dir=output_dir,
    )


def _device(config: TrainConfig) -> torch.device:
    """Return the device to train on, falling back to CPU when CUDA is absent."""
    if config.device.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(config.device)


def _train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    loss_function: nn.Module,
    optimiser: torch.optim.Optimizer,
    config: TrainConfig,
    device: torch.device,
) -> float:
    """Run one pass over a drawn epoch and return the mean loss."""
    model.train()
    total = 0.0
    batches = 0
    for windows, labels in data_loader:
        windows = windows.to(device)
        labels = labels.to(device)
        optimiser.zero_grad(set_to_none=True)
        loss = loss_function(model(windows), labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
        optimiser.step()
        total += float(loss.item())
        batches += 1
    return total / batches if batches else 0.0


@torch.no_grad()
def _validation_loss(
    model: nn.Module,
    probe: WindowDataset,
    loss_function: nn.Module,
    config: TrainConfig,
    device: torch.device,
) -> float:
    """Return the mean loss over the fixed probe set."""
    model.eval()
    data_loader = loader(
        probe,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        seed=config.seed,
    )
    total = 0.0
    batches = 0
    for windows, labels in data_loader:
        loss = loss_function(model(windows.to(device)), labels.to(device))
        total += float(loss.item())
        batches += 1
    return total / batches if batches else 0.0


@torch.no_grad()
def _probe_ap(
    model: nn.Module,
    probe: WindowDataset,
    probe_index: WindowIndex,
    config: TrainConfig,
    device: torch.device,
) -> float:
    """Return the life-level average precision on the probe set.

    The selection metric: reduced per life before it is scored, so sixty
    near-identical windows cannot stand in for sixty independent observations.
    """
    probabilities = _probabilities(model, probe, config, device)
    lives = life_scores(
        probe_index.life,
        probe_index.failed_lives(),
        probabilities,
    )
    if lives.positives == 0 or lives.positives == lives.life.shape[0]:
        return 0.0
    return metrics.average_precision(lives.label, lives.score)


@torch.no_grad()
def _probabilities(
    model: nn.Module,
    dataset: WindowDataset,
    config: TrainConfig,
    device: torch.device,
) -> np.ndarray:
    """Return the model's probability for every window in a dataset."""
    model.eval()
    data_loader = loader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        seed=config.seed,
    )
    collected: list[np.ndarray] = []
    for windows, _ in data_loader:
        logits = model(windows.to(device))
        collected.append(torch.sigmoid(logits).detach().cpu().numpy())
    return np.concatenate(collected) if collected else np.zeros(0)


def _write_checkpoint(
    path: Path,
    model: nn.Module,
    config: TrainConfig,
    manifest: RunManifest,
    epoch: Epoch,
    dataset: Dataset,
    device: torch.device,
    *,
    resumable: bool,
    optimiser: torch.optim.Optimizer | None = None,
    scheduler: object | None = None,
) -> None:
    """Write a checkpoint, with a golden batch for later verification.

    `resumable` marks whether the optimiser and scheduler state go in. They are
    needed to continue a run and are dead weight in `best.pt` — and, worse, a
    `best.pt` carrying them would invite someone to resume from a snapshot taken
    at a different point in the schedule.
    """
    golden = _golden_batch(dataset, config)
    save(
        path,
        Checkpoint(
            state_dict={key: value.detach().cpu() for key, value in model.state_dict().items()},
            config_json=config.to_json(),
            manifest_json=json.dumps(
                {
                    "run_id": manifest.run_id,
                    "dataset_fingerprint": manifest.dataset_fingerprint,
                },
                sort_keys=True,
            ),
            epoch=epoch.number,
            optimiser_state=optimiser.state_dict() if resumable and optimiser else None,
            scheduler_state=(
                scheduler.state_dict()  # type: ignore[attr-defined]
                if resumable and scheduler is not None
                else None
            ),
            metrics={
                "train_loss": epoch.train_loss,
                "validation_loss": epoch.validation_loss,
                "validation_life_ap": epoch.validation_life_ap,
            },
            # The sampled rate the model was trained at, and the natural rate it
            # must be corrected to. Both travel with the weights so the
            # correction can be reconstructed without the config.
            trained_rate=config.target_positive_rate,
            natural_rate=_natural_rate(dataset),
            split=Split.TRAIN,
            golden_windows=golden,
            golden_probabilities=_golden_probabilities(model, golden, device),
        ),
    )


def _natural_rate(dataset: Dataset) -> float:
    """Return the training split's own prevalence, for the prior correction.

    Measured over the same trainable rows Phase 3 reports, so the number the
    correction shifts towards is the number the report already published rather
    than a second, subtly different one.
    """
    minutes = dataset.minutes_to_onset
    trainable = (minutes == NEVER_FAILS) | (minutes > 0)
    positives = (minutes > 0) & (minutes <= HORIZON_MINUTES) & trainable
    total = int(trainable.sum())
    return float(positives.sum()) / total if total else 0.0


def _golden_batch(dataset: Dataset, config: TrainConfig) -> torch.Tensor:
    """Return a small fixed batch used to prove a checkpoint still works.

    Read from the first life of the artifact by row position, which is stable
    for a given dataset — the manifest's fingerprint is what detects a dataset
    that has moved underneath it.
    """
    count = min(config.batch_size, dataset.rows - WINDOW_MINUTES)
    if count < 1:
        raise ExperimentError("The artifact is too short to hold a golden batch.")

    ends = np.arange(WINDOW_MINUTES, WINDOW_MINUTES + count, dtype=np.int64)
    index = WindowIndex(
        ends=ends,
        life=np.zeros(count, dtype=np.int32),
        machine=np.zeros(count, dtype=np.int32),
        position=np.arange(WINDOW_MINUTES, WINDOW_MINUTES + count, dtype=np.int32),
        label=np.zeros(count, dtype=np.int64),
    )
    windows = WindowDataset(
        np.asarray(dataset.signals, dtype=np.float32), dataset.normalization, index
    )
    return torch.stack([windows[position][0] for position in range(count)])


@torch.no_grad()
def _golden_probabilities(
    model: nn.Module, golden: torch.Tensor, device: torch.device
) -> torch.Tensor:
    """Return the model's probabilities for the golden batch."""
    model.eval()
    return torch.sigmoid(model(golden.to(device))).detach().cpu()


def _resume_state(
    output_dir: Path,
    model: nn.Module,
    optimiser: torch.optim.Optimizer,
    scheduler: object,
    config: TrainConfig,
    resume: bool,
    device: torch.device,
) -> int:
    """Restore a checkpoint if asked, and return the epoch to start from.

    The scheduler is restored *before* the optimiser, per PyTorch's documented
    ordering — loading the optimiser first lets its own learning rates overwrite
    the restored ones, and the run silently continues at the wrong rate.
    """
    del config
    path = output_dir / LAST_CHECKPOINT
    if not resume or not path.exists():
        return 0

    checkpoint = load(path, map_location=str(device))
    model.load_state_dict(checkpoint.state_dict)

    # Scheduler first, then optimiser — PyTorch's documented order. Loading the
    # optimiser first lets its stored learning rates overwrite the restored
    # ones, and the run continues at the wrong rate with nothing to show for it.
    if checkpoint.scheduler_state is not None and scheduler is not None:
        scheduler.load_state_dict(checkpoint.scheduler_state)  # type: ignore[attr-defined]
    if checkpoint.optimiser_state is not None:
        optimiser.load_state_dict(checkpoint.optimiser_state)

    return checkpoint.epoch + 1


def _append_metrics(output_dir: Path, epoch: Epoch) -> None:
    """Append one line to the metrics log.

    Append-only, so an interrupted run keeps the history it had rather than
    losing it to the next write.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / METRICS_LOG).open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "epoch": epoch.number,
                    "train_loss": epoch.train_loss,
                    "validation_loss": epoch.validation_loss,
                    "validation_life_ap": epoch.validation_life_ap,
                    "learning_rate": epoch.learning_rate,
                },
                sort_keys=True,
            )
            + "\n"
        )
