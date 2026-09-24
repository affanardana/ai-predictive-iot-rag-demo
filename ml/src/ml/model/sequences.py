"""A `Dataset` that slices windows out of the flat artifact.

This is the entire torch-side of sequence generation, and it is thin on purpose.
It does not decide which windows exist, what they are labelled, or when a life
begins — `ml.experiment.index` decided all of that, torch-free, and this reads
its arrays. The only thing here that could change a number is the standardisation
and the dtype, and a test pins both.

## Why the signals are held in RAM

`read_artifact` memory-maps them, which is right for Phase 3's analysis and
wrong here. On Colab the artifact arrives on a Drive mount, and a FUSE-backed
memmap turns each window slice into strided reads over the network — millions of
them per epoch. It is slow rather than broken, which is the kind of failure that
costs an afternoon. The whole array is 58 MB, so the notebook copies it local
and this holds it in memory.
"""

from __future__ import annotations

import random

import numpy as np
import numpy.typing as npt
import torch
from torch.utils.data import DataLoader, Dataset

from ml.dataset.artifacts import Normalization
from ml.dataset.windows import WINDOW_MINUTES
from ml.experiment.errors import ExperimentError
from ml.experiment.index import WindowIndex

Float32Array = npt.NDArray[np.float32]


class WindowDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """One `(sequence, label)` pair per window in the index."""

    def __init__(
        self,
        signals: Float32Array,
        normalization: Normalization,
        index: WindowIndex,
        positions: npt.NDArray[np.int64] | None = None,
    ) -> None:
        """Bind the artifact to a set of windows.

        Args:
            signals: the whole flat signal array, in RAM, as `float32`. Raw, not
                standardised — the statistics are applied per window here.
            normalization: the artifact's own training statistics.
            index: which windows exist and what they are labelled.
            positions: an optional subset of the index to expose, used to
                present a sampled epoch rather than everything.

        Raises:
            ExperimentError: if the window width disagrees with the signals, or
                a position falls outside the index.
        """
        self._signals = signals
        self._index = index
        self._positions = np.arange(len(index), dtype=np.int64) if positions is None else positions
        if self._positions.size and (
            int(self._positions.min()) < 0 or int(self._positions.max()) >= len(index)
        ):
            raise ExperimentError("A drawn position falls outside the window index.")

        # Precomputed once as `float32`. `Normalization.apply` is declared on
        # `float64`, and calling it per window would upcast every sequence and
        # drag 64-bit arithmetic through the hottest loop in training.
        self._mean = normalization.means.astype(np.float32)
        self._std = normalization.stds.astype(np.float32)
        if np.any(self._std == 0.0):
            raise ExperimentError("A normalisation standard deviation is zero.")

    def __len__(self) -> int:
        """Return how many windows this dataset exposes."""
        return int(self._positions.shape[0])

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the standardised window and its label.

        The slice is built from `window_slice`, which is the function that
        already guarantees a window never crosses a life boundary and never
        includes a post-onset row. Nothing here re-derives that.
        """
        position = int(self._positions[item])
        end = int(self._index.ends[position])
        window = self._signals[end - WINDOW_MINUTES + 1 : end + 1]
        standardised = (window - self._mean) / self._std
        return (
            torch.from_numpy(np.ascontiguousarray(standardised, dtype=np.float32)),
            torch.tensor(float(self._index.label[position]), dtype=torch.float32),
        )


def seed_worker(worker_id: int) -> None:
    """Reseed a loader worker from torch's own seed.

    Without this, worker processes inherit divergent RNG state and "same seed,
    same result" quietly stops being true — which is the sort of failure that
    shows up as an unreproducible number weeks later rather than as an error.
    """
    del worker_id
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def loader(
    dataset: WindowDataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
    """Build a `DataLoader` with reproducibility switched on.

    The generator and the worker init function go together: one seeds the parent
    and the other propagates it, and setting only one is a common way to be
    reproducible on a single-worker run and not on a parallel one.
    """
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=generator,
        drop_last=False,
    )


def standardised_signals(
    signals: npt.NDArray[np.floating], normalization: Normalization
) -> Float32Array:
    """Return the whole signal array standardised, as `float32`.

    Used by prediction, where every window of a split is scored and the
    per-window transform would be repeated a few hundred thousand times for no
    reason. Training uses the per-window path so that the statistics cannot
    drift away from the ones stored in the artifact.
    """
    scaled = (signals.astype(np.float32) - normalization.means.astype(np.float32)) / (
        normalization.stds.astype(np.float32)
    )
    return np.ascontiguousarray(scaled, dtype=np.float32)
