"""Loading a checkpoint once, and scoring windows.

The model is loaded at startup and held, not reloaded per request. It is 1.5 MB,
so this is about latency rather than memory: reading it from disk on every call
would dominate the cost of the forward pass.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch

from inference.settings import Settings
from ml.dataset.artifacts import read_normalization
from ml.dataset.features import FEATURE_COLUMNS
from ml.dataset.windows import WINDOW_MINUTES
from ml.evaluation.prior import PriorShift
from ml.experiment.config import TrainConfig
from ml.model.checkpoint import load
from ml.model.network import build


class InsufficientHistoryError(ValueError):
    """Raised when a request carries too few readings to form a window."""


@dataclass(frozen=True, slots=True)
class Prediction:
    """What the service returns, and nothing more."""

    failure_probability: float
    model_version: str


class Scorable(Protocol):
    """What the app needs from a scorer, so tests can supply their own.

    The concrete `Scorer` loads a checkpoint and runs a network; the tests want
    neither. Depending on the shape rather than the class is what lets the HTTP
    contract be tested without a model on disk.
    """

    @property
    def model_version(self) -> str:
        """Return the version of the loaded model."""
        ...

    @property
    def window(self) -> int:
        """Return how many readings a window holds."""
        ...

    def score(self, readings: Sequence[Sequence[float]]) -> Prediction:
        """Score one window of readings."""
        ...


class Scorer:
    """A loaded model, ready to score 60-reading windows."""

    def __init__(self, settings: Settings) -> None:
        checkpoint = load(settings.checkpoint)
        config = TrainConfig.from_json(checkpoint.config_json)
        normalization = read_normalization(settings.artifact_dir)

        model = build(config, features=len(FEATURE_COLUMNS))
        model.load_state_dict(checkpoint.state_dict)
        model.eval()

        self._model = model
        self._mean = normalization.means.astype(np.float32)
        self._std = normalization.stds.astype(np.float32)
        self._run_id = checkpoint.run_id

        # The model was trained at the sampler's prevalence, not the real one,
        # so its raw output is calibrated to a world where 20% of windows
        # precede a failure. The product compares the probability against bands
        # meant for the real 4.4%, so the correction has to happen before it
        # leaves this service. Without it a CRITICAL band edge of 0.80 would
        # mean nothing.
        self._shift = PriorShift(
            trained_rate=checkpoint.trained_rate,
            natural_rate=checkpoint.natural_rate,
        )

    @property
    def model_version(self) -> str:
        """Return the run id of the loaded checkpoint."""
        return self._run_id

    @property
    def window(self) -> int:
        """Return how many readings a window holds."""
        return WINDOW_MINUTES

    def score(self, readings: Sequence[Sequence[float]]) -> Prediction:
        """Score one window of readings.

        `readings` are ordered oldest first, one row of six signals per minute,
        in `FEATURE_COLUMNS` order.

        Raises:
            InsufficientHistoryError: if there are fewer than `window` readings.
            ValueError: if a reading does not carry six signals.
        """
        if len(readings) < WINDOW_MINUTES:
            raise InsufficientHistoryError(
                f"A prediction needs {WINDOW_MINUTES} minutes of history and "
                f"{len(readings)} arrived."
            )

        window = np.asarray(readings[-WINDOW_MINUTES:], dtype=np.float32)
        if window.ndim != 2 or window.shape[1] != len(FEATURE_COLUMNS):
            raise ValueError(
                f"Each reading must carry {len(FEATURE_COLUMNS)} signals, got shape {window.shape}."
            )

        standardised = (window - self._mean) / self._std
        tensor = torch.from_numpy(np.ascontiguousarray(standardised))[None, ...]
        with torch.no_grad():
            logit = self._model(tensor)
            raw = float(torch.sigmoid(logit).item())

        corrected = float(self._shift.apply(np.asarray([raw], dtype=np.float64))[0])
        return Prediction(
            failure_probability=min(max(corrected, 0.0), 1.0),
            model_version=self._run_id,
        )
