"""The configuration of a training run, and its identity.

Every hyperparameter lives in one frozen dataclass, which is serialised
canonically and hashed. The digest is what the manifest records, so "which
settings produced this checkpoint" is answered by comparing a hash rather than
by reading prose, and a run that cannot be reproduced from its own manifest is
detectable rather than merely disappointing.

`window` and `horizon` are not tunable. `PRD.md` §25.8 and §25.9 fix both at
sixty minutes, so they are fields only so that a mismatch against the artifact's
own metadata can be *checked* — a dataset built for a different horizon would
otherwise train happily and mean something else.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from ml.dataset.labelling import HORIZON_MINUTES
from ml.dataset.windows import WINDOW_MINUTES
from ml.experiment.errors import ExperimentError

#: The share of positives the sampler aims for in a training epoch. Twenty
#: percent is high enough that the positive class carries weight in the loss and
#: low enough that the gradient still sees the shape of normal operation.
DEFAULT_TARGET_POSITIVE_RATE = 0.20

CONFIG_FILENAME = "config.json"


@dataclass(frozen=True, slots=True)
class TrainConfig:
    """Everything that determines a training run."""

    seed: int = 20260923
    window: int = WINDOW_MINUTES
    horizon: int = HORIZON_MINUTES
    hidden_size: int = 128
    layers: int = 2
    dropout: float = 0.20
    batch_size: int = 512
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    gradient_clip: float = 1.0
    max_epochs: int = 60
    patience: int = 8
    target_positive_rate: float = DEFAULT_TARGET_POSITIVE_RATE
    positive_repeats: float = 3.0
    validation_probe_per_life: int = 60
    num_workers: int = 2
    device: str = "cuda"

    def __post_init__(self) -> None:
        """Reject a configuration that cannot produce a meaningful run."""
        if self.window != WINDOW_MINUTES:
            raise ExperimentError(
                f"window must be {WINDOW_MINUTES} (PRD section 25.9), got {self.window}."
            )
        if self.horizon != HORIZON_MINUTES:
            raise ExperimentError(
                f"horizon must be {HORIZON_MINUTES} (PRD section 25.8), got {self.horizon}."
            )
        if not 0.0 < self.target_positive_rate < 1.0:
            raise ExperimentError(
                "target_positive_rate must be strictly between 0 and 1, got "
                f"{self.target_positive_rate}."
            )
        for name in ("hidden_size", "layers", "batch_size", "max_epochs", "patience"):
            if getattr(self, name) < 1:
                raise ExperimentError(f"{name} must be at least 1, got {getattr(self, name)}.")
        if self.positive_repeats <= 0.0:
            raise ExperimentError(
                f"positive_repeats must be positive, got {self.positive_repeats}."
            )
        if not 0.0 <= self.dropout < 1.0:
            raise ExperimentError(f"dropout must be in [0, 1), got {self.dropout}.")
        if self.learning_rate <= 0.0:
            raise ExperimentError(f"learning_rate must be positive, got {self.learning_rate}.")
        if self.num_workers < 0:
            raise ExperimentError(f"num_workers cannot be negative, got {self.num_workers}.")

    def epoch_size(self, available_positives: int) -> tuple[int, int]:
        """Return how many positives and negatives an epoch should draw.

        The class counts are derived from the target rate rather than from the
        split's own prevalence, so the trained prior is a number the config
        states rather than one the data happens to imply — which is what makes
        the correction in `ml.evaluation.prior` exact instead of estimated.
        """
        positives = max(1, round(self.positive_repeats * available_positives))
        negatives = round(positives * (1.0 - self.target_positive_rate) / self.target_positive_rate)
        return positives, max(1, negatives)

    def to_json(self) -> str:
        """Return a canonical serialisation.

        Sorted keys and no whitespace, because the digest is over this string:
        two configurations that differ only in field order must hash the same,
        and two that differ in value must not.
        """
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))

    @property
    def digest(self) -> str:
        """Return a stable fingerprint of these settings."""
        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()

    @classmethod
    def from_json(cls, text: str) -> TrainConfig:
        """Build a configuration from a serialised one.

        Fields the payload omits take their defaults, so adding a hyperparameter
        with a sensible default does not invalidate every config file already
        written.

        Raises:
            ExperimentError: if the payload names a field this build does not
                have. Silently ignoring it would mean running something other
                than what the file says.
        """
        payload = json.loads(text)
        known = {field.name for field in fields(cls)}
        unknown = sorted(set(payload) - known)
        if unknown:
            raise ExperimentError(
                f"Unknown configuration fields: {unknown}. This build has {sorted(known)}."
            )
        return cls(**payload)

    @classmethod
    def from_file(cls, path: Path) -> TrainConfig:
        """Read a configuration from a JSON file."""
        return cls.from_json(path.read_text(encoding="utf-8"))

    def write(self, path: Path) -> Path:
        """Write the configuration beside a checkpoint."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return path

    @classmethod
    def tiny(cls) -> TrainConfig:
        """Return the smoke-test configuration.

        Small enough to run in seconds on a CPU, and its whole purpose is to
        exercise every code path — index, sampler, loader, network, loss, early
        stopping, checkpoint, resume — before a GPU session is spent on a shape
        mismatch.
        """
        return cls(
            hidden_size=8,
            layers=1,
            dropout=0.0,
            batch_size=64,
            max_epochs=2,
            patience=2,
            positive_repeats=1.0,
            validation_probe_per_life=20,
            num_workers=0,
            device="cpu",
        )
