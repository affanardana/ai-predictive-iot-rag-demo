"""The LSTM.

`PRD.md` §25.7 fixes the family — an LSTM, not a sequence model of the
implementer's choosing — and `MASTERPLAN.md` puts multivariate transformers
explicitly out of scope. So this is one standard configuration, chosen once. It
is not the first step of a search: the owner's instruction is that model
accuracy is not the point, and `AI_AGENT_GUIDE.md` ranks it last of seven
priorities.

## The interface contract

`forward` returns **logits, never probabilities**. That is stated here rather
than left to convention because it forces the two transforms that matter — the
sigmoid that turns a logit into a probability, and the prior shift that
re-calibrates it to the real failure rate — into two named places instead of
scattering sigmoids through the training loop, the prediction path and the
evaluation. `BCEWithLogitsLoss` is also the numerically stable form, and a
hand-applied sigmoid before it is a classic way to silently lose precision on
confident examples.
"""

from __future__ import annotations

import torch
from torch import nn

from ml.experiment.config import TrainConfig


class LSTMClassifier(nn.Module):
    """A recurrent classifier over a fixed-length window of signals."""

    def __init__(
        self,
        *,
        features: int,
        hidden_size: int,
        layers: int,
        dropout: float,
    ) -> None:
        """Build the network.

        Args:
            features: how many signals a timestep carries — six in this project.
            hidden_size: the recurrent state width.
            layers: how many stacked LSTM layers.
            dropout: applied to the final hidden state, and between LSTM layers
                when there is more than one.
        """
        super().__init__()
        self.features = features
        self.hidden_size = hidden_size
        self.layers = layers

        self.lstm = nn.LSTM(
            input_size=features,
            hidden_size=hidden_size,
            num_layers=layers,
            batch_first=True,
            # `nn.LSTM` only applies this *between* layers, so with one layer it
            # is inert rather than wrong.
            dropout=dropout if layers > 1 else 0.0,
        )
        # Normalising the pooled state before the head keeps the single linear
        # layer from having to cope with the recurrent state's scale drifting
        # across a long training run.
        self.norm = nn.LayerNorm(hidden_size)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, sequences: torch.Tensor) -> torch.Tensor:
        """Return one logit per sequence.

        Args:
            sequences: `(batch, window, features)`, already standardised.

        Returns:
            `(batch,)` logits. Not probabilities — see the module docstring.
        """
        output, _ = self.lstm(sequences)
        # The final timestep is the one that has seen the whole window.
        pooled = output[:, -1, :]
        # Bound to a local so the squeeze is checked against the return type
        # rather than inferred as `Any` at the boundary.
        logits: torch.Tensor = self.head(self.drop(self.norm(pooled))).squeeze(-1)
        return logits


def build(config: TrainConfig, *, features: int) -> LSTMClassifier:
    """Build the network a configuration describes."""
    return LSTMClassifier(
        features=features,
        hidden_size=config.hidden_size,
        layers=config.layers,
        dropout=config.dropout,
    )
