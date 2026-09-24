"""The torch layer — and the only place in `ml` that imports it.

This package is deliberately small. Everything that decides whether a published
number is trustworthy already lives in `ml.experiment` (the window index, the
labels, the sampler, the config, the prediction table) and `ml.evaluation` (the
metrics, the grouping, the lead time), and both of those run without torch
installed. What is left here is the network, the training loop, and a
`torch.utils.data.Dataset` that is a thin adapter over arrays someone else built.

Two import-linter contracts enforce the boundary in both directions, so an
import added here for convenience cannot quietly make the evaluation half
unusable on the machine that is supposed to reproduce it.

`torch` is an optional extra, not a core dependency:

    uv sync --locked --all-packages --extra train
"""

from __future__ import annotations

__all__: list[str] = []
