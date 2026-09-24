"""Everything about a training run that is not the training.

The window index and its labels, the sampler's draw, the run configuration, the
prediction table and the run manifest all live here, and none of them import
torch.

That is deliberate. Training happens on a GPU somewhere else, so anything that
could make a published number wrong has to be buildable and testable where the
model cannot run. What is left in `ml.model` is a thin adapter over arrays this
package produces.

The other thing this package exists for is the seam: `evaluate` takes a
prediction table, not a model. A reviewer with no GPU re-derives every reported
figure by running a command over files.
"""

from __future__ import annotations

__all__: list[str] = []
