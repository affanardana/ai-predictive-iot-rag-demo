"""Dataset construction.

The pipeline runs in one direction and each stage has one job:

    lives     -> compose a fleet timeline out of sequential lives
    generation-> drive the simulator across it, writing telemetry and truth
    labelling -> read the truth, and say when each life entered failure
    splits    -> assign machines to train / validation / test
    windows   -> cut 60-step sequences that never cross a life boundary
    artifacts -> normalise, and write the tensors a model trains on
"""

from __future__ import annotations

__all__: list[str] = []
