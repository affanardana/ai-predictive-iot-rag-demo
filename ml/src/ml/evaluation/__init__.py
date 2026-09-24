"""Measuring a model, and refusing to measure it dishonestly.

This package is numpy-only and knows nothing about PyTorch. That is not a
preference, it is what makes the phase verifiable: training happens on a GPU
somewhere else, so everything that decides whether a published number is
trustworthy has to run on a machine where the model cannot.

Arrays in, numbers out. It does not read the dataset artifact, it does not
build windows, and it does not know a model exists — a `Predictions` table and
a set of labels are all it takes to produce every figure in the report.
"""

from __future__ import annotations

__all__: list[str] = []
