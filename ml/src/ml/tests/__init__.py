"""Tests for the dataset pipeline.

Inside the package rather than in a sibling `tests/` directory, for the reason
recorded in the root `pyproject.toml`: two directories both named `tests` would
produce two modules named `tests.conftest`, which pytest's default import mode
rejects. `ml.tests.*` cannot collide with anything.
"""
