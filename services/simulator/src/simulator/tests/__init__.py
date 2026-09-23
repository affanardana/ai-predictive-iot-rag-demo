"""Test suite for the telemetry simulator.

Lives inside the package rather than in a sibling `tests/` directory: two
directories both named `tests` would produce two modules named `tests.conftest`,
which pytest's default import mode rejects as an import-file mismatch.
`simulator.tests.*` cannot collide with anything.
"""
