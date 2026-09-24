"""Shared fixtures: a small fleet, generated once and read by many tests.

The real configuration is a hundred machines over thirty days, which takes
minutes. Everything structural is visible in twelve machines over one day, which
takes seconds — so the suite uses the small one and asserts the properties that
do not depend on scale.

The fleet is deliberately nine machines rather than three. Stratification splits
each primary scenario's group separately, and a group of one has nothing to
divide: every machine would land in training and the validation and test splits
would be empty, so tests about splitting would pass by measuring nothing.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from ml.dataset.artifacts import Dataset, build_dataset
from ml.dataset.generation import generate
from ml.dataset.lives import MachinePlan, iter_machine_ids
from ml.dataset.plan import PLAN_FILENAME, document_for, write_plan

FIXED_START = datetime(2026, 1, 1, tzinfo=UTC)
TEST_SEED = 7
#: Two days, not one. A one-day timeline often yields a single life per
#: machine, which is too few for the `NORMAL` share to be reliably represented —
#: and several tests need a fleet that actually contains both failing and
#: never-failing lives.
TEST_TIMELINE = timedelta(days=2)
MAIN_MACHINES = 9
SHIFT_MACHINES = 3


def build_fleet(
    directory: Path,
    *,
    machines: int = MAIN_MACHINES,
    shift: int = SHIFT_MACHINES,
    seed: int = TEST_SEED,
    timeline: timedelta = TEST_TIMELINE,
) -> tuple[Path, tuple[MachinePlan, ...]]:
    """Generate a small dataset on disk, and return its directory and plans."""
    document = document_for(
        seed=seed,
        started_at=FIXED_START,
        main_machines=tuple(iter_machine_ids(machines)),
        shift_machines=tuple(iter_machine_ids(shift, first=machines + 1)),
        timeline=timeline,
    )
    plans = document.build()
    generate(plans, output=directory)
    write_plan(directory / PLAN_FILENAME, document)
    return directory, plans


@pytest.fixture(scope="session")
def fleet(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, tuple[MachinePlan, ...]]:
    """A generated fleet, built once for the whole session."""
    return build_fleet(tmp_path_factory.mktemp("fleet"))


@pytest.fixture(scope="session")
def dataset(fleet: tuple[Path, tuple[MachinePlan, ...]]) -> Dataset:
    """The training artifact built from that fleet."""
    return build_dataset(fleet[0])


@pytest.fixture(scope="session")
def plans(fleet: tuple[Path, tuple[MachinePlan, ...]]) -> tuple[MachinePlan, ...]:
    """The fleet's plans."""
    return fleet[1]


@pytest.fixture
def fresh_fleet(tmp_path: Path) -> Iterator[tuple[Path, tuple[MachinePlan, ...]]]:
    """A fleet in a per-test directory, for tests that write to it."""
    yield build_fleet(tmp_path)
