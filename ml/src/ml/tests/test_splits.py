"""Splitting the fleet by machine, and stratifying it by failure mode."""

from __future__ import annotations

from ml.dataset.lives import iter_machine_ids
from ml.dataset.splits import (
    Split,
    assign_splits,
    machines_in,
    split_of,
    training_machines,
)
from simulator.domain.scenario import Scenario

SCENARIOS = (
    Scenario.BEARING_DEGRADATION,
    Scenario.OVERHEATING,
    Scenario.OVERLOAD,
)


def _machines(count: int) -> list[tuple[str, Scenario]]:
    """A fleet whose failure modes cycle, as `ml.dataset.lives` builds them."""
    return [
        (machine_id, SCENARIOS[position % len(SCENARIOS)])
        for position, machine_id in enumerate(iter_machine_ids(count))
    ]


def test_every_machine_lands_in_exactly_one_split() -> None:
    assignments = assign_splits(_machines(100))
    assigned = [item.machine_id for item in assignments]

    assert len(assigned) == len(set(assigned)) == 100


def test_every_split_receives_every_failure_mode() -> None:
    """Without stratification, an unlucky draw can leave a mode unrepresented.

    A test set with no overload machines says nothing about overload, and pooled
    metrics would hide the gap.
    """
    assignments = assign_splits(_machines(99))

    for split in (Split.TRAIN, Split.VALIDATION, Split.TEST):
        seen = {item.primary_scenario for item in assignments if item.split is split}
        assert seen == set(SCENARIOS), f"{split.value} is missing a failure mode."


def test_the_shares_are_approximately_seventy_fifteen_fifteen() -> None:
    assignments = assign_splits(_machines(99))

    assert len(machines_in(assignments, Split.TRAIN)) == 69
    assert len(machines_in(assignments, Split.VALIDATION)) == 15
    assert len(machines_in(assignments, Split.TEST)) == 15


def test_a_small_group_still_reaches_all_three_splits() -> None:
    """Nine machines is three per failure mode, so each group has three to divide.

    The shares alone would not manage it: `round(0.15 * 3)` is zero, so the
    group would go entirely to training, validation and test would both be
    empty, and every test about splitting would pass while measuring nothing.
    """
    assignments = assign_splits(_machines(9))

    assert len(machines_in(assignments, Split.TRAIN)) == 3
    assert len(machines_in(assignments, Split.VALIDATION)) == 3
    assert len(machines_in(assignments, Split.TEST)) == 3


def test_a_group_too_small_to_divide_stays_in_training() -> None:
    """A group of one machine cannot be divided three ways.

    Three machines cycling three failure modes are three groups of one, so
    validation would be a single machine — which is not a measurement.
    """
    assignments = assign_splits(_machines(3))

    assert len(machines_in(assignments, Split.TRAIN)) == 3
    assert not machines_in(assignments, Split.VALIDATION)
    assert not machines_in(assignments, Split.TEST)


def test_shift_machines_are_held_out_entirely() -> None:
    """The duration-shift probe must not touch training, or it measures nothing."""
    machines = _machines(20)
    shift = frozenset(("M018", "M019", "M020"))
    assignments = assign_splits(machines, shift=shift)

    shifted = set(machines_in(assignments, Split.TEST_SHIFT))

    assert shifted == set(shift)
    assert not shifted & set(machines_in(assignments, Split.TRAIN))
    assert not shifted & set(machines_in(assignments, Split.VALIDATION))
    assert not shifted & set(machines_in(assignments, Split.TEST))


def test_only_training_machines_may_supply_statistics() -> None:
    """The frozenset that keeps a global mean from becoming a leak."""
    assignments = assign_splits(_machines(30))
    training = training_machines(assignments)

    assert training == frozenset(machines_in(assignments, Split.TRAIN))
    assert len(training) < 30


def test_the_split_lookup_covers_every_machine() -> None:
    assignments = assign_splits(_machines(30))
    lookup = split_of(assignments)

    assert set(lookup) == {item.machine_id for item in assignments}
