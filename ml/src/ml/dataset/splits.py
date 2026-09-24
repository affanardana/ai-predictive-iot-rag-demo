"""Assigning machines to train, validation and test.

**Split by machine, never by time.** This is the single decision that makes
every number Phase 4 reports worth reading.

A machine's nominal operating point, ambient temperature and susceptibility are
stable across all of its lives — that is what makes a machine a machine rather
than a bag of readings. So a machine appearing in two splits is memorised, not
generalised: the model learns "M047 runs about 6 rpm slow and heats up quickly"
and reports it as skill. Splitting by time leaks the same way and more quietly,
because consecutive windows of one life are near-duplicates of each other to
begin with.

Holding out entire machines is the pessimistic choice, and `ml report` says so.

## Stratification

Machines are grouped by their primary degradation mode and each group is split
separately, so all three modes appear in all three splits. Without it, an
unlucky draw could leave the test set with no overload machines at all, and the
pooled metrics would say nothing about the mode the model never met.

The modes are far from equally hard. Overload announces itself in `load` and
`current` almost immediately; bearing wear has to be inferred from a slower,
noisier signature. Pooled numbers are therefore dominated by the easy mode, and
the report carries per-mode figures alongside the pooled ones.

## The duration shift

A fourth split holds machines whose lives are drawn from a different range
entirely. It is not there to measure accuracy — it is there to answer a specific
question. The simulator's degradation is a deterministic function of elapsed
time, so a model can score well by inferring how long a machine has been running
rather than by reading what is wrong with it. A model that has learned the main
regime's timing collapses on lives of six and forty-eight hours; one that has
learned the signals degrades gracefully. Given this simulator, that is the only
evaluation that tells the two apart.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from simulator.domain.scenario import Scenario


class Split(StrEnum):
    """Which part of the dataset a machine belongs to."""

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"
    TEST_SHIFT = "test_shift"


#: Fractions of the main fleet, applied within each primary scenario. Validation
#: and test are equal: validation is for early stopping and threshold selection,
#: so it needs enough events to be stable, and test is what the numbers are
#: finally reported from.
TRAIN_SHARE = 0.70
VALIDATION_SHARE = 0.15


@dataclass(frozen=True, slots=True)
class MachineAssignment:
    """One machine's place in the split."""

    machine_id: str
    primary_scenario: Scenario
    split: Split


def assign_splits(
    machines: Sequence[tuple[str, Scenario]],
    *,
    shift: frozenset[str] = frozenset(),
) -> tuple[MachineAssignment, ...]:
    """Assign every machine to a split, stratified by primary scenario.

    `shift` names machines drawn from the shifted duration regime; they are
    assigned to `Split.TEST_SHIFT` and take no part in the stratified draw. They
    are held out of training deliberately — the point of the split is to ask how
    a model behaves on a regime it never saw.
    """
    assignments: list[MachineAssignment] = []
    grouped: dict[Scenario, list[str]] = {}

    for machine_id, scenario in machines:
        if machine_id in shift:
            assignments.append(MachineAssignment(machine_id, scenario, Split.TEST_SHIFT))
        else:
            grouped.setdefault(scenario, []).append(machine_id)

    for scenario, machine_ids in grouped.items():
        assignments.extend(_stratify(machine_ids, scenario))

    return tuple(assignments)


def _boundaries(count: int) -> tuple[int, int]:
    """Return where training ends and where validation ends.

    Counts are computed once and the test set takes the remainder, so the three
    parts always account for every machine exactly once — a rounding slip here
    would silently drop a machine rather than fail loudly.

    Both boundaries are also floored at one machine per split. The shares alone
    do not guarantee that: a group of three gives `round(0.15 * 3)` machines of
    validation, which is none, and a split nobody lands in is a split that will
    not be noticed until a metric comes back empty. Groups below three keep
    everything in training, since there is nothing to divide.
    """
    if count < 3:
        return count, count
    train_end = min(max(1, round(TRAIN_SHARE * count)), count - 2)
    validation_end = min(max(train_end + 1, train_end + round(VALIDATION_SHARE * count)), count - 1)
    return train_end, validation_end


def _stratify(machine_ids: Sequence[str], scenario: Scenario) -> list[MachineAssignment]:
    """Split one scenario's machines into train, validation and test."""
    train_end, validation_end = _boundaries(len(machine_ids))

    return [
        MachineAssignment(
            machine_id=machine_id,
            primary_scenario=scenario,
            split=(
                Split.TRAIN
                if position < train_end
                else Split.VALIDATION
                if position < validation_end
                else Split.TEST
            ),
        )
        for position, machine_id in enumerate(machine_ids)
    ]


def machines_in(assignments: Sequence[MachineAssignment], split: Split) -> tuple[str, ...]:
    """Return the machines assigned to `split`, in order."""
    return tuple(item.machine_id for item in assignments if item.split is split)


def split_of(assignments: Sequence[MachineAssignment]) -> dict[str, Split]:
    """Return a machine id to split lookup."""
    return {item.machine_id: item.split for item in assignments}


def training_machines(assignments: Sequence[MachineAssignment]) -> frozenset[str]:
    """Return the machines whose statistics may be computed.

    Only these. A mean taken over the whole dataset carries information about
    the validation and test machines into training — the classic silent leak,
    invisible because nothing about the code looks wrong.
    """
    return frozenset(machines_in(assignments, Split.TRAIN))
