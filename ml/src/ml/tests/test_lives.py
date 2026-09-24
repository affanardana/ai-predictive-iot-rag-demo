"""Composing a machine's timeline out of lives."""

from __future__ import annotations

from datetime import timedelta

import pytest

from ml.dataset.lives import (
    MAIN_LIFE_MAX_HOURS,
    MAIN_LIFE_MIN_HOURS,
    MIN_LIFE,
    SHIFT_LIFE_HOURS,
    MachinePlan,
    Regime,
    plan_fleet,
    plan_machine,
)
from ml.tests.conftest import FIXED_START, TEST_SEED

#: A week, not a day. These tests only plan — no telemetry is generated — and a
#: one-day timeline yields a single life often enough that the range assertions
#: would pass by being vacuous.
TIMELINE = timedelta(days=7)


def _plan(machine_id: str = "M001", regime: Regime = Regime.MAIN) -> MachinePlan:
    return plan_machine(
        machine_id,
        base_seed=TEST_SEED,
        regime=regime,
        started_at=FIXED_START,
        timeline=TIMELINE,
    )


def test_lives_fill_the_timeline_exactly() -> None:
    """A machine is observed for its whole timeline, with no gap and no overlap."""
    plan = _plan()

    assert plan.duration == TIMELINE
    assert plan.lives[0].started_at == FIXED_START
    for previous, following in zip(plan.lives, plan.lives[1:], strict=False):
        assert previous.started_at + previous.duration == following.started_at


def test_lives_are_indexed_in_order() -> None:
    plan = _plan()

    assert [life.index for life in plan.lives] == list(range(len(plan.lives)))


def test_no_life_is_shorter_than_the_minimum() -> None:
    """A life too short to hold a window would add a row and nothing else."""
    for machine_id in ("M001", "M002", "M003", "M004"):
        for life in _plan(machine_id).lives:
            assert life.duration >= MIN_LIFE


def test_main_lives_are_drawn_from_the_main_range() -> None:
    """Except the last, which absorbs whatever remains."""
    lives = _plan().lives

    for life in lives[:-1]:
        hours = life.duration.total_seconds() / 3600.0
        assert MAIN_LIFE_MIN_HOURS <= hours <= MAIN_LIFE_MAX_HOURS


def test_shift_lives_use_a_different_distribution() -> None:
    """The held-out regime, whose whole purpose is to be out of range."""
    lives = _plan("M101", Regime.SHIFT).lives

    for life in lives[:-1]:
        hours = life.duration.total_seconds() / 3600.0
        assert hours in SHIFT_LIFE_HOURS


def test_a_machine_keeps_its_failure_mode_across_its_lives() -> None:
    """Which is what makes a machine-level split mean anything."""
    plan = _plan("M004")

    degrading = [life.scenario for life in plan.lives if life.scenario.value != "NORMAL"]

    assert degrading, "a machine with no degradation lives cannot produce a failure."
    assert set(degrading) == {plan.primary_scenario}


def test_healthy_lives_appear_among_the_degrading_ones() -> None:
    """Every negative example comes from here, so the mix cannot collapse."""
    scenarios = {
        life.scenario.value
        for machine_id in ("M001", "M002", "M003", "M004", "M005")
        for life in _plan(machine_id).lives
    }

    assert "NORMAL" in scenarios


def test_the_machine_profile_is_stable_across_its_lives() -> None:
    """A repair does not change a machine's rated speed.

    `SimulationSession.create` derives each machine from the session seed, so a
    fresh session per life would give the same machine a different nominal point
    every time it was repaired — and identity is exactly what the split relies
    on.
    """
    plan = _plan("M006")
    rebuilt = _plan("M006")

    assert plan.profile == rebuilt.profile
    assert plan.profile.machine_id == "M006"
    assert plan.profile.susceptibility > 0.0


def test_the_timeline_is_reproducible_from_its_inputs() -> None:
    """The plan file stores four values and re-derives the rest, so this holds."""
    assert _plan("M007").lives == _plan("M007").lives


def test_a_machine_is_unaffected_by_its_neighbours() -> None:
    """Adding a machine to a later run leaves the earlier ones byte-identical."""
    alone = _plan("M003")

    together = plan_fleet(
        ["M001", "M002", "M003"],
        base_seed=TEST_SEED,
        regime=Regime.MAIN,
        started_at=FIXED_START,
        timeline=TIMELINE,
    )

    assert together[2].lives == alone.lives


@pytest.mark.parametrize(
    ("machine_id", "expected"),
    [
        ("M001", "BEARING_DEGRADATION"),
        ("M002", "OVERHEATING"),
        ("M003", "OVERLOAD"),
        ("M004", "BEARING_DEGRADATION"),
        ("M101", "OVERHEATING"),
    ],
)
def test_the_failure_modes_cycle_across_the_fleet(machine_id: str, expected: str) -> None:
    """So a stratified split can put all three in every split."""
    assert _plan(machine_id).primary_scenario.value == expected
