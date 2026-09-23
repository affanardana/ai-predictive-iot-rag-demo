"""Degradation physics: the coupling that makes telemetry coherent."""

from __future__ import annotations

from dataclasses import asdict

import pytest

from simulator.domain.engine import MachineSimulator
from simulator.domain.physics import (
    FAILURE_HEALTH_THRESHOLD,
    degradation_at,
    degradation_progress,
    health_index,
    is_failure_imminent,
)
from simulator.domain.readings import NON_NEGATIVE_SIGNALS, SIGNAL_NAMES
from simulator.domain.scenario import Scenario, profile_for
from simulator.domain.session import SimulationSession
from simulator.domain.state import DegradationState
from simulator.tests.conftest import FIXED_START, STANDARD_DURATION
from simulator.tests.support import mean_signals, readings_between

#: A window at each end of a run, averaged. Comparing trend rather than single
#: samples keeps the assertion about the physics rather than about the noise.
_WINDOW = 10


def _trend(scenario: Scenario) -> tuple[dict[str, float], dict[str, float]]:
    """Return mean signals at the start and at the end of a run."""
    session = SimulationSession.create(
        machine_ids=["M001"],
        scenario=scenario,
        seed=4242,
        duration=STANDARD_DURATION,
        started_at=FIXED_START,
    )
    simulator = MachineSimulator(session, session.machines[0])
    total = simulator.tick_count
    return (
        mean_signals(readings_between(simulator, 0, _WINDOW)),
        mean_signals(readings_between(simulator, total - _WINDOW, total)),
    )


def test_progress_starts_at_zero_and_finishes_at_one() -> None:
    """The curve is normalised, so a run's endpoints mean something."""
    assert degradation_progress(0.0) == pytest.approx(0.0, abs=1e-9)
    assert degradation_progress(1.0) == pytest.approx(1.0, abs=1e-9)


def test_progress_is_bounded_outside_the_run() -> None:
    """Positions beyond the run clamp rather than diverging."""
    assert degradation_progress(-5.0) == pytest.approx(0.0, abs=1e-9)
    assert degradation_progress(5.0) == pytest.approx(1.0, abs=1e-9)


def test_progress_accelerates_rather_than_progressing_steadily() -> None:
    """Damage feeds on itself, so the curve is not a straight line.

    A linear ramp would make a demonstration look like a steady drift rather
    than a machine going wrong. The shape is checked at the quartiles rather
    than at the midpoint: a logistic is symmetric about its centre, so
    `progress(0.5)` is exactly 0.5 and asserts nothing about acceleration. What
    does is lagging at the first quarter and running ahead at the third.
    """
    assert degradation_progress(0.25) < 0.25
    assert degradation_progress(0.75) > 0.75


def test_progress_never_decreases() -> None:
    """Degradation is monotonic, sampled densely enough to catch a wobble."""
    values = [degradation_progress(step / 200) for step in range(201)]

    assert values == sorted(values)


def test_the_normal_scenario_degrades_nothing() -> None:
    """A healthy machine stays healthy, which is what makes it a baseline."""
    profile = profile_for(Scenario.NORMAL)
    state = degradation_at(1.0, profile, susceptibility=1.3)

    assert state == DegradationState()


@pytest.mark.parametrize(
    "scenario",
    [Scenario.BEARING_DEGRADATION, Scenario.OVERHEATING, Scenario.OVERLOAD],
)
def test_each_degradation_scenario_engages_at_least_one_channel(scenario: Scenario) -> None:
    """A scenario that drove nothing would produce a NORMAL run under another name."""
    state = degradation_at(1.0, profile_for(scenario), susceptibility=1.0)

    assert state.worst() > 0.0


def test_bearing_degradation_moves_four_signals_together() -> None:
    """PRD section 11's correlation, asserted rather than assumed.

    "bearing degradation should be capable of producing correlated changes such
    as: vibration up, temperature up, current up, RPM down". This is the
    requirement that separates a simulation from four independent random walks,
    and `MASTERPLAN.md` §3.6 names the independent version as a prohibited
    shortcut.
    """
    early, late = _trend(Scenario.BEARING_DEGRADATION)

    assert late["vibration"] > early["vibration"]
    assert late["temperature"] > early["temperature"]
    assert late["current"] > early["current"]
    assert late["rpm"] < early["rpm"]


def test_overheating_raises_temperature_hardest() -> None:
    """Each scenario has its own signature, not the same one relabelled.

    Each run is measured against its own starting point rather than against a
    nominal temperature, because the fleet's machines are jittered and a
    hardcoded baseline would be comparing profiles rather than scenarios.
    """
    early_bearing, late_bearing = _trend(Scenario.BEARING_DEGRADATION)
    early_thermal, late_thermal = _trend(Scenario.OVERHEATING)

    bearing_rise = late_bearing["temperature"] - early_bearing["temperature"]
    thermal_rise = late_thermal["temperature"] - early_thermal["temperature"]

    assert thermal_rise > bearing_rise


def test_overload_drives_load_and_current() -> None:
    """Overload's tell is mechanical, not thermal."""
    _, bearing = _trend(Scenario.BEARING_DEGRADATION)
    _, load = _trend(Scenario.OVERLOAD)

    assert load["load"] > bearing["load"]
    assert load["current"] > bearing["current"]


def test_a_healthy_run_stays_healthy() -> None:
    """Nothing degrades, so nothing reaches the failure threshold."""
    session = SimulationSession.create(
        machine_ids=["M001"],
        scenario=Scenario.NORMAL,
        seed=7,
        duration=STANDARD_DURATION,
        started_at=FIXED_START,
    )
    simulator = MachineSimulator(session, session.machines[0])
    final = simulator.tick(simulator.tick_count - 1).ground_truth

    assert final.health_index > 0.99
    assert not final.failure_imminent


def test_a_degradation_run_reaches_imminent_failure() -> None:
    """The demonstration has to actually demonstrate something."""
    session = SimulationSession.create(
        machine_ids=["M001"],
        scenario=Scenario.BEARING_DEGRADATION,
        seed=7,
        duration=STANDARD_DURATION,
        started_at=FIXED_START,
    )
    simulator = MachineSimulator(session, session.machines[0])
    final = simulator.tick(simulator.tick_count - 1).ground_truth

    assert final.failure_imminent
    assert final.health_index < FAILURE_HEALTH_THRESHOLD


def test_health_tracks_the_worst_channel_not_the_average() -> None:
    """A machine is as healthy as its worst problem.

    Averaging would let a badly worn bearing hide behind two healthy channels,
    which is precisely the machine an operator most needs warned about. The two
    states below carry the same total damage; only its distribution differs, and
    the concentrated one must read as worse.
    """
    concentrated = DegradationState(bearing_wear=0.9)
    spread = DegradationState(bearing_wear=0.3, thermal_stress=0.3, load_stress=0.3)

    assert sum(asdict(concentrated).values()) == pytest.approx(sum(asdict(spread).values()))
    assert health_index(concentrated) < health_index(spread)


def test_failure_threshold_is_a_band_not_a_point() -> None:
    """Health just above the threshold is not yet a failure."""
    assert not is_failure_imminent(FAILURE_HEALTH_THRESHOLD)
    assert is_failure_imminent(FAILURE_HEALTH_THRESHOLD - 0.001)


#: The lowest each signal may reach before it is suspiciously close to the
#: physical floor. Expressed per signal rather than as one number, because the
#: signals are on different scales: a load of 0.7 is healthy and an rpm of 0.7
#: is a stopped machine.
MINIMUM_PLAUSIBLE: dict[str, float] = {
    "vibration": 0.4,
    "rpm": 1000.0,
    "load": 0.4,
    "voltage": 350.0,
}


@pytest.mark.parametrize("scenario", list(Scenario))
def test_no_scenario_drives_a_signal_onto_its_physical_floor(scenario: Scenario) -> None:
    """The floor in `synthesise_reading` is a guard, not a correction.

    Every non-negative signal must stay well clear of zero across a whole run.
    If a physics change pushed one down towards the floor, the clamp would hide
    it and the telemetry would look plausible while being wrong — so the margin
    is asserted rather than the floor itself.
    """
    session = SimulationSession.create(
        machine_ids=["M001"],
        scenario=scenario,
        seed=11,
        duration=STANDARD_DURATION,
        started_at=FIXED_START,
    )
    simulator = MachineSimulator(session, session.machines[0])
    readings = readings_between(simulator, 0, simulator.tick_count)

    for name in NON_NEGATIVE_SIGNALS:
        lowest = min(getattr(reading, name) for reading in readings)
        assert lowest > MINIMUM_PLAUSIBLE[name], (
            f"'{name}' fell to {lowest}, close enough to the floor that the clamp "
            "may be hiding a physics change."
        )


def test_every_signal_is_generated_for_every_tick() -> None:
    """Guard against a signal silently dropping out of the synthesis loop."""
    session = SimulationSession.create(
        machine_ids=["M001"],
        scenario=Scenario.BEARING_DEGRADATION,
        seed=3,
        duration=STANDARD_DURATION,
        started_at=FIXED_START,
    )
    reading = MachineSimulator(session, session.machines[0]).tick(0).telemetry.reading

    assert set(reading.as_row()) == set(SIGNAL_NAMES)
