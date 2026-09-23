"""Machine profiles: validation, fleet variety, and the nominal point."""

from __future__ import annotations

import random

import pytest

from simulator.domain.errors import SimulationValidationError
from simulator.domain.machine import (
    MAX_MACHINE_ID_LENGTH,
    NOMINAL_CURRENT,
    NOMINAL_LOAD,
    NOMINAL_RPM,
    NOMINAL_VIBRATION,
    NOMINAL_VOLTAGE,
    SUSCEPTIBILITY_MAX,
    SUSCEPTIBILITY_MIN,
    MachineProfile,
    build_profile,
)
from simulator.domain.readings import SensorReading
from simulator.domain.seeding import derive_seed


def _profile(**overrides: object) -> MachineProfile:
    """Build a profile, overriding only what a test cares about."""
    values: dict[str, object] = {"machine_id": "M001", "name": "Motor M001"}
    values.update(overrides)
    return MachineProfile(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize("machine_id", ["M001", "M003", "PUMP012", "A999"])
def test_identifiers_matching_the_shared_pattern_are_accepted(machine_id: str) -> None:
    """The same pattern the API enforces, so a fleet crosses the boundary intact."""
    assert _profile(machine_id=machine_id).machine_id == machine_id


@pytest.mark.parametrize(
    "machine_id",
    ["", "m001", "M01", "M0001", "MOTOR1", "M 003", "M-003", "003M"],
)
def test_malformed_identifiers_are_rejected(machine_id: str) -> None:
    """Rejected at construction, not discovered when the API refuses them."""
    with pytest.raises(SimulationValidationError):
        _profile(machine_id=machine_id)


def test_an_identifier_longer_than_the_column_is_rejected() -> None:
    """The API's machine_id column is 16 characters."""
    with pytest.raises(SimulationValidationError):
        _profile(machine_id="M" * (MAX_MACHINE_ID_LENGTH + 1))


def test_a_blank_name_is_rejected() -> None:
    """A nameless machine is a data-entry mistake, not a machine."""
    with pytest.raises(SimulationValidationError):
        _profile(name="   ")


@pytest.mark.parametrize("field", ["rpm_nominal", "voltage_nominal", "susceptibility"])
@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_non_finite_nominals_are_rejected(field: str, value: float) -> None:
    """A nominal point has to be a real number, or every reading derived is not."""
    with pytest.raises(SimulationValidationError):
        _profile(**{field: value})


def test_a_non_positive_susceptibility_is_rejected() -> None:
    """Zero susceptibility would mean a machine that can never degrade."""
    with pytest.raises(SimulationValidationError):
        _profile(susceptibility=0.0)


def test_the_default_profile_matches_the_documented_operating_point() -> None:
    """The nominal point is the one the API's own test fixtures assume.

    The two services do not share code, so this is where the agreement is
    recorded: a healthy machine reads about 65 degrees, 1.4 mm/s, 1480 rpm,
    12.5 A, 0.72 load, and 400 V. Change one and the other is now wrong.
    """
    reading = MachineProfile(machine_id="M001", name="Motor M001").nominal_reading()

    assert reading == SensorReading(
        temperature=65.0,
        vibration=NOMINAL_VIBRATION,
        rpm=NOMINAL_RPM,
        current=NOMINAL_CURRENT,
        load=NOMINAL_LOAD,
        voltage=NOMINAL_VOLTAGE,
    )


def test_a_colder_workshop_produces_colder_readings() -> None:
    """Ambient temperature is a parameter rather than a constant.

    The rise is measured from ambient, so a machine in a cold plant room reads
    genuinely colder than one beside a furnace, rather than both reading 65.
    """
    cold = _profile(temperature_ambient=5.0).nominal_reading()
    hot = _profile(temperature_ambient=40.0).nominal_reading()

    assert cold.temperature < hot.temperature


def test_building_a_profile_is_reproducible_from_its_seed() -> None:
    """Same seed, same machine — the foundation of a reproducible fleet."""
    first = build_profile("M001", random.Random(derive_seed(1234, "M001")))
    second = build_profile("M001", random.Random(derive_seed(1234, "M001")))

    assert first == second


def test_different_machines_differ() -> None:
    """A fleet of identical units would make fleet-level results meaningless."""
    machines = [
        build_profile(f"M{index:03d}", random.Random(derive_seed(1234, index)))
        for index in range(1, 8)
    ]

    assert len({machine.rpm_nominal for machine in machines}) == len(machines)


def test_a_built_profile_stays_within_its_documented_spread() -> None:
    """The jitter is bounded, so a fleet stays physically plausible."""
    for index in range(1, 40):
        profile = build_profile(f"M{index:03d}", random.Random(derive_seed(99, index)))

        assert SUSCEPTIBILITY_MIN <= profile.susceptibility <= SUSCEPTIBILITY_MAX
        assert 1400.0 < profile.rpm_nominal < 1560.0
        assert 380.0 < profile.voltage_nominal < 420.0
