"""Per-machine configuration.

A `MachineProfile` is a machine's nominal operating point — what a healthy unit
of that type reads at rated load — plus how susceptible it is to degradation.
Profiles are built from a seed, so a fleet is reproducible while still
containing strong and weak units rather than copies of one machine.
"""

from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass

from simulator.domain.errors import SimulationValidationError
from simulator.domain.readings import SensorReading

#: Identical to the pattern `apps/api` enforces. Machine identifiers cross the
#: service boundary, so a fleet the simulator invents has to be one the API
#: will accept.
MACHINE_ID_PATTERN = re.compile(r"^[A-Z]{1,4}\d{3}$")

#: Fits the `String(16)` column in the Phase 1 schema.
MAX_MACHINE_ID_LENGTH = 16

#: A four-pole induction motor on a 50 Hz supply, running slightly below
#: synchronous speed under load.
NOMINAL_RPM = 1480.0
NOMINAL_VOLTAGE = 400.0
NOMINAL_LOAD = 0.72
NOMINAL_CURRENT = 12.5
NOMINAL_VIBRATION = 1.4

#: A machine in a plant room, not in a furnace. Rise is measured from ambient so
#: a cold workshop produces genuinely colder readings than a hot one.
AMBIENT_TEMPERATURE = 25.0
NOMINAL_TEMPERATURE_RISE = 40.0

#: How far a machine's nominal point may stray from the type's nominal values.
RPM_SPREAD = 15.0
VOLTAGE_SPREAD = 6.0
LOAD_SPREAD = 0.06
CURRENT_SPREAD = 1.0
VIBRATION_SPREAD = 0.15
AMBIENT_SPREAD = 3.0

#: A fleet contains machines that degrade faster and slower than the scenario's
#: nominal pace, which is what gives predictions varying lead times.
#:
#: The floor is 1.0 rather than something lower, deliberately. Susceptibility
#: scales the amplitude a scenario reaches, not just its speed, so a machine
#: below 1.0 would never cross the failure threshold no matter how long the run
#: lasted — and a degradation scenario that does not reliably produce a failure
#: is useless as a positive example and cannot drive the PRD §12 demonstration.
#: Variation in *when* the failure arrives is where the interest lies.
SUSCEPTIBILITY_MIN = 1.0
SUSCEPTIBILITY_MAX = 1.35


@dataclass(frozen=True, slots=True)
class MachineProfile:
    """A machine's nominal operating point and its susceptibility."""

    machine_id: str
    name: str
    rpm_nominal: float = NOMINAL_RPM
    voltage_nominal: float = NOMINAL_VOLTAGE
    load_nominal: float = NOMINAL_LOAD
    current_nominal: float = NOMINAL_CURRENT
    vibration_nominal: float = NOMINAL_VIBRATION
    temperature_ambient: float = AMBIENT_TEMPERATURE
    susceptibility: float = 1.0

    def __post_init__(self) -> None:
        """Validate the identifier and the nominal operating point."""
        if len(self.machine_id) > MAX_MACHINE_ID_LENGTH:
            raise SimulationValidationError(
                f"Machine id '{self.machine_id}' exceeds {MAX_MACHINE_ID_LENGTH} characters."
            )
        if not MACHINE_ID_PATTERN.match(self.machine_id):
            raise SimulationValidationError(
                f"Machine id '{self.machine_id}' is malformed. Expected one to four "
                "uppercase letters followed by three digits, e.g. 'M003'."
            )
        if not self.name.strip():
            raise SimulationValidationError("Machine name must not be blank.")

        for field in (
            "rpm_nominal",
            "voltage_nominal",
            "load_nominal",
            "current_nominal",
            "vibration_nominal",
            "temperature_ambient",
            "susceptibility",
        ):
            value = getattr(self, field)
            if math.isnan(value) or math.isinf(value):
                raise SimulationValidationError(f"'{field}' must be a finite number.")

        if self.susceptibility <= 0.0:
            raise SimulationValidationError(
                f"susceptibility must be positive, got {self.susceptibility}."
            )

    def nominal_reading(self) -> SensorReading:
        """Return what a healthy machine of this type reads at rated load."""
        return SensorReading(
            temperature=self.temperature_ambient + NOMINAL_TEMPERATURE_RISE,
            vibration=self.vibration_nominal,
            rpm=self.rpm_nominal,
            current=self.current_nominal,
            load=self.load_nominal,
            voltage=self.voltage_nominal,
        )


def build_profile(machine_id: str, rng: random.Random) -> MachineProfile:
    """Build a machine with a plausible spread around the type's nominal point.

    Drawing the spread from the supplied generator is what makes a fleet
    reproducible from a seed: the same seed yields the same machines, and
    therefore the same telemetry.
    """
    return MachineProfile(
        machine_id=machine_id,
        name=f"Motor {machine_id}",
        rpm_nominal=NOMINAL_RPM + rng.uniform(-RPM_SPREAD, RPM_SPREAD),
        voltage_nominal=NOMINAL_VOLTAGE + rng.uniform(-VOLTAGE_SPREAD, VOLTAGE_SPREAD),
        load_nominal=NOMINAL_LOAD + rng.uniform(-LOAD_SPREAD, LOAD_SPREAD),
        current_nominal=NOMINAL_CURRENT + rng.uniform(-CURRENT_SPREAD, CURRENT_SPREAD),
        vibration_nominal=NOMINAL_VIBRATION + rng.uniform(-VIBRATION_SPREAD, VIBRATION_SPREAD),
        temperature_ambient=AMBIENT_TEMPERATURE + rng.uniform(-AMBIENT_SPREAD, AMBIENT_SPREAD),
        susceptibility=rng.uniform(SUSCEPTIBILITY_MIN, SUSCEPTIBILITY_MAX),
    )
