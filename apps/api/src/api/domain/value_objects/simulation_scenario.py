"""The simulation scenarios a run may be asked for.

**A deliberate duplicate of `simulator.domain.scenario.Scenario`, and the
duplication is the point.** `apps/api` does not depend on `simulator` and must
not: the API container installs eight packages by name and copies
`apps/api/src/api`, so an import of the simulator package would pass locally --
where the whole workspace is installed -- and fail at startup in production with
a `ModuleNotFoundError` naming a package the API never declared.

The two enums are therefore a cross-service contract, kept in step by
`tests/integration/test_simulation_contract.py`, which asserts the values match
the simulator's. That test is the only place the two services are checked
against each other, and it is what turns a drift into a failing build rather
than a run that produces nothing.

`incident_type.py` holds a third enum with three of these same names. It is not
this one and must not be confused with it: `IncidentType` describes what is
failing in a *machine*, this describes what a *simulation* is configured to do.
"""

from __future__ import annotations

from enum import StrEnum

from api.domain.errors import DomainValidationError


class SimulationScenario(StrEnum):
    """What a run drives the machine towards."""

    NORMAL = "NORMAL"
    BEARING_DEGRADATION = "BEARING_DEGRADATION"
    OVERHEATING = "OVERHEATING"
    OVERLOAD = "OVERLOAD"

    @classmethod
    def from_name(cls, name: str) -> SimulationScenario:
        """Resolve a scenario by name, case-insensitively.

        Raises:
            DomainValidationError: if the name matches no scenario, so a typo is
                a clear message rather than a run that starts and does nothing.
        """
        try:
            return cls(name.strip().upper())
        except ValueError:
            known = ", ".join(member.value for member in cls)
            raise DomainValidationError(
                f"Unknown scenario '{name}'. Known scenarios: {known}."
            ) from None


#: The scenario the canonical demonstration uses, per PRD section 12.
DEMO_SCENARIO = SimulationScenario.BEARING_DEGRADATION

#: The demonstration's seed, fixed so the sequence is always the same one. Its
#: value is arbitrary and its constancy is the whole point -- PRD section 12
#: asks that a reviewer reproduce a recognizable degradation sequence, and a
#: random seed would make every demonstration a different one.
DEMO_SEED = 20_260_923
