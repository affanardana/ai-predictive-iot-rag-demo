"""Degradation scenarios.

A scenario says *which* failure mechanisms a run drives and how hard, not how
they progress over time — that is `physics.degradation_at`. Keeping the two
apart means a new scenario is a few numbers here rather than new logic.

The four scenarios are the ones `MASTERPLAN.md` §6 and PRD §11 name. The
scenario a machine is running is **ground truth**: it is never written to the
telemetry stream.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

from simulator.domain.errors import UnknownScenarioError


class Scenario(StrEnum):
    """A degradation pattern a machine can be driven through."""

    NORMAL = "NORMAL"
    BEARING_DEGRADATION = "BEARING_DEGRADATION"
    OVERHEATING = "OVERHEATING"
    OVERLOAD = "OVERLOAD"

    @classmethod
    def from_name(cls, name: str) -> Scenario:
        """Resolve a scenario from a case-insensitive name.

        Raises:
            UnknownScenarioError: if the name matches no scenario.
        """
        try:
            return cls(name.strip().upper())
        except ValueError as exc:
            known = tuple(scenario.value for scenario in cls)
            raise UnknownScenarioError(name, known) from exc


@dataclass(frozen=True, slots=True)
class ScenarioProfile:
    """How hard a scenario drives each degradation channel.

    Weights are relative intensities on `[0, 1]`, multiplied by how far the run
    has progressed and by the machine's own susceptibility. A weight of zero
    means the scenario does not engage that mechanism at all.
    """

    bearing_weight: float
    thermal_weight: float
    load_weight: float


#: A worn bearing raises friction, so vibration climbs hardest; the extra drag
#: heats the machine and draws more current while slowing the shaft. The
#: secondary thermal term is what makes the signature realistic rather than
#: a single sensor moving on its own.
BEARING_DEGRADATION_PROFILE = ScenarioProfile(
    bearing_weight=1.0,
    thermal_weight=0.25,
    load_weight=0.0,
)

#: Heat is the primary mechanism. A hot winding draws more current, and the
#: bearing suffers secondary damage from thermal expansion.
OVERHEATING_PROFILE = ScenarioProfile(
    bearing_weight=0.10,
    thermal_weight=1.0,
    load_weight=0.15,
)

#: Sustained overload drives current and load hardest, and the extra work shows
#: up as heat.
OVERLOAD_PROFILE = ScenarioProfile(
    bearing_weight=0.10,
    thermal_weight=0.50,
    load_weight=1.0,
)

#: Nothing degrades. Used for baselines and for proving that a healthy machine
#: stays inside the risk bands the API defines.
NORMAL_PROFILE = ScenarioProfile(
    bearing_weight=0.0,
    thermal_weight=0.0,
    load_weight=0.0,
)

SCENARIO_PROFILES: Mapping[Scenario, ScenarioProfile] = {
    Scenario.NORMAL: NORMAL_PROFILE,
    Scenario.BEARING_DEGRADATION: BEARING_DEGRADATION_PROFILE,
    Scenario.OVERHEATING: OVERHEATING_PROFILE,
    Scenario.OVERLOAD: OVERLOAD_PROFILE,
}


def profile_for(scenario: Scenario) -> ScenarioProfile:
    """Return the degradation weights a scenario applies."""
    return SCENARIO_PROFILES[scenario]
