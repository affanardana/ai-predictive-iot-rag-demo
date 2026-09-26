"""Read models assembled by use cases.

These combine several aggregates into the shape an operator actually needs --
a machine with its current condition, or a prediction alongside the incident
it raised. They are assembled here rather than in the domain because they are
views over state, not state themselves.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from api.domain.entities.incident import Incident
from api.domain.entities.machine import Machine
from api.domain.entities.prediction import Prediction
from api.domain.entities.simulation_run import SimulationRun
from api.domain.entities.telemetry import TelemetryRecord
from api.domain.value_objects.risk_level import RiskLevel


@dataclass(frozen=True, slots=True)
class SimulationRunView:
    """A simulation run, with the two facts a client cannot derive.

    Staleness needs a clock and a timeout, and the presentation layer has
    neither. Computing it here means every client agrees about whether a run is
    alive, using the same clock the runs were recorded against -- rather than
    each browser guessing from its own and disagreeing.
    """

    run: SimulationRun
    is_active: bool
    is_stale: bool


@dataclass(frozen=True, slots=True)
class MachineSummary:
    """A machine with its most recent observed and predicted condition.

    `latest_reading` and `latest_prediction` are both optional. A machine
    registered but not yet reporting has neither; one that has stopped
    reporting retains its last known values, which is what an operator needs
    to see even though the data is stale.
    """

    machine: Machine
    latest_reading: TelemetryRecord | None
    latest_prediction: Prediction | None
    risk_level: RiskLevel | None
    open_incident_count: int

    @property
    def is_reporting(self) -> bool:
        """Whether any telemetry has ever been received for this machine."""
        return self.latest_reading is not None


@dataclass(frozen=True, slots=True)
class MachineDetail:
    """Everything the machine detail page needs about one machine."""

    summary: MachineSummary
    recent_incidents: Sequence[Incident]
