"""Domain ports.

Interfaces the domain defines and infrastructure implements. Dependencies
point inward: infrastructure knows the domain, never the reverse.

Every method is `async`. That choice is recorded in
`docs/adr/0002-async-end-to-end.md` because it is visible in every
application-layer signature and is therefore a one-way door.
"""

from api.domain.ports.clock import Clock
from api.domain.ports.health import HealthProbe, HealthStatus
from api.domain.ports.repositories import (
    IncidentRepository,
    MachineRepository,
    PredictionRepository,
    TelemetryRepository,
)
from api.domain.ports.unit_of_work import UnitOfWork, UnitOfWorkFactory

__all__ = [
    "Clock",
    "HealthProbe",
    "HealthStatus",
    "IncidentRepository",
    "MachineRepository",
    "PredictionRepository",
    "TelemetryRepository",
    "UnitOfWork",
    "UnitOfWorkFactory",
]
