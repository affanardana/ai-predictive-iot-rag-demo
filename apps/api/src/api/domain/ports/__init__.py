"""Domain ports.

Interfaces the domain defines and infrastructure implements. Dependencies
point inward: infrastructure knows the domain, never the reverse.

Every method is `async`. That choice is recorded in
`docs/adr/0002-async-end-to-end.md` because it is visible in every
application-layer signature and is therefore a one-way door.
"""

from api.domain.ports.clock import Clock
from api.domain.ports.events import (
    EventKind,
    EventPublisher,
    EventSubscriber,
    MachineEvent,
    NullEventPublisher,
    StreamItem,
    StreamTick,
)
from api.domain.ports.health import HealthProbe, HealthStatus
from api.domain.ports.repositories import (
    IncidentRepository,
    MachineRepository,
    PredictionRepository,
    SimulationRunRepository,
    TelemetryRepository,
)
from api.domain.ports.unit_of_work import UnitOfWork, UnitOfWorkFactory

__all__ = [
    "Clock",
    "EventKind",
    "EventPublisher",
    "EventSubscriber",
    "HealthProbe",
    "HealthStatus",
    "IncidentRepository",
    "MachineEvent",
    "MachineRepository",
    "NullEventPublisher",
    "PredictionRepository",
    "SimulationRunRepository",
    "StreamItem",
    "StreamTick",
    "TelemetryRepository",
    "UnitOfWork",
    "UnitOfWorkFactory",
]
