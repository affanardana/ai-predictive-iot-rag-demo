"""Persistence models.

Deliberately separate from the domain entities. `CODING_STANDARDS.md` forbids
the domain from importing SQLAlchemy, and imperative mapping (binding domain
classes directly to tables) would force domain entities to satisfy SQLAlchemy's
instrumentation requirements -- which conflicts with the immutable-value-object
rule and reads as implicit.

The cost is duplicated field declarations. What keeps that duplication honest
is `alembic check` in CI, which fails the build when models and migrations
drift apart.
"""

from api.infrastructure.persistence.sql.models.incident import IncidentModel
from api.infrastructure.persistence.sql.models.knowledge import (
    KnowledgeChunkModel,
    KnowledgeDocumentModel,
)
from api.infrastructure.persistence.sql.models.machine import MachineModel
from api.infrastructure.persistence.sql.models.prediction import PredictionModel
from api.infrastructure.persistence.sql.models.simulation_run import SimulationRunModel
from api.infrastructure.persistence.sql.models.telemetry import TelemetryModel

__all__ = [
    "IncidentModel",
    "KnowledgeChunkModel",
    "KnowledgeDocumentModel",
    "MachineModel",
    "PredictionModel",
    "SimulationRunModel",
    "TelemetryModel",
]
