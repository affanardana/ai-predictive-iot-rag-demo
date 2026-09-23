"""Domain entities.

Entities have identity and a lifecycle; two entities with identical field
values but different identifiers are distinct things. Contrast with value
objects, which are interchangeable when their fields match.
"""

from api.domain.entities.incident import Incident
from api.domain.entities.machine import Machine
from api.domain.entities.prediction import Prediction
from api.domain.entities.telemetry import TelemetryRecord

__all__ = ["Incident", "Machine", "Prediction", "TelemetryRecord"]
