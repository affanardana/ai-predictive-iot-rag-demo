"""Domain value objects.

Value objects are immutable, validate their own invariants at construction,
and carry no identity -- two instances with equal fields are interchangeable.
"""

from api.domain.value_objects.chunk_match import ChunkMatch
from api.domain.value_objects.citation import Citation
from api.domain.value_objects.document_category import (
    PRD_DOCUMENT_CATEGORIES,
    DocumentCategory,
)
from api.domain.value_objects.evidence import Evidence, EvidenceKind
from api.domain.value_objects.failure_probability import FailureProbability
from api.domain.value_objects.incident_status import IncidentStatus
from api.domain.value_objects.incident_type import IncidentType
from api.domain.value_objects.machine_id import MachineId
from api.domain.value_objects.risk_level import IncidentSeverity, RiskLevel
from api.domain.value_objects.risk_thresholds import RiskThresholds
from api.domain.value_objects.sensor_reading import SensorReading
from api.domain.value_objects.text_line import TextLine
from api.domain.value_objects.time_window import SeriesResolution, TimeWindow

__all__ = [
    "PRD_DOCUMENT_CATEGORIES",
    "ChunkMatch",
    "Citation",
    "DocumentCategory",
    "Evidence",
    "EvidenceKind",
    "FailureProbability",
    "IncidentSeverity",
    "IncidentStatus",
    "IncidentType",
    "MachineId",
    "RiskLevel",
    "RiskThresholds",
    "SensorReading",
    "SeriesResolution",
    "TextLine",
    "TimeWindow",
]
