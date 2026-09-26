"""Domain services.

Business rules that do not naturally belong to a single entity or value
object. Each service is constructed with its configuration, so the rules stay
testable without any framework or infrastructure present.
"""

from api.domain.services.evidence_sufficiency import Sufficiency, assess_sufficiency
from api.domain.services.incident_policy import DefaultIncidentPolicy, IncidentPolicy
from api.domain.services.knowledge_chunking import TextChunk, chunk_document
from api.domain.services.risk_level_classifier import RiskLevelClassifier
from api.domain.services.similarity import cosine_similarity

__all__ = [
    "DefaultIncidentPolicy",
    "IncidentPolicy",
    "RiskLevelClassifier",
    "Sufficiency",
    "TextChunk",
    "assess_sufficiency",
    "chunk_document",
    "cosine_similarity",
]
