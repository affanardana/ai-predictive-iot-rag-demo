"""The single table mapping domain failures onto HTTP statuses.

Keeping this in one place is what stops infrastructure vocabulary and driver
messages from reaching a client: every failure that crosses the boundary has to
be classified here, and anything unclassified becomes a generic 500.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from api.domain.errors import (
    ChunkTooLongError,
    DocumentContentConflictError,
    DomainError,
    DomainValidationError,
    EmbeddingModelMismatchError,
    IncidentNotFoundError,
    InsufficientTelemetryHistoryError,
    InvalidIncidentTransitionError,
    InvalidRunTransitionError,
    InvalidTelemetryError,
    KnowledgeDocumentNotFoundError,
    MachineNotFoundError,
    PersistenceError,
    PredictionUnavailableError,
    RepositoryUnavailableError,
    RetrievalUnavailableError,
    SimulationAlreadyRunningError,
    SimulationRunActiveError,
    SimulationRunNotFoundError,
    SimulationUnavailableError,
    TooManySimulationsError,
)


@dataclass(frozen=True, slots=True)
class ErrorMapping:
    """HTTP status and stable error code for one kind of failure."""

    status_code: int
    code: str


ERROR_CATALOG: Mapping[type[DomainError], ErrorMapping] = {
    MachineNotFoundError: ErrorMapping(404, "machine_not_found"),
    IncidentNotFoundError: ErrorMapping(404, "incident_not_found"),
    # A conflict rather than a validation error: the request was well-formed,
    # the incident's current state simply does not permit the move.
    InvalidIncidentTransitionError: ErrorMapping(409, "invalid_incident_transition"),
    SimulationRunNotFoundError: ErrorMapping(404, "simulation_run_not_found"),
    # A conflict, not a validation error: the request was fine, and the
    # machine's or the fleet's current state is what refuses it.
    SimulationAlreadyRunningError: ErrorMapping(409, "simulation_already_running"),
    TooManySimulationsError: ErrorMapping(409, "too_many_simulations"),
    SimulationRunActiveError: ErrorMapping(409, "simulation_active"),
    InvalidRunTransitionError: ErrorMapping(409, "invalid_run_transition"),
    # The simulator container is down or refused. 503 for the same reason
    # inference gets one: retryable, and not the caller's mistake.
    SimulationUnavailableError: ErrorMapping(503, "simulation_unavailable"),
    InvalidTelemetryError: ErrorMapping(422, "invalid_telemetry"),
    # The request was fine and the machine exists; its history is too short to
    # build a window. Mapped explicitly rather than left to the MRO walk, which
    # would otherwise report it as a 400 `domain_error` and tell a caller
    # nothing about what to do next.
    InsufficientTelemetryHistoryError: ErrorMapping(422, "insufficient_history"),
    DomainValidationError: ErrorMapping(422, "invalid_value"),
    KnowledgeDocumentNotFoundError: ErrorMapping(404, "document_not_found"),
    # A conflict rather than a validation error: the request was well formed and
    # the version exists, but rewriting it would change what its citations
    # resolve to. The caller's remedy is a version bump.
    DocumentContentConflictError: ErrorMapping(409, "document_content_conflict"),
    # A dependency outage for the same reason inference gets one: retryable, and
    # not the caller's mistake. Covers both the embedding and the reranking
    # service, since neither can carry out a retrieval on its own.
    RetrievalUnavailableError: ErrorMapping(503, "retrieval_unavailable"),
    # Not an outage: the service is up and running a model the corpus was not
    # embedded with. 503 because the deployment has to change, not the request.
    EmbeddingModelMismatchError: ErrorMapping(503, "embedding_model_mismatch"),
    # Should be unreachable -- the chunker caps chunk length -- so a 500 is the
    # honest answer: something is wrong with this service, not with the request.
    ChunkTooLongError: ErrorMapping(500, "chunk_too_long"),
    RepositoryUnavailableError: ErrorMapping(503, "repository_unavailable"),
    # A dependency outage, not our fault and not the caller's. 503 says the
    # request was fine and the system could not answer, which is what lets a
    # caller retry instead of treating it as a bad request.
    PredictionUnavailableError: ErrorMapping(503, "inference_unavailable"),
    PersistenceError: ErrorMapping(500, "persistence_error"),
    DomainError: ErrorMapping(400, "domain_error"),
}


def resolve(error: DomainError) -> ErrorMapping:
    """Find the mapping for `error`, walking its class hierarchy.

    Walking the MRO means a new subclass inherits its parent's status unless it
    registers its own, so adding a domain error cannot accidentally produce an
    unhandled exception.
    """
    for klass in type(error).__mro__:
        mapping = ERROR_CATALOG.get(klass)
        if mapping is not None:
            return mapping
    return ErrorMapping(500, "internal_error")
