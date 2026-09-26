"""Domain error hierarchy.

Two distinct kinds of failure are modelled here, because callers handle them
differently:

* `DomainValidationError` -- a value is malformed at construction time. It also
  inherits `ValueError`, so Pydantic surfaces it as a validation error at the
  presentation boundary with no extra wiring.
* Everything else under `DomainError` -- a business rule or state conflict. The
  presentation layer maps these onto specific HTTP statuses.

Infrastructure exceptions must be translated into one of these before they
reach an upper layer; see `api.infrastructure.error_translation`.
"""

from __future__ import annotations


class DomainError(Exception):
    """Base class for every error raised by the domain layer."""


class DomainValidationError(DomainError, ValueError):
    """A value object was constructed with a value that violates its invariant."""


class MachineNotFoundError(DomainError):
    """Raised when a machine identifier does not resolve to a known machine."""

    def __init__(self, machine_id: str) -> None:
        self.machine_id = machine_id
        super().__init__(f"Machine '{machine_id}' was not found.")


class IncidentNotFoundError(DomainError):
    """Raised when an incident identifier does not resolve to a known incident."""

    def __init__(self, incident_id: str) -> None:
        self.incident_id = incident_id
        super().__init__(f"Incident '{incident_id}' was not found.")


class InvalidTelemetryError(DomainError):
    """Raised when a telemetry record violates a domain invariant."""


class PredictionUnavailableError(DomainError):
    """Inference could not be performed.

    Raised when the model service is unreachable, refuses the request, or
    answers with something that is not a probability. It exists so a dependency
    outage is reported as a dependency outage rather than as an internal error:
    the request was well formed, the machine exists, and the system could not
    answer.
    """


class SimulationRunNotFoundError(DomainError):
    """No run carries this identifier."""

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        super().__init__(f"Simulation run '{session_id}' was not found.")


class SimulationUnavailableError(DomainError):
    """The simulator service could not be reached or refused the request.

    A dependency outage, not a bad request: the run was well formed, the machine
    exists, and the service that would execute it did not answer. Reported
    distinctly so a caller can retry rather than hunt for a mistake in their
    configuration -- the same reasoning as `PredictionUnavailableError`.
    """


class SimulationAlreadyRunningError(DomainError):
    """The machine already has an active run.

    Two runs on one machine would interleave readings from two scenarios, and the
    risk band they produced would describe neither. A conflict rather than a
    validation error: the request was fine, the machine's current state is what
    refuses it.
    """

    def __init__(self, machine_id: str, session_id: str) -> None:
        self.machine_id = machine_id
        self.session_id = session_id
        super().__init__(f"{machine_id} already has an active run ('{session_id}'). Stop it first.")


class TooManySimulationsError(DomainError):
    """The concurrent-run ceiling has been reached.

    Enforced because runs cost CPU on a box that shares one core between the
    API, the orchestrator and the model service, and because the endpoint that
    starts them is reachable by anyone who finds the hostname.
    """

    def __init__(self, limit: int) -> None:
        self.limit = limit
        super().__init__(f"At most {limit} simulations may run at once, and that many are active.")


class SimulationRunActiveError(DomainError):
    """The run is still going, and the caller asked for something terminal."""

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        super().__init__(f"Simulation run '{session_id}' is still active.")


class InvalidRunTransitionError(DomainError):
    """A run's lifecycle does not permit the requested move.

    A conflict rather than a validation error: the request was well formed, the
    run's current state is what refuses it -- the same distinction
    `InvalidIncidentTransitionError` draws.
    """

    def __init__(self, current: str, requested: str) -> None:
        self.current = current
        self.requested = requested
        super().__init__(f"A run cannot move from {current} to {requested}.")


class InsufficientTelemetryHistoryError(DomainError):
    """There are too few stored readings to build a prediction window.

    Not a `DomainValidationError`: the request was well formed and the machine
    exists. What is lacking is the machine's own history, which is a state
    conflict rather than a malformed value.

    The counts go in the message because the error envelope's `details` field is
    `dict[str, str]` and the domain-error handler passes none -- the same
    phrasing the inference service uses for the same condition.
    """

    def __init__(self, have: int, need: int) -> None:
        self.have = have
        self.need = need
        super().__init__(f"A prediction needs {need} readings of history and {have} arrived.")


class InvalidIncidentTransitionError(DomainError):
    """Raised when an incident status transition is not permitted."""

    def __init__(self, current: str, requested: str) -> None:
        self.current = current
        self.requested = requested
        super().__init__(f"An incident in status '{current}' cannot transition to '{requested}'.")


class KnowledgeDocumentNotFoundError(DomainError):
    """No document carries this key and version.

    `version=None` means "no version of this key at all", which is what
    withdrawing a document asks for.
    """

    def __init__(self, document_key: str, version: str | None) -> None:
        self.document_key = document_key
        self.version = version
        named = f"version '{version}'" if version is not None else "any version"
        super().__init__(f"Document '{document_key}' was not found ({named}).")


class DocumentContentConflictError(DomainError):
    """This key and version already exist, with different content.

    Refused rather than replaced: a version is a promise that version N is this
    text, and silently re-writing it would make every citation recorded against
    it resolve to something the reader never saw.
    """

    def __init__(self, document_key: str, version: str) -> None:
        self.document_key = document_key
        self.version = version
        super().__init__(
            f"Document '{document_key}' version '{version}' already exists with different "
            "content. Bump the version, or re-ingest with replace enabled."
        )


class RetrievalUnavailableError(DomainError):
    """Embedding or reranking could not be performed.

    A dependency outage rather than a bad request: the query was well formed
    and the corpus is stored, but the service that turns text into vectors did
    not answer. Reported distinctly so a caller can retry instead of hunting
    for a mistake in the request -- the same reasoning as
    `PredictionUnavailableError`.
    """


class EmbeddingModelMismatchError(DomainError):
    """The embedding service is not running the model the index was built with.

    Refused rather than tolerated: vectors from two different models are not
    comparable, so ranking across them would return plausible-looking nonsense
    with nothing logged anywhere.
    """

    def __init__(self, expected: str, actual: str) -> None:
        self.expected = expected
        self.actual = actual
        super().__init__(
            f"The embedding service is running '{actual}', but the corpus was embedded "
            f"with '{expected}'. Re-embed the corpus or point the service at the right model."
        )


class ChunkTooLongError(DomainError):
    """A chunk is longer than the embedding model will read.

    Should be unreachable: the chunker caps chunk length. It is raised anyway,
    because the embedder's failure mode is to truncate silently, and a chunk
    embedded without its tail would be retrieved by words that are not in it.
    """

    def __init__(self, actual: int, limit: int) -> None:
        self.actual = actual
        self.limit = limit
        super().__init__(
            f"A chunk of {actual} characters exceeds the {limit}-character embedding limit."
        )


# ---------------------------------------------------------------------------
# Storage contract failures
#
# These live in the domain, not in application or infrastructure, because the
# domain owns the repository ports. If the domain defines the storage contract,
# it also defines how that contract fails -- which keeps infrastructure able to
# raise these without importing a sibling layer (the layer contract forbids
# infrastructure and application from depending on each other).
# ---------------------------------------------------------------------------


class PersistenceError(DomainError):
    """A storage operation failed.

    Carries a human-readable summary only. The original driver exception is
    deliberately not chained onto this message, because driver errors routinely
    embed connection strings and credentials; adapters log the original
    separately, through the redacting logger.
    """


class RepositoryUnavailableError(PersistenceError):
    """The backing store could not be reached, as opposed to rejecting a query."""
