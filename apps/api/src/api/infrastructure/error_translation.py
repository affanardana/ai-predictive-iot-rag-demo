"""Translate driver exceptions into domain storage errors.

CODING_STANDARDS.md requires errors to be translated between layers and
forbids leaking infrastructure exceptions to users. A SQLAlchemy or psycopg
exception can carry a full connection string in its message, so letting one
reach the API would leak credentials as well as break the abstraction.

Adapters wrap their queries in `translating_persistence_errors()`; the original
exception is logged through the redacting logger and the caller sees only a
`PersistenceError`.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy.exc import InterfaceError, OperationalError, SQLAlchemyError

from api.domain.errors import PersistenceError, RepositoryUnavailableError

logger = logging.getLogger(__name__)

#: Exceptions that mean "the database is not reachable" rather than "the query
#: was rejected". A readiness probe should treat the two differently: the first
#: is a dependency outage, the second is a bug.
_UNAVAILABILITY_EXCEPTIONS = (OperationalError, InterfaceError)


def translate(exc: SQLAlchemyError) -> PersistenceError:
    """Convert a driver exception into a domain storage error."""
    if isinstance(exc, _UNAVAILABILITY_EXCEPTIONS):
        return RepositoryUnavailableError("The database is not reachable.")
    return PersistenceError("A storage operation failed.")


@contextmanager
def translating_persistence_errors() -> Iterator[None]:
    """Translate any SQLAlchemy error raised inside the block.

    The original exception is logged (redacted) and chained with `from`, so it
    remains available in the traceback for debugging while the message a caller
    sees stays free of credentials.
    """
    try:
        yield
    except SQLAlchemyError as exc:
        logger.exception(
            "infrastructure.persistence.error",
            extra={"exception_type": type(exc).__name__},
        )
        raise translate(exc) from exc


def is_unavailable_error(exc: BaseException) -> bool:
    """Whether an exception indicates the database could not be reached."""
    return isinstance(exc, (RepositoryUnavailableError, *_UNAVAILABILITY_EXCEPTIONS))


__all__ = [
    "is_unavailable_error",
    "translate",
    "translating_persistence_errors",
]
