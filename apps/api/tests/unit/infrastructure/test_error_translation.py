"""Driver exception translation."""

from __future__ import annotations

import pytest
from sqlalchemy.exc import InterfaceError, OperationalError, ProgrammingError, SQLAlchemyError

from api.domain.errors import PersistenceError, RepositoryUnavailableError
from api.infrastructure.error_translation import (
    is_unavailable_error,
    translate,
    translating_persistence_errors,
)

STATEMENT = "SELECT 1"


def _sqlalchemy_error(kind: type[SQLAlchemyError]) -> SQLAlchemyError:
    """Build the given SQLAlchemy error around a driver-level cause."""
    return kind(STATEMENT, {}, Exception("driver detail"))


@pytest.mark.parametrize("kind", [OperationalError, InterfaceError])
def test_unreachable_database_maps_to_unavailable(kind: type[SQLAlchemyError]) -> None:
    """Connection-level failures are distinguishable from query failures.

    A readiness probe treats the two differently: an outage is a dependency
    problem, whereas a rejected query is a bug.
    """
    translated = translate(_sqlalchemy_error(kind))

    assert isinstance(translated, RepositoryUnavailableError)
    assert is_unavailable_error(translated)


@pytest.mark.parametrize("kind", [ProgrammingError])
def test_other_driver_errors_map_to_generic_persistence_errors(
    kind: type[SQLAlchemyError],
) -> None:
    """Query-level failures are not reported as outages."""
    translated = translate(_sqlalchemy_error(kind))

    assert isinstance(translated, PersistenceError)
    assert not isinstance(translated, RepositoryUnavailableError)


def test_translation_preserves_the_cause_for_debugging() -> None:
    """The original exception stays chained in the traceback.

    It is logged rather than shown, so a developer can still reach the driver
    detail while the message a caller sees stays free of credentials.
    """
    with (
        pytest.raises(PersistenceError) as caught,
        translating_persistence_errors(),
    ):
        raise _sqlalchemy_error(OperationalError)

    assert caught.value.__cause__ is not None


def test_non_driver_exceptions_pass_through_untouched() -> None:
    """Only driver errors are translated.

    A domain error raised inside the block is a legitimate signal and must not
    be rewritten into a storage failure.
    """
    with (
        pytest.raises(ValueError, match="domain problem"),
        translating_persistence_errors(),
    ):
        raise ValueError("domain problem")


def test_successful_blocks_are_unaffected() -> None:
    """The wrapper is transparent when nothing goes wrong."""
    with translating_persistence_errors():
        result = 1 + 1

    assert result == 2


def test_domain_errors_are_not_reported_as_unavailable() -> None:
    """A domain failure is never mistaken for an infrastructure outage."""
    assert not is_unavailable_error(PersistenceError("storage failed"))
    assert not is_unavailable_error(ValueError("unrelated"))
