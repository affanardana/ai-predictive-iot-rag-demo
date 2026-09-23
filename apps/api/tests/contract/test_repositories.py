"""The repository contract, run against both adapters.

Every contract class is paired with an adapter mixin supplying a unit-of-work
factory. If the in-memory double and the SQL implementation ever disagree about
behaviour, one of these classes fails.
"""

from __future__ import annotations

import pytest

from api.domain.ports.unit_of_work import UnitOfWorkFactory
from api.infrastructure.persistence.memory.unit_of_work import InMemoryUnitOfWorkFactory
from api.infrastructure.persistence.sql.unit_of_work import SqlUnitOfWorkFactory
from tests.contract.repository_contract import (
    IncidentRepositoryContract,
    MachineRepositoryContract,
    PredictionRepositoryContract,
    TelemetryRepositoryContract,
)


class InMemoryAdapterMixin:
    """Pairs a contract class with the in-memory adapter."""

    @pytest.fixture
    def uow_factory(self, in_memory_uow_factory: InMemoryUnitOfWorkFactory) -> UnitOfWorkFactory:
        """Return the in-memory factory."""
        return in_memory_uow_factory


class SqliteAdapterMixin:
    """Pairs a contract class with the SQLite-backed SQL adapter."""

    @pytest.fixture
    def uow_factory(self, sqlite_uow_factory: SqlUnitOfWorkFactory) -> UnitOfWorkFactory:
        """Return the SQLite-backed factory."""
        return sqlite_uow_factory


class TestInMemoryMachineRepository(InMemoryAdapterMixin, MachineRepositoryContract):
    """The in-memory machine repository satisfies the contract."""


class TestSqliteMachineRepository(SqliteAdapterMixin, MachineRepositoryContract):
    """The SQL machine repository satisfies the contract."""


class TestInMemoryTelemetryRepository(InMemoryAdapterMixin, TelemetryRepositoryContract):
    """The in-memory telemetry repository satisfies the contract."""


class TestSqliteTelemetryRepository(SqliteAdapterMixin, TelemetryRepositoryContract):
    """The SQL telemetry repository satisfies the contract."""


class TestInMemoryPredictionRepository(InMemoryAdapterMixin, PredictionRepositoryContract):
    """The in-memory prediction repository satisfies the contract."""


class TestSqlitePredictionRepository(SqliteAdapterMixin, PredictionRepositoryContract):
    """The SQL prediction repository satisfies the contract."""


class TestInMemoryIncidentRepository(InMemoryAdapterMixin, IncidentRepositoryContract):
    """The in-memory incident repository satisfies the contract."""


class TestSqliteIncidentRepository(SqliteAdapterMixin, IncidentRepositoryContract):
    """The SQL incident repository satisfies the contract."""
