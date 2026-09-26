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
    ChunkSearchContract,
    IncidentRepositoryContract,
    KnowledgeRepositoryContract,
    MachineRepositoryContract,
    PredictionRepositoryContract,
    SimulationRunRepositoryContract,
    TelemetryRepositoryContract,
)
from tests.support.factories import DEFAULT_EMBEDDING_MODEL, make_embedding


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


class TestInMemorySimulationRunRepository(InMemoryAdapterMixin, SimulationRunRepositoryContract):
    """The in-memory simulation run repository satisfies the contract."""


class TestSqliteSimulationRunRepository(SqliteAdapterMixin, SimulationRunRepositoryContract):
    """The SQL simulation run repository satisfies the contract."""


class TestInMemoryKnowledgeRepository(InMemoryAdapterMixin, KnowledgeRepositoryContract):
    """The in-memory knowledge repository satisfies the contract."""


class TestSqliteKnowledgeRepository(SqliteAdapterMixin, KnowledgeRepositoryContract):
    """The SQL knowledge repository satisfies the contract on SQLite."""


class TestInMemoryChunkSearch(InMemoryAdapterMixin, ChunkSearchContract):
    """The in-memory adapter ranks vectors with the domain's cosine.

    It is the only offline adapter that can rank at all, which is why it earns
    its place as the second implementation the search contract is asserted
    against.
    """


async def test_vector_search_is_refused_on_sqlite(sqlite_uow_factory: SqlUnitOfWorkFactory) -> None:
    """SQLite stores embeddings as JSON, so it cannot rank them.

    Asserted rather than left implicit. An adapter that returned nothing, or
    that quietly fell back to insertion order, would look like a passing search
    with no way to tell from the result.
    """
    async with sqlite_uow_factory() as uow:
        with pytest.raises(NotImplementedError, match="PostgreSQL"):
            await uow.knowledge.similar_chunks(
                make_embedding(1.0),
                embedding_model=DEFAULT_EMBEDDING_MODEL,
                limit=5,
            )
