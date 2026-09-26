"""The same contract, run against a real PostgreSQL database.

SQLite cannot faithfully emulate PostgreSQL, and the divergences land on this
project's riskiest code: `ON CONFLICT ... RETURNING`, real CHECK constraints,
`timestamptz` semantics, and the `to_timestamp(extract(epoch ...))` bucketing
expression. Those are exactly the paths the default offline run cannot verify,
so they are checked here.

Deselected by default (`addopts = "-m 'not postgres'"`). Run with:

    uv run pytest -m postgres
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from datetime import timedelta
from uuid import uuid4

import pytest
from sqlalchemy import text
from sqlalchemy.exc import DataError, IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine

from api.domain.ports.unit_of_work import UnitOfWorkFactory
from api.domain.services.similarity import cosine_similarity
from api.domain.value_objects.machine_id import MachineId
from api.domain.value_objects.time_window import Aggregation
from api.infrastructure.persistence.sql.base import Base
from api.infrastructure.persistence.sql.models import KnowledgeChunkModel, PredictionModel
from api.infrastructure.persistence.sql.session import create_session_factory
from api.infrastructure.persistence.sql.unit_of_work import SqlUnitOfWorkFactory
from tests.contract.repository_contract import (
    ChunkSearchContract,
    IncidentRepositoryContract,
    KnowledgeRepositoryContract,
    MachineRepositoryContract,
    PredictionRepositoryContract,
    TelemetryRepositoryContract,
)
from tests.support.factories import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_NOW,
    make_embedding,
    make_knowledge_chunk,
    make_knowledge_document,
    make_machine,
    make_telemetry,
)

pytestmark = pytest.mark.postgres

TEST_DATABASE_URL_ENV = "TEST_DATABASE_URL"


def _test_database_url() -> str:
    """Return the configured test database URL.

    Fails rather than skips when it is missing. Reaching this function means the
    `postgres` tier was explicitly selected, so a skip would report green while
    having verified nothing -- this tier exists to cover what SQLite cannot, and
    silently not running it is worse than not having it.
    """
    url = os.environ.get(TEST_DATABASE_URL_ENV)
    if not url:
        pytest.fail(
            f"{TEST_DATABASE_URL_ENV} is not set, but the postgres tier was "
            "selected. Set it to a session-pooler URL (port 5432), or deselect "
            "the tier with `-m 'not postgres'`."
        )
    return url


async def _assert_tables_landed_in(engine: AsyncEngine, schema: str) -> None:
    """Fail if the schema rewrite did not take effect.

    Isolation that silently stops working is worse than none: the tests keep
    passing while writing to the real tables. This turns that into an immediate
    failure, before any test has had a chance to run.

    `to_regclass` is asked directly, with an explicit schema-qualified name,
    because `text()` statements are not rewritten by the translate map.
    """
    async with engine.connect() as connection:
        for table in Base.metadata.sorted_tables:
            result = await connection.execute(
                text("SELECT to_regclass(:qualified)"),
                {"qualified": f'"{schema}"."{table.name}"'},
            )
            if result.scalar() is None:
                pytest.fail(
                    f"Table '{table.name}' was not created in '{schema}', so the "
                    "schema translate map had no effect and these tests would run "
                    "against the real database. Refusing to continue."
                )


@pytest.fixture
async def postgres_engine() -> AsyncIterator[AsyncEngine]:
    """An engine scoped to a throwaway schema.

    Isolation uses `schema_translate_map`, which rewrites schema names while SQL
    is compiled. The original approach set `search_path` through a connection
    `options` parameter, and **that does not work here**: Supabase's pooler
    silently drops the parameter, leaving every session on `public`. Queries
    simply ran against the real tables, and the development database was
    polluted before anyone noticed. The translate map needs no connection-level
    state, so there is nothing for a pooler to discard.

    The schema is dropped on teardown, including when a test fails.
    """
    base_url = _test_database_url()
    schema = f"test_{uuid4().hex[:12]}"

    admin_engine = create_async_engine(base_url, pool_pre_ping=True)
    async with admin_engine.begin() as connection:
        await connection.execute(text(f'CREATE SCHEMA "{schema}"'))

    scoped_engine = create_async_engine(base_url, pool_pre_ping=True).execution_options(
        schema_translate_map={None: schema}
    )
    try:
        async with scoped_engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)

        await _assert_tables_landed_in(scoped_engine, schema)

        yield scoped_engine
    finally:
        await scoped_engine.dispose()
        async with admin_engine.begin() as connection:
            await connection.execute(text(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))
        await admin_engine.dispose()


@pytest.fixture
def postgres_uow_factory(postgres_engine: AsyncEngine) -> UnitOfWorkFactory:
    """A unit-of-work factory over the throwaway schema."""
    return SqlUnitOfWorkFactory(postgres_engine)


@pytest.fixture
async def postgres_session(postgres_engine: AsyncEngine) -> AsyncIterator[AsyncSession]:
    """A raw session, for assertions that need to bypass the repositories."""
    session_factory = create_session_factory(postgres_engine)
    async with session_factory() as session:
        yield session


class TestPostgresMachineRepository(MachineRepositoryContract):
    """The machine repository satisfies the contract on PostgreSQL."""

    @pytest.fixture
    def uow_factory(self, postgres_uow_factory: UnitOfWorkFactory) -> UnitOfWorkFactory:
        """Return the PostgreSQL-backed factory."""
        return postgres_uow_factory


class TestPostgresTelemetryRepository(TelemetryRepositoryContract):
    """The telemetry repository satisfies the contract on PostgreSQL.

    This is where the `ON CONFLICT ... RETURNING` idempotency path and the
    epoch-bucketing expression are genuinely exercised rather than approximated.
    """

    @pytest.fixture
    def uow_factory(self, postgres_uow_factory: UnitOfWorkFactory) -> UnitOfWorkFactory:
        """Return the PostgreSQL-backed factory."""
        return postgres_uow_factory


class TestPostgresPredictionRepository(PredictionRepositoryContract):
    """The prediction repository satisfies the contract on PostgreSQL."""

    @pytest.fixture
    def uow_factory(self, postgres_uow_factory: UnitOfWorkFactory) -> UnitOfWorkFactory:
        """Return the PostgreSQL-backed factory."""
        return postgres_uow_factory


class TestPostgresIncidentRepository(IncidentRepositoryContract):
    """The incident repository satisfies the contract on PostgreSQL."""

    @pytest.fixture
    def uow_factory(self, postgres_uow_factory: UnitOfWorkFactory) -> UnitOfWorkFactory:
        """Return the PostgreSQL-backed factory."""
        return postgres_uow_factory


async def test_bucket_keys_are_timezone_aware(
    postgres_uow_factory: UnitOfWorkFactory,
) -> None:
    """The PostgreSQL bucketing expression yields aware instants.

    Asserted separately from the contract because the type of the bucket key is
    precisely what differs between dialects: PostgreSQL returns a `timestamptz`,
    while SQLite returns a naive string that the mapper has to reconcile.
    """
    async with postgres_uow_factory() as uow:
        await uow.machines.add(make_machine("M001"))
        await uow.telemetry.add_many_idempotent(
            [make_telemetry(event_id="evt-1", recorded_at=DEFAULT_NOW)]
        )
        buckets = await uow.telemetry.window_bucketed(
            MachineId("M001"),
            DEFAULT_NOW - timedelta(minutes=5),
            DEFAULT_NOW + timedelta(minutes=5),
            bucket_seconds=300,
            aggregation=Aggregation.MEAN,
        )

    assert len(buckets) == 1
    bucket_start = buckets[0][0]
    assert bucket_start.tzinfo is not None
    assert bucket_start.utcoffset() is not None


async def _seed_machine(uow_factory: UnitOfWorkFactory) -> None:
    """Register M001 and commit, so a raw insert has a valid foreign key."""
    async with uow_factory() as uow:
        await uow.machines.add(make_machine("M001"))
        await uow.commit()


async def test_check_constraint_rejects_an_impossible_probability(
    postgres_uow_factory: UnitOfWorkFactory,
    postgres_session: AsyncSession,
) -> None:
    """The database refuses a probability outside [0, 1].

    The domain clamps such values before they reach storage, so this asserts the
    second line of defence: a CHECK constraint that still holds if some future
    code path bypasses the domain. A raw `IntegrityError` is expected rather
    than a `PersistenceError`, because this session deliberately skips the
    translating wrapper the repositories use.
    """
    await _seed_machine(postgres_uow_factory)
    postgres_session.add(
        PredictionModel(
            prediction_id="pred-invalid",
            machine_id="M001",
            predicted_at=DEFAULT_NOW,
            failure_probability=1.5,
            risk_level="CRITICAL",
            horizon_seconds=3600,
            model_version="lstm-v1",
        )
    )

    with pytest.raises(IntegrityError):
        await postgres_session.flush()

    await postgres_session.rollback()


async def test_check_constraint_rejects_an_unknown_risk_level(
    postgres_uow_factory: UnitOfWorkFactory,
    postgres_session: AsyncSession,
) -> None:
    """The risk level column only accepts values the domain enum defines.

    The allowed values are generated from `RiskLevel` at import time, so this
    also confirms that generation produced a constraint PostgreSQL accepts.
    """
    await _seed_machine(postgres_uow_factory)
    postgres_session.add(
        PredictionModel(
            prediction_id="pred-unknown-risk",
            machine_id="M001",
            predicted_at=DEFAULT_NOW,
            failure_probability=0.5,
            risk_level="CATASTROPHIC",
            horizon_seconds=3600,
            model_version="lstm-v1",
        )
    )

    with pytest.raises(IntegrityError):
        await postgres_session.flush()

    await postgres_session.rollback()


class TestPostgresKnowledgeRepository(KnowledgeRepositoryContract):
    """The knowledge repository satisfies the contract on PostgreSQL."""

    @pytest.fixture
    def uow_factory(self, postgres_uow_factory: UnitOfWorkFactory) -> UnitOfWorkFactory:
        """Return the PostgreSQL-backed factory."""
        return postgres_uow_factory


class TestPostgresChunkSearch(ChunkSearchContract):
    """The PostgreSQL adapter ranks vectors with pgvector."""

    @pytest.fixture
    def uow_factory(self, postgres_uow_factory: UnitOfWorkFactory) -> UnitOfWorkFactory:
        """Return the PostgreSQL-backed factory."""
        return postgres_uow_factory


async def test_pgvector_cosine_agrees_with_the_domain(
    postgres_uow_factory: UnitOfWorkFactory,
) -> None:
    """`<=>` and `cosine_similarity` return the same similarity.

    The in-memory adapter ranks with the domain's pure-Python cosine; production
    ranks in SQL. Two implementations of one formula drift, so this asserts them
    against each other rather than trusting that they agree -- the same move as
    `test_open_counts_agrees_with_is_open`.
    """
    query = make_embedding(1.0, 0.5, 0.25)
    vectors = [
        make_embedding(1.0),
        make_embedding(0.0, 1.0),
        make_embedding(1.0, 1.0, 1.0),
        make_embedding(-1.0, 0.5),
    ]
    async with postgres_uow_factory() as uow:
        document = make_knowledge_document(is_active=True)
        await uow.knowledge.add_document(document)
        await uow.knowledge.add_chunks(
            [
                make_knowledge_chunk(
                    document_id=document.document_id,
                    chunk_index=index,
                    content=f"Passage {index}.",
                    embedding=vector,
                )
                for index, vector in enumerate(vectors)
            ]
        )

    async with postgres_uow_factory() as uow:
        matches = await uow.knowledge.similar_chunks(
            query,
            embedding_model=DEFAULT_EMBEDDING_MODEL,
            limit=len(vectors),
        )

    assert len(matches) == len(vectors)
    for match in matches:
        assert match.score == pytest.approx(
            cosine_similarity(query, match.chunk.embedding),
            # `vector` is single precision where the domain computes in double,
            # so the two agree to about float32 epsilon rather than exactly.
            abs=1e-6,
        )


async def test_the_vector_column_is_a_pgvector_type(
    postgres_uow_factory: UnitOfWorkFactory,
    postgres_session: AsyncSession,
) -> None:
    """A vector of the wrong width is refused by the database.

    Behavioural rather than introspective: it asserts that the extension is
    installed and the column really is `vector(384)`, without asking the
    catalogue what type it thinks it has. A JSON fallback would accept this row
    happily, which is exactly the failure being ruled out.

    `DataError`, not `IntegrityError` as the CHECK-constraint tests above use:
    the width is checked by pgvector's own input function, so the server reports
    a data exception rather than a constraint violation. This assertion was
    `IntegrityError` when it was written, and CI was the first thing to run it --
    the `postgres` tier is opt-in, so the offline gate deselects the only test
    that could have caught it.
    """
    async with postgres_uow_factory() as uow:
        document = make_knowledge_document(is_active=True)
        await uow.knowledge.add_document(document)
        await uow.commit()

    postgres_session.add(
        KnowledgeChunkModel(
            chunk_id="chunk-wrong-width",
            document_id=document.document_id,
            chunk_index=99,
            section="",
            page=1,
            content="Too short a vector.",
            char_count=20,
            embedding=[0.0, 0.0, 0.0],
            embedding_model=DEFAULT_EMBEDDING_MODEL,
        )
    )

    with pytest.raises(DataError):
        await postgres_session.flush()

    await postgres_session.rollback()
