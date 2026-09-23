"""
Alembic environment.

The URL comes from application settings rather than `alembic.ini`, so it is
configured in exactly one place and never committed.

Migrations run **synchronously**. The `psycopg` dialect supports both sync and
async through the same URL, so there is no need for the async template's
`run_sync` plumbing here -- Alembic is a batch tool, not a request path, and a
sync engine keeps `env.py` readable.
"""

from __future__ import annotations

from logging.config import fileConfig

from alembic import context
from pydantic import ValidationError
from sqlalchemy import engine_from_config, pool

from api.infrastructure.config import Settings

# Importing the models registers their tables on `Base.metadata`. Without this
# import, autogenerate would see an empty schema and try to drop everything.
from api.infrastructure.persistence.sql import models  # noqa: F401
from api.infrastructure.persistence.sql.base import Base

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def _database_url() -> str:
    """Resolve the database URL from application settings.

    Reports a readable message rather than a pydantic traceback when nothing is
    configured. This is the first command a fresh checkout runs, so a wall of
    validation output that never mentions `.env` is a poor introduction.
    """
    try:
        return Settings().database_url
    except ValidationError as exc:
        raise SystemExit(
            "No DATABASE_URL is configured.\n"
            "Copy .env.example to .env and fill in your Supabase session-pooler "
            "URL (port 5432), or export DATABASE_URL in the environment.\n\n"
            f"{exc}"
        ) from exc


def run_migrations_offline() -> None:
    """Emit the SQL without connecting to a database."""
    context.configure(
        url=_database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=config.get_main_option("compare_type") == "true",
        compare_server_default=config.get_main_option("compare_server_default") == "true",
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Connect and apply the migrations."""
    section = config.get_section(config.config_ini_section, {})
    connectable = engine_from_config(
        section,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
        url=_database_url(),
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=config.get_main_option("compare_type") == "true",
            compare_server_default=config.get_main_option("compare_server_default") == "true",
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
