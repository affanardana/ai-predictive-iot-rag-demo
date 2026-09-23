"""Declarative base and schema conventions.

Constraint naming is configured explicitly. Without it, PostgreSQL invents
names for indexes and constraints, and Alembic then cannot reliably drop or
alter them by name in a later migration -- migrations that work on a fresh
database and fail on an existing one.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, ClassVar

from sqlalchemy import TIMESTAMP, MetaData
from sqlalchemy.ext.asyncio import AsyncAttrs
from sqlalchemy.orm import DeclarativeBase

NAMING_CONVENTION = {
    "ix": "ix_%(table_name)s_%(column_0_N_name)s",
    "uq": "uq_%(table_name)s_%(column_0_N_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(AsyncAttrs, DeclarativeBase):
    """Base class for every persistence model.

    `AsyncAttrs` allows `await obj.awaitable_attrs.relationship` for lazy loads
    inside async code. Timestamps default to `TIMESTAMP(timezone=True)` so
    every stored instant carries its zone, matching the domain's insistence on
    aware datetimes.
    """

    metadata = MetaData(naming_convention=NAMING_CONVENTION)

    #: Annotated `ClassVar[Any]` because SQLAlchemy already declares this a
    #: ClassVar over a union-keyed dict that a narrower annotation would fail to
    #: satisfy. `Any` costs nothing here -- SQLAlchemy reads the mapping
    #: reflectively -- and the `ClassVar` marker is what stops ruff treating a
    #: mutable dict as an accidental shared default.
    type_annotation_map: ClassVar[Any] = {
        datetime: TIMESTAMP(timezone=True),
    }
