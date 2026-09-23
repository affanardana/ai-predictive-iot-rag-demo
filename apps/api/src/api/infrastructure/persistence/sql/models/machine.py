"""Persistence model for the machine registry."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column

from api.infrastructure.persistence.sql.base import Base

#: Matches the domain's `MachineId` pattern (up to four letters + three digits)
#: with headroom for a longer scheme later.
MACHINE_ID_MAX_LENGTH = 16
MACHINE_NAME_MAX_LENGTH = 128


class MachineModel(Base):
    """A registered machine."""

    __tablename__ = "machines"

    machine_id: Mapped[str] = mapped_column(String(MACHINE_ID_MAX_LENGTH), primary_key=True)
    name: Mapped[str] = mapped_column(String(MACHINE_NAME_MAX_LENGTH))
    registered_at: Mapped[datetime]
