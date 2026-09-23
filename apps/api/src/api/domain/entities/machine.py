"""Machine entity."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from api.domain.timestamps import ensure_aware
from api.domain.value_objects.machine_id import MachineId


@dataclass
class Machine:
    """A monitored machine, as registered in the fleet.

    This holds registry data only. A machine's *current* condition --
    temperature, vibration, failure probability, risk level -- is assembled
    from telemetry and predictions, because those are separate records with
    their own history rather than attributes of the machine.
    """

    id: MachineId
    name: str
    registered_at: datetime

    def __post_init__(self) -> None:
        """Validate the timestamp."""
        ensure_aware(self.registered_at, "registered_at")

    def rename(self, name: str) -> None:
        """Change the display name."""
        if not name.strip():
            raise ValueError("Machine name cannot be blank.")
        self.name = name
