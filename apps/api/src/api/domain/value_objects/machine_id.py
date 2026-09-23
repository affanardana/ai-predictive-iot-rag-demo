"""Machine identifier value object."""

from __future__ import annotations

import re
from dataclasses import dataclass

from api.domain.errors import DomainValidationError

#: Deliberately strict: one to four uppercase letters followed by exactly three
#: digits, e.g. ``M003`` or ``PUMP012``. Validating at the boundary means the
#: rest of the system never has to defend against malformed identifiers.
MACHINE_ID_PATTERN = re.compile(r"^[A-Z]{1,4}\d{3}$")


@dataclass(frozen=True, slots=True)
class MachineId:
    """Stable identifier for a monitored machine."""

    value: str

    def __post_init__(self) -> None:
        """Validate the identifier format."""
        if not MACHINE_ID_PATTERN.match(self.value):
            raise DomainValidationError(
                f"Machine id '{self.value}' is malformed. "
                "Expected one to four uppercase letters followed by three digits, e.g. 'M003'."
            )

    def __str__(self) -> str:
        """Return the raw identifier."""
        return self.value
