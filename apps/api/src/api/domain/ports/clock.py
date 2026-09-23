"""Clock port.

Time is a dependency. Injecting it means tests can pin "now" to a fixed
instant without monkeypatching `datetime` globally, which is what makes
window-boundary and age calculations deterministic.
"""

from __future__ import annotations

from datetime import datetime
from typing import Protocol


class Clock(Protocol):
    """Supplies the current time."""

    def now(self) -> datetime:
        """Return the current instant, always timezone-aware."""
        ...
