"""A reproducible simulation run.

A session names everything that determines the output: which machines, which
scenario, which seed, when the run starts, how often it samples, and how long it
lasts. Two sessions with the same values produce identical telemetry, which is
what makes a demonstration reproducible and a dataset regenerable.
"""

from __future__ import annotations

import random
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta

from simulator.domain.errors import SimulationValidationError
from simulator.domain.machine import MachineProfile, build_profile
from simulator.domain.scenario import Scenario
from simulator.domain.seeding import derive_seed
from simulator.domain.timestamps import ensure_aware

#: One reading per minute, matching the dataset definition in `MASTERPLAN.md` §6.
DEFAULT_SAMPLE_INTERVAL = timedelta(minutes=1)

#: The canonical demonstration runs for about ten minutes of simulated time
#: (PRD §12), long enough for a machine to visibly go wrong.
DEFAULT_DURATION = timedelta(minutes=10)

#: Bounded so a session id and an event id both fit the `String(64)` column in
#: the Phase 1 schema. The arithmetic: 32 for the session, one separator, 16 for
#: the machine (its own maximum), one separator, eight for a zero-padded index.
MAX_SESSION_ID_LENGTH = 32
MAX_EVENT_ID_LENGTH = 64

#: Eight digits of index covers ~190 years at one reading per minute.
_INDEX_DIGITS = 8


@dataclass(frozen=True, slots=True)
class SimulationSession:
    """Everything that determines a run's output."""

    session_id: str
    scenario: Scenario
    seed: int
    started_at: datetime
    sample_interval: timedelta
    duration: timedelta
    machines: tuple[MachineProfile, ...]

    def __post_init__(self) -> None:
        """Validate the identifiers and the run's shape."""
        if not self.session_id.strip():
            raise SimulationValidationError("session_id must not be blank.")
        if len(self.session_id) > MAX_SESSION_ID_LENGTH:
            raise SimulationValidationError(
                f"session_id '{self.session_id}' exceeds {MAX_SESSION_ID_LENGTH} "
                "characters, which would push event ids past the 64-character column."
            )
        if self.sample_interval <= timedelta(0):
            raise SimulationValidationError("sample_interval must be positive.")
        if self.duration <= timedelta(0):
            raise SimulationValidationError("duration must be positive.")
        if not self.machines:
            raise SimulationValidationError("A session must include at least one machine.")
        ensure_aware(self.started_at, "started_at")

        seen = {profile.machine_id for profile in self.machines}
        if len(seen) != len(self.machines):
            raise SimulationValidationError("A session must not contain duplicate machines.")

    @property
    def tick_count(self) -> int:
        """How many samples the run produces per machine.

        The final instant is exclusive: 30 days at one reading per minute is
        43,200 records, matching the masterplan's dataset arithmetic.
        """
        return int(self.duration / self.sample_interval)

    def event_id(self, machine_id: str, index: int) -> str:
        """Return the idempotency key for one sample.

        Derived from the session rather than drawn at random, so re-running a
        session regenerates the same event ids — which is what makes a
        redelivered message recognisable as a duplicate rather than as new data.
        """
        return f"{self.session_id}-{machine_id}-{index:0{_INDEX_DIGITS}d}"

    @classmethod
    def create(
        cls,
        *,
        machine_ids: Iterable[str],
        scenario: Scenario,
        seed: int,
        duration: timedelta = DEFAULT_DURATION,
        started_at: datetime,
        sample_interval: timedelta = DEFAULT_SAMPLE_INTERVAL,
        session_id: str | None = None,
    ) -> SimulationSession:
        """Build a session, deriving a machine fleet from the seed.

        Each machine's nominal operating point is drawn from its own generator,
        seeded on `(seed, machine_id)`, so adding a machine to a later run does
        not change the machines that were already there.
        """
        machines = tuple(
            build_profile(machine_id, random.Random(derive_seed(seed, machine_id)))
            for machine_id in machine_ids
        )
        return cls(
            # `is not None` rather than a truthiness check: an explicitly empty
            # id is a caller mistake, and silently substituting the default
            # would hide it. `__post_init__` rejects it instead.
            session_id=(
                session_id if session_id is not None else default_session_id(scenario, seed)
            ),
            scenario=scenario,
            seed=seed,
            started_at=started_at,
            sample_interval=sample_interval,
            duration=duration,
            machines=machines,
        )


def default_session_id(scenario: Scenario, seed: int) -> str:
    """Return a readable, bounded session id.

    Readable because a session id appears in every event id and in log output,
    so `sim-bd-1234` is far more useful to a person than a UUID — and bounded
    because it has to fit the column alongside a machine id and an index.
    """
    return f"sim-{_abbreviate(scenario)}-{seed}"


def _abbreviate(scenario: Scenario) -> str:
    """Return a short tag for a scenario, for use in identifiers."""
    return "".join(part[0] for part in scenario.value.split("_")).lower()
