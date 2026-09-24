"""The generation plan, on disk — and why it is tiny.

A fleet of a hundred machines running thirty days has a few thousand lives. The
obvious plan file lists them all. It does not need to: every life is a pure
function of four values, because each machine's timeline is drawn from its own
stream seeded on ``(base_seed, machine_id)`` and nothing consults another
machine's draws. So the plan stores the four values and re-derives the rest.

That is not only smaller. It means the plan cannot drift from the generator,
because the plan *is* the generator's input rather than a transcription of its
output — a transcription is one more thing that can be stale, and a stale plan
would mis-attribute every row in the dataset while looking perfectly valid.

`ml.dataset.artifacts` verifies the re-derived plan against the file's own
session ids before trusting any of it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from ml.dataset.errors import DatasetError
from ml.dataset.lives import (
    DEFAULT_TIMELINE,
    MachinePlan,
    Regime,
    plan_fleet,
)

PLAN_FILENAME = "plan.json"
PLAN_VERSION = 1


@dataclass(frozen=True, slots=True)
class PlanDocument:
    """Everything needed to rebuild a fleet."""

    seed: int
    started_at: datetime
    timeline_minutes: int
    main_machines: tuple[str, ...]
    shift_machines: tuple[str, ...]

    @property
    def machine_count(self) -> int:
        """Return how many machines the plan describes."""
        return len(self.main_machines) + len(self.shift_machines)

    def build(self) -> tuple[MachinePlan, ...]:
        """Rebuild every machine's timeline, in the order generation used."""
        timeline = timedelta(minutes=self.timeline_minutes)
        return plan_fleet(
            self.main_machines,
            base_seed=self.seed,
            regime=Regime.MAIN,
            started_at=self.started_at,
            timeline=timeline,
        ) + plan_fleet(
            self.shift_machines,
            base_seed=self.seed,
            regime=Regime.SHIFT,
            started_at=self.started_at,
            timeline=timeline,
        )

    def regimes(self) -> dict[str, Regime]:
        """Return which duration regime each machine belongs to."""
        return {
            **dict.fromkeys(self.main_machines, Regime.MAIN),
            **dict.fromkeys(self.shift_machines, Regime.SHIFT),
        }


def document_for(
    *,
    seed: int,
    started_at: datetime,
    main_machines: tuple[str, ...],
    shift_machines: tuple[str, ...],
    timeline: timedelta = DEFAULT_TIMELINE,
) -> PlanDocument:
    """Build a plan document from the values a caller supplies."""
    return PlanDocument(
        seed=seed,
        started_at=started_at,
        timeline_minutes=int(timeline.total_seconds() // 60),
        main_machines=main_machines,
        shift_machines=shift_machines,
    )


def write_plan(path: Path, document: PlanDocument) -> None:
    """Write the plan as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "version": PLAN_VERSION,
                "seed": document.seed,
                "started_at": document.started_at.isoformat(),
                "timeline_minutes": document.timeline_minutes,
                "main_machines": list(document.main_machines),
                "shift_machines": list(document.shift_machines),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def read_plan(path: Path) -> PlanDocument:
    """Read a plan written by `write_plan`.

    Raises:
        DatasetError: if the file is missing or is not a plan this version
            understands.
    """
    if not path.exists():
        raise DatasetError(f"{path} is missing. Generate the dataset first: `ml dataset generate`.")
    payload = json.loads(path.read_text(encoding="utf-8"))
    version = payload.get("version")
    if version != PLAN_VERSION:
        raise DatasetError(
            f"{path} is plan version {version!r}; this build writes and reads "
            f"version {PLAN_VERSION}. Regenerate the dataset."
        )
    return PlanDocument(
        seed=int(payload["seed"]),
        started_at=datetime.fromisoformat(payload["started_at"]),
        timeline_minutes=int(payload["timeline_minutes"]),
        main_machines=tuple(payload["main_machines"]),
        shift_machines=tuple(payload["shift_machines"]),
    )
