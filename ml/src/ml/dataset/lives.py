"""Composing a fleet timeline out of sequential *lives*.

A Phase 2 session runs one machine through one scenario, and its degradation
curve is a logistic in ``elapsed / duration`` — so damage always reaches full at
the session's end. Generating thirty days as a single session per machine would
therefore yield exactly one failure event per machine: a positive rate near
0.07%, which is realistic and untrainable.

So a machine's thirty days is composed instead from several sequential **lives**.
A degradation life ends at the onset of failure, and the next life begins
immediately after — a repair. A ``NORMAL`` life simply runs healthy, which is
where the model's negatives come from. A machine gets 20-30 lives, hence 20-30
failure events, and a positive rate around 3%.

Two things this module deliberately does *not* do:

*It does not change the simulator.* Lives are composed from sessions that
already exist. The physics, the scenario profiles and the coupling table are
untouched.

*It does not decide what a life means for labelling.* That is
`ml.dataset.labelling`, which reads the truth rather than inferring it.
"""

from __future__ import annotations

import random
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum

from simulator.domain.machine import MachineProfile, build_profile
from simulator.domain.scenario import Scenario
from simulator.domain.seeding import derive_seed

#: Every machine is observed for this long, per `MASTERPLAN.md` §6's
#: "100 machines x 30 days". A parameter rather than a constant because tests
#: plan fleets of a day or two, and a fixed thirty days would make the suite
#: unusable.
DEFAULT_TIMELINE = timedelta(days=30)

#: The three degradation modes, in the order the fleet cycles through them.
DEGRADATION_SCENARIOS: tuple[Scenario, ...] = (
    Scenario.BEARING_DEGRADATION,
    Scenario.OVERHEATING,
    Scenario.OVERLOAD,
)

#: How often a life is healthy rather than degrading. A quarter is enough to
#: supply honest negatives without diluting the positive class below the point
#: where a model can be trained on it.
NORMAL_LIFE_SHARE = 0.25

#: The main regime's life lengths, in hours. Drawn continuously rather than from
#: a fixed set, because a machine whose every life lasted exactly 24 hours would
#: let a model read "hours since the last repair" off a calendar and skip the
#: signals entirely. Variation in *when* a machine fails is where the interest
#: lies.
MAIN_LIFE_MIN_HOURS = 16.0
MAIN_LIFE_MAX_HOURS = 32.0

#: The duration-shift regime. Lives well outside the main regime's range, on
#: machines held out from training entirely. A model that learned the main
#: regime's timing collapses here; one that learned the signals degrades
#: gracefully. Given a simulator whose degradation is a deterministic function
#: of elapsed time, this is the only evaluation that separates the two.
SHIFT_LIFE_HOURS = (6.0, 48.0)

#: The shortest life worth creating. Below this a life produces no 60-minute
#: window at all, so composing one adds a row to the life table and nothing
#: else. A short remainder is absorbed into the preceding life instead.
MIN_LIFE = timedelta(minutes=30)


class Regime(StrEnum):
    """Which distribution a machine's life lengths are drawn from."""

    MAIN = "MAIN"
    SHIFT = "SHIFT"


@dataclass(frozen=True, slots=True)
class Life:
    """One machine's run from a repair to the onset of its next failure."""

    machine_id: str
    index: int
    scenario: Scenario
    duration: timedelta
    started_at: datetime
    seed: int

    @property
    def life_id(self) -> str:
        """Return this life's identifier, used as the session id.

        Deliberately *not* the simulator's default ``sim-{scenario}-{seed}``.
        That form abbreviates the scenario into the identifier, and the
        identifier appears in every telemetry row — which would put a token
        derived from ground truth into the observation stream. This form names
        only the machine and the life's ordinal, neither of which says anything
        about which scenario is running.

        Bounded so the resulting event ids fit the 64-character column: twelve
        characters here leaves ample room for the machine and an eight-digit
        index.
        """
        return f"life-{self.machine_id}-{self.index:02d}"

    def __post_init__(self) -> None:
        """Reject a life the simulator would refuse to run."""
        if self.duration < MIN_LIFE:
            raise ValueError(
                f"Life {self.life_id} lasts {self.duration}, below the {MIN_LIFE} minimum."
            )


@dataclass(frozen=True, slots=True)
class MachinePlan:
    """One machine's whole timeline: a stable profile and the lives it runs."""

    machine_id: str
    primary_scenario: Scenario
    profile: MachineProfile
    lives: tuple[Life, ...]

    @property
    def duration(self) -> timedelta:
        """Return the total simulated time across this machine's lives."""
        return sum((life.duration for life in self.lives), timedelta(0))

    @property
    def failure_lives(self) -> tuple[Life, ...]:
        """Return the lives that are expected to reach failure."""
        return tuple(life for life in self.lives if life.scenario is not Scenario.NORMAL)


def primary_scenario(index: int) -> Scenario:
    """Return the degradation mode the `index`-th machine is prone to.

    A machine keeps its failure mode across its lives, which is both realistic —
    a misaligned shaft keeps wearing bearings — and the reason a machine-level
    split is meaningful at all. Cycling by position spreads all three modes
    evenly across the fleet, so a stratified split can put all three in every
    split.
    """
    return DEGRADATION_SCENARIOS[(index - 1) % len(DEGRADATION_SCENARIOS)]


def plan_fleet(
    machine_ids: Sequence[str],
    *,
    base_seed: int,
    regime: Regime,
    started_at: datetime,
    timeline: timedelta = DEFAULT_TIMELINE,
) -> tuple[MachinePlan, ...]:
    """Compose every machine's timeline.

    Each machine is planned from its own random stream, seeded on
    ``(base_seed, machine_id)``. That is what makes a fleet extensible: adding a
    machine to a later run leaves the machines already there byte-identical,
    exactly as `SimulationSession.create` guarantees for a single session.

    It also makes the whole fleet reconstructible from four values — the seed,
    the machine ids, the regime and the start instant — which is why the
    generation plan on disk is a few hundred bytes rather than a list of
    thousands of lives.
    """
    return tuple(
        plan_machine(
            machine_id,
            base_seed=base_seed,
            regime=regime,
            started_at=started_at,
            timeline=timeline,
        )
        for machine_id in machine_ids
    )


def plan_machine(
    machine_id: str,
    *,
    base_seed: int,
    regime: Regime,
    started_at: datetime,
    timeline: timedelta = DEFAULT_TIMELINE,
) -> MachinePlan:
    """Compose one machine's profile and its sequence of lives."""
    primary = primary_scenario(_ordinal(machine_id))
    return MachinePlan(
        machine_id=machine_id,
        primary_scenario=primary,
        # Built once per machine and reused for every life, so a machine's
        # nominal point and susceptibility are the stable identity that a
        # machine-level split exists to keep on one side of the boundary.
        profile=build_profile(machine_id, random.Random(derive_seed(base_seed, machine_id))),
        lives=_compose_lives(
            machine_id,
            primary=primary,
            base_seed=base_seed,
            regime=regime,
            started_at=started_at,
            timeline=timeline,
        ),
    )


def _compose_lives(
    machine_id: str,
    *,
    primary: Scenario,
    base_seed: int,
    regime: Regime,
    started_at: datetime,
    timeline: timedelta,
) -> tuple[Life, ...]:
    """Fill exactly `timeline` with consecutive lives.

    The last life absorbs whatever remains rather than being drawn, so a
    machine's timeline is exactly thirty days regardless of how the draws fall.
    A remainder too short to be worth a life is folded into its predecessor
    instead of becoming a life that could never produce a window.
    """
    rng = random.Random(derive_seed(base_seed, machine_id, "lives"))
    lives: list[Life] = []
    cursor = started_at
    remaining = timeline
    index = 0

    while remaining > timedelta(0):
        drawn = _draw_duration(rng, regime)
        duration = drawn if remaining - drawn >= MIN_LIFE else remaining
        lives.append(
            Life(
                machine_id=machine_id,
                index=index,
                scenario=_draw_scenario(rng, primary),
                duration=duration,
                started_at=cursor,
                # A distinct seed per life. Reusing one seed across a machine's
                # lives would make them byte-identical — same profile, same
                # duration, same physics — which is a copy of a training row
                # wearing a different timestamp.
                seed=derive_seed(base_seed, machine_id, "life", index),
            )
        )
        cursor += duration
        remaining -= duration
        index += 1

    return tuple(lives)


def _draw_duration(rng: random.Random, regime: Regime) -> timedelta:
    """Draw a life's length from the regime's distribution."""
    if regime is Regime.SHIFT:
        return timedelta(hours=rng.choice(SHIFT_LIFE_HOURS))
    return timedelta(minutes=round(rng.uniform(MAIN_LIFE_MIN_HOURS, MAIN_LIFE_MAX_HOURS) * 60.0))


def _draw_scenario(rng: random.Random, primary: Scenario) -> Scenario:
    """Draw a life's scenario: usually this machine's own failure mode."""
    if rng.random() < NORMAL_LIFE_SHARE:
        return Scenario.NORMAL
    return primary


def _ordinal(machine_id: str) -> int:
    """Return the number in a machine id like ``M003``."""
    return int(machine_id[-3:])


def iter_machine_ids(count: int, *, first: int = 1) -> Iterator[str]:
    """Yield `count` sequential simulator-valid machine ids."""
    for offset in range(count):
        yield f"M{first + offset:03d}"
