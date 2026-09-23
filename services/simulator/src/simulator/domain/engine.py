"""Advances one machine through a scenario."""

from __future__ import annotations

import math
import random

from simulator.domain.errors import SimulationValidationError
from simulator.domain.machine import MachineProfile
from simulator.domain.physics import (
    OPERATING_CYCLE_PERIOD,
    degradation_at,
    health_index,
    is_failure_imminent,
    synthesise_reading,
)
from simulator.domain.scenario import profile_for
from simulator.domain.seeding import derive_seed
from simulator.domain.session import SimulationSession
from simulator.domain.state import GroundTruthState, SimulationTick, TelemetrySample

_TWO_PI = 2.0 * math.pi


class MachineSimulator:
    """Produces one machine's series, tick by tick.

    A machine's condition is a pure function of how far into the run it is.
    Nothing accumulates between ticks, so ticks may be requested in any order, a
    run may be resumed at any index, and there is no hidden state to drift or to
    depend on call order. "Same seed, same series" is therefore true by
    construction rather than by care — and a realtime run that restarts can pick
    up exactly where it left off.
    """

    def __init__(self, session: SimulationSession, profile: MachineProfile) -> None:
        self._session = session
        self._profile = profile
        self._scenario_profile = profile_for(session.scenario)

    @property
    def profile(self) -> MachineProfile:
        """Return the machine this simulator drives."""
        return self._profile

    @property
    def tick_count(self) -> int:
        """Return how many samples the run produces for this machine."""
        return self._session.tick_count

    def tick(self, index: int) -> SimulationTick:
        """Return the sample at `index`, and the truth behind it.

        Raises:
            SimulationValidationError: if `index` lies outside the run.
        """
        if not 0 <= index < self.tick_count:
            raise SimulationValidationError(
                f"Tick index {index} is outside the run, which has {self.tick_count} ticks."
            )

        elapsed = self._session.sample_interval * index
        recorded_at = self._session.started_at + elapsed
        elapsed_fraction = elapsed / self._session.duration

        degradation = degradation_at(
            elapsed_fraction,
            self._scenario_profile,
            self._profile.susceptibility,
        )

        # A generator per tick, seeded from the tick index rather than advanced
        # across ticks. That is what makes `tick(5)` independent of whether
        # ticks 0 through 4 were ever requested.
        rng = random.Random(derive_seed(self._session.seed, self._profile.machine_id, index))
        cycle_phase = _TWO_PI * (elapsed / OPERATING_CYCLE_PERIOD)

        reading = synthesise_reading(self._profile, degradation, cycle_phase, rng)
        health = health_index(degradation)

        return SimulationTick(
            telemetry=TelemetrySample(
                event_id=self._session.event_id(self._profile.machine_id, index),
                machine_id=self._profile.machine_id,
                recorded_at=recorded_at,
                session_id=self._session.session_id,
                reading=reading,
            ),
            ground_truth=GroundTruthState(
                machine_id=self._profile.machine_id,
                recorded_at=recorded_at,
                scenario=self._session.scenario,
                degradation=degradation,
                health_index=health,
                failure_imminent=is_failure_imminent(health),
            ),
        )
