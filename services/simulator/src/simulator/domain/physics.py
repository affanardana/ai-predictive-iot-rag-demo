"""How degradation shows up in the signals.

This module is the reason the project is not a demo trick. `MASTERPLAN.md` §3.6
prohibits "fake telemetry generated independently for every sensor", and PRD §11
requires that a degradation mode produce a *correlated* signature. The coupling
table below is that requirement written as data: one physical cause moves several
signals together, because that is what physically happens.
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping
from datetime import timedelta

from simulator.domain.machine import MachineProfile
from simulator.domain.readings import NON_NEGATIVE_SIGNALS, SIGNAL_NAMES, SensorReading
from simulator.domain.scenario import ScenarioProfile
from simulator.domain.state import CHANNEL_NAMES, DegradationState

#: Steepness of the degradation ramp. Higher fails more abruptly; this value
#: puts a scenario's critical point at roughly two-thirds of the way through a
#: run, which reads as a machine going wrong rather than a steady drift.
DEGRADATION_STEEPNESS = 8.0

#: The operating cycle. Load — and therefore heat and current — drifts over a
#: period rather than sitting flat, which is what stops a healthy machine's
#: series looking like a straight line with jitter. Short enough to appear
#: within a demonstration, long enough not to be mistaken for degradation.
OPERATING_CYCLE_PERIOD = timedelta(minutes=20)

#: Below this health index, failure is imminent. Phase 3 labels training data
#: from this, and Phase 4 measures how early the model saw it coming.
FAILURE_HEALTH_THRESHOLD = 0.2

#: Extra units of each signal at full degradation, keyed by mechanism.
#:
#: The signs are the point. A worn bearing raises vibration hardest; the
#: friction heats the machine, the drag draws more current, and the shaft slows.
#: Those four movements from one cause are what PRD §11 asks for, and what an
#: independently-randomised sensor would never produce.
COUPLING: Mapping[str, Mapping[str, float]] = {
    "vibration": {
        "bearing_wear": 4.6,
        "thermal_stress": 0.5,
        "load_stress": 0.0,
    },
    "temperature": {
        "bearing_wear": 25.0,
        "thermal_stress": 45.0,
        "load_stress": 20.0,
    },
    "current": {
        "bearing_wear": 3.0,
        "thermal_stress": 2.0,
        "load_stress": 8.0,
    },
    "rpm": {
        "bearing_wear": -60.0,
        "thermal_stress": -20.0,
        "load_stress": -40.0,
    },
    "load": {
        "bearing_wear": 0.0,
        "thermal_stress": 0.0,
        "load_stress": 0.35,
    },
    # Supply voltage sags a little as the machine draws more current.
    "voltage": {
        "bearing_wear": 0.0,
        "thermal_stress": -0.5,
        "load_stress": -2.0,
    },
}

#: Standard deviation of per-sample measurement noise. Voltage is held tightly
#: by the supply; vibration is the noisiest thing on the machine.
NOISE_SIGMA: Mapping[str, float] = {
    "temperature": 0.8,
    "vibration": 0.05,
    "rpm": 2.0,
    "current": 0.25,
    "load": 0.01,
    "voltage": 0.4,
}

#: Peak amplitude of the operating cycle, per signal.
CYCLE_AMPLITUDE: Mapping[str, float] = {
    "temperature": 3.0,
    "vibration": 0.08,
    "rpm": 8.0,
    "current": 1.2,
    "load": 0.06,
    "voltage": 1.5,
}


def degradation_progress(elapsed_fraction: float) -> float:
    """Return how far a degradation scenario has progressed, on `[0, 1]`.

    A logistic ramp, not a straight line. Real degradation accelerates as damage
    feeds on itself — a worn bearing runs hotter, and running hotter wears it
    faster — so a linear ramp would read as a steady drift rather than a machine
    failing. The curve is normalised so that zero maps to exactly zero and one
    to exactly one, which keeps a run's endpoints meaningful.
    """
    position = min(max(elapsed_fraction, 0.0), 1.0)
    raw = 1.0 / (1.0 + math.exp(-DEGRADATION_STEEPNESS * (position - 0.5)))
    low = 1.0 / (1.0 + math.exp(DEGRADATION_STEEPNESS * 0.5))
    high = 1.0 / (1.0 + math.exp(-DEGRADATION_STEEPNESS * 0.5))
    return (raw - low) / (high - low)


def degradation_at(
    elapsed_fraction: float,
    profile: ScenarioProfile,
    susceptibility: float,
) -> DegradationState:
    """Return each mechanism's progress at a point in a run.

    A machine's susceptibility scales how fast it gets there, so a fleet
    contains units that last longer and units that fail sooner than the
    scenario's nominal pace.
    """
    progress = degradation_progress(elapsed_fraction) * susceptibility
    return DegradationState(
        bearing_wear=min(1.0, progress * profile.bearing_weight),
        thermal_stress=min(1.0, progress * profile.thermal_weight),
        load_stress=min(1.0, progress * profile.load_weight),
    )


def health_index(degradation: DegradationState) -> float:
    """Return overall condition on `[0, 1]`, where 1 is healthy.

    Derived from the worst mechanism rather than the average: a machine with a
    badly worn bearing is in trouble even if its thermal and load channels are
    fine, and averaging would let the healthy channels hide it.
    """
    return 1.0 - degradation.worst()


def is_failure_imminent(health: float) -> bool:
    """Whether a health index means failure is imminent."""
    return health < FAILURE_HEALTH_THRESHOLD


def synthesise_reading(
    profile: MachineProfile,
    degradation: DegradationState,
    cycle_phase: float,
    rng: random.Random,
) -> SensorReading:
    """Combine the nominal point, degradation, the operating cycle, and noise.

    The four terms are deliberately distinct. Degradation is the signal a model
    is meant to learn; the cycle is normal operation it must not mistake for
    degradation; noise is what makes the readings look like measurements rather
    than arithmetic.
    """
    nominal = profile.nominal_reading()
    values: dict[str, float] = {}

    for name in SIGNAL_NAMES:
        coupled = sum(
            COUPLING[name][channel] * getattr(degradation, channel) for channel in CHANNEL_NAMES
        )
        cycle = CYCLE_AMPLITUDE[name] * math.sin(cycle_phase)
        noise = rng.gauss(0.0, NOISE_SIGMA[name])
        values[name] = getattr(nominal, name) + coupled + cycle + noise

    # A physical floor, not a correction: none of these can go negative. The
    # scenarios are tuned so this never actually engages -- a test asserts the
    # margin -- so a physics change that drove a signal negative would show up
    # as a failing test rather than as silently clamped telemetry.
    for name in NON_NEGATIVE_SIGNALS:
        values[name] = max(0.0, values[name])

    return SensorReading(
        temperature=values["temperature"],
        vibration=values["vibration"],
        rpm=values["rpm"],
        current=values["current"],
        load=values["load"],
        voltage=values["voltage"],
    )
