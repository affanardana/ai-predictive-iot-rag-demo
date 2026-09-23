"""The simulation model. Pure, deterministic, and free of I/O.

Everything here is a function of its inputs: given the same seed and
configuration, the same series is produced, in any process, on any machine, in
any order. That property is what makes a demonstration reproducible and a
dataset regenerable, so it is enforced by design rather than left to care.
"""

from simulator.domain.engine import MachineSimulator
from simulator.domain.errors import (
    SimulationError,
    SimulationValidationError,
    UnknownScenarioError,
)
from simulator.domain.machine import MachineProfile, build_profile
from simulator.domain.physics import (
    FAILURE_HEALTH_THRESHOLD,
    degradation_at,
    degradation_progress,
    health_index,
    is_failure_imminent,
    synthesise_reading,
)
from simulator.domain.readings import SIGNAL_NAMES, SensorReading, signal_names
from simulator.domain.scenario import Scenario, ScenarioProfile, profile_for
from simulator.domain.seeding import derive_seed
from simulator.domain.session import (
    DEFAULT_DURATION,
    DEFAULT_SAMPLE_INTERVAL,
    SimulationSession,
    default_session_id,
)
from simulator.domain.state import (
    CHANNEL_NAMES,
    DegradationState,
    GroundTruthState,
    SimulationTick,
    TelemetrySample,
)

__all__ = [
    "CHANNEL_NAMES",
    "DEFAULT_DURATION",
    "DEFAULT_SAMPLE_INTERVAL",
    "FAILURE_HEALTH_THRESHOLD",
    "SIGNAL_NAMES",
    "DegradationState",
    "GroundTruthState",
    "MachineProfile",
    "MachineSimulator",
    "Scenario",
    "ScenarioProfile",
    "SensorReading",
    "SimulationError",
    "SimulationSession",
    "SimulationTick",
    "SimulationValidationError",
    "TelemetrySample",
    "UnknownScenarioError",
    "build_profile",
    "default_session_id",
    "degradation_at",
    "degradation_progress",
    "derive_seed",
    "health_index",
    "is_failure_imminent",
    "profile_for",
    "signal_names",
    "synthesise_reading",
]
