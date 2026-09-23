"""The guarantee that ground truth cannot leak into the telemetry stream.

`MASTERPLAN.md` §3.2 requires that hidden ground-truth state never be exposed to
the model as an input feature, and §25.15 repeats it as a product constraint.

The design enforces it structurally: observations and ground truth are separate
types flowing into separate sinks, and dataset mode writes them to separate
files. These tests guard the structure, because the moment it erodes the leak is
silent — a model trained on a leaked feature scores well and is worthless.
"""

from __future__ import annotations

import dataclasses
import json
from typing import get_type_hints

from simulator.domain.engine import MachineSimulator
from simulator.domain.machine import MachineProfile
from simulator.domain.readings import SIGNAL_NAMES
from simulator.domain.scenario import Scenario
from simulator.domain.session import SimulationSession
from simulator.domain.state import GroundTruthState, TelemetrySample
from simulator.tests.conftest import FIXED_START, STANDARD_DURATION

#: Everything an observation is allowed to contain. Adding to this set is a
#: deliberate act, which is the point of asserting on it.
TELEMETRY_CONTRACT_FIELDS = frozenset(
    {"event_id", "machine_id", "recorded_at", "session_id", "reading"}
)


def _bearing_degradation_session() -> SimulationSession:
    """A session whose scenario is known and identifiable by name."""
    return SimulationSession.create(
        machine_ids=["M003"],
        scenario=Scenario.BEARING_DEGRADATION,
        seed=99,
        duration=STANDARD_DURATION,
        started_at=FIXED_START,
    )


def test_telemetry_sample_declares_only_observation_fields() -> None:
    """No field may be added to an observation without a deliberate decision.

    This is the moment a leak would happen: someone needs a scenario label or a
    health index alongside the reading, adds a field, and every downstream
    consumer silently gains access to ground truth. Failing here forces that
    conversation instead.
    """
    declared = {field.name for field in dataclasses.fields(TelemetrySample)}

    assert declared == TELEMETRY_CONTRACT_FIELDS


def test_observation_serialises_to_the_signals_and_identifiers_only() -> None:
    """What a sink receives carries exactly the six signals and their identity."""
    session = _bearing_degradation_session()
    sample = MachineSimulator(session, session.machines[0]).tick(0).telemetry

    assert set(sample.as_row()) == {
        "event_id",
        "machine_id",
        "recorded_at",
        "session_id",
        *SIGNAL_NAMES,
    }


def test_scenario_name_never_appears_in_a_serialised_observation() -> None:
    """The strongest form of the guarantee: not the field, the value.

    A field-set assertion would miss a leak that reused an existing column — a
    scenario label smuggled into `session_id`, for instance. Searching the
    serialised observation for the scenario's name catches that.
    """
    session = _bearing_degradation_session()
    sample = MachineSimulator(session, session.machines[0]).tick(0).telemetry
    rendered = json.dumps(sample.as_row(), default=str).lower()

    for token in ("bearing", "degrad", "thermal", "overload", "health"):
        assert token not in rendered, f"Observation leaked the token '{token}'."


def test_ground_truth_records_the_scenario() -> None:
    """The scenario lives on the hidden side, where labelling can reach it."""
    session = _bearing_degradation_session()
    tick = MachineSimulator(session, session.machines[0]).tick(0)

    assert tick.ground_truth.scenario is Scenario.BEARING_DEGRADATION
    assert "scenario" in tick.ground_truth.as_row()


def test_ground_truth_declares_no_signal_fields() -> None:
    """The split runs both ways: hidden state does not carry observations either.

    A ground-truth row that duplicated the signals would let a Phase 3 join
    reintroduce them from the labels file, which is the same leak arriving by a
    different route.
    """
    declared = {field.name for field in dataclasses.fields(GroundTruthState)}

    assert declared == {
        "machine_id",
        "recorded_at",
        "scenario",
        "degradation",
        "health_index",
        "failure_imminent",
    }
    assert not declared & set(SIGNAL_NAMES)


def test_each_sink_accepts_exactly_one_channel() -> None:
    """Every shipped sink is bound to one half of the split.

    Checked by inspecting the type each `write` accepts, not by name: the two
    interfaces have the same method names and differ only in what they take, so
    a sink that drifted to the wrong channel would otherwise look identical.
    """
    from simulator.infrastructure.sinks import (
        ConsoleGroundTruthSink,
        ConsoleTelemetrySink,
        JsonLinesGroundTruthSink,
        JsonLinesTelemetrySink,
        ParquetGroundTruthSink,
        ParquetTelemetrySink,
    )

    telemetry_sinks = (
        ConsoleTelemetrySink,
        JsonLinesTelemetrySink,
        ParquetTelemetrySink,
    )
    ground_truth_sinks = (
        ConsoleGroundTruthSink,
        JsonLinesGroundTruthSink,
        ParquetGroundTruthSink,
    )

    for telemetry_sink in telemetry_sinks:
        assert get_type_hints(telemetry_sink.write)["sample"] is TelemetrySample
    for ground_truth_sink in ground_truth_sinks:
        assert get_type_hints(ground_truth_sink.write)["state"] is GroundTruthState


def test_a_machine_profile_is_not_part_of_an_observation() -> None:
    """Nominal values and susceptibility are configuration, not measurement.

    A nominal speed is arguably harmless, but susceptibility is not: it would
    tell a model how fast this particular unit degrades, which is exactly the
    kind of hidden advantage that makes an evaluation meaningless.

    `machine_id` is shared on purpose — it is how a reading is attributed — so it
    is excluded from the comparison.
    """
    session = _bearing_degradation_session()
    sample = MachineSimulator(session, session.machines[0]).tick(0).telemetry

    configuration_fields = {field.name for field in dataclasses.fields(MachineProfile)} - {
        "machine_id"
    }

    assert not configuration_fields & set(sample.as_row())
