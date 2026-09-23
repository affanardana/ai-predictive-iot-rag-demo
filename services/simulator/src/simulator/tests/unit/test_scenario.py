"""Scenarios and the mechanisms each one drives."""

from __future__ import annotations

import pytest

from simulator.domain.errors import UnknownScenarioError
from simulator.domain.scenario import SCENARIO_PROFILES, Scenario, profile_for


def test_the_four_scenarios_from_the_masterplan_exist() -> None:
    """`MASTERPLAN.md` §6 and PRD §11 name exactly these."""
    assert {scenario.value for scenario in Scenario} == {
        "NORMAL",
        "BEARING_DEGRADATION",
        "OVERHEATING",
        "OVERLOAD",
    }


@pytest.mark.parametrize(
    "name",
    ["bearing_degradation", "BEARING_DEGRADATION", "Bearing_Degradation", " bearing_degradation "],
)
def test_names_are_resolved_case_insensitively(name: str) -> None:
    """A command line flag should not have to match the enum's casing exactly."""
    assert Scenario.from_name(name) is Scenario.BEARING_DEGRADATION


def test_an_unknown_name_lists_what_is_available() -> None:
    """The failure tells the caller what to type instead."""
    with pytest.raises(UnknownScenarioError) as caught:
        Scenario.from_name("EXPLODING")

    assert "EXPLODING" in str(caught.value)
    assert "BEARING_DEGRADATION" in str(caught.value)


def test_every_scenario_has_a_profile() -> None:
    """A scenario without weights would fail at the first tick, not here."""
    assert set(SCENARIO_PROFILES) == set(Scenario)


@pytest.mark.parametrize("scenario", list(Scenario))
def test_weights_are_fractions(scenario: Scenario) -> None:
    """Weights are relative intensities, so they live on `[0, 1]`."""
    weights = profile_for(scenario)

    for value in (weights.bearing_weight, weights.thermal_weight, weights.load_weight):
        assert 0.0 <= value <= 1.0


def test_the_normal_scenario_drives_nothing() -> None:
    """A baseline that degraded would not be a baseline."""
    weights = profile_for(Scenario.NORMAL)

    assert (weights.bearing_weight, weights.thermal_weight, weights.load_weight) == (0.0, 0.0, 0.0)


@pytest.mark.parametrize(
    ("scenario", "channel"),
    [
        (Scenario.BEARING_DEGRADATION, "bearing_weight"),
        (Scenario.OVERHEATING, "thermal_weight"),
        (Scenario.OVERLOAD, "load_weight"),
    ],
)
def test_each_degradation_scenario_has_a_primary_mechanism(
    scenario: Scenario,
    channel: str,
) -> None:
    """Each scenario is named for the mechanism it drives hardest.

    A scenario whose namesake channel was not its strongest would be
    mislabelled — and, being ground truth, would mislabel every dataset built
    from it.
    """
    weights = profile_for(scenario)

    assert getattr(weights, channel) == 1.0
