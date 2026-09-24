"""The feature whitelist itself."""

from __future__ import annotations

from ml.dataset.features import FEATURE_COLUMNS, IDENTIFIER_COLUMNS, is_feature
from simulator.domain.readings import SIGNAL_NAMES


def test_the_features_are_exactly_the_signals() -> None:
    """Sourced from the simulator, so the two cannot drift apart."""
    assert FEATURE_COLUMNS == SIGNAL_NAMES


def test_the_identifiers_are_named_so_a_test_can_assert_their_absence() -> None:
    """`ml.dataset.features` explains what each one would give away."""
    assert set(IDENTIFIER_COLUMNS) == {
        "event_id",
        "machine_id",
        "recorded_at",
        "session_id",
    }


def test_only_the_signals_are_features() -> None:
    for name in FEATURE_COLUMNS:
        assert is_feature(name)
    for name in IDENTIFIER_COLUMNS:
        assert not is_feature(name)
    assert not is_feature("minutes_to_onset")
    assert not is_feature("health_index")
