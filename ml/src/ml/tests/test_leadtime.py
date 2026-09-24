"""Lead time, which is only a number because an alarm has a definition."""

from __future__ import annotations

import numpy as np
import pytest

from ml.dataset.labelling import NEVER_FAILS
from ml.evaluation import leadtime
from ml.evaluation.errors import EvaluationError

WINDOWS = np.arange(0, 200, dtype=np.int32)


def _one_life(scores: np.ndarray, onset: int | None = 200) -> leadtime.Alarms:
    life = np.zeros(scores.shape[0], dtype=np.int32)
    return leadtime.alarms(
        scores.astype(np.float64),
        life,
        WINDOWS[: scores.shape[0]],
        np.array([NEVER_FAILS if onset is None else onset], dtype=np.int64),
        rule=leadtime.AlarmRule(threshold=0.5, consecutive=5),
    )


def test_a_sustained_run_is_an_alarm_with_the_expected_lead() -> None:
    scores = np.where(WINDOWS >= 150, 0.9, 0.1)
    result = _one_life(scores, onset=200)

    detected = result.detected
    assert len(detected) == 1
    assert detected[0].alarm == 150
    assert detected[0].lead_time == 50
    assert result.detection_rate == 1.0


def test_a_run_shorter_than_the_debounce_is_not_an_alarm() -> None:
    """Four windows above threshold is not a smaller alarm, it is not one.

    At this prevalence a single window crossing the threshold is close to a
    coin flip, which is the entire reason the rule exists.
    """
    scores = np.where(WINDOWS >= 196, 0.9, 0.1)
    result = _one_life(scores, onset=200)

    assert result.detected == ()
    assert result.detection_rate == 0.0


def test_a_run_that_breaks_before_the_failure_is_not_credited() -> None:
    """The decisive test of the definition.

    An early spike that vanishes is not a warning. Measured naively it would
    score a far better lead time than a late, reliable alarm, which is exactly
    backwards.
    """
    scores = np.where((WINDOWS >= 60) & (WINDOWS < 70), 0.9, 0.1)
    scores = np.where(WINDOWS >= 190, 0.9, scores)
    result = _one_life(scores, onset=200)

    assert len(result.detected) == 1
    assert result.detected[0].alarm == 190
    assert result.detected[0].lead_time == 10
    assert result.abandoned == 1


def test_an_abandoned_crossing_alone_detects_nothing() -> None:
    scores = np.where((WINDOWS >= 60) & (WINDOWS < 70), 0.9, 0.1)
    result = _one_life(scores, onset=200)

    assert result.detected == ()
    assert result.abandoned == 1


def test_lead_time_is_measured_from_the_window_end() -> None:
    """The prediction is made at the end instant, so that is when the clock starts."""
    scores = np.where(WINDOWS >= 100, 0.9, 0.1)
    result = _one_life(scores, onset=175)

    assert result.detected[0].lead_time == 75


def test_a_never_failing_life_counts_false_alarms_per_machine_day() -> None:
    """The rate needs a denominator in time, not in lives.

    A 6-hour healthy life and a 48-hour healthy life are not the same amount of
    opportunity to raise a false alarm.
    """
    scores = np.where(WINDOWS >= 190, 0.9, 0.1)
    result = _one_life(scores, onset=None)

    assert result.events == ()
    assert result.non_event_alarms == 1
    # 200 minutes of observation is 200/1440 of a machine-day, so one alarm in
    # that window is 7.2 per machine-day.
    assert result.false_alarms_per_machine_day == pytest.approx(1440 / 200)


def test_two_separate_runs_count_as_two_false_alarms() -> None:
    scores = np.where((WINDOWS >= 50) & (WINDOWS < 60), 0.9, 0.1)
    scores = np.where(WINDOWS >= 150, 0.9, scores)
    result = _one_life(scores, onset=None)

    assert result.non_event_alarms == 2


def test_lead_time_quartiles_over_several_events() -> None:
    life = np.zeros(400, dtype=np.int32)
    life[200:] = 1
    positions = np.concatenate([np.arange(0, 200), np.arange(0, 200)]).astype(np.int32)
    scores = np.concatenate(
        [np.where(np.arange(200) >= 100, 0.9, 0.1), np.where(np.arange(200) >= 180, 0.9, 0.1)]
    ).astype(np.float64)
    onset = np.array([200, 200], dtype=np.int64)

    result = leadtime.alarms(
        scores, life, positions, onset, rule=leadtime.AlarmRule(threshold=0.5, consecutive=5)
    )

    assert result.lead_times.tolist() == [100, 20]
    assert result.median_lead_time == pytest.approx(60.0)


def test_the_summary_carries_the_four_reportable_figures() -> None:
    """A lead time without a detection rate and a false-alarm rate is not reportable."""
    scores = np.where(WINDOWS >= 150, 0.9, 0.1)
    summary = _one_life(scores, onset=200).summary()

    for key in (
        "detection_rate",
        "median_lead_time",
        "lead_time_q1",
        "lead_time_q3",
        "abandoned_crossings",
        "false_alarms_per_machine_day",
    ):
        assert key in summary


def test_an_impossible_rule_is_refused() -> None:
    with pytest.raises(EvaluationError, match="threshold"):
        leadtime.AlarmRule(threshold=1.5)
    with pytest.raises(EvaluationError, match="consecutive"):
        leadtime.AlarmRule(threshold=0.5, consecutive=0)


def test_mismatched_arrays_are_refused() -> None:
    with pytest.raises(EvaluationError, match="same windows"):
        leadtime.alarms(
            np.zeros(4),
            np.zeros(4, dtype=np.int32),
            np.zeros(3, dtype=np.int32),
            np.array([10], dtype=np.int64),
            rule=leadtime.AlarmRule(threshold=0.5),
        )


def test_a_sweep_produces_one_result_per_threshold() -> None:
    """The curve that lets a reader pick an operating point rather than accept one."""
    scores = np.where(WINDOWS >= 150, 0.9, 0.1)
    life = np.zeros(200, dtype=np.int32)

    results = leadtime.sweep(
        scores.astype(np.float64),
        life,
        WINDOWS,
        np.array([200], dtype=np.int64),
        thresholds=np.array([0.5, 0.95]),
    )

    assert len(results) == 2
    assert results[0].detection_rate == 1.0
    assert results[1].detection_rate == 0.0
