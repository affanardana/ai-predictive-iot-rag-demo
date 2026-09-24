"""Lead time, which is not a number until an alarm is defined.

"Median lead time 47 minutes" is meaningless on its own. A model that spikes
above the threshold once, forty minutes early, and then falls silent has not
warned anyone of anything — and measured naively it scores better than a model
that fires late and holds. So an alarm is defined as a *sustained* crossing, and
the definition is the whole content of this module.

## The definition

An alarm, on a life that fails, is the **first window of the final unbroken run
of windows scoring at or above the threshold that reaches the life's last
pre-onset window**, and it only counts if that run is at least `consecutive`
windows long.

Read backwards from the end, that is: take the maximal suffix of the life where
every window is at or above the threshold; if it is long enough, its first
window is the alarm. Anything earlier is discarded, deliberately. A model that
fires early, drops below the threshold, and fires again later is giving two
answers, and the one worth reporting is the one it still holds when the machine
is about to fail.

`consecutive` defaults to five rather than one because a single window crossing
a threshold is a coin flip at this prevalence, and to sixty — a full hour — would
make the lead time an artefact of the window length.

## What gets reported with it

A lead time without a false-alarm rate is not reportable either: a model that
alarms constantly has a wonderful median lead time and is useless. So the report
carries four numbers together — the distribution over *detected* events, the
detection rate, the number of alarms that were raised and then abandoned, and
false alarms per machine-day on the lives that never failed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ml.dataset.labelling import NEVER_FAILS
from ml.evaluation.errors import EvaluationError

Float64Array = npt.NDArray[np.float64]
Int32Array = npt.NDArray[np.int32]
Int64Array = npt.NDArray[np.int64]

MINUTES_PER_DAY = 1440


@dataclass(frozen=True, slots=True)
class AlarmRule:
    """When a run of scores counts as an alarm."""

    threshold: float
    consecutive: int = 5

    def __post_init__(self) -> None:
        """Reject a rule that cannot ever fire."""
        if not 0.0 <= self.threshold <= 1.0:
            raise EvaluationError(f"threshold must be a probability, got {self.threshold}.")
        if self.consecutive < 1:
            raise EvaluationError(f"consecutive must be at least 1, got {self.consecutive}.")

    def describe(self) -> str:
        """Return a one-line summary, for the report's table."""
        return f"p>={self.threshold:.2f} for {self.consecutive} windows"


@dataclass(frozen=True, slots=True)
class EventAlarm:
    """One failing life, and what the model did about it."""

    life: int
    onset: int
    alarm: int | None
    lead_time: int | None
    run_length: int
    abandoned: int

    @property
    def detected(self) -> bool:
        """Whether the model sustained an alarm to the failure."""
        return self.alarm is not None


@dataclass(frozen=True, slots=True)
class Alarms:
    """Every alarm the rule produced, and the rates derived from them."""

    rule: AlarmRule
    events: tuple[EventAlarm, ...]
    non_event_alarms: int
    non_event_minutes: int

    @property
    def detected(self) -> tuple[EventAlarm, ...]:
        """Return the events the model warned about."""
        return tuple(event for event in self.events if event.detected)

    @property
    def lead_times(self) -> Float64Array:
        """Return the lead time of every detected event, in minutes."""
        return np.asarray(
            [event.lead_time for event in self.detected if event.lead_time is not None],
            dtype=np.float64,
        )

    @property
    def detection_rate(self) -> float:
        """Return the share of failing lives that were warned about."""
        return len(self.detected) / len(self.events) if self.events else 0.0

    @property
    def abandoned(self) -> int:
        """Return how many early crossings were raised and then dropped.

        Counted rather than ignored. A model with a high abandoned count is
        flapping, which is a different failure from being late and shows up
        nowhere else.
        """
        return sum(event.abandoned for event in self.events)

    @property
    def median_lead_time(self) -> float:
        """Return the median lead time over detected events."""
        values = self.lead_times
        return float(np.median(values)) if values.shape[0] else 0.0

    @property
    def lead_time_quartiles(self) -> tuple[float, float]:
        """Return the first and third quartiles of the lead time."""
        values = self.lead_times
        if not values.shape[0]:
            return 0.0, 0.0
        low, high = np.quantile(values, [0.25, 0.75])
        return float(low), float(high)

    @property
    def false_alarms_per_machine_day(self) -> float:
        """Return alarms raised on never-failing lives, per machine-day.

        The denominator is the *observed* time across the lives that never
        failed, in minutes, converted to machine-days — not the number of lives,
        because a 6-hour healthy life and a 48-hour healthy life are not the
        same amount of opportunity to raise a false alarm.
        """
        if self.non_event_minutes <= 0:
            return 0.0
        return self.non_event_alarms / (self.non_event_minutes / MINUTES_PER_DAY)

    def summary(self) -> dict[str, float]:
        """Return the four figures that make a lead time reportable."""
        low, high = self.lead_time_quartiles
        values = self.lead_times
        return {
            "detection_rate": self.detection_rate,
            "median_lead_time": self.median_lead_time,
            "lead_time_q1": low,
            "lead_time_q3": high,
            "lead_time_min": float(values.min()) if values.shape[0] else 0.0,
            "detected": float(len(self.detected)),
            "events": float(len(self.events)),
            "abandoned_crossings": float(self.abandoned),
            "non_event_alarms": float(self.non_event_alarms),
            "false_alarms_per_machine_day": self.false_alarms_per_machine_day,
        }


def alarms(
    scores: Float64Array,
    life: Int32Array,
    position: Int32Array,
    onset: Int64Array,
    *,
    rule: AlarmRule,
) -> Alarms:
    """Apply `rule` to every life and collect what fired.

    Args:
        scores: one score per window.
        life: the life each window belongs to.
        position: the window's end minute within its life, which is what a lead
            time is measured against.
        onset: the onset minute per life, or `NEVER_FAILS` for a life that never
            failed.
        rule: the sustained-crossing definition to apply.

    Raises:
        EvaluationError: if the arrays disagree in length, or a life index has
            no entry in `onset`.
    """
    if not (scores.shape == life.shape == position.shape):
        raise EvaluationError(
            "scores, life and position must describe the same windows: "
            f"{scores.shape}, {life.shape}, {position.shape}."
        )

    events: list[EventAlarm] = []
    non_event_alarms = 0
    non_event_minutes = 0

    for index in np.unique(life):
        identifier = int(index)
        if identifier >= onset.shape[0]:
            raise EvaluationError(f"life {identifier} has no onset entry.")
        chosen = np.flatnonzero(life == index)
        # Sorted by time, so "the final run" means the final run.
        minutes = position[chosen]
        values = scores[chosen]
        order = np.argsort(minutes, kind="stable")
        minutes = minutes[order]
        values = values[order]

        life_onset = int(onset[identifier])
        runs = _runs_at_or_above(values, rule.threshold, rule.consecutive)

        if life_onset == NEVER_FAILS:
            non_event_alarms += len(runs)
            if minutes.shape[0]:
                non_event_minutes += int(minutes[-1]) + 1
            continue

        # The suffix run: the last run, if it ends at the life's final window.
        alarm: int | None = None
        run_length = 0
        if runs and runs[-1][1] == values.shape[0] - 1:
            start, stop = runs[-1]
            run_length = stop - start + 1
            alarm = int(minutes[start])

        events.append(
            EventAlarm(
                life=identifier,
                onset=life_onset,
                alarm=alarm,
                lead_time=None if alarm is None else life_onset - alarm,
                run_length=run_length,
                abandoned=len(runs) - (1 if alarm is not None else 0),
            )
        )

    return Alarms(
        rule=rule,
        events=tuple(events),
        non_event_alarms=non_event_alarms,
        non_event_minutes=non_event_minutes,
    )


def _runs_at_or_above(
    values: Float64Array, threshold: float, minimum: int
) -> list[tuple[int, int]]:
    """Return the `(start, stop)` index bounds of every run of at least `minimum`.

    Bounds are inclusive, and only runs long enough to count as an alarm are
    returned — a shorter crossing is not a smaller alarm, it is not an alarm.
    """
    above = values >= threshold
    if not bool(above.any()):
        return []

    # Rising and falling edges found separately. Taking both from a single
    # differencing pass is the obvious shortcut and it is wrong: `diff` reports
    # a rise and a fall as the same kind of event, so the two lists come out
    # different lengths and every run is paired with the wrong partner.
    previous = np.concatenate(([False], above[:-1]))
    following = np.concatenate((above[1:], [False]))
    starts = np.flatnonzero(above & ~previous)
    ends = np.flatnonzero(above & ~following)

    return [
        (int(start), int(stop))
        for start, stop in zip(starts, ends, strict=True)
        if stop - start + 1 >= minimum
    ]


def sweep(
    scores: Float64Array,
    life: Int32Array,
    position: Int32Array,
    onset: Int64Array,
    *,
    thresholds: Float64Array,
    consecutive: int = 5,
) -> tuple[Alarms, ...]:
    """Apply a range of thresholds, for the lead-time-versus-threshold curve.

    The curve is what lets a reader choose an operating point rather than accept
    one: lead time and false alarms trade against each other along it, and a
    single threshold hides which side of that trade the model is on.
    """
    return tuple(
        alarms(
            scores,
            life,
            position,
            onset,
            rule=AlarmRule(threshold=float(value), consecutive=consecutive),
        )
        for value in thresholds
    )
