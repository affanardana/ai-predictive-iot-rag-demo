"""Turning hidden truth into a label, and saying what the label means.

## What counts as a failure

Twice now the obvious answer has been wrong, so it is worth stating exactly.

The failure event is **the first instant a life reports `failure_imminent`** —
the crossing of `health_index` below `FAILURE_HEALTH_THRESHOLD`, which
`simulator.domain.physics` sets at 0.2. That is the simulator's own definition
of a machine going critical, and Phase 2 recorded the intention next to the
constant: *"Phase 3 labels training data from this."*

The tempting alternative is to call the end of a life the failure, since that is
where damage saturates. It is wrong for two reasons.

**It makes lead time meaningless.** The product's headline number is lead time.
If a window is positive only in the sixty minutes before a life ends, then a
perfect model's lead time is capped at sixty minutes and is uniform on
``[0, 60]`` — the metric would be reporting the label definition back at us. On
the threshold definition, lead time runs from the alarm to the point where the
damage actually saturates, which is hours, and which is the number the product
promises.

**It punishes exactly the behaviour the product wants.** A machine whose worst
channel has reached 0.9 and is still climbing is one an operator needs to hear
about, and it is the machine with the most remaining lead time. Under an
end-of-life label that window is scored *negative*, so an early, correct,
useful alarm counts as a false positive.

## What is not labelled

Rows at and after onset are **neither** positive nor negative, and are excluded
rather than assigned a class.

The machine has already entered failure. "Will it fail in the next sixty
minutes" is no longer the question being asked — it is the question that was
answered. Calling those rows negative would contradict the ground truth outright.
Calling them positive would flood the positive class with saturated,
trivially-separable rows and inflate every metric computed from it. They are
carried in the artifact with `is_trainable` false, and the report states what
fraction of rows they are, so the exclusion is visible rather than silent.

## The encoding

`minutes_to_onset` is one signed integer per timestep, and everything else is
derived from it:

======  ==================================================================
>= 1    the life enters failure in this many minutes
0       this timestep *is* the onset
< 0     onset was this many minutes ago
-32768  this life never enters failure at all
======  ==================================================================

One column carries the whole labelling, so a horizon other than sixty minutes
is a different expression over the same array rather than a regeneration.
"""

from __future__ import annotations

from collections.abc import Sequence

#: PRD §25.8: "Prediction horizon is 60 minutes".
HORIZON_MINUTES = 60

#: Marks a life that never enters failure — every `NORMAL` life, and any
#: degradation life whose observation window closes before its onset. The
#: sentinel is `int16`'s minimum, far below any value a real lifetime could
#: produce: a 48-hour life is 2,880 minutes, so nothing legitimate reaches this.
NEVER_FAILS = -32768


def onset_index(imminent: Sequence[bool]) -> int | None:
    """Return the first index at which failure is imminent, or `None`.

    `imminent` is the ground-truth column for one life, in tick order. The
    simulator's degradation is monotonic — damage never heals between repairs —
    so this is the crossing the rest of the module is built around. A later
    reversal would mean the life had been composed from more than one session,
    which `windows` exists to prevent.
    """
    for index, flag in enumerate(imminent):
        if flag:
            return index
    return None


def minutes_to_onset(count: int, onset: int | None) -> list[int]:
    """Return one signed offset per tick, relative to the life's onset."""
    if onset is None:
        return [NEVER_FAILS] * count
    return [onset - index for index in range(count)]


def label(minutes: int, horizon: int = HORIZON_MINUTES) -> int:
    """Return whether failure falls within the next `horizon` minutes.

    Strictly greater than zero: at onset the machine is failing *now*, which is
    a different question from whether it will fail soon, and those rows are
    excluded rather than labelled.
    """
    return int(0 < minutes <= horizon)


def is_trainable(minutes: int) -> bool:
    """Whether this timestep poses the question the model is asked to answer."""
    return minutes == NEVER_FAILS or minutes > 0


def is_post_onset(minutes: int) -> bool:
    """Whether the machine has already entered failure at this timestep."""
    return minutes != NEVER_FAILS and minutes <= 0


def positive_rate(all_minutes: Sequence[int], horizon: int = HORIZON_MINUTES) -> float:
    """Return the prevalence of the positive class over the trainable rows.

    Reported rather than assumed. The composition that produces this number is
    derived from the life lengths and the `NORMAL` share — see `lives` — and a
    change to either moves it.
    """
    trainable = [minutes for minutes in all_minutes if is_trainable(minutes)]
    if not trainable:
        return 0.0
    return sum(label(minutes, horizon) for minutes in trainable) / len(trainable)
