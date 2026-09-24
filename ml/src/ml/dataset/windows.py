"""Cutting sixty-step sequences, and the two rules that bound them.

## Why this is a function and not a stored tensor

The obvious artifact for a sequence model is a pre-expanded array of windows:
one row per `(window, label)` pair, ready to feed a `DataLoader`. It is 45 times
larger than it needs to be. Consecutive windows overlap by 59 of their 60
timesteps, so the same reading is stored sixty times over — the 4.84 million
windows this dataset produces would be 3.5 GB at `float16`, against 58 MB for the
flat per-timestep signals they are
sliced from. On Colab's free tier that difference is the whole budget.

So the artifact stores one row per `(life, timestep)` and this module turns those
rows into sequences. Windowing is index arithmetic, which is cheap; storage is
not, which is why the trade goes this way.

## The first rule: a window never crosses a life boundary

A life ends at the onset of failure, and the next begins immediately after — a
repair. Between them the signals step discontinuously, because a repaired machine
is a different machine. That step is an artefact of how the dataset was
composed, not physics, and a window spanning it would hand the model a jump it
could learn to read as a feature. Sixty timesteps of one life, always.

Enforced structurally rather than by care: windows are expressed as positions
*within* a life, and the caller offsets them into the flat arrays. There is no
way to express a window that crosses a boundary, so there is none to check for.

## The second rule: a window never includes the post-onset tail

Rows at and after onset are excluded — see `ml.dataset.labelling` for why. Since
a window ending at position `e` covers `[e - 59, e]`, requiring `e < onset` keeps
the whole span pre-onset. The end position is the only thing that needs
checking, because onset is by construction the first excluded row.
"""

from __future__ import annotations

from collections.abc import Iterator

#: PRD §25.9: "Sequence window is 60 minutes". Matches the prediction horizon,
#: so a model sees exactly as far back as it is asked to look forward.
WINDOW_MINUTES = 60


def window_ends(count: int, onset: int | None, *, window: int = WINDOW_MINUTES) -> range:
    """Return the positions within a life at which a valid window can end.

    Args:
        count: how many timesteps the life has.
        onset: the position of the life's first failed timestep, or `None` if
            the life never fails. Per `ml.dataset.labelling`.
        window: how many timesteps a sequence spans.

    Returns:
        End positions in ascending order, empty when the life is too short to
        hold a single complete window.
    """
    # The last usable end is the timestep before onset. Onset itself is already
    # the failing instant, so a window ending there would have to answer a
    # question that has been settled.
    last = count - 1 if onset is None else onset - 1
    return range(window - 1, last + 1)


def window_slice(end: int, *, window: int = WINDOW_MINUTES) -> slice:
    """Return the slice of a life's own rows that ends at `end`, inclusive."""
    return slice(end - window + 1, end + 1)


def global_slice(life_offset: int, local: slice) -> slice:
    """Translate a life-local slice into one over the flat artifact arrays."""
    return slice(life_offset + local.start, life_offset + local.stop)


def iter_windows(
    count: int,
    onset: int | None,
    *,
    stride: int = 1,
    window: int = WINDOW_MINUTES,
) -> Iterator[slice]:
    """Yield each valid window of a life, as a slice of its own rows.

    `stride` of one is the honest default. A larger stride thins the
    near-duplicates described below, but it also thins them unevenly, so it is
    something the caller asks for rather than something applied quietly.

    ## Sixty positives per failure are not sixty samples

    Consecutive windows share 59 timesteps and are about 98% identical to one
    another. Treating them as independent understates confidence intervals by
    roughly `sqrt(60)` and lets a single hard life contribute sixty false
    positives — so it is penalised sixty times harder than an easy one. The
    artifact carries machine and life identifiers precisely so Phase 4 can group
    by them and report per-failure-event figures; metrics computed per window
    will flatter the model, and the report says so.
    """
    if stride < 1:
        raise ValueError(f"stride must be at least 1, got {stride}.")
    for end in window_ends(count, onset, window=window)[::stride]:
        yield window_slice(end, window=window)
