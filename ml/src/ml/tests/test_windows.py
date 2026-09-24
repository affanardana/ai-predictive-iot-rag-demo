"""Sequence generation: sixty steps, one life, before onset."""

from __future__ import annotations

import pytest

from ml.dataset.windows import (
    WINDOW_MINUTES,
    global_slice,
    iter_windows,
    window_ends,
    window_slice,
)


def test_a_healthy_life_can_be_windowed_from_its_first_full_step() -> None:
    """No onset means no upper bound but the life's own end."""
    assert window_ends(200, None) == range(WINDOW_MINUTES - 1, 200)


def test_a_failing_life_stops_one_step_before_its_onset() -> None:
    """The onset timestep is already the failure, so a window may not end there."""
    ends = window_ends(200, onset=150)

    assert ends.start == WINDOW_MINUTES - 1
    assert ends.stop == 150
    assert max(ends) == 149


def test_a_life_shorter_than_the_window_yields_nothing() -> None:
    """Rather than a window padded across a repair."""
    assert len(window_ends(30, None)) == 0
    assert len(window_ends(WINDOW_MINUTES - 1, None)) == 0
    assert len(window_ends(WINDOW_MINUTES, None)) == 1


def test_a_life_that_fails_immediately_yields_nothing() -> None:
    assert len(window_ends(500, onset=10)) == 0


@pytest.mark.parametrize("end", [59, 100, 1439])
def test_a_window_is_exactly_sixty_steps_ending_at_the_given_position(end: int) -> None:
    window = window_slice(end)

    assert window.stop - window.start == WINDOW_MINUTES
    assert window.stop == end + 1
    assert window.start == end - WINDOW_MINUTES + 1


def test_windows_are_translated_into_the_flat_arrays() -> None:
    """A life's rows are contiguous, so the offset is all that is needed."""
    window = window_slice(100)

    assert global_slice(1_000, window) == slice(1_000 + 41, 1_000 + 101)


def test_iteration_walks_every_valid_window() -> None:
    windows = list(iter_windows(200, None))

    assert len(windows) == 200 - WINDOW_MINUTES + 1
    assert windows[0] == slice(0, WINDOW_MINUTES)
    assert windows[-1] == slice(200 - WINDOW_MINUTES, 200)


def test_stride_thins_the_near_duplicates() -> None:
    """A coarser pass over the near-duplicates is available, but never implicit.

    Consecutive windows share 59 of their 60 timesteps, so thinning them is
    sometimes what a caller wants — and always something a caller should ask for.
    """
    every_tenth = list(iter_windows(200, None, stride=10))

    assert len(every_tenth) < len(list(iter_windows(200, None)))
    assert all(window.stop - window.start == WINDOW_MINUTES for window in every_tenth)


def test_a_stride_below_one_is_rejected() -> None:
    with pytest.raises(ValueError, match="stride"):
        list(iter_windows(200, None, stride=0))
