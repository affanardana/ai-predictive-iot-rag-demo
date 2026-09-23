"""Timestamp guards."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

import pytest

from api.domain.errors import DomainValidationError
from api.domain.timestamps import ensure_aware, utc_now


def test_accepts_aware_datetimes() -> None:
    """A UTC datetime passes through unchanged."""
    moment = datetime(2026, 9, 23, 12, 0, tzinfo=UTC)

    assert ensure_aware(moment, "recorded_at") is moment


def test_accepts_a_non_utc_offset() -> None:
    """Any real offset is acceptable; only naivety is rejected.

    The system stores UTC, but rejecting a +07:00 datetime would be wrong --
    it is unambiguous, and converting it is a storage concern.
    """
    moment = datetime(2026, 9, 23, 19, 0, tzinfo=timezone(timedelta(hours=7)))

    assert ensure_aware(moment, "recorded_at") == moment


def test_rejects_naive_datetimes() -> None:
    """A naive datetime fails with the field name in the message.

    Naive values are a latent bug: comparing one against an aware value raises
    `TypeError`, and persisting one against a `timestamptz` column silently
    assumes the server's zone.
    """
    with pytest.raises(DomainValidationError) as caught:
        ensure_aware(datetime(2026, 9, 23, 12, 0), "detected_at")

    assert "detected_at" in str(caught.value)


def test_utc_now_is_aware_and_utc() -> None:
    """The helper always returns a UTC-aware instant."""
    now = utc_now()

    assert now.tzinfo is not None
    assert now.utcoffset() == timedelta(0)
