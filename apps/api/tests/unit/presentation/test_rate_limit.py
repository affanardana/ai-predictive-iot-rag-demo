"""The limiter's arithmetic, with a clock the test owns.

No sleeps: the bucket is refilled from elapsed time, so the test advances a
number rather than waiting for one. That is also why `clock` is injectable on
the dataclass rather than read inside the methods.
"""

from __future__ import annotations

import pytest

from api.presentation.errors.rate_limit import RateLimitError
from api.presentation.rate_limit import ClientRateLimiter, Cost, Limit

LIMITS = {
    Cost.GLOBAL: Limit(per_minute=600, burst=120),
    Cost.COPILOT: Limit(per_minute=1, burst=2),
    Cost.SIMULATION: Limit(per_minute=1, burst=2),
    Cost.SEARCH: Limit(per_minute=6, burst=5),
}


class Clock:
    """A monotonic clock a test can move by hand."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        """Move time forward."""
        self.now += seconds


@pytest.fixture
def clock() -> Clock:
    """The injected clock."""
    return Clock()


@pytest.fixture
def limiter(clock: Clock) -> ClientRateLimiter:
    """A limiter over the deployment's own limits."""
    return ClientRateLimiter(limits=LIMITS, clock=clock)


def test_a_new_caller_may_spend_the_whole_burst(limiter: ClientRateLimiter) -> None:
    """A first visit is not an attack.

    Starting a bucket empty would refuse the demonstration's own first click,
    which is the one caller this must never turn away.
    """
    for _ in range(2):
        limiter.charge("203.0.113.9", Cost.COPILOT)


def test_the_burst_ends_and_the_wait_is_reported(limiter: ClientRateLimiter) -> None:
    """The refusal carries a real countdown, not a constant."""
    limiter.charge("203.0.113.9", Cost.COPILOT)
    limiter.charge("203.0.113.9", Cost.COPILOT)

    with pytest.raises(RateLimitError) as raised:
        limiter.charge("203.0.113.9", Cost.COPILOT)

    assert raised.value.limit == "copilot"
    # One a minute, so a whole minute of waiting after the burst is spent.
    assert raised.value.retry_after_seconds == 60


def test_time_refills_the_bucket(limiter: ClientRateLimiter, clock: Clock) -> None:
    """A sustained rate, not a hard ceiling for ever."""
    limiter.charge("203.0.113.9", Cost.COPILOT)
    limiter.charge("203.0.113.9", Cost.COPILOT)
    clock.advance(60)

    limiter.charge("203.0.113.9", Cost.COPILOT)


def test_a_long_idle_does_not_bank_more_than_the_burst(
    limiter: ClientRateLimiter, clock: Clock
) -> None:
    """Capacity is capped, or a caller who waits an hour gets an hour's tokens."""
    clock.advance(3600)

    for _ in range(2):
        limiter.charge("203.0.113.9", Cost.COPILOT)
    with pytest.raises(RateLimitError):
        limiter.charge("203.0.113.9", Cost.COPILOT)


def test_budgets_are_independent_for_one_caller(limiter: ClientRateLimiter) -> None:
    """Spending the Copilot budget leaves the others alone.

    This is what stops a Copilot-heavy visitor from also losing the budget the
    dashboard's polling depends on.
    """
    limiter.charge("203.0.113.9", Cost.COPILOT)
    limiter.charge("203.0.113.9", Cost.COPILOT)
    with pytest.raises(RateLimitError):
        limiter.charge("203.0.113.9", Cost.COPILOT)

    limiter.charge("203.0.113.9", Cost.SEARCH)
    limiter.charge("203.0.113.9", Cost.GLOBAL)


def test_one_caller_does_not_spend_anothers_budget(limiter: ClientRateLimiter) -> None:
    """The whole point: a budget is per address."""
    limiter.charge("203.0.113.9", Cost.COPILOT)
    limiter.charge("203.0.113.9", Cost.COPILOT)
    with pytest.raises(RateLimitError):
        limiter.charge("203.0.113.9", Cost.COPILOT)

    limiter.charge("198.51.100.4", Cost.COPILOT)


def test_memory_is_bounded_and_eviction_fails_open(
    clock: Clock,
) -> None:
    """The bound is real, and what it costs is stated rather than discovered.

    An evicted caller gets a fresh bucket -- they are allowed their burst again.
    That is the deliberate direction: the alternative is a limiter whose memory
    grows with every address that has ever visited, on a box where memory is the
    binding constraint.
    """
    limiter = ClientRateLimiter(limits=LIMITS, max_tracked_clients=10, clock=clock)

    for index in range(50):
        limiter.charge(f"198.51.100.{index}", Cost.COPILOT)

    assert limiter.tracked_clients <= 10

    # The very first caller was evicted, so their budget starts over.
    limiter.charge("198.51.100.0", Cost.COPILOT)
    limiter.charge("198.51.100.0", Cost.COPILOT)


def test_the_least_recently_used_bucket_is_the_one_evicted(clock: Clock) -> None:
    """Recency, not insertion order, or a chatty caller would be dropped first."""
    limiter = ClientRateLimiter(limits=LIMITS, max_tracked_clients=2, clock=clock)
    limiter.charge("first", Cost.COPILOT)
    limiter.charge("second", Cost.COPILOT)
    # Touching "first" makes "second" the least recently used.
    limiter.charge("first", Cost.COPILOT)
    limiter.charge("third", Cost.COPILOT)

    assert limiter.tracked_clients == 2
    # "first" survived: it still has a spent bucket rather than a fresh one.
    with pytest.raises(RateLimitError):
        limiter.charge("first", Cost.COPILOT)


def test_a_missing_budget_fails_at_startup() -> None:
    """A cost with no limit is a configuration error, not a silent free pass."""
    with pytest.raises(ValueError, match="global"):
        ClientRateLimiter(limits={Cost.COPILOT: Limit(per_minute=1, burst=1)})


def test_a_limit_that_could_never_be_satisfied_is_rejected() -> None:
    """Both fields are validated where they are set."""
    with pytest.raises(ValueError):
        Limit(per_minute=0, burst=1)
    with pytest.raises(ValueError):
        Limit(per_minute=1, burst=0)
