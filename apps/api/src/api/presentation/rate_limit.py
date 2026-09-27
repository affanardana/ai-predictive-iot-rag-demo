"""Per-caller budgets for the routes that cost something.

`ADR 0007`, `0008` and `0009` each left an unguarded route behind with bounds in
place of a token: reads are open because a browser reads them, and a token in a
browser bundle is a boundary in appearance only. Phase 11 keeps that decision —
a login a reviewer cannot pass would delete the demonstration — and this is what
takes the guard's place on the three routes that spend real resources.

**What this is.** A sustained-rate ceiling per source address, with a bounded
burst, on a box with one core and no swap. It stops one caller monopolising the
Copilot's single answer slot so that everyone else gets `409 busy` forever; it
stops a start/reset loop; it stops a scraper walking the public read routes. It
costs one dictionary lookup per request and nothing when idle.

**What this is not.** It is not a defence against many addresses: an attacker
with an IPv6 /48 or a botnet spends this box regardless, and what bounds *that*
damage remains the concurrent-run ceiling, the single Copilot slot and the
container's memory limit. It is not durable -- a restart empties every bucket,
so a deploy is a free burst for everyone, which is the correct direction to fail
(a limiter that fails closed turns a deploy into an outage for the reviewer). It
is not per user: a household behind one address shares a budget, which is also
the honest unit, because that is the address a person is.

The state is in-process, and that is exact rather than approximate here only
because `Dockerfile.api` starts a single worker — the same property the event
broadcaster relies on, and `test_the_api_runs_as_a_single_process` fails the
suite if it ever changes.
"""

from __future__ import annotations

import math
import time
from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum

from api.presentation.errors.rate_limit import RateLimitError

#: How long a burst may be spent over. A shape decision rather than a host fact,
#: which is why it is a constant and the *rate* is a setting: bursts let a
#: person click twice, and the rate is what bounds the box.
_SECONDS_PER_MINUTE = 60.0


class Cost(StrEnum):
    """The budgets a request can be charged to.

    Named rather than keyed by path, so that a Copilot-heavy caller does not
    exhaust the budget the dashboard's own polling depends on. Every name here
    must be present in the limits mapping; a missing one raises at startup.
    """

    GLOBAL = "global"
    COPILOT = "copilot"
    SIMULATION = "simulation"
    SEARCH = "search"


@dataclass(frozen=True, slots=True)
class Limit:
    """A sustained rate and the burst allowed above it."""

    per_minute: float
    burst: int

    def __post_init__(self) -> None:
        """Reject a limit that could never be satisfied."""
        if self.per_minute <= 0:
            raise ValueError("A rate limit must allow more than zero requests.")
        if self.burst < 1:
            raise ValueError("A burst of zero would refuse every caller outright.")


@dataclass(slots=True)
class _Bucket:
    """One caller's remaining tokens for one budget."""

    tokens: float
    updated_at: float


@dataclass(slots=True)
class ClientRateLimiter:
    """Token buckets per (caller, budget), bounded in both directions.

    Lazy refill rather than a timer: the bucket is topped up by however much
    time has passed when it is next read. A background sweep task on a one-core
    box would be a second thing to hold memory and a second thing to stop
    silently.
    """

    limits: Mapping[Cost, Limit]
    max_tracked_clients: int = 4096
    #: `time.monotonic`, not the domain's `Clock`: that one reports business
    #: time as datetimes, and reusing it here would make the limiter jump
    #: whenever the host's clock was corrected.
    clock: Callable[[], float] = time.monotonic
    _buckets: OrderedDict[tuple[str, Cost], _Bucket] = field(default_factory=OrderedDict)

    def __post_init__(self) -> None:
        """Fail at startup if a budget has no limit."""
        missing = [cost.value for cost in Cost if cost not in self.limits]
        if missing:
            raise ValueError(f"No rate limit is configured for: {', '.join(missing)}.")

    def charge(self, key: str, cost: Cost) -> None:
        """Spend one token from `key`'s budget for `cost`.

        Raises:
            RateLimitError: if the budget is spent, carrying the wait.
        """
        limit = self.limits[cost]
        bucket_key = (key, cost)
        now = self.clock()
        bucket = self._buckets.get(bucket_key)

        if bucket is None:
            # A new caller starts with a full burst: a first visit is not an
            # attack, and starting empty would refuse the demonstration's own
            # first click.
            bucket = _Bucket(tokens=float(limit.burst), updated_at=now)
        else:
            elapsed = max(0.0, now - bucket.updated_at)
            refill = elapsed * limit.per_minute / _SECONDS_PER_MINUTE
            bucket.tokens = min(float(limit.burst), bucket.tokens + refill)
            bucket.updated_at = now
            self._buckets.move_to_end(bucket_key)

        if bucket.tokens < 1.0:
            self._buckets[bucket_key] = bucket
            wait = math.ceil((1.0 - bucket.tokens) * _SECONDS_PER_MINUTE / limit.per_minute)
            raise RateLimitError(
                retry_after_seconds=max(1, wait),
                limit=cost.value,
            )

        bucket.tokens -= 1.0
        self._buckets[bucket_key] = bucket
        self._evict()

    @property
    def tracked_clients(self) -> int:
        """How many buckets are held. Exists for a test and for a diagnosis."""
        return len(self._buckets)

    def _evict(self) -> None:
        """Drop the least recently used buckets, keeping memory bounded.

        An evicted caller gets a fresh bucket, which is fail-open: they are
        allowed a burst again. That is deliberate -- the alternative is a
        limiter whose memory grows with the number of addresses that have ever
        visited, on a box where memory is the binding constraint -- and it is
        stated here rather than discovered.
        """
        while len(self._buckets) > self.max_tracked_clients:
            self._buckets.popitem(last=False)
