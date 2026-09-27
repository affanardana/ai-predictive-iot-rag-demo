"""The failure of having asked too often.

Beside `AuthenticationError` rather than in `api.domain.errors`, and for the
reason that module gives: being over a budget is a property of how a request
arrived, not of any business rule. The domain's own capacity bounds --
`ChatBusyError`, `TooManySimulationsError` -- are about the *box's* state and
would exist with no network at all; this one is meaningless without a caller.

Unlike `AuthenticationError`, it carries detail. Telling a caller how long the
wait is leaks nothing they cannot measure, and a client that knows does not
retry into the same wall.
"""

from __future__ import annotations


class RateLimitError(Exception):
    """The caller has spent their budget.

    The message states the wait and not the rate. A rate written into prose is a
    second copy of a configured number, and the failure mode is a 429 that
    explains a limit nobody set any more -- so the rate travels as a field, and
    the number a client actually needs is the one it can act on.
    """

    def __init__(self, *, retry_after_seconds: int, limit: str) -> None:
        """Record the wait and which budget was spent."""
        super().__init__(
            f"Too many requests from this address. Try again in {retry_after_seconds} seconds."
        )
        self.retry_after_seconds = retry_after_seconds
        #: The budget's name -- `copilot`, `simulation`, `search`, `global`.
        #: Debugging a 429 means knowing which ceiling was hit, and it is not
        #: always the obvious one: the global budget is shared by every route.
        self.limit = limit
