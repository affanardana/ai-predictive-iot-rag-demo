"""Event loop policy compatibility."""

from __future__ import annotations

import asyncio

from api.infrastructure.event_loop import use_psycopg_compatible_event_loop


def test_never_leaves_a_proactor_policy_in_place() -> None:
    """The resulting policy must be one psycopg 3 can drive.

    On Windows that means replacing the Proactor default, which lacks
    `add_reader`; elsewhere the platform default already works and must be left
    untouched, so that production behaviour cannot be altered by a helper meant
    only for development machines.
    """
    before = asyncio.get_event_loop_policy()

    use_psycopg_compatible_event_loop()

    after = asyncio.get_event_loop_policy()
    proactor = getattr(asyncio, "WindowsProactorEventLoopPolicy", None)

    if proactor is None:
        assert after is before, "non-Windows platforms should be left untouched"
    else:
        assert not isinstance(after, proactor), "a Proactor loop cannot run psycopg 3"


def test_is_idempotent() -> None:
    """Calling it twice leaves the same kind of policy installed.

    Both the application entrypoint and the test suite call it, and in some
    invocations both will have run.

    Identity is deliberately not asserted: the helper installs a fresh instance
    on each call, and a new instance of the same policy is equivalent. Asserting
    identity would be testing an implementation detail rather than the
    behaviour that matters.
    """
    use_psycopg_compatible_event_loop()
    policy_type = type(asyncio.get_event_loop_policy())

    use_psycopg_compatible_event_loop()

    assert type(asyncio.get_event_loop_policy()) is policy_type


def test_the_installed_policy_can_build_a_usable_loop() -> None:
    """A loop can be created and closed under the installed policy.

    This is the property the policy exists to guarantee; asserting on the class
    name alone would pass even if the object were unusable.
    """
    use_psycopg_compatible_event_loop()

    loop = asyncio.new_event_loop()
    try:
        assert not loop.is_closed()
    finally:
        loop.close()
