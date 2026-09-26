"""Event loop compatibility for Windows development.

psycopg 3's async support registers readers and writers on the running event
loop. Windows' default `ProactorEventLoop` does not implement those, so every
async database call fails with:

    Psycopg cannot use the 'ProactorEventLoop' to run in async mode

Linux and macOS default to a selector-based loop, so production and CI are
untouched. This exists so the application and its PostgreSQL tests can run on a
Windows development machine.
"""

from __future__ import annotations

import asyncio


def use_psycopg_compatible_event_loop() -> None:
    """Install a selector-based loop policy where psycopg 3 requires one.

    A no-op on Linux and macOS, where the default policy is already
    selector-based, so this cannot change production behaviour.

    **The lookup decides, and there is deliberately no `sys.platform` test.**
    The policy class is reached by name because it does not exist in `asyncio` on
    other platforms -- and its absence is exactly the condition this function
    cares about. An earlier version tested `sys.platform` first and then looked
    the class up, which reads more obviously and is a platform-dependent bug:
    mypy evaluates `sys.platform` against the platform it is running on, so on
    Linux the guard is always true, everything after it is unreachable, and
    `warn_unreachable` says so -- while the same code passes on a Windows
    development machine. CI caught it; the author's `uv run mypy` never could.

    Keying on the lookup removes the branch and the platform dependence with it.
    """
    policy_class = getattr(asyncio, "WindowsSelectorEventLoopPolicy", None)
    if policy_class is None:
        return

    # Instantiated, not passed as the class: `set_event_loop_policy` takes a
    # policy object.
    asyncio.set_event_loop_policy(policy_class())
