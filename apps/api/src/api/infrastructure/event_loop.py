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
import sys


def use_psycopg_compatible_event_loop() -> None:
    """Install a selector-based loop policy where psycopg 3 requires one.

    A no-op on Linux and macOS, where the default policy is already
    selector-based, so this cannot change production behaviour.

    The Windows policy class is looked up by name rather than referenced
    directly: it does not exist in the `asyncio` namespace on other platforms,
    so a direct reference would fail to type check and fail at import time
    everywhere else.
    """
    if sys.platform != "win32":
        return

    policy_class = getattr(asyncio, "WindowsSelectorEventLoopPolicy", None)
    if policy_class is not None:
        asyncio.set_event_loop_policy(policy_class())
