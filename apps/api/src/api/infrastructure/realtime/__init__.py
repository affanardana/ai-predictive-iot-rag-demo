"""Realtime adapters.

The in-process event broadcaster behind the dashboard's live update stream. A
cross-process implementation would live beside it and swap in at the
composition root without any other file changing.
"""

from api.infrastructure.realtime.broadcaster import InProcessEventBroadcaster

__all__ = ["InProcessEventBroadcaster"]
