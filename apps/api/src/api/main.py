"""Application entrypoint.

Run with:

    uv run uvicorn api.main:app --reload --app-dir apps/api/src
"""

from __future__ import annotations

from api.composition import build_container
from api.infrastructure.config import Settings
from api.infrastructure.event_loop import use_psycopg_compatible_event_loop
from api.infrastructure.logging import configure_logging
from api.presentation.app import create_app

# Before anything creates an event loop. Uvicorn imports this module and only
# then runs `asyncio.run`, so setting the policy here is early enough. On
# Windows it is also required: psycopg 3 cannot drive a Proactor loop, and the
# failure is a runtime error on the first query rather than anything at startup.
use_psycopg_compatible_event_loop()

_settings = Settings()
configure_logging(_settings.log_level)

app = create_app(build_container(_settings))
