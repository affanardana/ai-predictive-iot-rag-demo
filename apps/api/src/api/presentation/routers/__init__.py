"""HTTP routers."""

from api.presentation.routers.events import router as events_router
from api.presentation.routers.health import router as health_router
from api.presentation.routers.incidents import router as incidents_router
from api.presentation.routers.knowledge import router as knowledge_router
from api.presentation.routers.machines import router as machines_router
from api.presentation.routers.simulations import router as simulations_router
from api.presentation.routers.telemetry import router as telemetry_router

__all__ = [
    "events_router",
    "health_router",
    "incidents_router",
    "knowledge_router",
    "machines_router",
    "simulations_router",
    "telemetry_router",
]
