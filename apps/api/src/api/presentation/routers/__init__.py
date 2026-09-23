"""HTTP routers."""

from api.presentation.routers.health import router as health_router
from api.presentation.routers.incidents import router as incidents_router
from api.presentation.routers.machines import router as machines_router

__all__ = ["health_router", "incidents_router", "machines_router"]
