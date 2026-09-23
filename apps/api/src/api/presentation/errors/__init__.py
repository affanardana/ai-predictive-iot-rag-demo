"""Error translation from domain failures to HTTP responses."""

from api.presentation.errors.error_catalog import ErrorMapping, resolve
from api.presentation.errors.handlers import register_error_handlers

__all__ = ["ErrorMapping", "register_error_handlers", "resolve"]
