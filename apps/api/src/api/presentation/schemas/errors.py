"""Error response envelope."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ErrorDetail(BaseModel):
    """Machine-readable error information.

    `code` is a stable identifier the frontend can branch on; `message` is
    human-readable and may change. Clients should never parse `message`.
    """

    model_config = ConfigDict(frozen=True)

    code: str = Field(description="Stable identifier for the failure kind.")
    message: str = Field(description="Human-readable explanation.")
    details: dict[str, str] | None = Field(
        default=None,
        description="Optional field-level validation details.",
    )


class ErrorResponse(BaseModel):
    """The single shape every error response takes."""

    model_config = ConfigDict(frozen=True)

    error: ErrorDetail
