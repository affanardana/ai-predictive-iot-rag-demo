"""The metrics endpoint.

Mounted beside `/health`, outside `/api/v1`, and excluded from the OpenAPI
document: it returns a text exposition rather than a schema, and declaring a
response model would put it in the document the frontend's generated types are
diffed against -- for a route no browser calls.

**Guarded when a token is configured, public when one is not.** The repository's
own test for whether a surface takes a credential is whether a *browser* reads
it (ADR 0007): reads are open because the dashboard is a browser, writes are
guarded because a browser cannot hold the token safely. No browser reads this
one. A scraper is a machine, and `ingest_api_token`'s docstring already names
that pattern -- "one machine proving to another that it is the expected caller".

The guard is optional rather than required because some hosted scrapers accept a
URL and nothing else, and a 401 that a collector can never satisfy fails as
"target down" with nothing in this container's logs to explain it. Unset means
public, which is also what local development and CI want.
"""

from __future__ import annotations

import hmac
from typing import Annotated

from fastapi import APIRouter, Header, Response
from prometheus_client import CONTENT_TYPE_LATEST

from api.observability.metrics import render
from api.presentation.dependencies import ContainerDep
from api.presentation.errors.authentication import AuthenticationError

router = APIRouter(tags=["observability"])

#: The scheme every scraper and agent already sends for a bearer credential.
_BEARER_PREFIX = "bearer "


@router.get(
    "/metrics",
    include_in_schema=False,
    summary="Prometheus metrics",
    response_class=Response,
)
async def metrics(
    container: ContainerDep,
    authorization: Annotated[str | None, Header()] = None,
) -> Response:
    """Render the registry in the Prometheus exposition format.

    Raises:
        AuthenticationError: if a metrics token is configured and the request
            does not carry it.
    """
    expected = container.settings.metrics_token
    if expected is not None:
        presented = authorization or ""
        if not presented.casefold().startswith(_BEARER_PREFIX):
            raise AuthenticationError
        # Compared as bytes, for the reason `require_ingest_token` gives: a
        # `str` holding a non-ASCII character makes `compare_digest` raise
        # `TypeError`, which would turn a malformed header into a 500.
        if not hmac.compare_digest(
            presented[len(_BEARER_PREFIX) :].encode("utf-8"), expected.encode("utf-8")
        ):
            raise AuthenticationError

    return Response(content=render(), media_type=CONTENT_TYPE_LATEST)
