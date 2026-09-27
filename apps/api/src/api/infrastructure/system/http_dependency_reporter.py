"""Asks every service this API depends on what it is.

The two side services publish no host port -- only the API and n8n do -- so
nothing outside the Compose bridge can reach them, and a scraper or a person
asking "is the model service up, and which model is it running" has no way to
find out. This adapter is that view: it calls each service's own endpoint from
inside the network and reports what came back.

**Nothing here raises.** A dependency that is down is the thing being reported,
so a failure to reach one is an answer rather than an error -- and an endpoint
whose entire job is to say which parts are broken cannot itself be broken by
one of them being broken.
"""

from __future__ import annotations

import logging

import httpx

from api.domain.ports.health import DependencyReport, HealthProbe
from api.request_context import correlation_headers

logger = logging.getLogger(__name__)

#: Short. A dependency report is read by a person or scraped on a schedule, and
#: an answer that takes longer than this is `unreachable` either way.
TIMEOUT_SECONDS = 5.0


class HttpDependencyReporter:
    """Reports on the database, the model service and the simulator."""

    def __init__(
        self,
        *,
        database_probe: HealthProbe,
        inference_url: str,
        simulator_url: str,
        client: httpx.AsyncClient,
    ) -> None:
        self._database_probe = database_probe
        self._inference_url = inference_url.rstrip("/")
        self._simulator_url = simulator_url.rstrip("/")
        self._client = client

    async def report(self) -> list[DependencyReport]:
        """Return one entry per dependency, in a stable order."""
        return [
            await self._database(),
            await self._inference(),
            await self._simulator(),
        ]

    async def _database(self) -> DependencyReport:
        """Report the database, through the same probe readiness uses."""
        status = await self._database_probe.check()
        return DependencyReport(
            name="database",
            reachable=status.healthy,
            detail=status.detail,
            models={},
        )

    async def _inference(self) -> DependencyReport:
        """Report the model service and the identities it loaded."""
        body = await self._get_object(f"{self._inference_url}/health")
        if body is None:
            return DependencyReport(
                name="inference",
                reachable=False,
                detail="The model service did not answer.",
                models={},
            )
        # Named exactly as the service names them, minus its own `status` field:
        # this endpoint's job is to report identities, not to reinterpret them.
        models = {
            key: str(value)
            for key, value in body.items()
            if key != "status" and isinstance(value, (str, int, float))
        }
        return DependencyReport(
            name="inference",
            reachable=True,
            detail="The model service is answering.",
            models=models,
        )

    async def _simulator(self) -> DependencyReport:
        """Report the simulator.

        `/runs` rather than a health route, because the service has none -- its
        Compose healthcheck uses the same endpoint for the same reason. An empty
        list is a healthy simulator: doing nothing is what it is for.
        """
        # `_get`, not `_get_object`: this endpoint answers with a *list* of runs,
        # and reading it as an object would report a healthy simulator as
        # unreachable.
        body = await self._get(f"{self._simulator_url}/runs")
        if body is None:
            return DependencyReport(
                name="simulator",
                reachable=False,
                detail="The simulator did not answer.",
                models={},
            )
        active = len(body) if isinstance(body, list) else 0
        return DependencyReport(
            name="simulator",
            reachable=True,
            detail=f"The simulator is answering with {active} run(s) on its books.",
            models={},
        )

    async def _get_object(self, url: str) -> dict[str, object] | None:
        """Fetch and parse a JSON *object*, or return None having logged why not."""
        parsed = await self._get(url)
        return parsed if isinstance(parsed, dict) else None

    async def _get(self, url: str) -> object | None:
        """Fetch and parse JSON, or return None having logged why not."""
        try:
            response = await self._client.get(
                url,
                headers=correlation_headers(),
                timeout=TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            # Annotated because `json()` is typed `Any`, and returning it
            # directly would switch the return type off at this call's edge.
            parsed: object = response.json()
            return parsed
        except (httpx.HTTPError, ValueError) as exc:
            # Logged at info, not warning. This endpoint exists to report
            # outages, so every call to it while something is down would
            # otherwise write a warning on a schedule -- noise that trains a
            # reader to ignore the level.
            logger.info(
                "infrastructure.dependencies.unreachable",
                extra={"url": url, "error_type": type(exc).__name__},
            )
            return None
