"""The simulator service, reached over HTTP.

Mirrors `HttpPredictor` in shape and in its error handling: a connection failure
becomes a domain error naming the dependency, so an outage reads as an outage
rather than as a bug in whichever request happened to hit it first.
"""

from __future__ import annotations

import httpx

from api.domain.errors import SimulationUnavailableError
from api.domain.ports.simulation import SimulationPlan

#: The service's own control prefix. `simulator.control` serves these.
RUNS_PATH = "/runs"

#: Long enough for a container to accept a run on a loaded box, short enough
#: that a dashboard press does not hang. Runs are accepted, not awaited, so this
#: is a round-trip timeout rather than a run duration.
DEFAULT_TIMEOUT_SECONDS = 10.0


class HttpSimulationController:
    """Starts and stops runs by calling the simulator's control surface."""

    def __init__(self, base_url: str, client: httpx.AsyncClient) -> None:
        self._base_url = base_url.rstrip("/")
        self._client = client

    async def start(self, plan: SimulationPlan) -> None:
        """Ask the simulator to begin a run.

        Raises:
            SimulationUnavailableError: if the service could not be reached, or
                answered with anything other than success. A 409 -- the machine
                already running -- arrives here too, and is reported as an
                outage rather than silently succeeding: from the caller's point
                of view the run it asked for is not the run that is happening,
                and saying so is better than a success it cannot rely on.
        """
        await self._post(
            RUNS_PATH,
            {
                "session_id": plan.session_id,
                "machine_id": plan.machine_id.value,
                # The `.value` matters: the simulator parses its own enum, and a
                # raw `str()` of a StrEnum would send the same text only by
                # accident of implementation.
                "scenario": plan.scenario.value,
                "seed": plan.seed,
                "started_at": plan.started_at.isoformat(),
                "duration_minutes": int(plan.duration.total_seconds() // 60),
                "sample_interval_seconds": int(plan.sample_interval.total_seconds()),
                "tick_seconds": plan.tick_seconds,
                "start_index": plan.start_index,
            },
        )

    async def stop(self, session_id: str) -> None:
        """Ask the simulator to end a run.

        A 404 is success. The run is not there, which is the state stopping it
        was trying to reach -- and a stop that failed because the run had
        already finished would be an escape hatch that refuses to open.

        Raises:
            SimulationUnavailableError: if the service could not be reached.
        """
        try:
            response = await self._client.delete(f"{self._base_url}{RUNS_PATH}/{session_id}")
        except httpx.HTTPError as error:
            raise SimulationUnavailableError(
                f"The simulator service at {self._base_url} could not be reached: {error}"
            ) from error

        if response.status_code == httpx.codes.NOT_FOUND:
            return
        if response.is_error:
            raise SimulationUnavailableError(
                f"The simulator refused to stop '{session_id}' with status {response.status_code}."
            )

    async def aclose(self) -> None:
        """Close the underlying client."""
        await self._client.aclose()

    async def _post(self, path: str, payload: dict[str, object]) -> None:
        """Send a request, translating transport failures into a domain error."""
        try:
            response = await self._client.post(f"{self._base_url}{path}", json=payload)
        except httpx.HTTPError as error:
            raise SimulationUnavailableError(
                f"The simulator service at {self._base_url} could not be reached: {error}"
            ) from error

        if response.is_error:
            raise SimulationUnavailableError(
                f"The simulator refused the request with status {response.status_code}: "
                f"{response.text[:200]}"
            )
