r"""Modal deployment for the API.

The last piece Phase 6's exit condition needs. n8n runs in Modal and cannot
reach a laptop, so "continuous cloud telemetry flows through the complete
backend pipeline" is not demonstrable until this is reachable from the internet.
That makes it Phase 11's container work arriving a few phases early.

Deploying is deliberately not something this repository does on your behalf:

    modal secret create pdm-api \
        DATABASE_URL="postgresql+psycopg://...@...pooler.supabase.com:5432/postgres" \
        INFERENCE_SERVICE_URL=https://<workspace>--pdm-inference-fastapi-app.modal.run \
        INGEST_API_TOKEN=<the same value n8n holds as PDM_INGEST_TOKEN>

    modal deploy apps/api/modal_app.py

Apply migrations **before** deploying, from a machine with network access, and
not from inside the container:

    uv run alembic -c apps/api/alembic.ini upgrade head

Migrations are a batch operation with one writer. Running them from a web
container means every replica races to apply the same revision on a cold start,
and the loser's failure is a confusing startup error rather than a migration
problem.

`INFERENCE_SERVICE_URL` is a secret rather than a constant in this file because
the deployed address of the inference service is not recorded anywhere in the
repository -- it is `https://<workspace>--pdm-inference-fastapi-app.modal.run`,
where the workspace is whatever `modal deploy` printed. Writing a remembered
value here would be a guess, and a wrong one fails as a request timeout rather
than as a configuration error. `Settings` defaults it to `http://localhost:8001`
for local work, which on a deployed container is a host that does not exist.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastapi import FastAPI

import modal

APP_NAME = "pdm-api"
#: The name of the Modal Secret, not its contents. `S105` reads the word
#: "secret" next to an assignment and reports a hardcoded password; this is a
#: label.
SECRET_NAME = "pdm-api"  # noqa: S105

#: Every setting the API cannot start without, or must not get wrong.
#:
#: `DATABASE_URL` has no default and the API exits without it. The other two do
#: have defaults, which is the problem: a deployed container that silently falls
#: back to `http://localhost:8001` or to running unauthenticated looks healthy
#: and fails in a way that points somewhere else entirely.
REQUIRED_KEYS = ["DATABASE_URL", "INFERENCE_SERVICE_URL", "INGEST_API_TOKEN"]

#: The API is conversational-sized work: a few queries and one HTTP call, with
#: no model in this process at all. That is the whole point of keeping torch out
#: of it, and it is what makes a small container viable.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "fastapi>=0.115",
        "uvicorn[standard]>=0.32",
        "pydantic>=2.9",
        "pydantic-settings>=2.6",
        "sqlalchemy[asyncio]>=2.0.36",
        # The binary wheel bundles libpq, which debian_slim does not have. The
        # source distribution would try to link against a library that is not
        # there and fail at import rather than at install.
        "psycopg[binary]>=3.2",
        "httpx>=0.28",
    )
    # Every build step precedes the local directory below; Modal rejects an
    # image that runs one afterwards, and local files added last are what keep
    # a source change from rebuilding the dependency layers.
    #
    # `APP_ENV=production` is set here rather than in the secret because it is
    # not a secret and must not be overridable by accident: it is what makes
    # the settings validator require an ingest token and refuse to start
    # without one.
    .env({"PYTHONPATH": "/root", "APP_ENV": "production"})
    .add_local_dir("apps/api/src/api", remote_path="/root/api")
)

app = modal.App(APP_NAME, image=image)


@app.function(
    secrets=[modal.Secret.from_name(SECRET_NAME)],
    # Scale to zero. Telemetry arrives at roughly one message a second while a
    # demonstration runs, which keeps a container warm on its own; between
    # demonstrations there is nothing to pay for.
    min_containers=0,
    max_containers=4,
    # A request that ingests a batch does one insert and a handful of counts.
    # The only slow path is scoring, which calls the inference service and can
    # meet a cold start there.
    timeout=120,
)
@modal.asgi_app()
def fastapi_app() -> FastAPI:
    """The application, built the way it is built locally."""
    from api.composition import build_container
    from api.infrastructure.config import Settings
    from api.infrastructure.event_loop import use_psycopg_compatible_event_loop
    from api.presentation.app import create_app

    # Before anything creates an event loop. The function body runs inside one
    # Modal has already created, so this is the last moment it can take effect:
    # psycopg 3 cannot drive the Proactor loop Windows uses, and while a Linux
    # container will not have that particular problem, setting the policy in
    # one place keeps the deployed process and the local one on the same code
    # path rather than hoping they agree.
    use_psycopg_compatible_event_loop()

    return create_app(build_container(Settings()))


@app.local_entrypoint()
def check() -> None:
    """Report whether the Secret holds what the API needs, before deploying.

    The API reads `DATABASE_URL` at startup and exits without it, and refuses to
    start outside `local` without `INGEST_API_TOKEN`. Both live in the secret,
    so a missing one is a deployment that boots and immediately dies -- which
    reads as a Modal problem rather than as a missing variable.
    """
    print(f"secret {SECRET_NAME}")
    try:
        modal.Secret.from_name(SECRET_NAME, required_keys=list(REQUIRED_KEYS))
    except modal.exception.NotFoundError:
        print(f"  MISSING  create it with: modal secret create {SECRET_NAME} ...")
    else:
        print(f"  ok  {', '.join(REQUIRED_KEYS)} are all set")
        print("      INGEST_API_TOKEN must equal n8n's PDM_INGEST_TOKEN")
