r"""Modal deployment for n8n, the pipeline's orchestrator.

Nothing else depends on this file. n8n is self-hosted rather than bought because
`MASTERPLAN.md` §25 names "n8n Cloud handles workflow orchestration" and n8n
Cloud is not free; running it here keeps the named stack rather than dropping a
stated product constraint.

**This is an unusual thing to do.** Modal is built for request-scoped work that
scales to zero, and n8n is a stateful Node application whose MQTT Trigger holds
a persistent connection to a broker. `min_containers=1` is what makes that work,
and it is also the cost: a container is warm around the clock whether or not any
telemetry is flowing. Set it to 0 between demonstrations to stop paying for it,
and accept that telemetry published while it is down is not delivered -- an
MQTT subscription that is not connected is not a queue.

Deploying is deliberately not something this repository does on your behalf:

    modal secret create pdm-n8n \
        PDM_INGEST_TOKEN=<the same value the API has> \
        PDM_API_BASE_URL=https://<your-deployed-api>

    modal volume create pdm-n8n-data
    modal deploy infra/n8n/modal_app.py

**Do not put `N8N_ENCRYPTION_KEY` in the secret.** n8n generates one on first
boot and writes it to `/data/.n8n/config` on the Volume. Setting the environment
variable is optional and is only correct when it matches that stored value --
and when it does not, n8n does not degrade gracefully, it refuses to start:

    Error: Mismatching encryption keys. The encryption key in the settings file
    /data/.n8n/config does not match the N8N_ENCRYPTION_KEY env var.

The variable exists for restoring an instance whose Volume you still have; for
a first deployment it is a way to break a working instance. That is not
hypothetical: this project's own first deployment was broken by it.

The trade is that the key lives only on the Volume, so losing the Volume loses
the stored credentials. For a demonstration instance that is the right side of
the trade; a real one would back the key up somewhere and restore it through
this variable.

`PDM_API_BASE_URL` and `PDM_INGEST_TOKEN` are read by the workflow itself,
through `$env`. Secrets belong in the environment, not in a workflow file that
is committed.
"""

from __future__ import annotations

import subprocess

import modal

APP_NAME = "pdm-n8n"
VOLUME_NAME = "pdm-n8n-data"
#: The name of the Modal Secret, not its contents. `S105` reads the word
#: "secret" next to an assignment and reports a hardcoded password; this is a
#: label.
SECRET_NAME = "pdm-n8n"  # noqa: S105

#: Where n8n keeps its SQLite database and its encrypted credentials.
#:
#: Set explicitly rather than left to default. n8n resolves its data directory
#: from the running user's home -- `/home/node/.n8n` as the image's `node` user,
#: `/root/.n8n` as root -- and a Volume mounted at the wrong one would look like
#: a working deployment that forgot everything on every restart.
DATA_DIR = "/data"

N8N_PORT = 5678

#: The official image tag, pinned. `latest` would mean a redeploy silently
#: changes the Node version, the node library, and the editor with no diff to
#: review.
N8N_IMAGE = "docker.io/n8nio/n8n:1.115.3"

#: What n8n reads from its environment.
N8N_ENVIRONMENT = {
    "N8N_USER_FOLDER": DATA_DIR,
    "N8N_PORT": str(N8N_PORT),
    # Binding every interface is required, not sloppy: Modal routes to the
    # container from outside it, so a server listening on localhost would never
    # answer. `S104` cannot tell the difference.
    "N8N_LISTEN_ADDRESS": "0.0.0.0",  # noqa: S104
    "N8N_PROTOCOL": "https",
    # Modal terminates TLS in front of the container, so the cookies n8n sets
    # are already travelling over https by the time a browser sees them.
    "N8N_SECURE_COOKIE": "true",
    # `$env` is how the workflow reads its API URL and token. n8n blocks
    # environment access from nodes by default, which would leave both
    # expressions empty and every request pointed at "undefined/api/v1/...".
    "N8N_BLOCK_ENV_ACCESS_IN_NODE": "false",
    # Telemetry arrives at roughly one message per second per machine. Keeping
    # an execution record for every successful one would grow the SQLite file
    # without ever being read; failures are kept, because a failure is the
    # thing worth looking at.
    "EXECUTIONS_DATA_SAVE_ON_SUCCESS": "none",
    "EXECUTIONS_DATA_SAVE_ON_ERROR": "all",
    "EXECUTIONS_DATA_PRUNE": "true",
    "EXECUTIONS_DATA_MAX_AGE": "168",
    "GENERIC_TIMEZONE": "UTC",
    "TZ": "UTC",
}

image = modal.Image.from_registry(
    N8N_IMAGE,
    # A Modal function *is* Python -- the body below shells out to n8n -- so
    # Modal needs a Python interpreter it can identify inside the container.
    # The n8n image is Node and has none.
    #
    # **The image is Alpine, and that is why this took four attempts.** `apk`,
    # not `apt-get`; musl, not glibc. Modal's documented remedy,
    # `add_python="3.12"`, installs a *glibc* standalone CPython build -- which
    # cannot execute on musl at all. So the build showed `COPY /python/.` and
    # `ln -s .../python3` succeeding, and the deploy then reported "unable to
    # determine the version of Python installed in the Image": there was a
    # Python file there, and it could not run.
    #
    # The other wrong turn is worth recording too. Building on `debian_slim`
    # and installing n8n from npm satisfies Modal completely and produces a
    # container where n8n dies on startup -- Node 22.18+ type-strips TypeScript
    # by default and cannot apply the decorators n8n's dependency injection is
    # built on, while Node 20 cannot build `isolated-vm`, which its tree needs.
    # The official image exists so nobody has to discover that.
    setup_dockerfile_commands=[
        # The image runs as `node`. Modal's build steps assume root.
        "USER root",
        # The official image sets an entrypoint that runs n8n itself; Modal
        # launches the process, so the two would both try.
        "ENTRYPOINT []",
        # Alpine's own Python, which is what matches this image's libc.
        "RUN apk add --no-cache python3 py3-pip",
        *(f"ENV {key}={value}" for key, value in N8N_ENVIRONMENT.items()),
    ],
)

app = modal.App(APP_NAME, image=image)
data_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


@app.cls(
    volumes={DATA_DIR: data_volume},
    secrets=[modal.Secret.from_name(SECRET_NAME)],
    # The whole point of this file. n8n's MQTT Trigger holds a subscription
    # open, so the process has to be running when a message arrives -- there is
    # no request to wake it.
    min_containers=1,
    # One container. n8n's SQLite cannot be shared between replicas, and two
    # editors over one database is a corruption waiting to happen.
    max_containers=1,
    # Modal's default would recycle the container every few minutes, breaking
    # the broker connection on a schedule. Daily is a compromise: long enough
    # not to interrupt a demonstration, short enough to pick up an image change.
    timeout=24 * 60 * 60,
)
@modal.concurrent(
    # Modal's default is one input at a time per container, which n8n cannot
    # work under: its editor opens a persistent websocket at `/rest/push` for
    # live updates, and that single connection then occupies the container
    # forever. Every other request queues behind it, including the one that
    # renders the workflows page. Raising `max_containers` would not help; it
    # would put two n8n replicas on one SQLite, which is what `max_containers=1`
    # above exists to prevent.
    #
    # This has to be a class. Stacking `@modal.concurrent` directly over
    # `@modal.web_server` on a plain `@app.function` is silently ignored -- the
    # deploy succeeds, both decorators return a `_PartialFunction`, and only one
    # of them survives. The symptom is a container that works when idle and
    # hangs the moment a browser opens it, with the queued requests appearing
    # nowhere in n8n's logs because Modal never forwards them.
    #
    # These are proxied HTTP requests, not calls into the Python, which runs
    # once to start n8n and returns. n8n is a web server and handles concurrency
    # itself; this only stops Modal from serialising it.
    max_inputs=100,
)
class N8nServer:
    """n8n, launched once per container and left running."""

    @modal.web_server(
        port=N8N_PORT,
        # The default is five seconds, and n8n spends longer than that running
        # its migrations on a cold Volume. Without this the container starts,
        # n8n comes up a moment later, and Modal has already given up on the
        # port.
        startup_timeout=300,
    )
    def serve(self) -> None:
        """Start n8n.

        Deliberately not `n8n start --tunnel`, and deliberately not waiting:
        Modal returns as soon as this returns, and it is the port opening that
        signals readiness.
        """
        # S607: `n8n` is resolved from PATH rather than given as an absolute
        # path, which is how the image installs it. The argv is a literal, so
        # S603 does not fire.
        subprocess.Popen(["n8n", "start"])  # noqa: S607


@app.local_entrypoint()
def check() -> None:
    """Report whether the Secret holds what the workflow needs, before deploying.

    `N8N_ENCRYPTION_KEY` is deliberately not required. n8n writes its own key to
    the Volume on first boot, and an environment variable that disagrees with it
    stops the container from starting at all.
    """
    print(f"secret {SECRET_NAME}")
    try:
        modal.Secret.from_name(SECRET_NAME, required_keys=["PDM_INGEST_TOKEN", "PDM_API_BASE_URL"])
    except modal.exception.NotFoundError:
        print(f"  MISSING  create it with: modal secret create {SECRET_NAME} ...")
    else:
        print("  ok  PDM_INGEST_TOKEN and PDM_API_BASE_URL are set")
        print("      PDM_INGEST_TOKEN must equal the API's INGEST_API_TOKEN")
