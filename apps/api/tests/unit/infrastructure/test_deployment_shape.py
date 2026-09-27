"""Constraints the deployment must keep, asserted rather than remembered.

A gate rather than a comment, because every one of these fails *silently*:
nothing errors, the deployment simply behaves in a way nobody can reproduce
locally, or exposes something nobody meant to expose.

The compose assertions exist for a sharper reason. `simulator` runs an HTTP
control surface with **no authentication**, which is only defensible because the
service publishes no host port and is reachable solely over the Compose bridge.
A `ports:` line added in a moment of debugging would put an open "start a run"
endpoint on the internet, and nothing would fail -- so this does.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tomllib
from pathlib import Path

import pytest
import yaml

#: The repository root. `__file__` walks up rather than being written absolute,
#: so the test works from any checkout -- the same approach
#: `test_n8n_workflow.py` takes to find the workflow export.
COMPOSE_DIR = Path(__file__).resolve().parents[5] / "infra" / "compose"

#: The API package, whose declared dependencies the image has to install.
API_PACKAGE = Path(__file__).resolve().parents[3]

#: Flags that start more than one server process.
WORKER_FLAGS = re.compile(r"(?:^|\s)(?:--workers|-w)(?:\s|=)")

#: A compose value that interpolates a variable, and its default.
INTERPOLATION = re.compile(r"\$\{(?P<name>[A-Z_]+)(?::-(?P<default>[^}]*))?\}")

#: Peak resident memory of the two jobs in the inference container, measured on
#: the deployment host with `docker stats` during a real request: the LSTM and
#: the two sentence encoders, and the Copilot's model with its KV cache.
RETRIEVAL_FOOTPRINT_MIB = 506
CHAT_FOOTPRINT_MIB = 1724


def _chat_model_file(value: str) -> str:
    """Return the file name a compose value resolves to.

    The same expression for both sides of the assertion, because that is the
    property under test: neither service writes the name itself, they both read
    it from the one variable.
    """
    match = INTERPOLATION.search(value)
    assert match is not None, f"{value!r} is hard-coded rather than shared"
    assert match.group("name") == "CHAT_MODEL_FILE"
    return match.group("default")


@pytest.fixture(scope="module")
def compose() -> dict[str, object]:
    """Return the parsed compose file."""
    text = (COMPOSE_DIR / "compose.yaml").read_text(encoding="utf-8")
    parsed = yaml.safe_load(text)
    assert isinstance(parsed, dict)
    return parsed


@pytest.fixture(scope="module")
def services(compose: dict[str, object]) -> dict[str, dict[str, object]]:
    """Return the compose file's services."""
    found = compose["services"]
    assert isinstance(found, dict)
    return found


def _dockerfile_text(name: str) -> str:
    return (COMPOSE_DIR / name).read_text(encoding="utf-8")


def _instructions(text: str) -> list[str]:
    """Return a Dockerfile's instructions, with comments removed.

    Comments matter here: these Dockerfiles explain the very rules the tests
    enforce, so a test searching the raw text fails on its own warning. Only
    what Docker would actually execute counts.
    """
    return [
        line for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")
    ]


def _cmd_line(text: str) -> str:
    """Return a Dockerfile's `CMD` line."""
    for line in _instructions(text):
        if line.startswith("CMD"):
            return line
    raise AssertionError("No CMD found.")


# --- The API container -------------------------------------------------------


def test_the_api_runs_as_a_single_process() -> None:
    """No `--workers`, because the event stream is fanned out in memory.

    `InProcessEventBroadcaster` delivers each change to the subscribers attached
    to the process that published it. A second worker would not fail: it would
    quietly deliver each event to an arbitrary subset of connected browsers, so
    one dashboard tab would update and another would sit still, and the bug
    would be blamed on the browser, the network, or the frontend.

    Making the bus cross-process is the fix if this ever needs to change --
    PostgreSQL `LISTEN`/`NOTIFY` or Redis -- and that is a deliberate piece of
    work rather than a flag.
    """
    assert not WORKER_FLAGS.search(_cmd_line(_dockerfile_text("Dockerfile.api")))


def test_the_image_installs_what_the_package_declares() -> None:
    """Every declared dependency is in the image's own pip list.

    The two lists are maintained by hand and drifted once already: `pgvector`
    was declared in `apps/api/pyproject.toml` and missing from
    `Dockerfile.api`, so the container crash-looped on import while every test
    passed -- CI installs the workspace, and only the image installs that list.
    """
    declared = tomllib.loads((API_PACKAGE / "pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]["dependencies"]
    installed = "\n".join(_instructions(_dockerfile_text("Dockerfile.api")))

    # Split on the version specifier and any extras, so `psycopg[binary]>=3.2`
    # is looked for as `psycopg`.
    missing = [
        re.split(r"[<>=!\[]", dependency)[0].strip()
        for dependency in declared
        if re.split(r"[<>=!\[]", dependency)[0].strip() not in installed
    ]

    assert missing == [], f"declared in pyproject.toml but not installed in the image: {missing}"


def test_no_secret_file_is_tracked() -> None:
    """§8: "Secrets are not committed". Asked of git, not of `.gitignore`.

    Reading the ignore rules would only prove they exist. Asking git whether a
    path is tracked also catches a file added with `git add -f`, and a rule
    deleted in a later commit -- both of which leave the ignore file looking
    correct.
    """
    if shutil.which("git") is None:
        pytest.skip("git is not on PATH, so this cannot be checked here.")

    repository = COMPOSE_DIR.parents[1]
    #: `env.template`, not `.env.example`: the latter is gitignored too, which is
    #: the whole reason the template has the name it has -- a server gets it
    #: through `git clone`, and `.env.*` does not travel. This list was written
    #: the other way round first and the assertion below is what corrected it.
    must_not_be_tracked = (".env", "infra/compose/.env", ".env.example", "infra/compose/.env")
    must_be_tracked = ("infra/compose/env.template",)

    def tracked(name: str) -> bool:
        """Whether git knows about `name`."""
        result = subprocess.run(  # noqa: S603 - a fixed argv, no shell
            ["git", "ls-files", "--error-unmatch", name],  # noqa: S607 - PATH lookup intended
            cwd=repository,
            capture_output=True,
            text=True,
            check=False,
        )
        return result.returncode == 0

    for name in must_not_be_tracked:
        assert not tracked(name), f"{name} is tracked by git, and it holds secrets."
    for name in must_be_tracked:
        assert tracked(name), f"{name} is not tracked, so a fresh clone has no template."


def test_the_deployed_commit_reaches_the_image() -> None:
    """`/health` can only report the commit if the build carries it.

    A build argument and an environment variable, because the container has no
    `.git` and the host's working tree is not what was built.
    """
    instructions = _instructions(_dockerfile_text("Dockerfile.api"))

    assert any(line.startswith("ARG SOURCE_COMMIT") for line in instructions)
    assert any(line.startswith("ENV") and "SOURCE_COMMIT" in line for line in instructions)


def test_the_api_container_does_not_sync_the_whole_workspace() -> None:
    """Packages are installed by name, never with `uv sync --all-packages`.

    The workspace resolves torch from PyPI, which on Linux is the CUDA build:
    roughly 2.9 GB of download for compute this service does not perform. The
    Windows development machine never notices, because the nvidia dependencies
    carry `sys_platform == 'linux'` markers and are skipped there.
    """
    assert not any("uv sync" in line for line in _instructions(_dockerfile_text("Dockerfile.api")))


# --- The inference container -------------------------------------------------


def test_the_copilot_weights_are_not_baked_into_the_image() -> None:
    """The GGUF is mounted, not fetched during the build.

    Baked in, it would be a 940 MB download on a one-core host every time
    anything in the image changed -- and swapping models would mean rebuilding
    rather than replacing a file. The two sentence encoders *are* fetched at
    build time, and deliberately: they are small, they are pinned to a revision
    the corpus records, and they are not meant to be swapped by hand.
    """
    instructions = _instructions(_dockerfile_text("Dockerfile.inference"))

    assert not any(".gguf" in line.lower() for line in instructions)


def test_the_copilot_model_is_named_once(
    services: dict[str, dict[str, object]],
) -> None:
    """One variable, two readers, so the two services cannot disagree.

    The API reports `COPILOT_MODEL` as the model that wrote each answer; the
    inference service loads whatever `INFERENCE_CHAT_MODEL` points at, and
    reports its file name. Two independent variables would drift, and the drift
    is invisible: every answer would name a model that did not write it, which
    is worse than naming none, because it reads as provenance.

    The file's name is also pinned literally. It is what `Settings.chat_model_id`
    returns and what the page prints, so a rename here is a rename of something
    a reader sees.
    """
    api = services["api"]["environment"]
    inference = services["inference"]["environment"]
    assert isinstance(api, dict) and isinstance(inference, dict)

    path = str(inference["INFERENCE_CHAT_MODEL"])
    assert path.startswith("/model/chat/"), "the weights belong in the read-only model mount"
    assert _chat_model_file(path) == _chat_model_file(str(api["COPILOT_MODEL"]))
    assert _chat_model_file(path) == "qwen2.5-1.5b-instruct-q4_k_m.gguf"


def test_the_inference_limit_fits_the_copilot_model(
    services: dict[str, dict[str, object]],
) -> None:
    """The one bound on this host whose failure is a kill rather than a slowdown.

    There is no swap here. A container that exceeds its limit is not paged out,
    it is OOM-killed -- most likely mid-answer, having already spent twenty
    seconds. So the limit is asserted against the measured footprint of both
    jobs the service now does, rather than being left as a number someone
    lowers again when the box looks tight.
    """
    limit = str(services["inference"]["mem_limit"])
    assert limit.endswith("m"), f"expected a megabyte limit, got {limit!r}"

    assert int(limit.removesuffix("m")) >= RETRIEVAL_FOOTPRINT_MIB + CHAT_FOOTPRINT_MIB


# --- The simulator container -------------------------------------------------


def test_the_simulator_runs_as_a_single_process() -> None:
    """No `--workers`, and here the failure is worse than the API's.

    The run registry is in-process state. A second worker would answer `GET
    /runs` with an empty list and accept a second run for a machine already
    running in the first -- two scenarios interleaving readings on one machine,
    with nothing logged to say why.
    """
    assert not WORKER_FLAGS.search(_cmd_line(_dockerfile_text("Dockerfile.simulator")))


def test_the_simulator_container_does_not_carry_pyarrow() -> None:
    """The Parquet sinks are imported on demand, so this image need not.

    PyArrow is around 40 MB of wheel for a service that publishes to a broker
    and never writes a file. `test_lazy_sinks.py` is what keeps this true; this
    is what states why it matters.
    """
    instructions = _instructions(_dockerfile_text("Dockerfile.simulator"))

    assert not any("pyarrow" in line.lower() for line in instructions)
    assert not any("uv sync" in line for line in instructions)


# --- The stack ---------------------------------------------------------------


def test_the_simulator_publishes_no_host_port(
    services: dict[str, dict[str, object]],
) -> None:
    """The control surface must stay on the bridge.

    It has no authentication, so a published port is an unauthenticated "start a
    run" endpoint reachable by anyone who finds the host -- and a run costs CPU
    on a box that already shares one core between n8n, the API and torch.

    `expose` is deliberately not flagged: it documents a port for the reader
    without publishing it to the host.
    """
    assert "ports" not in services["simulator"]


def test_every_service_has_a_healthcheck(services: dict[str, dict[str, object]]) -> None:
    """Compose reports health, so `depends_on` can mean something.

    Two services are exempt, both one-shot jobs that exit rather than run:
    `n8n-permissions` is a `chown` whose health signal is
    `depends_on: service_completed_successfully`, and `ingest` is behind a
    profile, is started by hand, and reports success by exiting zero. A
    healthcheck on either would be a check on a process that is supposed to be
    gone.
    """
    one_shot = {"n8n-permissions", "ingest"}
    for name, service in services.items():
        if name in one_shot:
            continue
        assert "healthcheck" in service, f"{name} has no healthcheck"


def test_the_simulator_carries_resource_limits(
    services: dict[str, dict[str, object]],
) -> None:
    """A run must not be able to starve the rest of the box.

    The limits are estimates rather than measurements and should be tuned from
    `docker stats` during a real run. Their purpose is to bound the damage from a
    pace set too high, which is exactly the failure the Modal deployment hit --
    it logged `waiting to be scheduled on a CPU worker` during a live run.
    """
    simulator = services["simulator"]

    assert "cpus" in simulator
    assert "mem_limit" in simulator
    assert simulator.get("restart") == "unless-stopped"


def test_the_simulator_can_reach_the_api_and_the_broker(
    services: dict[str, dict[str, object]],
) -> None:
    """The two variables that must agree with the rest of the stack.

    The topic prefix is the one that fails silently: n8n subscribes to
    `pdm/demo/+/telemetry`, so a run published under any other prefix is
    accepted by the broker and ignored by the orchestrator, with nothing logged
    anywhere. The API URL is the service name rather than a public hostname, the
    same choice n8n's own configuration makes and for the same reason -- the
    call stays on the bridge, with no public round trip and no dependency on DNS.
    """
    environment = services["simulator"]["environment"]
    assert isinstance(environment, dict)

    assert environment["MQTT_TOPIC_PREFIX"] in {"${MQTT_TOPIC_PREFIX:-pdm/demo}", "pdm/demo"}
    assert environment["PDM_API_BASE_URL"] == "http://api:8000"
    assert "PDM_INGEST_TOKEN" in environment
