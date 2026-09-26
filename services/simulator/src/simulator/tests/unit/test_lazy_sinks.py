"""What the sinks package does and does not pull in when it is imported.

Run in a subprocess rather than in-process, and the reason is the whole point of
the test: `sys.modules` inside a running suite is already polluted by whatever
was imported first, so an assertion about what an import *adds* would pass for
the wrong reason. A fresh interpreter is the only place the question has an
answer.

These matter because they are deployment facts, not style. The control container
installs `paho` and FastAPI and deliberately does **not** install `pyarrow`;
without the lazy import in `sinks/__init__.py`, importing the MQTT sink would
drag a columnar library into an image that never writes a file -- or fail at
startup with an `ImportError` naming a module the caller never mentioned.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

#: Long enough for a cold interpreter to import the package, short enough that a
#: hang is a failure rather than a stuck suite.
TIMEOUT_SECONDS = 60


def _modules_after(statement: str) -> set[str]:
    """Return the modules loaded by a fresh interpreter after `statement`."""
    completed = subprocess.run(  # noqa: S603 - fixed argv, no shell, no input
        [sys.executable, "-c", f"import sys; {statement}; print('\\n'.join(sys.modules))"],
        capture_output=True,
        text=True,
        timeout=TIMEOUT_SECONDS,
        check=True,
    )
    return set(completed.stdout.split())


@pytest.mark.parametrize("module", ["pyarrow", "paho", "fastapi"])
def test_importing_the_package_alone_pulls_in_nothing_heavy(module: str) -> None:
    """Importing `simulator` costs nothing optional.

    The narrowest form of the claim: the package itself must not decide which of
    its optional backends a caller has installed.
    """
    assert module not in _modules_after("import simulator")


def test_importing_the_sinks_package_does_not_import_pyarrow() -> None:
    """The MQTT sink is importable without a columnar library.

    `import simulator.infrastructure.sinks` is what the control container does,
    transitively, on its way to `MqttTelemetrySink`.
    """
    assert "pyarrow" not in _modules_after("import simulator.infrastructure.sinks")


def test_importing_the_mqtt_sink_does_not_import_pyarrow() -> None:
    """Naming the sink the container actually uses is still pyarrow-free.

    The precise path: a submodule import executes its parent package first, so a
    regression in `sinks/__init__.py` would surface here even though this
    statement never mentions Parquet.
    """
    loaded = _modules_after("from simulator.infrastructure.sinks.mqtt import MqttTelemetrySink")

    assert "pyarrow" not in loaded


def test_the_parquet_sink_still_works_when_asked_for() -> None:
    """Deferring an import must not break the thing being deferred.

    A lazy import that never resolves is indistinguishable from a deletion, and
    the failure would land on whoever next asks for a dataset -- in a different
    phase, with nothing pointing here. So the name is asserted to arrive, and to
    bring pyarrow with it.

    No marker: `pyarrow` is a core dependency of this package, so this belongs in
    the default tier rather than beside the tests that need optional extras.
    """
    loaded = _modules_after("from simulator.infrastructure.sinks import ParquetTelemetrySink")

    assert "pyarrow" in loaded
