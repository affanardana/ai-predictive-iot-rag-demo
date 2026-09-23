"""Telemetry simulator for the AI Predictive Maintenance platform.

Generates temporally coherent motor telemetry and the hidden ground-truth state
behind it. Layers, from the inside out:

* ``simulator.domain`` -- the simulation model: machine profiles, degradation
  scenarios, the physics that couples them to sensor signals, and the engine
  that advances one machine. Pure Python; imports nothing from the standard
  library's I/O modules and nothing third-party.
* ``simulator.application`` -- orchestrates a run: profiles, engine, sinks.
* ``simulator.infrastructure`` -- sinks and defaults.
* ``simulator.cli`` -- the command line.

The layering is enforced by the ``import-linter`` contracts in the repository's
root ``pyproject.toml``.
"""

__version__ = "0.2.0"
