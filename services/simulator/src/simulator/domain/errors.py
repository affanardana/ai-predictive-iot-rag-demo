"""Simulation errors."""

from __future__ import annotations


class SimulationError(Exception):
    """Base class for every error raised by the simulation model."""


class SimulationValidationError(SimulationError, ValueError):
    """A simulation input or intermediate value violates an invariant.

    Also a `ValueError` so it behaves as callers expect from argument
    validation, while remaining catchable as a simulation failure.
    """


class UnknownScenarioError(SimulationError):
    """A scenario name does not match any known scenario."""

    def __init__(self, name: str, known: tuple[str, ...]) -> None:
        self.name = name
        self.known = known
        super().__init__(f"Unknown scenario '{name}'. Known scenarios: {', '.join(known)}.")


class RunAlreadyActiveError(SimulationError):
    """A machine already has a run in this process.

    A machine's series is a pure function of its own session, so two runs on
    different machines cannot interfere -- but two on the same machine would
    interleave readings for one machine from two scenarios, and the risk band
    they produce would describe neither.
    """
