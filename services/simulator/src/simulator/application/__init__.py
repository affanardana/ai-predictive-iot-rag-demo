"""Run orchestration.

Drives the simulation model and hands its two output channels to their sinks.
Imports the domain and nothing else: which sink is used, and where it writes, is
decided by the command line and never here.
"""

from simulator.application.run_simulation import RunSummary, run_session, stream_session

__all__ = ["RunSummary", "run_session", "stream_session"]
