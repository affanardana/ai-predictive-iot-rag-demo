"""Use cases.

Each use case is a small object constructed with its dependencies and invoked
through `execute`. They are wired by the composition root, never instantiated
inside another layer.
"""

from api.application.use_cases.get_machine_detail import GetMachineDetail
from api.application.use_cases.get_prediction_history import GetPredictionHistory
from api.application.use_cases.get_telemetry_history import GetTelemetryHistory
from api.application.use_cases.list_incidents import ListIncidents
from api.application.use_cases.list_machines import ListMachines

__all__ = [
    "GetMachineDetail",
    "GetPredictionHistory",
    "GetTelemetryHistory",
    "ListIncidents",
    "ListMachines",
]
