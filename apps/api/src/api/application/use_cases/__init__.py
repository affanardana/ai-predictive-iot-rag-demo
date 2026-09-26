"""Use cases.

Each use case is a small object constructed with its dependencies and invoked
through `execute`. They are wired by the composition root, never instantiated
inside another layer.
"""

from api.application.use_cases.get_machine_detail import GetMachineDetail
from api.application.use_cases.get_prediction_history import GetPredictionHistory
from api.application.use_cases.get_telemetry_history import GetTelemetryHistory
from api.application.use_cases.ingest_telemetry import IngestResult, IngestTelemetry
from api.application.use_cases.list_incidents import ListIncidents
from api.application.use_cases.list_machines import ListMachines
from api.application.use_cases.list_simulations import ListSimulations
from api.application.use_cases.record_prediction import PredictionResult, RecordPrediction
from api.application.use_cases.register_machine import RegisterMachine, RegistrationResult
from api.application.use_cases.report_simulation_state import ReportSimulationState
from api.application.use_cases.reset_simulation import ResetSimulation
from api.application.use_cases.start_simulation import StartSimulation
from api.application.use_cases.stop_simulation import StopSimulation
from api.application.use_cases.update_incident_status import UpdateIncidentStatus

__all__ = [
    "GetMachineDetail",
    "GetPredictionHistory",
    "GetTelemetryHistory",
    "IngestResult",
    "IngestTelemetry",
    "ListIncidents",
    "ListMachines",
    "ListSimulations",
    "PredictionResult",
    "RecordPrediction",
    "RegisterMachine",
    "RegistrationResult",
    "ReportSimulationState",
    "ResetSimulation",
    "StartSimulation",
    "StopSimulation",
    "UpdateIncidentStatus",
]
