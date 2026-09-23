# Simulator

**Status: not implemented — Phase 2.**

A Python service that generates temporally coherent motor telemetry and owns the
hidden ground-truth machine state.

Planned scope (MASTERPLAN §6, Phase 2):

- temporal state machine over six signals — temperature, vibration, rpm,
  current, load, voltage;
- correlated sensors, so a degradation mode produces a coherent signature
  (bearing degradation: vibration ↑, temperature ↑, current ↑, rpm ↓) rather
  than six independently randomised values;
- scenarios: `NORMAL`, `BEARING_DEGRADATION`, `OVERHEATING`, `OVERLOAD`;
- configurable random seeds and identifiable simulation sessions, so a demo is
  reproducible;
- dataset mode (bulk historical generation) and realtime mode (continuous
  publishing).

## Two rules this service exists to satisfy

1. **The simulator owns ground truth.** Ground-truth state labels the training
   data and is used for evaluation. It must **never** be exposed as a model
   input feature.
2. **Telemetry is not RAG knowledge.** Simulated telemetry belongs in
   PostgreSQL, not in the vector store.

This directory is intentionally empty until Phase 2. It is not yet a `uv`
workspace member — it will be added to `members` in the root `pyproject.toml`
when it gains real dependencies.
