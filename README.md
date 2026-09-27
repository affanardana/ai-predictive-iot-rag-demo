# AI Predictive Maintenance & Maintenance Copilot

A predictive maintenance platform for industrial electric motors. Simulated
telemetry is ingested over MQTT into PostgreSQL, an LSTM estimates the
probability of failure within the next 60 minutes, and an incident is raised
when risk crosses a configured threshold.

A Maintenance Copilot answers questions about machine condition, history, and
maintenance procedures, distinguishing what was observed, what was predicted,
and what is documented — built with FastAPI, React, PyTorch, PostgreSQL and
Supabase.

> In development: the backend, database, telemetry simulator, predictive model,
> dashboard and maintenance-document retrieval are in place. The Copilot that
> answers questions over them is not yet built. All telemetry and documentation
> are synthetic demonstration material.
