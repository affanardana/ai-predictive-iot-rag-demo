# AI Predictive Maintenance & Maintenance Copilot

A predictive maintenance platform for industrial electric motors. Simulated
telemetry is ingested over MQTT into PostgreSQL, an LSTM estimates the
probability of failure within the next 60 minutes, and an incident is raised
when risk crosses a configured threshold.

A Maintenance Copilot answers questions about machine condition, history, and
maintenance procedures, distinguishing what was observed, what was predicted,
and what is documented — built with FastAPI, React, PyTorch, PostgreSQL and
Supabase.

> Early development: the backend and database are in place. The simulator,
> model, dashboard, and Copilot are not yet built, and all telemetry and
> documentation are synthetic demonstration material.
