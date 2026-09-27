# AI Predictive Maintenance & Maintenance Copilot

A predictive maintenance platform for industrial electric motors. Simulated
telemetry is ingested over MQTT into PostgreSQL, an LSTM estimates the
probability of failure within the next 60 minutes, and an incident is raised
when risk crosses a configured threshold.

A Maintenance Copilot answers questions about machine condition, history, and
maintenance procedures, distinguishing what was observed, what was predicted,
and what is documented — built with FastAPI, React, PyTorch, PostgreSQL and
Supabase.

> In development, and self-hosted end to end: the backend, database, telemetry
> simulator, predictive model, dashboard, maintenance-document retrieval and the
> Copilot are all deployed and answering. The Copilot's language model runs on
> the same host as everything else — no hosted API, and no per-question cost —
> so its answers are plainer than a large model's and its grounding is checked
> rather than assumed. All telemetry and documentation are synthetic
> demonstration material.
