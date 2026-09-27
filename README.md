# AI Predictive Maintenance & Maintenance Copilot

A predictive maintenance platform for industrial electric motors. Simulated
telemetry is ingested over MQTT into PostgreSQL, an LSTM estimates the
probability of failure within the next 60 minutes, and an incident is raised
when risk crosses a configured threshold.

A Maintenance Copilot answers questions about machine condition, history, and
maintenance procedures, distinguishing what was observed, what was predicted,
and what is documented — built with FastAPI, React, PyTorch, PostgreSQL and
Supabase.

> Self-hosted end to end: the backend, database, telemetry simulator, predictive
> model, dashboard, maintenance-document retrieval and the Copilot are all
> deployed and answering. The Copilot's language model runs on the same host as
> everything else — no hosted API, and no per-question cost — so its answers are
> plainer than a large model's and its grounding is checked rather than assumed.
> All telemetry and documentation are synthetic demonstration material.

**Dashboard:** <https://pdm-sim.vercel.app> · **API and docs:**
<https://pdm-api.72-61-214-194.sslip.io/docs>

The demonstration is four steps, and the fourth is the one worth waiting for:

1. Open the dashboard and select **M003**.
2. Start the canonical run — *Bearing Degradation*, demo mode. Readings arrive
   about one a second; the risk band climbs; an incident is raised.
3. Watch it on the machine page. The event stream pushes updates; it does not
   poll.
4. Ask the Copilot *"Why is M003 becoming risky and what should I inspect
   according to the SOP?"* — about twenty seconds, most of it spent watching the
   sources being read. The answer separates what was **observed**, what was
   **predicted**, what the **documentation** says and what was **inferred**,
   and carries the evidence it was written from.

Then ask something the manuals do not cover — *"What is the torque spec for the
M003 gearbox output shaft?"* — and it refuses rather than inventing a procedure.
That refusal is the feature: `PRD.md` §19, enforced by the system not asking the
model at all.

`docs/adr/` records the decisions behind each of those, including the ones taken
against the obvious choice.
