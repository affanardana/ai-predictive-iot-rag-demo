# Infrastructure

**Status: implemented — the backend runs on a self-hosted VPS.**

Cross-cutting deployment and orchestration assets that are not application code.

Present today:

- `.github/workflows/ci.yml` — the CI pipeline (format, lint, types,
  architecture contracts, offline tests, and a PostgreSQL job that applies
  migrations and checks for model/migration drift).
- `compose/` — **the deployment.** A Docker Compose stack running Caddy, the
  API, the model service and n8n on one VPS, with `README.md` as the runbook.
- `n8n/` — the orchestrator's workflow export and the runbook for configuring
  it, plus the Modal deployment it used to run under.

Not yet built:

- **the Vercel deployment itself.** `apps/web/vercel.json` and the build are in
  place, and `apps/web/README.md` has the two settings the project needs — but
  creating the project, setting `VITE_API_BASE_URL` and deploying are the owner's
  actions, and the app has never run against the hosted API.

**The `modal_app.py` files are superseded but not deleted.** They are the
specification the Dockerfiles were translated from, they still work, and they
are the rollback until the VPS has been in use long enough to trust. They live
beside the code they deploy rather than here, because they are application code:
`services/inference/modal_app.py`, `apps/api/modal_app.py`,
`infra/n8n/modal_app.py`.

## Where everything runs

```text
VPS (Docker Compose)     Caddy · API · model service · simulator · n8n
Supabase                 PostgreSQL
broker.emqx.io           MQTT
Vercel                   the web app (not yet deployed)
```

The simulator joined this list in Phase 8, and its absence before that was a
requirement the project had been failing rather than a gap in this diagram:
`PRD.md` AC-010 requires the demonstration to run without a locally running
simulator, and both runbooks used to instruct one. See `../docs/adr/0008`.

Migrations are applied from a developer machine rather than from a container —
a batch operation with one writer, so a cold start must not race another replica
to the same revision. `compose/README.md` has the command.

## The Phase 6 question, and how it was settled

This directory carried an open question from Phase 1:

> MASTERPLAN §Phase 6 reads literally as n8n writing telemetry rows into
> PostgreSQL itself. Taken literally, n8n would have to reimplement the
> idempotency rule, the risk-band mapping, and the incident policy in
> JavaScript — duplicating the domain and hollowing out the architecture.
>
> The recommended alternative is that n8n calls an authenticated internal API
> endpoint instead, keeping the domain the single source of truth. That is a real
> deviation from a literal reading of the masterplan and needs the repository
> owner's explicit decision before Phase 6 begins.

**The owner chose the recommended alternative.** n8n validates shape and
forwards; the API owns idempotency, persistence, readiness, inference, risk
classification, and incident policy. The reasoning, the alternatives, and what
the decision costs are recorded in
[ADR 0004](../docs/adr/0004-n8n-is-transport.md) rather than here.

## What changed in the stack, and why

Three components the specs name are not free, or not the right shape, and this
project runs on free tiers:

- **n8n Cloud → self-hosted n8n.** `PRD.md` §25 item 5 names n8n Cloud
  specifically, so dropping n8n was the alternative; self-hosting keeps the
  named component.
- **EMQX Cloud → `broker.emqx.io`.** EMQX's public broker needs no credentials
  and carries a publicly trusted certificate, so TLS costs nothing. `PRD.md`
  §25 item 4 names EMQX Cloud, so this is a substitution rather than a match —
  though the public broker is still EMQX's own service.
- **Modal → a self-hosted VPS.** `MASTERPLAN.md` §5 names Modal under Backend.
  The move is recorded in [ADR 0005](../docs/adr/0005-self-hosted-vps.md), along
  with what it costs and what stays on Modal. `Docker` is also a §5 component,
  so the mechanism is inside the stated stack even though the host is not.

Neither of the first two is a silent substitution. Both are visible in
`simulator/cli.py` and the Compose stack, and the second has a consequence worth
restating: a public broker has no authentication, so the topic prefix is the
only isolation there is. The API's token stops a stranger writing to the
database; it does not stop a stranger publishing plausible readings into the
topic. Self-hosting EMQX in the same Compose stack would close that, and is one
more service in a file that already exists.

