# ADR 0004 — n8n is transport; the API owns the domain

**Status:** Accepted (Phase 6)
**Date:** 2026-09-24

## Context

`MASTERPLAN.md` §Phase 6 draws the pipeline as:

```text
Simulator → MQTT → EMQX → n8n → validation → idempotency check → PostgreSQL
          → inference → prediction → incident evaluation
```

Read one way, n8n **is** the ingestion tier: it validates, checks idempotency,
and writes telemetry rows into PostgreSQL itself, with the "n8n Cloud" box in
§4's architecture diagram sitting directly above "PostgreSQL / Supabase".

Read that way, n8n has to be told what a duplicate is, when a risk band warrants
an incident, and which readings count as a full window. Those are the rules
`api.domain` exists to hold — and reimplementing them in JavaScript creates a
second copy that can drift from the first without anything failing.

`infra/README.md` recorded this as an open question from Phase 1 onward, and the
changelog carried it as the highest-impact unresolved item.

## Decision

**n8n validates shape and forwards. The API owns every rule.**

The workflow: MQTT Trigger → validate required keys → `POST /api/v1/telemetry` →
if the response names machines holding a full window, `POST
/api/v1/machines/{id}/predictions` for each.

The API: idempotency (`add_many_idempotent`, backed by the primary key on
`event_id`), persistence, readiness, inference, risk classification, and
incident policy.

## Rationale

**The domain stays in one place.** `DefaultIncidentPolicy.should_raise` decides
what deserves an incident; `RiskLevelClassifier` decides what a probability is
called; the primary key on `telemetry.event_id` decides what a duplicate is.
None of those has a JavaScript counterpart, so none of them can disagree with
one.

**The rules are already enforced by the schema.** The changelog recorded this in
Phase 1: *"`telemetry.event_id` is the primary key, so redelivery cannot create
duplicate rows — the non-functional idempotency requirement is enforced by the
schema rather than by application code."* Nothing about idempotency needs
re-deciding in the orchestrator; a single `ON CONFLICT DO NOTHING` expresses it,
and `add_many_idempotent` is that statement.

**A transaction boundary cannot cross HTTP.** `domain/ports/unit_of_work.py`
says a prediction and the incident it raises *"either commit together or not at
all"*. That guarantee is available inside one process holding one database
session. An orchestrator that wrote telemetry and then called an API to raise an
incident would have to build its own compensation logic, badly.

**Credentials.** The literal reading needs n8n to hold the operational database
password. This one needs a single token scoped to three endpoints, and the API
can narrow it further without touching the workflow.

## Consequences

**A new ingest endpoint exists.** `POST /api/v1/telemetry` is a write surface
the API did not have, guarded by a shared secret. Phase 11 owns real
authentication; this is one token for one caller, and `require_ingest_token`
documents that limit.

**Readiness is the API's answer, not n8n's.** The ingest response reports which
machines hold a full window, so the 60-reading rule lives in `IngestTelemetry`
rather than in the workflow. It also means the "when should we score" question
has one answer instead of one per orchestrator.

**n8n is still load-bearing.** It is the MQTT subscriber, the retry point, and
the place the pipeline is observable — the execution list is where a failed hop
is found. It is transport, not a formality.

**The masterplan's stage list is satisfied, but not one-stage-per-box.** Every
named stage happens; the API performs five of them. §Phase 6's own ordering puts
`inference → prediction → incident evaluation` *after* PostgreSQL, which is
where this reading places them too.

**Two hops remain unverified by the test suite.** The MQTT broker hop is covered
by an opt-in `mqtt` tier and the container is not covered at all. What is
pinned automatically is the agreement the workflow must not drift out of: the
topic prefix, the route paths, and the header name, each asserted against the
code that owns it — see `simulator/tests/integration/test_n8n_workflow.py`.
