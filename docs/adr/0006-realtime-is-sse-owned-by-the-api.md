# ADR 0006 — Realtime is server-sent events, owned by the API

**Status:** Accepted (Phase 7)
**Date:** 2026-09-26

## Context

`spec_driven_development/docs/CHANGELOG.md` carried this as an open question from
v0.2.0 onward:

> **Supabase Realtime or FastAPI-owned SSE** for dashboard updates? (Phase 7.)

Two documents pulled in different directions. `MASTERPLAN.md` §4 draws the
frontend's only edge as `HTTPS / API` into FastAPI — no websocket, no
browser-to-database path. But §5 lists Supabase Realtime under **Data**, and
`PRD.md` §21 says it "may be used as the primary mechanism for UI updates". `PRD.md`
FR-014 is stricter: the dashboard "shall update relevant operational information
without requiring manual refresh", and §21 names the events that must propagate —
new telemetry, changed machine risk, new prediction, new incident.

## Decision

**Server-sent events, served by the API at `GET /api/v1/events`.** The browser
never talks to Supabase.

Three reasons decided it:

1. **One data path.** A browser subscribed to Postgres changes bypasses the API
   entirely, which means two representations of the same fleet that can disagree,
   and a second place for the domain rules to be re-derived.
2. **No credentials in the browser.** Supabase Realtime needs a key shipped to
   every client, and with no row-level security configured — Phase 11's work — that
   key can read every row. A stream of hints from the API needs nothing.
3. **`MASTERPLAN.md` §4 already draws it this way.** §5's mention of Supabase
   Realtime is the only text pointing elsewhere, and PRD §21 says "may".

## Consequences

**The payload is a hint, not the data.** A frame names a machine and what changed
about it; the client refetches the REST endpoint that owns that representation.
This is the load-bearing part of the decision, and it is what makes the rest
cheap:

- There is no second wire format. The presentation layer stays the only place a
  domain object becomes JSON, so a streamed payload cannot drift from the schema
  served at `/api/v1/machines`.
- **A dropped frame is survivable**, because the next one repairs it. That is
  what makes a bounded queue with a drop-on-overflow policy legal rather than
  lossy, and it is why `publish` never blocks and never raises.
- **Reconnecting needs no replay buffer.** `Last-Event-ID` is deliberately
  ignored: the recovery for a gap and the recovery for a fresh connection are the
  same action, so per-client sequence tracking, replay windows and their memory
  bounds are all deleted at no correctness cost.
- PRD §24 holds by construction — the stream ships no historical data.

**The API must remain a single process.** `InProcessEventBroadcaster` fans out in
memory, so an event published on one worker reaches only that worker's
subscribers. `infra/compose/Dockerfile.api` runs uvicorn with no `--workers`, and
a test parses that file and fails if a count appears. The failure this prevents
is silent rather than loud: a second worker would deliver each change to an
arbitrary subset of browsers, and the symptom would be blamed on the frontend.
Going multi-process means `LISTEN`/`NOTIFY` or Redis.

**The stream is a latency optimisation, never the source of correctness.** Every
query in the client carries a poll interval behind it, so a missed event costs
freshness rather than accuracy. This is also what makes the single-worker
constraint a degradation rather than an outage.

**It cannot be authenticated, now or later.** `EventSource` cannot set request
headers, so no `X-Ingest-Token` could ever be presented to this route. Phase 11's
operator authentication has to account for that — a cookie would work where a
header cannot.

**Rejected: polling alone.** It needs no new machinery and would satisfy the exit
condition, but it makes `GET /api/v1/machines` the hot path on a one-core box for
every open tab, and it does not satisfy FR-014's "without requiring manual
refresh" in the sense the demonstration needs — telemetry arriving a poll late is
not the live progression `MASTERPLAN.md` §7 step 12 describes.
