# ADR 0007 — The dashboard's incident actions are unguarded

**Status:** Accepted (Phase 7)
**Date:** 2026-09-26

## Context

Phase 6 left every read endpoint open, and recorded why: `infra/compose/caddy.snippet`
says reads are unguarded because the dashboard reads them from a browser, and
requiring a shared secret there would mean shipping it to every client. The three
write endpoints — telemetry, machine registration, and scoring — require
`X-Ingest-Token`, because the only caller is the pipeline.

Phase 7 adds the first write a *browser* makes: `PATCH /api/v1/incidents/{id}`,
which moves an incident through `OPEN → ACKNOWLEDGED → RESOLVED | DISMISSED`.
Its place in the phase is not cosmetic. `RecordPrediction` raises an incident
only when a machine has none open, so before this route existed an incident was
permanent — every later excursion on that machine was suppressed by the first
one, forever. Without it the incident list can only ever hold one row per
machine, and the suppression rule that makes the list readable also makes it
permanently stale.

The options were:

1. Guard it with the ingest token.
2. Leave it open and record the exposure.
3. Defer the whole feature to Phase 11, and accept permanent suppression until
   then.

## Decision

**Leave it open, and record it.**

Option 1 is not a security boundary, it is the appearance of one. The token
would have to be in the browser bundle for the dashboard to present it, where
anyone can read it out of the JavaScript — so it would stop an honest client and
nothing else, while breaking the demonstration for a reviewer who opens the
deployed site.

Option 3 would leave a documented, user-visible defect in the product for the
sake of avoiding a bounded and reversible exposure.

## Consequences

**Anyone who finds the hostname can resolve or dismiss incidents.** The blast
radius is bounded and worth stating precisely: incidents are the only mutable
resource reachable without a token. Closing one cannot inject telemetry, register
a machine, create a prediction, or corrupt stored data. It can misrepresent the
current state of the fleet, which corrupts the *narrative* of a demonstration
rather than the data behind it — and the effect is reversible, since a dismissed
incident can be raised again by the next scoring that crosses a band.

**Phase 11 owns closing it**, alongside operator authentication. The constraint
to carry forward is that the event stream cannot be authenticated by header at
all — `EventSource` cannot set request headers — so whatever scheme Phase 11
chooses has to work for a cookie or a query parameter as well.

**CORS was widened for the same reason.** `api/presentation/app.py` allows every
origin, because reads are open, writes are guarded by a header rather than a
cookie, and Vercel gives every preview deployment its own generated origin — an
allowlist would break previews on every push and surface as an opaque browser
error rather than a server-side one. `allow_credentials=False` is what keeps the
wildcard from combining with ambient credentials later.

**This is a decision, not an oversight**, which is why it is here. The router's
own docstring says so at the point of use, so a reader of the code meets the
reasoning rather than a surprise at review.
