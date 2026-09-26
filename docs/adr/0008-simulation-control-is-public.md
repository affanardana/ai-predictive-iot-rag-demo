# ADR 0008 — Simulation control is public, and bounded instead

**Status:** Accepted (Phase 8)
**Date:** 2026-09-26

## Context

Phase 8 adds `POST /api/v1/simulations`, which starts a simulated run on a
machine. The dashboard calls it from a browser, so it inherits the constraint
ADR 0007 already established: the API's reads are unguarded because a browser
reads them, and a token in a browser bundle is not a boundary but the appearance
of one.

This route is a larger exposure than ADR 0007's incident PATCH, and in a
different way. Resolving an incident misrepresents state; starting a run
**consumes CPU on a host that has one core**, shared with n8n, the API and the
model service. `infra/compose/README.md` warns about exactly this load, in a note
written when the simulator still ran on a developer's laptop.

So the question was not "guarded or not" — the browser settles that — but what
takes the guard's place.

## Decision

**The route is unguarded, and four bounds take the token's place.**

| Bound | Where | What it prevents |
|---|---|---|
| One active run per machine | Partial unique index in the schema | Two scenarios interleaving readings on one machine, producing a risk band that describes neither |
| A fleet-wide ceiling (default 3) | `StartSimulation`, from settings | A fleet-wide demonstration from starving the API and the orchestrator |
| A 60-minute floor on duration | `StartSimulation`, derived from the model's window | A run that cannot produce a prediction, and so cannot show anything |
| Container limits | `compose.yaml` — `cpus`, `mem_limit`, `pids_limit` | A run at a bad pace from taking the whole box |

The ceiling is a **settings value**, not a constant, because it is a property of
the host rather than of the domain: three runs is right for one core, and the
right way to change it should be an environment variable rather than a deploy.

## Consequences

**Anyone who finds the hostname can start a run.** They cannot exceed three, and
each is bounded to a machine that is not already running. The worst case is that
a stranger occupies the demonstration's CPU for a few minutes, which is
recoverable by waiting — not by restoring anything. Nothing they can do through
this route writes telemetry the pipeline did not produce, and every reading it
causes is a genuine simulation of a genuine scenario.

**Stop is an escape hatch, and is built like one.** `StopSimulation` terminalises
the run whatever the simulator says, recording an unreachable service in the
run's `detail` rather than raising. A stop that could fail because the container
was down would leave a run stuck at `RUNNING` with no way out but editing the
database — which is a worse outcome than an unauthenticated endpoint, and the
one this design most needed to avoid.

**The state-report route is guarded**, and it is the only route on this router
that is. The asymmetry is deliberate and is the reason the guard sits on the
route rather than the router: the reporter is a service, exactly like n8n, and
`ingest_api_token`'s docstring describes it as "one machine proving to another
that it is the expected caller". A browser is structurally unable to forge a run
completion, which is something the three dashboard routes cannot claim.

**Reset cannot delete telemetry, and this is a structural limit rather than
caution.** Discarding stored readings is a destructive write, so it would have to
be guarded, and a browser cannot present the guard. A destructive reset is
therefore unreachable from the button meant to offer it. Reset clears the run
record and nothing else, which means a second run **appends** to a machine's
charts. The windowed charts hide it in practice, and `infra/compose/README.md`
says it out loud.

**Phase 11 owns closing this**, alongside operator authentication. One constraint
to carry forward: the event stream cannot be authenticated by header at all —
`EventSource` cannot set request headers — so whatever scheme Phase 11 chooses
has to work for a cookie or a query parameter as well.
