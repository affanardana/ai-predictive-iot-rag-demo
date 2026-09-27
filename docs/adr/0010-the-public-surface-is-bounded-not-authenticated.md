# ADR 0010 — The public surface is bounded, not authenticated

**Status:** Accepted (Phase 11)
**Date:** 2026-09-27

## Context

`MASTERPLAN.md` §6 asks Phase 11 for *"authentication/authorization where
appropriate"*, and four ADRs before this one deferred to it by name: `0006`
(row-level security, and the event stream's inability to carry a header),
`0007` (the incident write route), `0008` (simulation control) and `0009` (the
Copilot). Every one of them ends with the same sentence in some form: *Phase 11
owns the guard.*

The obvious reading is "add a login". §8's Demonstration line is what makes that
hard:

> A reviewer can open the deployed application and observe the complete chain —
> Simulation → Telemetry → Prediction → Risk → Incident → Dashboard → Copilot →
> Evidence — without needing to run the system locally.

A login a reviewer cannot pass does not protect that chain; it deletes it. The
portfolio's central asset is a stranger opening a URL and watching the system
work, and every route worth guarding is on that path.

So the question was not whether to authenticate but **what takes the guard's
place**, and what an honest alternative looks like on a host with one core.

## Decision

**The public routes stay public, and five things bound them instead.**

| Bound | Where | What it stops |
|---|---|---|
| Per-address rate limits | `presentation/rate_limit.py`, per route plus a global ceiling | One caller monopolising the Copilot's answer slot, a start/reset loop, a scraper walking the read routes |
| One answer at a time | `AskCopilot.claim()` (ADR 0009) | Two answers halving each other on one core |
| Concurrent-run ceiling | `StartSimulation` (ADR 0008) | A fleet-wide demonstration starving everything else |
| Container limits | `compose.yaml` | An OOM kill taking the demonstration with it |
| Row-level security | Migration `0004` | A leaked Supabase anon key reading every table |

The last one is the only *authorization* left standing, and it is the one that
matters. The browser never receives a Supabase key — the API is the only path to
this data — but a Supabase project is reachable without the API: PostgREST
answers at the project's own hostname with the project's anon key, which is
*designed* to be published in frontends. With row-level security off, that key
reads every row. Enabling it with no policies at all closes that: a table owner
bypasses row-level security unless `FORCE ROW LEVEL SECURITY` is set, the API
connects as the owner, and `anon` has nothing granting it anything.

**`FORCE` is deliberately not used**, and that is not an oversight. Forcing it
applies policies to the owner too, the product defines none, and every query in
the API would return empty. `test_row_level_security.py` asserts the owner still
reads, which is what fails if somebody later decides stricter is safer.

## What the bounds are and are not

**They are**: a sustained-rate ceiling per source address, with a bounded burst,
that costs one dictionary lookup per request and nothing when idle; and a
limiter whose own memory is bounded by a fixed key count, because an unbounded
map keyed by address is a memory leak with a security story.

**They are not** a defence against many addresses. An attacker with an IPv6 /48
or a botnet spends this box regardless; what bounds *that* damage is the
concurrent-run ceiling, the single answer slot and the container's memory limit
— the same bounds that protect it from a bad demonstration. They are not durable
across a restart: every bucket empties, so a deploy is a free burst for
everyone, which is the correct direction to fail (a limiter that fails closed
turns a deploy into an outage for the reviewer). They are not per user: a
household behind one address shares a budget, which is also the honest unit,
because that is the address a person is.

**The client address is the fragile part**, and the failure is silent. The API
publishes only to loopback, so every public request arrives through the host's
Caddy — and through Docker's bridge, which means the socket peer is the gateway
rather than `127.0.0.1`. uvicorn's own `ProxyHeadersMiddleware` trusts
`127.0.0.1,::1` by default, which the gateway is not, so a deployment that did
not override it would believe no forwarded header at all and charge every
visitor on earth to one bucket. Both sides read one variable,
`RATE_LIMIT_TRUSTED_PROXIES`, and the resolution takes the **rightmost entry
that is not itself a declared proxy**: each hop appends the address of whoever
connected to it, so the rightmost non-proxy entry is the one the nearest proxy
wrote itself. Reading it the conventional nginx way round — leftmost is the
client — would hand an attacker the key by forging one header.

**The pipeline is never charged.** n8n posts about one telemetry batch a second
per running machine, from inside the Compose network, for as long as a run
lasts. Metering that would return 429 to the ingest route — dropped readings,
holes in the charts — which is a worse failure than a slow dashboard. An
in-network caller is recognised by the absence of a forwarding header, which
Caddy always adds; the routes it uses are token-guarded separately.

## Monitoring: measured, not installed

**Metrics are exposed at `GET /metrics` in the Prometheus format, and collected
by an external service.** The alternative — Prometheus and Grafana in containers
— is roughly 400–600 MB and a slice of the only core, next to an inference
service that already peaks at 1.7 GB. Nothing in the code depends on which
collector is used.

Two rules keep it honest. **Every label value comes from a closed set defined in
`observability/metrics.py`**: no machine ids, no addresses, no paths, no
question text. The route label is a *template* — `/machines/{machine_id}` —
which is why the middleware reads it off the matched route rather than the
requested path. A test asserts that requesting `/api/v1/machines/M003` leaves
the string `M003` nowhere in the exposition. And **latency buckets are not the
defaults**: the library's top out at ten seconds, which would park every Copilot
answer in `+Inf` and hide the one latency this product is designed around.

`/metrics` is guarded by an optional `METRICS_TOKEN`, and the reasoning is the
repository's own: reads are open because a *browser* reads them, and no browser
reads metrics. A scraper is a machine, and `ingest_api_token`'s docstring
already names the pattern — one machine proving to another. It is optional
rather than required because a collector that accepts only a URL could never
satisfy it, and that failure presents as "target down" with nothing in this
container's logs.

`GET /health/dependencies` exists for the same reason and covers what metrics
cannot: **neither the inference service nor the simulator publishes a host
port**, so nothing outside the bridge can ask either of them anything. It
reports each dependency, and the model service's own view of which artefacts it
loaded — the LSTM's run id, the two encoders, and the Copilot's GGUF, none of
which this API can otherwise observe because they are placed by hand on the
host. It is deliberately **not** part of readiness: readiness gates n8n's
traffic, the inference container spends twenty seconds loading its chat model at
startup, and folding this in would stop the pipeline storing readings during
every deploy — a real outage caused by more information.

## Declined, with reasons

**Playwright** (`MASTERPLAN.md` §5). A suite driving a live deployment is slow
and flaky, and the runner minutes were judged worth more than the coverage. What
that leaves unverified is stated in `apps/web/README.md` and in the changelog
rather than papered over: routing, styling, ECharts, and anything that only
breaks in a browser. In its place, `AnswerBlock.test.tsx` renders the three
Copilot verdicts in jsdom — the page Phase 10 churned hardest, tested for what it
*says*.

**MLflow.** `model_version` already travels in every prediction, and the
knowledge corpus records its embedding model per chunk. A tracking server would
add a container and a database to store what is already stored, for a
demonstration with one trained model.

**Langfuse**, and this one is a real conflict rather than a preference. The
direction list names LLM observability, and Langfuse's hosted tier would ship
every Copilot prompt — machine identifiers, readings and retrieved procedure
text — to a third party. That contradicts the owner's decision that the model
runs locally and questions cost nothing, which ADR 0009 is built on.
Self-hosting it needs Clickhouse on a one-core box. The substitute is the one
this project would have anyway: the evidence and tool activity attached to every
answer, `model_id` in every response, and the metrics above.

**OpenTelemetry.** Optional in §5, and there is no collector to send to. The
correlation id already crosses containers on every outbound call, which is the
part of tracing this system can act on.

**Deployment automation.** The owner declined a registry and a CI deploy job.
The runbook is the deploy path, and it now passes `SOURCE_COMMIT` so the running
image can say which commit it is.

## Consequences

**Anyone who finds the hostname can spend this box's CPU, slowly.** One question
a minute, one simulation a minute, bounded bursts. The worst case is a stranger
occupying the demonstration for a few minutes, recoverable by waiting rather
than by restoring anything — the same conclusion ADR 0008 reached for simulation
control, extended to the Copilot.

**The rate limits are sized to clear the demonstration, not to shape it.** A
single dashboard tab polls about ten times a minute idle and fifty with a live
run; the global ceiling is thirty times the resting rate. A limit that fires
during the canonical demonstration has cost more than it saved, and the numbers
are recorded in `env.template` beside that reasoning.

**Phase 12, if there is one, inherits two specific constraints.** The event
stream cannot be authenticated by header at all — `EventSource` cannot set them
— so any scheme has to work for a cookie or a query parameter. And a login must
not be the only way in: a demonstration whose first screen is a password prompt
is a demonstration most reviewers will not see.
