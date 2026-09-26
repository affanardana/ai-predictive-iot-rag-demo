# ADR 0005 — Compute moves to a self-hosted VPS

**Status:** Accepted (between Phase 6 and Phase 7)
**Date:** 2026-09-26

## Context

Phases 5 and 6 deployed three services to Modal: the API, the model service, and
n8n. It worked, and Phase 6's exit condition was met there — telemetry flowed
from the simulator through EMQX, n8n, the API and PostgreSQL, through inference,
and produced a prediction and an incident, all verified live.

Three things pushed toward moving anyway.

**Modal is a poor shape for two of the three.** n8n is a stateful Node
application whose MQTT Trigger holds a persistent connection; running it on a
platform built for request-scoped Python needed an always-warm container, four
failed image builds before the Alpine/musl problem was found, and a class-based
workaround when `@modal.concurrent` was silently ignored on a plain function.
None of that is Modal's fault, and all of it is work that vanishes when n8n runs
in a container the way its authors intended. The API has the same problem in
miniature: `asgi_app` was a fine fit, but the deployment still needed a secret,
a Volume and a deploy step for something that is one uvicorn process.

**The compute is small and idle.** The model is two LSTM layers and a linear
head. The API is a handful of queries and one HTTP call. Modal's scale-to-zero
was the right answer when there was no server; it is less compelling when one
already exists and is paid for.

**A server makes the next phase cheaper.** Phase 7 needs a public place to serve
from, and Phase 9 needs pgvector. Both are simpler on a host the owner controls
than across three vendors' free tiers.

## Decision

**Move the compute to a single Ubuntu VPS running Docker Compose.** Caddy
terminates TLS; the API, the model service and n8n run as containers behind it.

**PostgreSQL stays on Supabase and the broker stays at `broker.emqx.io`.** Both
were offered as things to bring in-house and both were declined, which is what
keeps this to four containers on one core.

## Rationale

**Every service runs the way it was designed to.** n8n uses its official image
with its own entrypoint — no `ENTRYPOINT []`, no Python injected into a Node
image, no concurrency decorator. The API is one uvicorn process. The whole class
of problems recorded in `infra/n8n/README.md` is specific to hosting an
application inside a function platform, and it stops existing here.

**`Docker` is already in the stack.** `MASTERPLAN.md` §5 lists it under CI/CD.
This is that line being used rather than a new technology introduced.

**§3.1 is satisfied, and more literally than before.** *"Production execution
must not depend on a local computer… the deployed system must operate entirely
through cloud services."* A VPS is a cloud service. The rule's stated harm is a
dependency on the developer's laptop, and this removes it exactly as Modal did.

**The cost is bounded and known.** One core, 3.8 GB, 48 GB of disk — and the
two things that would have strained it are not here, because Supabase and the
broker stayed external.

## Consequences

**What is given up.** Modal scaled to zero between demonstrations and to four
containers under load; a VPS does neither. Its free tier absorbed the cost of
idleness, which a VPS does not — though a VPS is also flat-rate, so the trade is
predictability for elasticity. There is no longer a platform handling TLS,
certificates or restarts; Caddy and `restart: unless-stopped` do, and they are
now the owner's to operate.

**Operational responsibility moves.** Backups of the n8n volume, of
`N8N_ENCRYPTION_KEY`, and of the model artefacts are the owner's, where Modal's
Volumes were at least durable by default. The encryption key in particular is
set explicitly at first boot precisely so it can be backed up, rather than being
trapped in a volume as the only copy — which is the failure `infra/n8n/README.md`
records.

**`N8N_ENCRYPTION_KEY` guidance inverts.** `infra/n8n/README.md` says not to set
it. That was correct for an instance already running, where an environment
variable disagreeing with the stored key stops n8n from booting. It is wrong for
a fresh instance, and this deployment sets it before first boot.

**A deviation from `PRD.md` §25 deepens, and is narrower than it looks.** Items
4 and 5 name EMQX Cloud and n8n Cloud as mandatory, and this replaces the
*compute* host — Modal — which §25 does not name at all. Supabase is §25's
"PostgreSQL", kept. The broker is still EMQX's own public service, not a
competitor's. Both substitutions were already recorded in `infra/README.md`; the
one thing genuinely new is that n8n and the API now run on infrastructure the
owner operates rather than a managed platform.

**A trap was avoided in the build.** `uv.lock` resolves torch from PyPI, whose
Linux wheel is the CUDA build — about 2.9 GB of download for compute this
project does not do. On Windows the `sys_platform == 'linux'` markers that carry
those dependencies are false, so the problem is invisible from the development
machine. Both Dockerfiles install by name from the CPU index, mirroring what
`services/inference/modal_app.py` already did and nowhere else recorded.

## What this supersedes

**ADR 0003 stays in force for `psycopg 3`, which is unaffected.** Its closing
section is already titled "Note on portability" and says the decision touches no
use case. Only the session-pooler guard becomes inert — and it stays in the
code, because it is asserted by tests and is harmless against a non-Supabase
host.

**The `modal_app.py` files are not deleted.** They are the specification the
Dockerfiles were translated from, and they remain a working rollback until the
VPS is verified in production use.
