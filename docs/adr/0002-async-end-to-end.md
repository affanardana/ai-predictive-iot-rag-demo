# ADR 0002 — Async end-to-end

**Status:** Accepted (Phase 1)
**Date:** 2026-09-23

## Context

The stack could be synchronous (FastAPI with sync path operations, sync
SQLAlchemy sessions) or asynchronous throughout.

## Decision

Asynchronous throughout. Use cases and repository methods are `async def`;
infrastructure uses `create_async_engine` and `AsyncSession`.

## Rationale

The later phases are dominated by I/O-bound work. The Copilot calls an LLM, RAG
computes embeddings and reranks, and inference runs over the network on Modal.
Under a sync design, each of those calls either blocks a worker thread or forces
`asyncio.run` at a layer boundary — and `asyncio.run` inside a running event loop
raises outright.

Mixing the two styles is the failure mode worth avoiding: a sync use case that
needs one async collaborator tends to acquire a thread pool, and the boundary
between the two becomes a debugging hazard.

The choice is *visible* in every application-layer signature, which is why it is
recorded here rather than left as an implementation detail.

## Consequences

- Repository and use-case methods are `async def`; every call site awaits.
- Repository *implementation* is confined to infrastructure, so the async
  plumbing does not leak into the domain. The domain's Protocols declare `async`
  methods, which is a signature convention rather than an I/O dependency — the
  existing architecture tests continue to prove the domain imports no framework.
- `expire_on_commit=False` on the session factory, because refreshing attributes
  after a commit would trigger lazy I/O, which cannot happen implicitly in async
  code.
- Tests use `pytest-asyncio` in `asyncio_mode = "auto"`, so async test functions
  and fixtures need no explicit marker.

## Alternatives considered

**Sync core with async edges.** Would keep use cases simpler, but pushes the
`asyncio.run` problem to exactly the phases that need async most, and would have
to be reversed later.

**Sync throughout.** Simplest to read, and defensible for Phase 1's read-only
endpoints. Rejected because it would need rewriting at Phase 9, when the first
genuinely concurrent I/O arrives.
