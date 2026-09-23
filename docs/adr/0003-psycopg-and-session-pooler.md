# ADR 0003 — psycopg 3, and requiring the session pooler

**Status:** Accepted (Phase 1)
**Date:** 2026-09-23

## Context

Supabase is the development and production database. Supabase offers three ways
to reach a project's Postgres, and they are not interchangeable:

| Route | Port | Notes |
|---|---|---|
| Direct connection | 5432 | Resolves to IPv6 by default; may be unreachable on IPv4-only networks. |
| Session pooler (Supavisor, session mode) | 5432 | Supports server-side prepared statements. |
| Transaction pooler (Supavisor, transaction mode) | 6543 | Does **not** support prepared statements. |

Supabase's own documentation states that ORMs relying on server-side prepared
statements break behind the transaction pooler. SQLAlchemy relies on them.

Separately, the driver had to be chosen: `asyncpg` or `psycopg` 3.

## Decision

Use **psycopg 3** (`postgresql+psycopg://`), and target the **session pooler on
port 5432**. `Settings` raises at startup if `DATABASE_URL` points at a Supabase
pooler host on port 6543.

## Rationale

**Driver.** `asyncpg`'s prepared-statement cache is what produces
`DuplicatePreparedStatement` failures under transaction pooling; working around
it means remembering to disable that cache with a driver-specific connect
argument. psycopg 3 tolerates the pooler modes without special configuration.

A second, unplanned benefit settled the choice: psycopg 3 supports **both sync
and async through the same dialect name**. Alembic therefore runs synchronously
against the same URL the application uses asynchronously, with none of the
`run_sync` plumbing the async Alembic template requires.

**Pooler mode.** A hard failure at startup, rather than a warning, because the
symptom of getting it wrong is intermittent driver errors that say nothing about
the port. A developer would reasonably spend an afternoon blaming their queries.

## Consequences

- `.env.example` documents the session-pooler URL and warns against port 6543.
- The check is scoped by hostname (`*.pooler.supabase.com`), so a local or CI
  PostgreSQL on port 6543 is unaffected — the rule targets a specific
  misconfiguration, not the port number itself.
- `pool_pre_ping=True` on the engine, because managed poolers drop idle
  connections without telling the client; without a pre-flight check the first
  query after an idle period fails with a stale-connection error that looks like
  an outage.
- Reserved characters in the password must be percent-encoded.

## Note on portability

Nothing in this decision reaches the domain or application layers. Swapping to a
different PostgreSQL provider, or to another driver, is a change to
`infrastructure/persistence/sql/session.py` and configuration — not to any use
case.
