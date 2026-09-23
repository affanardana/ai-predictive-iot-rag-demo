# ADR 0001 — Separate persistence models with explicit mappers

**Status:** Accepted (Phase 1)
**Date:** 2026-09-23

## Context

`CODING_STANDARDS.md` forbids the domain layer from importing SQLAlchemy, so the
domain's representation of a machine cannot be the same class as its database
table. Two options:

**A. Separate ORM models plus explicit mappers.** A pure domain entity, a
separate persistence model, and a mapper pair that converts between them.

**B. Imperative mapping.** `registry.map_imperatively(MachineEntity, table)`
binds domain classes to tables without them inheriting from a SQLAlchemy base.

## Decision

Option A.

## Rationale

Option B removes the duplicated field declarations, which is a genuine and
attractive benefit. It fails on the project's stated priorities, though.

The coding standards rank maintainability and readability above cleverness, and
state *Explicit over Implicit*. Imperative mapping means a reader must know
SQLAlchemy's mapping machinery to answer "where is this attribute stored?" — the
answer is not in either file they are looking at.

It also constrains the domain. Entities bound imperatively must satisfy
SQLAlchemy's instrumentation requirements: mutable attributes, a particular
constructor contract, and no interference with `__setattr__`. The standards ask
for immutable value objects. The two pull against each other, and resolving the
conflict would mean weakening the domain model to accommodate the persistence
layer — the dependency direction inverted in practice if not in imports.

Option A keeps the domain exactly as expressive as the business rules require.

## Consequences

**Accepted cost.** Every field is declared twice — once in `domain/entities/`,
once in `infrastructure/persistence/sql/models/`. Mappers are the only place
that knows both shapes.

**Mitigation.** `alembic check` runs in CI against a real PostgreSQL instance and
reports when the models and the migrations disagree. The duplication therefore
tends to surface as a red build rather than a silent runtime failure, which is
the difference between an acceptable cost and an unacceptable one.

**Caveat, learned by getting it wrong.** `alembic check` does **not** compare
CHECK constraints — Alembic's autogenerate has no support for them. The first
version of the initial migration named its CHECK constraints wrongly (the naming
convention wrapped each name a second time, producing
`ck_predictions_ck_predictions_failure_probability_range`) and still passed
`check` cleanly. It was caught by reading `alembic upgrade head --sql`, not by
the guardrail.

So the guarantee is narrower than "models and migrations agree": it covers
tables, columns, types, nullability, indexes, foreign keys, and unique
constraints. **Review the generated DDL whenever a migration touches a CHECK
constraint**, because nothing else will.

**Follow-on.** Adding a field means editing the entity, the model, the mappers,
and generating a migration. That is four edits for one field, and it is a
deliberate trade: the moment a field is added to only one side, CI says so.
