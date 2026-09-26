# Architecture

This document explains how the backend is put together, why the boundaries fall
where they do, and what to change when adding to it. Decision records for the
choices worth revisiting live in [`adr/`](adr/).

## Layers

```text
                    ┌───────────────────┐
                    │   presentation    │  HTTP, schemas, error mapping
                    └─────────┬─────────┘
                              │
                    ┌─────────▼─────────┐
                    │    composition    │  the only place that knows every layer
                    └─────────┬─────────┘
                              │
            ┌─────────────────┴─────────────────┐
            │                                   │
   ┌────────▼────────┐                 ┌────────▼────────┐
   │ infrastructure  │                 │   application   │
   └────────┬────────┘                 └────────┬────────┘
            │                                   │
            └─────────────────┬─────────────────┘
                              │
                    ┌─────────▼─────────┐
                    │      domain       │
                    └───────────────────┘
```

`infrastructure` and `application` sit at the same level. They do not import each
other: infrastructure implements the domain's ports, and application orchestrates
through those same ports without knowing which adapter answers.

### `domain`

Entities (`Machine`, `TelemetryRecord`, `Prediction`, `Incident`), value objects
(`MachineId`, `SensorReading`, `FailureProbability`, `RiskLevel`, `RiskThresholds`,
`TimeWindow`, `Evidence`), two domain services (`RiskLevelClassifier`,
`IncidentPolicy`), and the ports that infrastructure implements.

**Imports no framework.** Not FastAPI, SQLAlchemy, Pydantic, psycopg, NumPy, or
PyTorch — nor any outer layer of this project.

### `application`

Eight use cases, each a small object taking a `UnitOfWork` by constructor
injection. No SQL, no HTTP types, no model loading. A `summaries` module holds
the machine-summary assembly shared by the fleet list and the detail view, so the
two cannot disagree about the same machine.

Two of them write, and they are the only two. `IngestTelemetry` persists a batch
of readings and reports which machines now hold enough history to score;
`RecordPrediction` reads a machine's window back out, asks the model, and
persists the prediction together with any incident it raises. `RegisterMachine`
adds to the fleet. Everything the pipeline does to the database passes through
one of the three.

### `infrastructure`

- `config` — `Settings` via `pydantic-settings`, with a guard that rejects
  Supabase's transaction pooler at startup.
- `logging` — structured JSON with a redaction filter. Both the message and any
  structured extra are scrubbed, because driver errors routinely embed the DSN.
- `persistence/memory` — in-memory repositories, held to the same contract suite
  as the SQL ones.
- `persistence/sql` — SQLAlchemy models, mappers, repositories, unit of work.
- `system` — `SystemClock`, `DatabaseHealthProbe`.
- `error_translation` — turns driver exceptions into domain storage errors.

### `presentation`

Routers, Pydantic response schemas, presenters (domain → schema), and the error
catalog. No business logic and no direct persistence access.

It also holds the one credential in the system: `require_ingest_token`, applied
to the three routes that write. It lives here rather than in `domain/errors.py`
because authentication is a property of how a request arrived, not a business
rule — the domain never sees it, and `error_catalog.py` is typed as a mapping
over `DomainError` to keep that boundary meaningful. Phase 11 owns real
authentication; this is one machine proving to another that it is the expected
caller.

### `composition`

`build_container(settings)` selects adapters and wires the use cases. Every seam
a later phase adds — a PyTorch predictor, an MQTT consumer, a pgvector retriever,
an LLM provider — plugs in here by adding a constructor argument. Nothing in
`application` or `presentation` needs to change.

## How the rules are enforced

Conventions decay. These are gates:

| Rule | Enforced by |
|---|---|
| Layer direction | `import-linter` `layers` contract |
| `domain` imports no framework | `import-linter` `forbidden` contract with `include_external_packages` |
| `application` imports no adapter or web framework | `import-linter` + AST tests |
| `presentation` never imports `infrastructure` | `import-linter` + AST tests |
| Models and migrations agree, except CHECK constraints | `alembic check` in CI |
| Both repository adapters behave identically | the shared contract suite |

The AST tests exist alongside `import-linter` because `lint-imports` is a
separate command. Without them, a violation could pass `pytest` and only surface
in a CI step someone might reorder away.

## Two representations of the same data

Domain entities and SQLAlchemy models are separate types, with explicit mappers
in `infrastructure/persistence/sql/mappers.py`.

The alternative — SQLAlchemy's imperative mapping, binding domain classes
directly to tables — avoids the duplicated field declarations but forces domain
entities to satisfy SQLAlchemy's instrumentation requirements, which conflicts
with the immutable-value-object rule and reads as implicit. The domain's
independence was judged worth the duplication.

What keeps the duplication honest is `alembic check`, which reports when the
models and the migrations drift apart, so the cost is usually a red build rather
than a silent runtime bug.

It does **not** cover CHECK constraints — Alembic's autogenerate has no support
for them, and a wrongly-named CHECK constraint passes `check` cleanly. When a
migration touches one, review `alembic upgrade head --sql`. See
[ADR 0001](adr/0001-separate-persistence-models.md).

## Testing strategy

Two tiers.

**Default tier — SQLite in memory.** `uv run pytest` needs no database, no
credentials, and no network. Tables come from `metadata.create_all`, not Alembic.

**PostgreSQL tier.** `uv run pytest -m postgres` against a real database,
covering what SQLite cannot: `ON CONFLICT ... RETURNING`, CHECK constraints,
`timestamptz` semantics, and the epoch-bucketing expression. Locally this runs
against a throwaway schema in the shared database, so it can never touch
development rows.

One **contract suite** (`tests/contract/repository_contract.py`) is run against
both adapters. This is what makes the in-memory double trustworthy: it cannot
quietly diverge from the SQL implementation, because both face the same
assertions.

Test doubles for ports are hand-written rather than `unittest.mock.Mock`. A
`Mock` accepts any attribute and any call, so a renamed port method would let
tests keep passing while production code failed. Hand-written doubles implement
the real Protocols and are type-checked against them.

Time is injected through the `Clock` port; no test patches `datetime`.

## Adding to the system

**A new provider** (predictor, retriever, LLM): define the port in
`domain/ports/`, implement it in `infrastructure/`, and add a constructor
argument in `composition/container.py`. Nothing above the composition root
changes — that is the point.

**A new use case**: add it to `application/use_cases/`, wire it in
`composition/container.py`, expose it through a router. The router calls the use
case, passes the result through a presenter, and returns a schema.

**A new domain rule**: put it in the entity or value object that owns it, or in a
domain service if it spans several. Add tests at the boundaries — the values
where behaviour changes are the ones worth asserting.

**A schema change**: edit the model, generate a migration, and let
`alembic check` confirm they agree.

## Known limitations

- **`GET /api/v1/machines` issues O(3n) queries** for `n` machines, because each
  summary reads its latest reading, prediction, and incidents separately. Fine at
  current scale, and the coding standards discourage premature optimisation, but
  it should be replaced by a bulk read port before the Phase 7 dashboard polls it.
- **`IncidentType` is always `UNCLASSIFIED`.** The model is a binary failure
  classifier and cannot say what is failing. Three resolutions are documented in
  `domain/value_objects/incident_type.py`; the stub is now reached by real
  incidents, so the gap is visible in the product rather than only in the type.
- **Incidents are suppressed while one is open.** `RecordPrediction` raises an
  incident only when the machine has none open, which is a rule the PRD does not
  state — §10 says only "create an incident when configured predictive-risk
  conditions are satisfied". Without it a machine that crosses HIGH stays there
  and files one incident per reading. The residual race, two concurrent scorings
  both finding none, is accepted rather than locked against.
- **The simulator's clock runs ahead of the wall clock.** `recorded_at` advances
  at `sample_interval × index` while the paced loop sleeps `tick_seconds` per
  tick, so a demo at 300× produces timestamps an hour ahead of now within twelve
  seconds. Anything bounded by `clock.now()` — every `GET
  /machines/{id}/telemetry` — returns nothing during such a run. Phase 6 works
  around it with count-based reads and a `--started-at` in the past; the fix is
  a "relative to newest record" query mode, and it belongs to Phase 7.
- **`list_for_machine` is used to count open incidents** in the fleet summary,
  capped at 500 rows. A dedicated count query is the right fix when incident
  volume grows.
