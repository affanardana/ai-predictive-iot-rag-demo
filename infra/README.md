# Infrastructure

**Status: partially implemented — deployment material arrives with later phases.**

Cross-cutting deployment and orchestration assets that are not application code.

Present today:

- `.github/workflows/ci.yml` — the CI pipeline (format, lint, types,
  architecture contracts, offline tests, and a PostgreSQL job that applies
  migrations and checks for model/migration drift).

Planned:

- `infra/n8n/` — exported n8n Cloud workflow definitions, version-controlled
  (Phase 6);
- container definitions for the API (Phase 11);
- deployment configuration for the simulator (Modal) and the web app (Vercel).

## Open question affecting this directory

MASTERPLAN §Phase 6 reads literally as n8n writing telemetry rows into
PostgreSQL itself. Taken literally, n8n would have to reimplement the
idempotency rule, the risk-band mapping, and the incident policy in JavaScript —
duplicating the domain and hollowing out the architecture.

The recommended alternative is that n8n calls an authenticated internal API
endpoint instead, keeping the domain the single source of truth. That is a real
deviation from a literal reading of the masterplan and needs the repository
owner's explicit decision before Phase 6 begins. It does not affect Phase 1.
