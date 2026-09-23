# Web application

**Status: not implemented — Phase 7.**

A React + TypeScript dashboard for fleet monitoring, machine detail, incidents,
simulation control, and the Maintenance Copilot.

Planned stack (MASTERPLAN §5): React, TypeScript, Vite, Tailwind CSS, Apache
ECharts, deployed to Vercel.

Planned routes (PRD §20):

```text
/                 fleet overview
/machines         machine list
/machines/:id     machine detail
/incidents        incident list
/simulation       simulation control
/copilot          Maintenance Copilot
```

The backend contract these routes consume is already served as OpenAPI at
`/api/v1/openapi.json` — the frontend client should be generated from that
document rather than hand-written, so a schema change cannot silently drift
from the UI.

This directory is intentionally empty until Phase 7.
