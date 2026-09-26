# Web application

The fleet dashboard: live telemetry, predictive risk, and maintenance
incidents for the simulated motors.

React + TypeScript + Vite + Tailwind + Apache ECharts, deployed to Vercel.

## Running it

The API must be running, because nothing here is mocked:

```bash
# from the repository root
uv run uvicorn api.main:app --reload --port 8000
```

Then:

```bash
cd apps/web
cp .env.example .env.local
npm install
npm run gen:api      # needs openapi.json — see below
npm run dev
```

### Generating the API types

`src/api/schema.d.ts` is generated, and **does not exist until you generate it**,
so the first `npm run typecheck` on a fresh clone fails until you have. Without
it every schema type resolves to `any`, and the errors that produces point at
the components rather than at the missing file.

One command does both halves — it exports the document from the API's own code,
then generates the types from it:

```bash
cd apps/web
npm run gen:api
```

The export needs no server and no database. Run it directly if you want the
document on its own:

```bash
uv run python apps/api/scripts/export_openapi.py --out openapi.json
```

**Use `--out`, not `> openapi.json`.** A shell redirect writes UTF-16 on Windows
PowerShell — every character separated by a null byte — and the result is a file
that looks like JSON to a human and that every parser rejects, with an error
naming a byte offset rather than the shell that produced it.

`openapi.json` is gitignored; `src/api/schema.d.ts` is committed. That
asymmetry is the point: CI re-exports the document, regenerates the types and
fails on any difference, so a schema change that was not reflected in the
frontend is a red build rather than an empty table in production.

## Routes

| Route | What it shows |
|---|---|
| `/` | Fleet summary and machine table |
| `/machines` | The machine list |
| `/machines/:id` | Current state, telemetry, prediction history, incidents |
| `/incidents` | Incident list, with acknowledge / resolve / dismiss |
| `/simulation` | Placeholder — Phase 8 owns simulation control |
| `/copilot` | Placeholder — Phase 10 owns the Copilot |

## How live updates work

`GET /api/v1/events` is a server-sent event stream. **Each frame names a machine
and what changed about it; it carries no data.** The client responds by
invalidating the relevant queries and refetching them, so the REST endpoints
stay the only source of truth and a dropped frame costs freshness rather than
correctness.

Three consequences worth knowing before changing anything here:

- **One `EventSource` for the whole app**, in `EventStreamProvider`. Browsers
  cap concurrent event streams per origin at six on HTTP/1.1, and reconnecting
  on every navigation would show stale data on each page change.
- **The stream is a latency optimisation, not a correctness mechanism.** Every
  query has a poll interval behind it, which is what makes it safe for the
  server to drop events when a client falls behind.
- **`src/realtime/types.ts` is hand-written**, the one place the generated-client
  rule is broken — a `text/event-stream` body cannot be described in OpenAPI.
  `apps/api/tests/integration/test_events_api.py` pins the field names and every
  event kind, so a rename on the server fails the Python suite rather than
  silently freezing this app.

## Testing

```bash
npm run typecheck
npm run test        # vitest
npm run build
```

Vitest covers the pure logic — the event-to-invalidation mapping, the risk
bands, timestamp staleness, and the chart option builders. It deliberately does
not render ECharts or snapshot markup. End-to-end testing is Phase 11's, per
`MASTERPLAN.md` §5.

## Deploying

**Vercel project settings:**

- **Root Directory: `apps/web`.** Not optional, and the first thing to check if a
  deploy fails oddly. Left at the repository root, Vercel runs framework
  detection there, finds the `pyproject.toml` at the top of the repo, decides
  the project is a FastAPI application, and offers to deploy the *backend* —
  which lives on the VPS and is not what is being hosted here.
- **Framework Preset: Vite**, Build Command `npm run build`, Output `dist`.
  All three are detected once the root directory is right.
- **Environment variable `VITE_API_BASE_URL`**, set for Production *before* the
  first build. Vite inlines it at build time, so it can be neither changed nor
  added without a redeploy.

`vercel.json` holds one SPA rewrite, so `/machines/M003` works when pasted into
a fresh tab instead of 404ing. Vercel applies rewrites **after** filesystem
matching, so `/assets/*` still resolves to real files. The file is deliberately
minimal: Vercel validates it against a strict schema that rejects unknown keys,
including a `//` key used as a comment — and JSON has no comment syntax anyway.

The build runs `tsc --noEmit && vite build`, so a type error fails the deploy
rather than shipping. That works because `src/api/schema.d.ts` is **committed**
rather than generated at build time, which means Vercel needs no Python
toolchain and no access to the API. `apps/api/tests`' CI drift check is what
keeps that file honest.

One environment variable:

```
VITE_API_BASE_URL = https://pdm-api.72-61-214-194.sslip.io
```

Vite inlines `VITE_`-prefixed variables at **build** time, so changing this
requires a **redeploy**, not a restart — and no secret may ever be given that
prefix, because everything with it ends up in the JavaScript.
