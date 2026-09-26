# The self-hosted backend

Four containers on one VPS: Caddy for TLS, the API, the model service, and n8n.
PostgreSQL stays on Supabase and the broker stays at `broker.emqx.io`, so
neither is a container here — which is most of the reason this fits on a single
core.

```text
caddy      :80 :443   TLS, and the only thing exposed
api        :8000      internal
inference  :8001      internal, torch CPU
n8n        :5678      internal

Supabase PostgreSQL   external, unchanged
broker.emqx.io        external, unchanged
```

Nothing here runs on the developer's machine, which is what `MASTERPLAN.md` §3.1
requires.

## Prerequisites

**Docker with the Compose plugin.** On Ubuntu:

```bash
curl -fsSL https://get.docker.com | sh
docker compose version
```

**The repository, committed.** This is worth stating before anything else,
because it is the one step that can silently give you the wrong code: as of this
writing the entire Phase 5 and Phase 6 body of work — `services/inference/`,
`apps/api/modal_app.py`, `infra/n8n/`, ADR 0004 — is **uncommitted**. A
`git clone` on the server would produce a tree without the inference service, the
ingest route, or the n8n workflow. Commit and push first.

**Two model files, copied by hand.** `best.pt` and `normalization.json` are
gitignored, so a clone cannot contain them and the inference container will not
start without them. From the development machine:

```powershell
ssh root@72.61.214.194 "mkdir -p /opt/pdm/infra/compose/model/artifact"
scp colab_report/best.pt                       root@72.61.214.194:/opt/pdm/infra/compose/model/
scp data/training/normalization.json           root@72.61.214.194:/opt/pdm/infra/compose/model/artifact/
```

The layout matters. `INFERENCE_CHECKPOINT` is `/model/best.pt` and
`INFERENCE_ARTIFACT_DIR` is `/model/artifact`, and the service checks for
`normalization.json` inside the latter before it will open its port.

## Deploying

```bash
cd /opt/pdm/infra/compose
cp env.template .env
nano .env                      # PDM_HOST, DATABASE_URL, the two generated secrets
docker compose config          # renders cleanly, no warnings
docker compose up -d --build
```

Two secrets to generate — `openssl rand -hex 32` for each:

- `INGEST_API_TOKEN` — one value, read by both the API and n8n. `compose.yaml`
  sets both names from it so they cannot drift.
- `N8N_ENCRYPTION_KEY` — set **before** n8n's first boot, and never changed
  afterwards.

**The first build installs torch, and this is a one-core box.** Expect it to
take several minutes. It installs from the CPU index deliberately — see the
comment at the top of `Dockerfile.inference` — because the default PyPI wheel
for Linux pulls roughly 2.9 GB of CUDA libraries this service cannot use.

Migrations are applied from a developer machine, not from a container:

```bash
docker compose run --rm api alembic -c /app/alembic.ini upgrade head
```

A batch operation with one writer, run from a web container, means every replica
races to apply the same revision on a cold start. Supabase is already migrated,
so this is only needed against a fresh database.

## Checking it

```bash
docker compose ps                                   # all healthy
curl https://api.$PDM_HOST/health                   # {"status":"ok","app_env":"production"}
curl https://api.$PDM_HOST/health/ready             # "The database is reachable."
curl -o /dev/null -w '%{http_code}\n' https://$PDM_HOST/healthz   # 200
curl -o /dev/null -w '%{http_code}\n' -X POST \
     https://api.$PDM_HOST/api/v1/telemetry \
     -H 'Content-Type: application/json' -d '{"records":[]}'      # 401
```

That last one returning **401** is the point. It proves the write guard is live
rather than the endpoint being open to anyone who finds the hostname.

## n8n first boot

The editor is at `https://$PDM_HOST` — **not at `/`**, which returns
`Cannot GET /` from n8n's Express router. The UI is at `/setup` on a fresh
instance and `/home`, `/signin`, `/workflows` afterwards.

1. **Create the owner account** at `/setup`. This is the only thing between the
   internet and a workflow that can write to your database, so do it before
   anything else.
2. **Credentials → New → MQTT.** Protocol `mqtts`, host `broker.emqx.io`, port
   `8883`, no username or password.
3. **Import** `infra/n8n/telemetry-ingest.json`.
4. **Open the `Telemetry topic` node and select the credential** from step 2.
   The export carries a placeholder id rather than a real one — it has to, since
   credentials are not in the workflow file — so this link is always manual.
5. **Activate the workflow.** The MQTT Trigger subscribes on activation, not on
   save, and the editor's trigger panel previews incoming messages either way.
   An inactive workflow shows you live traffic while subscribing to nothing.

## Running the demonstration

Register the machine first, or every reading is refused:

```bash
curl -X POST "https://api.$PDM_HOST/api/v1/machines" \
  -H "X-Ingest-Token: $INGEST_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"machine_id":"M003","name":"Demo motor"}'
```

Then, from the development machine:

```powershell
uv run python -m simulator realtime --demo --minutes 240 --tick-seconds 1 `
  --started-at (Get-Date).ToUniversalTime().AddHours(-4).ToString("yyyy-MM-ddTHH:mm:ss+00:00") `
  --sink mqtt
```

Reading 60 arrives about a minute in, and that is when the first prediction
fires.

**Run it only while demonstrating.** One core is shared between n8n, the API and
torch, and the simulator publishes one message per second indefinitely. The
Modal deployment was already CPU-starved at this workload — it logged
`waiting to be scheduled on a CPU worker` during a live run — and this box gives
everything less CPU, not more. It holds for a four-minute demonstration; it is
not meant to run for a week.

## When something is wrong

| Symptom | Cause |
|---|---|
| `401 unauthorized` on every write | `INGEST_API_TOKEN` and `PDM_INGEST_TOKEN` disagree. They are set from one variable, so this means `.env` changed without a restart. |
| `404 machine_not_found` | The machine is not registered. Every message before registration was refused; registration is idempotent, so just run it. |
| The workflows page spins forever | n8n's `/rest/push` websocket is not getting through. Check Caddy is proxying `Upgrade`. |
| n8n will not start: `Mismatching encryption keys` | `N8N_ENCRYPTION_KEY` changed after first boot. Restore the original, or wipe the `n8n-data` volume and start again — there is no third option. |
| Inference container restarts repeatedly | `best.pt` or `artifact/normalization.json` is missing from `MODEL_DIR`. The service exits rather than serving degraded, so the log names the file. |
| `inference_unavailable` (503) from the API | The inference container is down or still loading. On one core, a cold start takes a while. |
| API refuses to start, naming an unexpected variable | `Settings` uses `extra="forbid"`. Something is passing it a variable the API does not read — check nothing added an `env_file` to the `api` service. |
| Editor loads but requests hang | Should not happen here, unlike the Modal deployment: `@modal.concurrent` was needed there because Modal serialises inputs per container by default. Caddy and uvicorn have no such limit. |

## After it works

Stop the Modal apps — `modal app stop pdm-api` and `modal app stop pdm-n8n` —
once the VPS is verified. Keep `modal app stop` rather than deleting, so the
rollback exists until you are confident. The `pdm-model` volume is worth keeping
as an off-host copy of the checkpoint.
