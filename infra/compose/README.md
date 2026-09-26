# The self-hosted backend

Three containers on one VPS. PostgreSQL stays on Supabase and the broker stays
at `broker.emqx.io`, so neither is a container here — which is most of the
reason this fits on a single core.

```text
api        :8000   internal, published to 127.0.0.1:8100
inference  :8001   internal only
n8n        :5678   internal, published to 127.0.0.1:8101

Supabase PostgreSQL   external, unchanged
broker.emqx.io        external, unchanged
```

**There is no Caddy in `compose.yaml`, and that is deliberate.** This host
already runs another project — `~/ifne` — whose Caddy holds `0.0.0.0:80` and
`0.0.0.0:443` through `network_mode: host`. Two reverse proxies cannot share
those ports, and the one that starts second fails with `address already in use`
in a way that looks like a bug in whichever project lost.

One proxy per host is the right architecture: one place for certificates, one
ACME account. So the API and n8n publish to loopback and that Caddy proxies to
them, which is exactly how it already reaches its own API on `localhost:8000`.

Nothing here runs on the developer's machine, which is what `MASTERPLAN.md` §3.1
requires.

## Prerequisites

**Docker with the Compose plugin.** Already present on this host — `docker
--version` reports 29.8.1. On a fresh one:

```bash
curl -fsSL https://get.docker.com | sh
docker compose version
```

**The repository, committed.** The entire Phase 5 and Phase 6 body of work has
to be pushed before a clone can contain it — `services/inference/`,
`apps/api/modal_app.py`, `infra/n8n/`. Verify after cloning:

```bash
ls services/inference/modal_app.py infra/compose/compose.yaml infra/n8n/telemetry-ingest.json
```

All three must be listed.

**Two model files, copied by hand.** `best.pt` and `normalization.json` are
gitignored, so a clone cannot contain them and the inference container will not
start without them. From the development machine:

```powershell
ssh root@72.61.214.194 "mkdir -p /opt/pdm/infra/compose/model/artifact"
scp colab_report/best.pt             root@72.61.214.194:/opt/pdm/infra/compose/model/
scp data/training/normalization.json root@72.61.214.194:/opt/pdm/infra/compose/model/artifact/
```

The layout matters. `INFERENCE_CHECKPOINT` is `/model/best.pt` and
`INFERENCE_ARTIFACT_DIR` is `/model/artifact`, and the service checks for
`normalization.json` inside the latter before it will open its port.

## Deploying

```bash
cd /opt/pdm/infra/compose
cp env.template .env
nano .env                      # DATABASE_URL and the two generated secrets
docker compose config          # renders cleanly, no warnings
docker compose up -d --build
```

Two secrets to generate — `openssl rand -hex 32` for each:

- `INGEST_API_TOKEN` — one value, read by both the API and n8n. `compose.yaml`
  sets both names from it so they cannot drift.
- `N8N_ENCRYPTION_KEY` — set **before** n8n's first boot, and never changed
  afterwards.

**The first build installs torch, and this is a one-core box.** Expect several
minutes. It installs from the CPU index deliberately — see the top of
`Dockerfile.inference` — because the default PyPI wheel for Linux pulls roughly
2.9 GB of CUDA libraries this service cannot use.

Migrations are applied from a developer machine, not from a container:

```bash
docker compose run --rm api alembic -c /app/alembic.ini upgrade head
```

Supabase is already migrated, so this is only needed against a fresh database.

## Publishing it — editing the other project's Caddyfile

`caddy.snippet` holds two site blocks. **Back the file up first, and validate
before reloading**, because a syntax error takes down both projects:

```bash
cd ~/ifne/backend/deploy/vps
cp Caddyfile Caddyfile.bak

# Append the two blocks from /opt/pdm/infra/compose/caddy.snippet
nano Caddyfile
```

Then, in order:

```bash
docker exec vps-caddy-1 caddy validate --config /etc/caddy/Caddyfile
```

Read the output. If it says `Valid configuration`, continue. If it complains
about anything, restore with `cp Caddyfile.bak Caddyfile` and nothing has
changed — **do not run the next command on a failed validation.**

```bash
docker exec vps-caddy-1 caddy reload --config /etc/caddy/Caddyfile
```

That applies the new configuration without restarting the container, so the
other project's traffic is not interrupted. Caddy then requests certificates for
the two new hostnames on the first request to each.

## Checking it

```bash
cd /opt/pdm/infra/compose
docker compose ps
```

All three should read `healthy`.

```bash
curl -o /dev/null -w 'api health   %{http_code}\n' https://pdm-api.72-61-214-194.sslip.io/health
curl -s https://pdm-api.72-61-214-194.sslip.io/health/ready; echo
curl -o /dev/null -w 'n8n healthz  %{http_code}\n' https://pdm.72-61-214-194.sslip.io/healthz
```

Expect `200`, then `"The database is reachable."`, then `200`. **The first
request to each hostname is slower** — Caddy is obtaining a certificate.

The one that matters, which must print **401**:

```bash
curl -o /dev/null -w '%{http_code}\n' -X POST \
  https://pdm-api.72-61-214-194.sslip.io/api/v1/telemetry \
  -H 'Content-Type: application/json' -d '{"records":[]}'
```

That proves the write guard is live rather than the endpoint being open to
anyone who finds the hostname.

Confirm the shared Caddy did not break the other project:

```bash
curl -o /dev/null -w '%{http_code}\n' https://$(grep -oP '(?<=API_HOSTNAME=).*' ~/ifne/backend/deploy/vps/.env)
```

## n8n first boot

The editor is at `https://pdm.72-61-214-194.sslip.io` — **not at `/`**, which
returns `Cannot GET /` from n8n's Express router. The UI is at `/setup` on a
fresh instance and `/home`, `/signin`, `/workflows` afterwards.

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
curl -X POST "https://pdm-api.72-61-214-194.sslip.io/api/v1/machines" \
  -H "X-Ingest-Token: <your INGEST_API_TOKEN>" \
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
everything less CPU, not more.

## When something is wrong

| Symptom | Cause |
|---|---|
| `address already in use` on 80/443 | Something started its own Caddy. This stack must not have one — the host already does. |
| Caddy refuses to reload | A syntax error in the edited Caddyfile. Restore `Caddyfile.bak`; the running config is untouched until `reload` succeeds. |
| The other project's site 404s after an edit | Same. Restore the backup and reload again. |
| `401 unauthorized` on every write | `INGEST_API_TOKEN` and `PDM_INGEST_TOKEN` disagree. They are set from one variable, so this means `.env` changed without a restart. |
| `404 machine_not_found` | The machine is not registered. Registration is idempotent, so just run it. |
| n8n will not start: `Mismatching encryption keys` | `N8N_ENCRYPTION_KEY` changed after first boot. Restore the original, or wipe the `n8n-data` volume and start again — there is no third option. |
| Inference container restarts repeatedly | `best.pt` or `artifact/normalization.json` is missing from `MODEL_DIR`. The service exits rather than serving degraded, so the log names the file. |
| `inference_unavailable` (503) from the API | The inference container is down or still loading. On one core, a cold start takes a while. |
| API refuses to start, naming an unexpected variable | `Settings` uses `extra="forbid"`. Something is passing it a variable the API does not read — check nothing added an `env_file` to the `api` service. |
| The workflows page spins forever | n8n's `/rest/push` websocket is not getting through. Check the shared Caddy is proxying `Upgrade`. |

## After it works

Stop the Modal apps — `modal app stop pdm-api` and `modal app stop pdm-n8n` —
once the VPS is verified. Keep `modal app stop` rather than deleting, so the
rollback exists until you are confident. The `pdm-model` volume is worth keeping
as an off-host copy of the checkpoint.
