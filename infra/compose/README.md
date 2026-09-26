# The self-hosted backend

Four containers on one VPS. PostgreSQL stays on Supabase and the broker stays
at `broker.emqx.io`, so neither is a container here — which is most of the
reason this fits on a single core.

```text
api        :8000   internal, published to 127.0.0.1:8100
inference  :8001   internal only
simulator  :8002   internal only — no host port, deliberately
n8n        :5678   internal, published to 127.0.0.1:8101

Supabase PostgreSQL   external, unchanged
broker.emqx.io        external, unchanged
```

**The simulator's missing port is load-bearing.** Its control surface has no
authentication, so publishing it would put an open "start a run" endpoint on the
internet. `apps/api/tests/unit/infrastructure/test_deployment_shape.py` fails if
a `ports:` entry is ever added to that service, along with several other things
that are easy to remove and impossible to notice.

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

## Ingesting the maintenance corpus

Phase 9's retrieval does nothing until the corpus is in the database, and it is
ingested from a container rather than from a laptop — the same principle the
simulator follows:

```bash
cd /opt/pdm/infra/compose
docker compose --profile tools run --rm ingest
```

That parses the ten PDFs in `dummy_pdfs/`, sends them to the API inside the
compose network, and activates each version. The service is profile-gated, so
`docker compose up -d` never starts it: it is a job, and a job with a restart
policy would re-ingest on a loop. It needs the API up and migrated — the
migration is applied from a developer machine, above.

Two things to expect the first time. The build fetched ~180 MB of model weights
from HuggingFace, so the first request loads two encoders on one core and takes
a few seconds; and running it a second time reports `unchanged` for every
document, because re-ingesting identical content writes nothing and costs no
embedding at all.

Then measure that retrieval works, from the same image:

```bash
docker compose --profile tools run --rm ingest \
  python -m ml knowledge evaluate --api-url http://api:8000
```

It prints two columns — vector search alone and with the cross-encoder — over
`knowledge/eval/questions.json`, and those numbers are the phase's evidence.
`--out results.json` writes them down. The abstention row is the one PRD section
19 turns on: the share of questions the corpus cannot answer where the system
said so instead of returning its nearest neighbour.

The same check by hand: open the dashboard's **Knowledge** page and ask *"Vibration
is rising on M003. What should I inspect?"* — the Bearing Inspection SOP should
come back with its section and page. Ask for a gearbox torque specification and
it should say the documentation does not cover it.

## Running the demonstration

Register the machine first, or every reading is refused:

```bash
curl -X POST "https://pdm-api.72-61-214-194.sslip.io/api/v1/machines" \
  -H "X-Ingest-Token: <your INGEST_API_TOKEN>" \
  -H "Content-Type: application/json" \
  -d '{"machine_id":"M003","name":"Demo motor"}'
```

Then open the dashboard, pick M003, and press **Run the demonstration** on its
machine page — or use `/simulation` for the full form. The simulator runs in its
own container; **nothing needs starting on a laptop.** That is `PRD.md` AC-010,
which this stack did not satisfy until Phase 8.

Reading 60 arrives about a minute in, and that is when the first prediction
fires.

### Why the notional duration is 240 minutes

The form asks for *simulated* time, not how long you watch. At the default
one-minute sample interval, 240 minutes is 240 readings, and at one reading per
second that is four minutes of watching replaying four hours of degradation.

The minimum the form accepts is 60 minutes, and that floor is not arbitrary: the
model needs 60 consecutive readings before it will predict anything. A shorter
run produces no risk band and no incident, which is indistinguishable from a
broken pipeline. `MASTERPLAN.md` §Phase 8's own example says `10 minutes`, which
at these defaults would produce ten readings and never be scored — so the
example is read as the wall-clock length of the demonstration rather than as the
value for this field.

### One core, now with five containers

The simulator publishes about one message per second while a run is active, and
this box shares a single core between it, n8n, the API and torch. The Modal
deployment was already CPU-starved at this workload — it logged `waiting to be
scheduled on a CPU worker` during a live run — and this box gives everything
less CPU, not more.

So the simulator service carries explicit limits in `compose.yaml`, and the API
refuses more than three concurrent runs. Both are estimates rather than
measurements; tune them from `docker stats` during a real run.

**Runs are bounded, and an idle simulator costs nothing.** It sits healthy and
does nothing until the dashboard asks it to start, so unlike the old manual
workflow there is nothing to remember to stop.

### Running more than one machine at once

One run per machine, and several machines concurrently — M003 degrading while
M005 overheats. The fleet view is built for it: a single global run would leave
every other row frozen.

Two runs on the *same* machine are refused, by the API and again by a unique
index in the schema. They would interleave two scenarios' readings on one
machine, and the risk band they produced would describe neither.

### A second run appends to the charts

Reset clears a run's record; it does not delete the telemetry it produced, and
cannot — deleting stored readings is a destructive write, which would have to be
guarded by the ingest token, which a browser cannot hold. So a machine's charts
accumulate across runs.

The windows hide it in practice: `recorded_at` is wall-clock and the charts are
windowed, so an earlier run falls out of the one-hour view on its own.

## When something is wrong

| Symptom | Cause |
|---|---|
| `address already in use` on 80/443 | Something started its own Caddy. This stack must not have one — the host already does. |
| Caddy refuses to reload | A syntax error in the edited Caddyfile. Restore `Caddyfile.bak`; the running config is untouched until `reload` succeeds. |
| The other project's site 404s after an edit | Same. Restore the backup and reload again. |
| `401 unauthorized` on every write | `INGEST_API_TOKEN` and `PDM_INGEST_TOKEN` disagree. They are set from one variable, so this means `.env` changed without a restart. |
| `404 machine_not_found` | The machine is not registered. Registration is idempotent, so just run it. |
| n8n crash-loops: `EACCES: permission denied, mkdir '/data/.n8n'` | The `n8n-data` volume is owned by root and n8n runs as uid 1000. The `n8n-permissions` service fixes this on every `up`, so seeing it means that service did not run — check `docker compose ps -a` shows it `Exited (0)`, and `docker compose up -d n8n-permissions` to run it by hand. |
| n8n will not start: `Mismatching encryption keys` | `N8N_ENCRYPTION_KEY` changed after first boot. Restore the original, or wipe the `n8n-data` volume and start again — there is no third option. |
| Inference container restarts repeatedly | `best.pt` or `artifact/normalization.json` is missing from `MODEL_DIR`. The service exits rather than serving degraded, so the log names the file. |
| `inference_unavailable` (503) from the API | The inference container is down or still loading. On one core, a cold start takes a while. |
| Inference container restarts naming a model, after a rebuild | The build could not reach HuggingFace, so the weights were not baked in. `HF_HUB_OFFLINE=1` is set at runtime on purpose: a missing model fails loudly rather than reaching for the network mid-demonstration. Rebuild with the network up. |
| `retrieval_unavailable` (503) from `/knowledge/search` | The same container, one stage further along — embedding or reranking could not be performed. Note this is *not* the same as "no documents matched", which is a 200 with `sufficient: false`. |
| `/knowledge` shows no documents | The corpus has not been ingested. Run the `ingest` service above. |
| Ingest fails with `document_content_conflict` (409) | A version already exists with different text. Bump the version in `knowledge/corpus.json`, or re-run with `--allow-replace` if the change is a correction. |
| API refuses to start, naming an unexpected variable | `Settings` uses `extra="forbid"`. Something is passing it a variable the API does not read — check nothing added an `env_file` to the `api` service. |
| The workflows page spins forever | n8n's `/rest/push` websocket is not getting through. Check the shared Caddy is proxying `Upgrade`. |

## After it works

Stop the Modal apps — `modal app stop pdm-api` and `modal app stop pdm-n8n` —
once the VPS is verified. Keep `modal app stop` rather than deleting, so the
rollback exists until you are confident. The `pdm-model` volume is worth keeping
as an off-host copy of the checkpoint.
