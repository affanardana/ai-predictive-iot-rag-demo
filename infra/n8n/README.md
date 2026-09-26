# n8n — the pipeline's orchestrator

> **The Modal deployment in this directory is superseded.** n8n now runs as a
> container in `infra/compose/`, where it uses its official image and its own
> entrypoint — no Python injected into a Node image, no `ENTRYPOINT []`, no
> concurrency decorator. Everything below about the *workflow* still applies
> unchanged, and `telemetry-ingest.json` is imported into the Compose instance
> as-is. The sections about hosting n8n on Modal describe a deployment that is
> kept as a rollback and is no longer how the system runs. Start at
> `infra/compose/README.md`.

`MASTERPLAN.md` §Phase 6 puts n8n between the broker and the database:

```text
Simulator → MQTT → EMQX → n8n → validation → idempotency check → PostgreSQL
          → inference → prediction → incident evaluation
```

**n8n is transport here, not a second backend.** It validates shape, forwards
readings, and asks for a prediction when the API says a machine is ready. It
does not know what a risk band is, when an incident should be raised, or how
idempotency is enforced — all three stay in the API, which is why `add_many_idempotent`
and `DefaultIncidentPolicy` are Python rather than JavaScript. The alternative
reading, n8n writing rows itself, would put the same rules in two places that
could disagree; `docs/adr/0004-n8n-is-transport.md` records the decision.

## Contents

| File | What it is |
|---|---|
| `telemetry-ingest.json` | The workflow, exported. Import it into n8n. |
| `modal_app.py` | Runs n8n on Modal. Nothing else depends on it. |
| `README.md` | This file. |

Credentials are **not** in the workflow. It references them by name, which is
why the export is safe to commit — see [Credentials](#credentials).

## Deploying

n8n Cloud is a paid product and this project runs on free tiers, so n8n is
self-hosted on Modal instead. That keeps the stack the masterplan names, at the
cost of one always-warm container; the reasoning is in `modal_app.py`, along
with the commands to create the secret and the volume.

```powershell
modal secret create pdm-n8n `
    PDM_INGEST_TOKEN=<the same value the API has> `
    PDM_API_BASE_URL=https://<your-deployed-api>
modal volume create pdm-n8n-data
modal deploy infra/n8n/modal_app.py
```

`PDM_API_BASE_URL` is the deployed API, which is
`apps/api/modal_app.py` — see its docstring for the secret it needs. The two
must share a token: `PDM_INGEST_TOKEN` here is `INGEST_API_TOKEN` there, and a
mismatch is a 401 on every write.

**Do not add `N8N_ENCRYPTION_KEY` to this secret.** n8n generates one on first
boot and writes it to `/data/.n8n/config` on the Volume. The environment
variable is optional and is only correct when it matches that stored value —
and when it does not, n8n does not degrade gracefully, it refuses to start:

```
Error: Mismatching encryption keys. The encryption key in the settings file
/data/.n8n/config does not match the N8N_ENCRYPTION_KEY env var.
```

This is not hypothetical. The first deployment of this file was broken by
setting it: the secret was recreated to fix a different problem, the key
changed with it, and the container stopped booting. Recovering from it needs the
original key, which by then only existed on the Volume.

The variable is for restoring an instance whose Volume you still have. The trade
in leaving it unset is that the key lives only on the Volume, so losing the
Volume loses stored credentials. For a demonstration instance that is the right
side of the trade.

**One thing Modal needs that the n8n image does not provide, and it took four
attempts to get right.** A Modal function is Python code — the body shells out to
`n8n start` — so the image must carry an interpreter. The n8n image is Node and
has none, and it is **Alpine**, which is the part the documentation does not
prepare you for:
- Modal's documented remedy, `add_python="3.12"`, installs a **glibc** standalone
  CPython. Alpine is **musl**. The build shows `COPY /python/. /usr/local`
  succeeding and the deploy then fails with *"unable to determine the version of
  Python installed in the Image"*, because there is a Python file there and it
  cannot execute.
- Installing Python with `apt-get` fails too — Alpine has no `apt-get`. It is
  `apk add python3 py3-pip`, which is what `modal_app.py` does.

The other wrong turn is worth knowing about. Building on Modal's own
`debian_slim` and installing n8n from npm satisfies Modal completely and
produces a container where n8n dies on startup: Node 22.18+ type-strips
TypeScript by default and cannot apply the decorators n8n's dependency injection
is built on, while Node 20 cannot build `isolated-vm`, which n8n's tree requires.
The official image exists so nobody has to discover that.

**Modal serves one request at a time per container by default, and n8n cannot
work that way.** The editor opens a persistent websocket at `/rest/push` for
live updates; that single connection then occupies the container, and every
other request queues behind it — including the one that renders the workflows
page, which simply spins forever. The symptom is distinctive: the browser's own
requests succeed while anything else hangs, and the hung requests appear
**nowhere** in n8n's logs, because Modal never forwards them.

`@modal.concurrent(max_inputs=100)` is the fix, and it has to be on a **class**.
Stacking it directly over `@modal.web_server` on a plain `@app.function` is
silently ignored: the deploy succeeds, both decorators return a
`_PartialFunction`, and only one survives. That is why `N8nServer` is a class
with a `serve` method, and why the deployed URL ends in `n8nserver-serve`
rather than `n8n-server` — the naming changed with the shape.

**The editor is not at `/`.** The root path returns `Cannot GET /`, which is
n8n's Express router rather than an error. n8n's UI lives at `/setup` on a fresh
instance and `/home`, `/signin` and `/workflows` afterwards; `/healthz` reports
liveness. Setting `N8N_EDITOR_BASE_URL` would register the root route, and is
worth doing if webhooks are ever used — the MQTT Trigger does not need it.

## First boot

n8n is deployed at
`https://email-affanardana--pdm-n8n-n8nserver-serve.modal.run`. The hostname is
derived from the app name and the server method, so it changes if either does —
check `modal app list` or the deploy output rather than trusting a bookmark.

0. Deploy the API first (`apps/api/modal_app.py`), since `PDM_API_BASE_URL` has
   to point at something and every write from here is refused without it.
1. Open `/signin` and log in — or `/setup` on a fresh instance, where the editor
   is publicly reachable and an owner account should be created immediately.
2. **Credentials → New → MQTT.** Protocol `mqtts`, host `broker.emqx.io`, port
   `8883`, no username or password. That broker is unauthenticated and public.
3. **Import** `telemetry-ingest.json`.
4. Open **Telemetry topic** and select the MQTT credential from step 2. The
   import cannot carry a credential id, so this reference is the one manual step.
5. **Activate** the workflow. The MQTT Trigger connects on activation, not on
   save — an inactive workflow subscribes to nothing and looks like a broker
   problem.

## Registering machines first

Telemetry for an unregistered machine is refused with a 404, by design: the FK
would reject it anyway, but as a 500 that says nothing about which machine or
what to do. Register the fleet once before the first run, or the n8n execution
list fills with failures that look like real ones:

```powershell
curl.exe -X POST "$env:PDM_API_BASE_URL/api/v1/machines" `
    -H "X-Ingest-Token: $env:PDM_INGEST_TOKEN" `
    -H "Content-Type: application/json" `
    -d '{\"machine_id\":\"M003\",\"name\":\"Demo motor\"}'
```

The call is idempotent — 201 the first time, 200 after — so rerunning it is safe.

## Running the demonstration

`--started-at` matters more than it looks. The simulator advances `recorded_at`
faster than the wall clock, so a run started from *now* produces timestamps
racing ahead of real time, and any query bounded by `clock.now()` — every
`GET /machines/{id}/telemetry` — returns nothing. Start the run in the past so
the timestamps land at or behind now:

```powershell
# The mqtt extra belongs to the `simulator` member, not the virtual root, so
# `uv run --extra mqtt` fails with "Extra `mqtt` is not defined". Sync once with
# --all-packages --extra mqtt and then run plainly.
uv sync --locked --all-packages --extra mqtt

# 240 readings at one per second: four minutes of wall clock, four hours of
# simulated time, with the last reading landing on the present.
uv run python -m simulator realtime `
    --demo --minutes 240 --tick-seconds 1 `
    --started-at (Get-Date).ToUniversalTime().AddHours(-4).ToString("yyyy-MM-ddTHH:mm:ss+00:00") `
    --sink mqtt
```

A plain `uv sync` without `--extra mqtt` removes paho again, and the run then
fails at import rather than at publish.

The machine warms up at reading 60 — roughly a minute in — and the API starts
reporting it as `ready`. Predictions and incidents follow.

`--extra mqtt` installs paho for the run. The MQTT sink is an optional extra
because `ml` depends on `simulator` and `services/inference` depends on `ml`,
so a core dependency here would ship a transport library inside the inference
image. It is declared on the `simulator` member, which is why `uv run --extra
mqtt` does not find it — see the sync command above.

**Rerunning with the same flags stores nothing new**, and that is correct.
`event_id` is derived from the session, which `--demo` pins, so the second run
republishes identical identifiers and the API reports `accepted: 0`. Add
`--session-id <something-new>` to ingest a genuinely new run.

**Bound the stream.** At one reading per second this is roughly 3,600 executions
an hour, and the container is billed for as long as it is warm. Stop the
simulator when the demonstration ends rather than leaving it running.

## Diagnosing a failure

The PRD requires telemetry ingestion failures, inference failures, and workflow
failures to be diagnosable. Each hop has one place to look:

| Symptom | Where to look |
|---|---|
| Nothing arrives at all | n8n's **Telemetry topic** node — is the workflow active, and is the MQTT credential selected? |
| Messages arrive but are dropped | The **Validate** node's output. It drops anything missing a required key, and a drop is silent by design so one stray publish cannot stall the pipeline. |
| `accepted: 0` every time | Working as intended: the same `event_id`s are being redelivered. See `--session-id` above. |
| 404 `machine_not_found` | The machine is not registered. See [Registering machines first](#registering-machines-first). |
| 401 `unauthorized` | `PDM_INGEST_TOKEN` differs from the API's `INGEST_API_TOKEN`. |
| 422 `insufficient_history` | Fewer than 60 readings are stored. Expected during warm-up; the workflow should not be asking yet. |
| 503 `inference_unavailable` | The inference service is cold or down. Check Modal's app logs. |
| Executions failing in the n8n list | Successful executions are not saved, so this list holds failures only — which is what makes it readable. |

`x-request-id` is set to the n8n execution id on both HTTP nodes, and the API's
request middleware honours an inbound one, so a log line in the API can be
matched to the execution that caused it.

## What to verify on first import

The workflow was written against n8n's documented node types but has not been
imported into a running instance. Two things are worth confirming before
trusting it:

- **The MQTT Trigger's output shape.** The Validate node reads the payload from
  `message`, `payload`, or the item's own json, because the field has moved
  between n8n versions. If all three miss, the node outputs nothing and the
  execution ends quietly — check the trigger's output on the first message.
- **`$env` access.** `modal_app.py` sets `N8N_BLOCK_ENV_ACCESS_IN_NODE=false`,
  which the workflow depends on. If the URL resolves to `undefined/...`, that
  setting did not take effect.
