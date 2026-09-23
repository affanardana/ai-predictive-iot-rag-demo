# Simulator

Generates temporally coherent motor telemetry and the hidden ground-truth state
behind it. It is the source of the data the rest of the platform runs on.

```bash
# A historical dataset: two separate files, telemetry and ground truth
uv run python -m simulator dataset --machines 3 --minutes 60 --out data/raw

# Watch a machine degrade, as PRD section 12 specifies
uv run python -m simulator realtime --demo

# What can be simulated
uv run python -m simulator scenarios
```

## Why the sensors move together

PRD §11 requires a degradation mode to produce a *correlated* signature —
bearing degradation gives vibration ↑, temperature ↑, current ↑, rpm ↓ — "rather
than independently randomizing each sensor". `MASTERPLAN.md` §3.6 names the
independent version as a prohibited shortcut, because it would make the whole
project a demonstration trick.

So one physical cause drives several signals. A worn bearing raises friction,
which shows up first as vibration; the friction heats the machine; the drag
draws more current; and the shaft slows. That coupling is data in
`domain/physics.py`, and a test asserts it.

## Why there are two output files

The simulator knows both what the sensors report and what is actually happening
inside the machine. The second is ground truth: it labels training data and
evaluates the model, and `MASTERPLAN.md` §3.2 forbids it from ever being a model
input.

That is enforced structurally rather than by convention. Each tick is an
observation and a ground-truth record in separate objects; the sinks are two
separate interfaces; and dataset mode writes **two separate files**. A Phase 3
feature builder reading `telemetry.parquet` cannot reach a label that only
exists in `ground_truth.parquet`.

The scenario label lives on the hidden side only. It does not become an
incident's `incident_type` — that question remains open, because publishing the
simulator's label would be exactly the leak this design prevents.

## Why the output is reproducible

`MASTERPLAN.md` §3.5 requires datasets, experiments, and demonstrations to be
regenerable. Everything random derives from a seed through
`domain/seeding.py`, which uses SHA-256 rather than Python's `hash()` — string
hashing is randomised per process, so `hash()` would give different data on
every run, silently.

A machine's condition is a pure function of how far into a run it is. Nothing
accumulates between ticks, so `tick(5)` gives the same answer whether or not
ticks 0–4 were ever requested. That is what makes the demonstration repeatable
and what would let a realtime run resume mid-flight.

## Layout

```
src/simulator/
├── domain/          the simulation model. pure, deterministic, no I/O
│   ├── machine.py   nominal operating points, jittered per machine
│   ├── scenario.py  what each scenario drives
│   ├── physics.py   degradation curves and the signal coupling
│   ├── engine.py    advances one machine, one tick at a time
│   ├── session.py   the description of a run
│   ├── ports.py     the two sink interfaces
│   └── state.py     observations and ground truth, kept apart
├── application/     run orchestration
├── infrastructure/  sinks: console, JSON Lines, Parquet
├── cli.py           the command line
└── tests/           inside the package, to avoid a module-name collision
```

The layering is enforced by the `import-linter` contracts in the root
`pyproject.toml`. The domain may not import a sink, a dataframe library, or a
plotting library; were it to, the model would depend on where its output happens
to go.

## Not here yet

No MQTT, EMQX, or n8n — the transport is Phase 6, and realtime mode writes to a
pluggable sink until then. No database writes at all: telemetry reaches
PostgreSQL through the real pipeline rather than through a shortcut that would
later need removing. No large-scale dataset generation, which is Phase 3.

The domain is scalar and pure, which is what makes its tests fast and its
behaviour exact. Phase 3's ~4.3M-row target may want a vectorised implementation
behind the same interface; measuring then is better than optimising now.
