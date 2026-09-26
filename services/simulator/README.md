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

## Running as a service

```bash
uv run --extra control --extra mqtt python -m simulator.control
```

Serves the HTTP surface the API starts and stops runs through. This is how the
dashboard drives a simulation, and the reason `PRD.md` AC-010 is now satisfied —
the demonstration no longer needs a simulator on a developer's machine.

It is a **composition point**, a peer of `cli.py` rather than beneath it: both
wire the same application and infrastructure, and neither is the other's
dependency. That is enforced by the layers contract rather than by convention,
which is why the broker settings they share live a rung below both in
`infrastructure/broker.py`.

`control` is an extra, and for the reason the `mqtt` extra's comment already
gives: `ml` depends on `simulator` and `services/inference` depends on
`ml[train]`, so FastAPI as a core dependency would ride into the training and
inference images to serve a surface neither has.

### A run is stoppable and resumable

`stream_session` takes `should_stop` and `start_index`. Neither is a convenience:
`PRD.md` §20.5 requires a stop control, and there was no way to end a run other
than `KeyboardInterrupt` reaching the process; and resumption is what makes a
container restart survivable mid-demonstration.

Resuming is safe because of the property this package was built around — see
`MachineSimulator`'s docstring — and because every re-emitted tick carries an
`event_id` the API already holds.

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

**No database writes at all**, and that is a design choice rather than a gap:
telemetry reaches PostgreSQL through the real pipeline — MQTT, then the
orchestrator, then the API — rather than through a shortcut that would later
need removing. The simulator does not know the API's address for writing; it
only reports run *state*, which is a different thing on a different route.

No large-scale dataset generation beyond what Phase 3 produced.

The domain is scalar and pure, which is what makes its tests fast and its
behaviour exact. Phase 3's ~4.3M-row target may want a vectorised implementation
behind the same interface; measuring then is better than optimising now.
