# ADR 0009 — The Copilot runs a local model, selects its own tools, and is public and bounded

**Status:** Accepted (Phase 10)
**Date:** 2026-09-27

## Context

Phase 10 adds the Maintenance Copilot: `POST /api/v1/copilot/chat` takes a
question in plain language, reads whatever the question needs, and answers from
what it read. `PRD.md` section 20.6 asks for it, AC-007 names the question it
must answer, and sections 16 and 18 require the answer to separate what was
measured from what was predicted, what the documents say, and what was reasoned
from those — with the sources shown.

Three decisions had to be made before any of that could be built, and each of
them is the kind that is expensive to reverse. They are recorded together
because they constrain one another: the model choice sets the memory ceiling,
the memory ceiling rules out an agent loop, and both shape what the bound on the
endpoint has to be.

`ADR 0002` already justified making the whole API async by *"the Copilot calls
an LLM, RAG…"*, and `composition/container.py` named *"an LLM provider"* as a
seam that plugs in there. This is that promise coming due.

## Decision 1 — The model is local, and it is a 1.5B quantised GGUF

The owner's constraint was that a question must not cost money and must not
leave the host. So the language model runs in the inference container, beside
the retrieval models, on the same single core.

`Qwen2.5-1.5B-Instruct`, Q4_K_M, 940 MB on disk, loaded by `llama.cpp` through
`llama-cpp-python`. Measured on the deployment host before anything downstream
was designed:

| | Measured |
|---|---|
| Load | 20.6 s |
| Prefill, a real prompt | 8.9 s |
| Decode | 7.8 tokens/second |
| A five-sentence answer | ~21.8 s |
| Peak resident memory | **1724 MiB** |

The size is set by memory, not by quality. The host is 3.9 GB with **no swap**
and 506 MiB already committed to the LSTM and the two sentence encoders. A 3B
model is roughly 2 GB resident, which puts the container near 2.6 GiB against
2.4 GB available — an OOM kill with no swap to absorb it. That is a
demonstration that dies rather than one that is slow, and it is the wrong
failure for a project whose subject is reliability.

Two things make 1.5B fit rather than merely squeeze. The weights are **mmap'd**,
so the kernel can evict those pages under pressure instead of killing the
container — the closest thing to swap this host has, and it comes free with the
GGUF format. And `llama-cpp-python` publishes a **prebuilt CPU wheel**, so the
model costs its file size rather than a torch runtime it would never use.

Rejected: **a hosted API.** It would answer far better, and it would put a
per-question cost and an outbound dependency in a system whose entire claim is
that its answers are grounded. A copilot that stops working when someone else's
service is down cannot demonstrate reliability on a box that is meant to be
self-contained.

## Decision 2 — The system selects the tools; the model writes the prose

`MASTERPLAN.md` section 6 says the Copilot *"must dynamically determine which
information sources are required"*.

**It does not mean the model decides.** A CPU reliability benchmark of sub-2B
models reports roughly 50% tool-selection accuracy over a large tool set, rising
to about 60% with a hybrid tool presentation
([Beyond Fluent Generation](https://browse-export.arxiv.org/pdf/2609.07370)).
One of seven tools chosen correctly half the time is not a design; it is a
coin-flip with a fallback router bolted on, and the fallback router is the real
implementation.

So tool selection is `plan_question()`, a pure function in the domain, driven by
the question's own words and the fleet's identifiers. It returns an ordered tool
list, a machine id, a window and a search query. The determination is still
dynamic and still per-question — what changed is that it is a **tested table**
rather than a model that cannot be debugged. `MASTERPLAN.md`'s own worked
examples are its test table, including the negative one: *"What happened to M003
during the last five hours?"* must not search the documentation.

The model is given no tools and no ability to fetch anything. It receives
labelled evidence blocks and returns prose, which makes the blast radius of a
bad generation a bad paragraph rather than a bad read.

## Decision 3 — Derived statistics are `OBSERVED` at the endpoints and `INFERRED` in the fit

`CHANGELOG.md` recorded this as an open question for Phase 10. The answer is
that it is two questions, and `get_machine_trend()` returns both claims with
different kinds:

- *"Vibration was 1.42 mm/s then and 2.31 mm/s now"* — two readings and a
  subtraction. `PRD.md` section 15 lists exactly this under Observed, and a
  reader can check it against the charts.
- *"It fitted a rise of 0.89 mm/s over the window"* — a derivation whose answer
  depends on the method chosen. Section 16 forbids presenting that as directly
  observed.

A trend states its resolution (`"mean of 5-minute buckets"`) and its sample
count for the same reason `TelemetrySeries` distinguishes a measurement from an
aggregate: an aggregate that hid what it aggregated would undo that.

## Decision 4 — The route is public, and bounded instead

ADR 0008 is the precedent, and the reasoning carries over exactly: the dashboard
calls this from a browser, and a token in a browser bundle is a boundary in
appearance only.

| Bound | Where | What it prevents |
|---|---|---|
| One answer at a time | `AskCopilot.claim()`, from settings | A second question halving the speed of the first on one core |
| Question ≤ 500 characters | `AskRequest` | An unbounded prefill, which is half the wait |
| `max_tokens` 180, context 2048 | `AskCopilot`, inference `Settings` | An essay, and a KV cache the box cannot hold |
| ≤ 3 passages, ≤ 7 tool calls | `AskCopilot`, the planner | Unbounded fan-out into a prompt |
| Container memory 2560m | `compose.yaml` | The OOM bound, asserted by a test against the measurement |
| Request timeout 180 s | `HttpChat` | A hung model becoming a browser hanging forever |

The claim happens **before** the streaming response begins, which is why it is a
method the route calls rather than something the use case does for itself: a 409
that arrives as a frame inside a 200 is no longer a status code a client can act
on.

## Consequences

**Answer quality is the price, and it is visible.** A 1.5B model writes plain
prose and reasons poorly. This is defensible for a demonstration whose claim is
*grounding*, not eloquence — and the design leans on it deliberately: the
evidence blocks are attached by the system, so a reader can always see what the
model was given, and the page shows the tool activity that produced it. The
`model_id` travels in every answer so the two can be judged separately.

**The grounding check catches invented numbers, not invented reasoning.** Every
number in the answer must appear in the evidence it was given; one that does not
forces `FALLBACK`, which serves the deterministic rendering instead and reports
the offending values. A sentence that misreads true numbers is not caught. The
mitigation is structural rather than clever: PRD section 19's refusal is
enforced by *not calling the model*, so the one rule a small model would happily
break is a rule it is never in a position to break.

**Latency is the demonstration's real constraint.** Twenty seconds is acceptable
only because the answer streams — tool activity first, then tokens — and a
silent wait reads as a broken page. The knobs, in order, if it must get faster:
fewer passages, a lower token cap, a smaller model.

**Streaming a `POST` has no `EventSource`**, so the browser parses frames by
hand. The frame shape is therefore mirrored by hand on both sides, pinned by
`test_copilot_api.py` on the server and `stream.test.ts` on the client — the
same arrangement `ADR 0006` established for the event stream, for the same
reason: a `text/event-stream` body cannot be described in OpenAPI.

**The model loads at startup, adding ~21 seconds to the container's boot.** The
service's healthcheck `start_period` was raised to cover it, because a container
reported unhealthy while it is still loading is a false alarm during every
deploy. The alternative — loading on the first question — was rejected: it would
move the load into a reader's first request, where it is twenty seconds of
silence on top of twenty seconds of generation.

**A missing model file is a startup failure, not a degraded mode.** The
inference service runs without a chat model by default — it is the hungriest and
least essential of the three — but a path that is configured and not there stops
the container. On a deployment that is the correct direction: the alternative is
answering "the Copilot is unavailable" for a reason nobody can see.

**Phase 11 owns the guard.** The route joins the other unguarded ones, and the
constraint carries forward unchanged: `EventSource` cannot set headers, so
whatever authentication Phase 11 chooses has to work for a cookie or a query
parameter too.
