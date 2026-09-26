# Machine learning

Builds a labelled, leakage-free dataset from the simulator's output
(`MASTERPLAN.md` §6) and trains an LSTM on it.

```bash
# ~4.8M records: 100 machines over 30 days, plus 12 held out
uv run python -m ml dataset generate --out data/dataset

# label, split, normalise, and write the training artifact
uv run python -m ml dataset export --dataset data/dataset --out data/training

# what came out, and how hard it is
uv run python -m ml dataset report --dataset data/dataset --artifact data/training

# train, score, evaluate, prove it reproduces
uv run python -m ml model train    --config ml/configs/lstm.json --out data/models/run-1
uv run python -m ml model predict  --run data/models/run-1 --split test
uv run python -m ml model evaluate --predictions data/models/run-1/predictions_test.parquet
uv run python -m ml model verify   --run data/models/run-1

# the maintenance corpus (Phase 9). Needs the `knowledge` extra:
#   uv sync --locked --all-packages --extra knowledge
uv run python -m ml knowledge extract --document bearing-inspection-sop
uv run python -m ml knowledge ingest  --api-url http://localhost:8100 --activate
uv run python -m ml knowledge evaluate --api-url http://localhost:8100
```

`knowledge extract` prints what a document parses into — every line with its
page and font size — because the parse is the fragile half and the size is the
only structure these PDFs carry. `ingest` sends the corpus to the API, which
owns chunking, versioning and activation; re-running it over unchanged content
writes nothing. `evaluate` measures retrieval against `knowledge/eval/`
`questions.json` twice, with and without reranking, and prints both columns.

`generate` is CPU-bound and takes a few minutes; everything else is faster. The
steps are separate because labelling, splitting and normalisation are decisions
about *training* that can be revisited without regenerating the telemetry
underneath them.

## Results

Trained on Colab, one run, one seed. Every figure is measured on the same rows
as the clock baseline.

```text
test                    avg precision   roc auc   partial auc   brier   recall@p80
  lstm                          0.848     0.992         0.918   0.017        72.9%
  clock_baseline                0.160     0.836         0.547   0.043         0.0%

test_shift              avg precision   roc auc   partial auc   brier   recall@p80
  lstm                          0.691     0.985         0.852   0.030        28.5%
  clock_baseline                0.030     0.328         0.487   0.048         0.0%
  prevalence                    4.38%
```

**`test_shift` is the row that answers the question the phase was built
around.** Those machines run 6- and 48-hour lives, outside the 16-32 hour range
everything else uses. A model that had learned the training regime's timing
would collapse there; the clock baseline does, scoring 0.030 against a 4.38%
prevalence — worse than chance. The LSTM scores 0.691, twenty-three times the
floor, and holds across all twelve machines (min 0.651, median 0.870).

Degrading from 0.848 to 0.691 across an eightfold change in life length is the
signature of a model reading signals.

At the product's own bands on `test`: at 0.80 (CRITICAL) precision 0.820 and
recall 0.674, flagging 4.04% of windows against a 4.92% base rate. Lead time
over detected failures: median 62 minutes, IQR 45-82, 95.7% detection, zero
false alarms per machine-day.

Two things the report shows that are worth reading rather than skimming past:

- **The per-machine average precisions (0.956-0.996) all exceed the pooled
  0.848.** Within a machine the ranking is near-perfect; pooled across fifteen
  it is not. The scores are therefore not comparable *between* machines, and
  since PRD §9 applies one global set of bands, that is a real operational
  limitation rather than a curiosity. Per-machine calibration is the obvious
  next step.
- **It is one seed.** The floor is a deterministic number, so strictly a 5.3x
  margin over a single run cannot be separated from a lucky initialisation. At
  that distance luck is unlikely, but a second seed would settle it.

## Training is an extra, not a dependency

```bash
uv sync --locked --all-packages --extra train
```

torch is optional so that `uv sync --all-packages` does not install it into the
API's environment or into every CI job, and so that `ml dataset report` keeps
working where torch is absent. Two import-linter contracts enforce that: the
evaluation and experiment modules cannot import torch, and the dataset package
cannot import the model.

The practical consequence is that **the evaluation runs without a GPU**. It
takes a prediction table and the artifact and produces every published figure,
so a reviewer reproduces the numbers by running a command over files.

## How the numbers are reported

The clock baseline - a model given only minutes since the life began, no signals
- is computed on the same rows as the LSTM, and `render` refuses to print a
model's figures without it. `MASTERPLAN.md` requires ROC-AUC; at 4.6% prevalence
it reads near 0.99 for almost anything, so the partial AUC over the low
false-positive region is reported beside it.

Three other things follow from the near-duplicate problem (sixty windows before
a failure share 59 timesteps): life-level metrics use a fixed subsample so a
48-hour life cannot outvote a 6-hour one, the per-machine spread is shown
alongside the pooled figure, and early stopping reads a group-averaged metric.

Lead time is quoted as a distribution over detected events, with a detection
rate, a false-alarm rate per machine-day, and a count of alarms raised and then
abandoned — an alarm that fires once and vanishes is not a warning.

## Why a machine's timeline is made of *lives*

A simulator session runs one machine through one scenario, and its damage always
reaches full at the session's end. Thirty days as a single session per machine
would therefore yield one failure event per machine — a positive rate near
0.07%, which is realistic and untrainable.

So each machine's thirty days is composed from 20–30 sequential **lives**: a
degradation life ends when the machine enters failure, and the next begins
immediately after, as a repair. That raises the positive rate to around 3% and
the event count to 20-odd per machine, without changing the simulator at all.

## Why the labelling is not "the end of the life"

A failure is the first instant a life reports `failure_imminent` — the crossing
of `health_index` below the simulator's own threshold. Phase 2 recorded that
intention beside the constant.

Calling the end of a life the failure is the tempting alternative and it is
wrong twice over. It caps lead time at sixty minutes by construction, so the
product's headline metric would be measuring the label definition rather than
the model; and it scores an early warning on a visibly failing machine as a
false positive, which is the opposite of what the product is for.

Rows at and after onset are neither class. They are carried with
`is_trainable` false and reported as a share of the dataset.

## Why the artifact is flat

One row per `(life, timestep)`, not one row per window. Consecutive windows
overlap by 59 of their 60 timesteps, so a pre-expanded tensor stores the same
reading sixty times — 3.5 GB against 58 MB. `ml.dataset.windows` slices
sequences out of the flat arrays instead, which is index arithmetic and costs
nothing. On Colab's free tier that difference is the whole memory budget.

## What leakage prevention actually means here

`MASTERPLAN.md` §6 lists it as a deliverable, and prose is the wrong medium: a
leak produces a model that scores *better*, so nothing downstream complains.
`ml/src/ml/tests/test_leakage.py` states each route as a test that fails.

| Route | Prevention |
|---|---|
| A machine in two splits | Split by machine, never by time — a machine's nominal point and susceptibility are stable across its lives, so it would be memorised rather than learned |
| Normalisation statistics | Computed from training rows only, asserted against an independent recomputation *and* against the global statistics, which must differ |
| Positional labelling | The two channel files are compared key column by key column before any label is attached |
| `event_id` | Never a feature: its trailing digits are a tick's position within its life, which is a proxy for time-to-failure |
| `session_id` | Never a feature: every degradation life produces a positive and no `NORMAL` life ever does |
| `recorded_at` | Never a feature: any function of it reconstructs elapsed time |
| `elapsed_minutes` | Exists for the clock baseline only, and a test asserts it is not a feature |
| Windows | Never cross a life boundary, and never include a post-onset row |

## The number the report is really about

A model given only minutes since the life began — no signals at all — is fitted
on training lives and scored on validation, test, and the duration-shift split.
The simulator's degradation is a deterministic function of elapsed time, so a
model can score well by reading a clock. Phase 4's results are only meaningful
as a margin above that floor, and `ml dataset report` prints it beside the
prevalence it has to beat.

The duration-shift split exists for the same reason: twelve machines whose lives
are six or forty-eight hours, held out entirely. A model that learned the main
regime's timing collapses there; one that learned the signals degrades
gracefully.

## Layout

```text
src/ml/dataset/      Phase 3 - building the data
  features.py        the signal whitelist
  lives.py           composing a fleet timeline out of lives
  plan.py            the generation plan: four values, re-derived
  generation.py      running the fleet, writing telemetry and truth
  labelling.py       what a failure is, and the signed offset encoding
  splits.py          by-machine, stratified, plus the duration shift
  windows.py         sixty-step sequences that never cross a life
  artifacts.py       verification, normalisation, the flat artifact
  baseline.py        the clock model
  report.py          what came out
  errors.py          the error type, kept free of numpy

src/ml/evaluation/   numpy only, no torch
  metrics.py         precision, recall, F1, PR-AUC, ROC-AUC, partial AUC
  curves.py          reliability, Brier, risk bands
  prior.py           the logit offset that re-calibrates a rebalanced model
  events.py          per-life and per-machine grouping, cluster bootstrap
  leadtime.py        debounced alarm detection
  report.py          rendering, with the floor structurally attached
  pipeline.py        evaluate_run - reads a prediction table, writes a report

src/ml/experiment/   numpy + pyarrow, no torch
  config.py          hyperparameters, canonically serialised and hashed
  index.py           which windows exist, where they sit, what they are labelled
  sampling.py        the life-aware epoch draw
  predictions.py     the prediction table - the seam with evaluation
  manifest.py        dataset fingerprint, git revision, environment

src/ml/knowledge/    the maintenance corpus, no torch
  extraction.py      PDF to lines: page, text, and the size it was set at
  corpus.py          the manifest, held against the directory it describes
  ingest.py          parsing the corpus and posting it to the API
  evaluation.py      recall, MRR, nDCG, abstention - and the measured run
  errors.py          the error type, kept free of pypdf

src/ml/model/        torch, the optional extra
  sequences.py       a Dataset that slices windows out of the flat artifact
  network.py         the LSTM
  checkpoint.py      save and load under torch.load's safe default
  training.py        the loop, early stopping, resume
  pipeline.py        train, predict, verify
```

`dataset/features`, `labelling`, `lives`, `plan`, `splits`, `windows`, `errors`
are dependency-free, and an import-linter contract enforces it. Another keeps
`ml` free of `api`, so FastAPI and SQLAlchemy cannot follow a convenience import
into a training environment. A third keeps torch out of `evaluation` and
`experiment`, and a fourth keeps the model out of `dataset`.
