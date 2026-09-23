# Machine learning

**Status: not implemented — Phases 3 and 4.**

Dataset engineering and the predictive model.

Planned scope:

**Phase 3 — Dataset engineering** (MASTERPLAN §6)

Generate a reproducible synthetic dataset, then preprocess it:

```text
target   ~100 machines x 30 days x 1 reading/minute  (~4.32M records)
```

Pipeline: preprocessing → feature normalization → sequence generation →
failure-window labeling → train/validation/test split → **leakage prevention**.

**Phase 4 — LSTM**

Answer: *will this machine fail within the next 60 minutes?*

```text
input    the previous 60 minutes of temperature, vibration, rpm,
         current, load, voltage
output   failure_probability
```

Evaluate with precision, recall, F1, ROC-AUC, PR-AUC, calibration, confusion
matrix, and lead time. Track runs and artifacts with MLflow.

## The constraint that shapes this work

Ground truth is used to build **labels** and to **evaluate** — never as an input
feature. A dataset that leaks a future-derived column into the feature set will
score well and be worthless, which is why leakage prevention is an explicit
pipeline step rather than a review checklist item.

This directory is intentionally empty until Phase 3. It is not yet a `uv`
workspace member — it will be added to `members` in the root `pyproject.toml`
when it gains real dependencies.
