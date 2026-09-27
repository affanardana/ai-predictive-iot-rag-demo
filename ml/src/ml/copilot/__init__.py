"""Measuring what the Copilot does with questions.

The retrieval work has had `ml knowledge evaluate` since Phase 9; the Copilot
had a runbook telling an operator to ask two questions by hand and read the
frames. This package is the missing half.

It is **the online half only.** The planner's tool selection is a pure function
in `api.domain.services.copilot_plan`, and scoring it needs no stack -- but `ml`
is a separate package that does not depend on `api`, and the `ingest` image that
runs this does not contain it either. So the planner is scored by
`apps/api/tests/unit/domain/test_copilot_plan_questions.py`, in the suite where
it lives, against the same question set this reads.
"""
