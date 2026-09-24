"""Evaluating a run from its prediction table, with no model in sight.

The exit condition is "a versioned model produces reproducible predictions and
has documented evaluation results". This module is the second half of that made
mechanical: it takes a prediction table and the dataset artifact, and produces
every published figure. A reviewer with no GPU, no checkpoint and no torch runs
one command and gets the recorded numbers back — or gets different ones, which
is the point.

The clock baseline is recomputed here rather than read from Phase 3's report,
and on the **same rows** as the model. The published floor is per trainable row;
a model is scored per window, which is a smaller and differently-balanced subset
of the same lives. Comparing the two across that difference would be comparing
two questions and calling the gap skill.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ml.dataset import baseline
from ml.dataset.artifacts import read_artifact
from ml.dataset.labelling import NEVER_FAILS
from ml.evaluation import curves, events, leadtime
from ml.evaluation import metrics as metrics_module
from ml.evaluation.errors import EvaluationError
from ml.evaluation.report import (
    FLOOR_MODEL_NAME,
    RISK_BAND_EDGES,
    Bundle,
    ModelEvaluation,
    render,
)
from ml.experiment.predictions import Predictions, read_predictions

EVALUATION_FILENAME = "evaluation.json"

#: The alarm rule the lead-time figures are quoted at. Five consecutive windows
#: is a minute of sustained evidence; one would be a coin flip at this
#: prevalence.
DEFAULT_ALARM_THRESHOLD = 0.50
DEFAULT_ALARM_CONSECUTIVE = 5

#: The high-risk slice calibration is additionally measured on. ECE over the
#: whole range is dominated by the 95% of rows sitting near zero.
HIGH_RISK_SHARE = 0.05


@dataclass(frozen=True, slots=True)
class EvaluationBundle:
    """The rendered report and the structured figures behind it."""

    bundle: Bundle
    report: str
    document: dict[str, object]


def evaluate_run(
    *,
    predictions_path: Path,
    artifact_dir: Path,
    output_dir: Path | None = None,
    alarm_threshold: float = DEFAULT_ALARM_THRESHOLD,
    bootstrap_seed: int = 0,
) -> EvaluationBundle:
    """Evaluate one prediction table against the artifact that produced it.

    Raises:
        EvaluationError: if the table is empty or the artifact cannot be read.
    """
    predictions = read_predictions(predictions_path)
    dataset = read_artifact(artifact_dir)
    hazard = baseline.fit(dataset)

    # Recomputed on the model's own rows, which is what makes the comparison
    # mean anything.
    floor = hazard.score(predictions.position.astype(np.int64))
    model = _score(predictions)

    evaluations = (
        _evaluate(
            predictions,
            model,
            model=predictions.model,
            alarm_threshold=alarm_threshold,
            bootstrap_seed=bootstrap_seed,
        ),
        _evaluate(
            predictions,
            floor,
            model=FLOOR_MODEL_NAME,
            alarm_threshold=alarm_threshold,
            bootstrap_seed=bootstrap_seed,
        ),
    )

    bundle = Bundle(
        run_id=predictions.run_id,
        evaluations=evaluations,
        floor_rows={evaluation.split.value: evaluation.metrics for evaluation in evaluations},
    )
    report = render(bundle)
    document = _document(bundle)

    if output_dir is not None:
        _write(output_dir, report, document)

    return EvaluationBundle(bundle=bundle, report=report, document=document)


def _score(predictions: Predictions) -> np.ndarray:
    """Return the scores to evaluate.

    The prior-corrected column, because that is what the product would act on:
    a probability reported to an operator has to be calibrated to the rate
    failures actually occur at, not to the rate the sampler manufactured.

    Ranking metrics are unaffected by the choice — the correction is a constant
    shift in log-odds and cannot reorder anything — but the threshold metrics,
    the reliability table and the risk bands all are.
    """
    calibrated = predictions.calibrated
    if calibrated.shape[0] != len(predictions):
        raise EvaluationError("The calibrated column does not match the table's length.")
    return calibrated


def _evaluate(
    predictions: Predictions,
    scores: np.ndarray,
    *,
    model: str,
    alarm_threshold: float,
    bootstrap_seed: int,
) -> ModelEvaluation:
    """Compute every reported figure for one set of scores."""
    labels = predictions.label
    summary = metrics_module.summarise(
        labels, scores, thresholds=RISK_BAND_EDGES, target_precision=0.80
    )

    threshold = summary["best_f1_threshold"]
    matrix = metrics_module.confusion(labels, scores, threshold=threshold)
    reliability = curves.reliability(labels, scores)

    per_machine = events.per_machine_metric(
        labels,
        scores,
        predictions.machine.astype(np.int32),
        metric=metrics_module.average_precision,
    )
    interval = events.cluster_bootstrap(
        labels,
        scores,
        predictions.machine.astype(np.int32),
        metric=metrics_module.average_precision,
        resamples=400,
        seed=bootstrap_seed,
    )

    life = events.life_scores(
        predictions.life.astype(np.int32),
        _failed_lives(predictions),
        scores,
    )
    life_metrics = {
        "lives": float(life.life.shape[0]),
        "failing": float(life.positives),
        "average_precision": metrics_module.average_precision(life.label, life.score),
        "roc_auc": metrics_module.roc_auc(life.label, life.score),
    }

    rule = leadtime.AlarmRule(threshold=alarm_threshold, consecutive=DEFAULT_ALARM_CONSECUTIVE)
    alarms = leadtime.alarms(
        scores,
        predictions.life.astype(np.int32),
        predictions.position.astype(np.int32),
        _onset_lookup(predictions),
        rule=rule,
    )
    lead = alarms.summary()
    lead["threshold"] = alarm_threshold
    lead["consecutive"] = float(DEFAULT_ALARM_CONSECUTIVE)

    return ModelEvaluation(
        split=predictions.split,
        model=model,
        metrics=summary,
        reliability=reliability,
        confusion=matrix,
        life_metrics=life_metrics,
        per_machine=per_machine,
        bootstrap_low=interval.low,
        bootstrap_high=interval.high,
        lead_time=lead,
    )


def _failed_lives(predictions: Predictions) -> np.ndarray:
    """Return a 0/1 flag per life: did it ever fail?

    A life failed if any of its windows is labelled positive. Derived from the
    windows present rather than from the artifact's life table, so a life with
    no usable windows cannot contribute a label that nothing scores.
    """
    count = int(predictions.life.max()) + 1 if len(predictions) else 0
    failed = np.zeros(count, dtype=np.int64)
    for identifier in np.unique(predictions.life):
        failed[int(identifier)] = int(predictions.label[predictions.life == identifier].max())
    return failed


def _onset_lookup(predictions: Predictions) -> np.ndarray:
    """Return the onset minute of each life, or `NEVER_FAILS`.

    Recovered from the table rather than passed in, so `evaluate` stays a pure
    function of the prediction file and the artifact — no third input that could
    get out of step with the other two.

    The reconstruction is exact because a window is positive when its end sits
    in `[onset - 60, onset - 1]`, so a failing life's positives are contiguous
    and its **last** positive window ends exactly at `onset - 1`. That holds
    only at stride 1, which evaluation always uses; a thinned set would leave
    gaps and make the maximum an underestimate.
    """
    count = int(predictions.life.max()) + 1 if len(predictions) else 0
    onset = np.full(count, NEVER_FAILS, dtype=np.int64)
    for identifier in np.unique(predictions.life):
        positive = (predictions.life == identifier) & (predictions.label == 1)
        if bool(positive.any()):
            onset[int(identifier)] = int(predictions.position[positive].max()) + 1
    return onset


def _document(bundle: Bundle) -> dict[str, object]:
    """Return the machine-readable form of the report."""
    return {
        "run_id": bundle.run_id,
        "evaluations": [
            {
                "split": evaluation.split.value,
                "model": evaluation.model,
                "metrics": evaluation.metrics,
                "life_metrics": evaluation.life_metrics,
                "per_machine": list(evaluation.per_machine),
                "bootstrap": [evaluation.bootstrap_low, evaluation.bootstrap_high],
                "lead_time": evaluation.lead_time,
                "reliability": {
                    "brier": evaluation.reliability.brier_score,
                    "expected_calibration_error": (
                        evaluation.reliability.expected_calibration_error
                    ),
                    "bins": [
                        {
                            "lower": bin_.lower,
                            "upper": bin_.upper,
                            "count": bin_.count,
                            "mean_score": bin_.mean_score,
                            "observed_rate": bin_.observed_rate,
                        }
                        for bin_ in evaluation.reliability.bins
                    ],
                },
            }
            for evaluation in bundle.evaluations
        ],
    }


def _write(output_dir: Path, report: str, document: dict[str, object]) -> None:
    """Write the report and its structured form."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.txt").write_text(report + "\n", encoding="utf-8")
    (output_dir / EVALUATION_FILENAME).write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
