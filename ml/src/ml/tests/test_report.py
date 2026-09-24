"""The report, and the pipeline that produces it without a model.

The end-to-end test here fabricates a prediction table and runs the whole
evaluation. That is the producer/consumer proof that the reporting half of the
exit condition is complete: a reviewer with no GPU, no checkpoint and no torch
gets the documented figures from files alone.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ml.dataset.artifacts import Dataset
from ml.dataset.splits import Split
from ml.evaluation import metrics
from ml.evaluation.curves import Reliability
from ml.evaluation.errors import EvaluationError
from ml.evaluation.pipeline import evaluate_run
from ml.evaluation.report import (
    FLOOR_MODEL_NAME,
    RISK_BAND_EDGES,
    Bundle,
    ModelEvaluation,
    render,
)
from ml.experiment.index import build_index
from ml.experiment.predictions import Predictions, write_predictions


def _reliability() -> Reliability:
    return Reliability(
        bins=(),
        expected_calibration_error=0.01,
        maximum_calibration_error=0.02,
        brier_score=0.03,
        rows=100,
        positives=5,
    )


def _evaluation(model: str, split: Split = Split.TEST) -> ModelEvaluation:
    return ModelEvaluation(
        split=split,
        model=model,
        metrics={
            "prevalence": 0.05,
            "average_precision": 0.4,
            "roc_auc": 0.99,
            "partial_roc_auc_0.05": 0.6,
            "recall_at_precision_target": 0.2,
            **{f"precision_at_{e:.2f}": 0.5 for e in RISK_BAND_EDGES},
            **{f"recall_at_{e:.2f}": 0.3 for e in RISK_BAND_EDGES},
            **{f"f1_at_{e:.2f}": 0.35 for e in RISK_BAND_EDGES},
            **{f"predicted_positive_rate_at_{e:.2f}": 0.03 for e in RISK_BAND_EDGES},
            "best_f1_threshold": 0.42,
            "best_f1": 0.5,
        },
        reliability=_reliability(),
        confusion=None,
        life_metrics={"lives": 100.0, "failing": 70.0, "average_precision": 0.7, "roc_auc": 0.8},
        per_machine=(0.2, 0.5, 0.8),
        bootstrap_low=0.1,
        bootstrap_high=0.9,
        lead_time={"detection_rate": 0.6, "median_lead_time": 30.0, "threshold": 0.5},
    )


def test_a_model_row_cannot_be_rendered_without_the_floor() -> None:
    """The module's whole point, as a failure rather than a convention.

    The specification requires ROC-AUC; Phase 3's baseline argues it is
    uninformative at this prevalence. Both are honoured by making the number
    unpublishable on its own — so the conflict cannot be resolved by accident.
    """
    bundle = Bundle(run_id="r", evaluations=(_evaluation("lstm"),), floor_rows={})

    with pytest.raises(EvaluationError, match="clock_baseline"):
        render(bundle)


def test_the_floor_and_the_model_render_together() -> None:
    bundle = Bundle(
        run_id="r",
        evaluations=(_evaluation("lstm"), _evaluation(FLOOR_MODEL_NAME)),
        floor_rows={},
    )

    text = render(bundle)

    assert "lstm" in text
    assert FLOOR_MODEL_NAME in text
    assert "Per machine" not in text  # lower-case in the actual output
    assert "per machine" in text
    assert "per life" in text


def test_the_rendered_table_names_the_product_risk_bands() -> None:
    """PRD section 9's edges are the interface everything else uses."""
    bundle = Bundle(
        run_id="r",
        evaluations=(_evaluation("lstm"), _evaluation(FLOOR_MODEL_NAME)),
        floor_rows={},
    )

    text = render(bundle)

    for edge in RISK_BAND_EDGES:
        assert f"{edge:.2f}" in text


def test_the_floor_is_recomputed_on_the_models_own_rows(dataset: Dataset, tmp_path: Path) -> None:
    """Not read from Phase 3's report, which counts trainable rows instead.

    The end-to-end run, with no model: a fabricated prediction table over the
    real window set, evaluated entirely from files.
    """
    index = build_index(dataset, Split.TEST)
    rng = np.random.default_rng(0)
    scores = rng.random(len(index))

    predictions = Predictions(
        split=Split.TEST,
        run_id="fabricated",
        model="lstm",
        ends=index.ends,
        life=index.life,
        machine=index.machine,
        position=index.position,
        label=index.label,
        probability=scores,
        calibrated=scores,
    )
    path = write_predictions(tmp_path / "predictions.parquet", predictions)
    artifact = tmp_path / "artifact"
    _write_artifact(dataset, artifact)

    result = evaluate_run(predictions_path=path, artifact_dir=artifact, output_dir=tmp_path)

    models = {item.model for item in result.bundle.evaluations}
    assert models == {"lstm", FLOOR_MODEL_NAME}
    assert "clock_baseline" in result.report
    assert (tmp_path / "report.txt").exists()
    assert (tmp_path / "evaluation.json").exists()

    for item in result.bundle.evaluations:
        assert item.metrics["rows"] == float(len(index))
        assert set(item.metrics) >= {
            "average_precision",
            "roc_auc",
            "partial_roc_auc_0.05",
            "best_f1",
        }


def test_the_floor_strictly_beats_a_random_model(dataset: Dataset, tmp_path: Path) -> None:
    """A sanity check on the whole stack, not on the model.

    Random scores must score at about prevalence and the clock baseline must
    beat them, or something in the pipeline is inverted. It is the cheapest
    possible end-to-end assertion and it catches an enormous amount.
    """
    index = build_index(dataset, Split.TEST)
    rng = np.random.default_rng(1)
    scores = rng.random(len(index))

    predictions = Predictions(
        split=Split.TEST,
        run_id="fabricated",
        model="random",
        ends=index.ends,
        life=index.life,
        machine=index.machine,
        position=index.position,
        label=index.label,
        probability=scores,
        calibrated=scores,
    )
    path = write_predictions(tmp_path / "p.parquet", predictions)
    artifact = tmp_path / "artifact"
    _write_artifact(dataset, artifact)

    result = evaluate_run(predictions_path=path, artifact_dir=artifact)
    by_model = {item.model: item.metrics for item in result.bundle.evaluations}

    random_ap = by_model["random"]["average_precision"]
    floor_ap = by_model[FLOOR_MODEL_NAME]["average_precision"]

    assert random_ap == pytest.approx(by_model["random"]["prevalence"], abs=0.02)
    assert floor_ap > random_ap


def test_the_reconstruction_of_onset_is_exact(dataset: Dataset) -> None:
    """Lead time depends on recovering each life's onset from the table.

    The positives of a failing life are contiguous and its last one ends exactly
    at `onset - 1`, so the maximum positive position plus one is the onset.
    Checked against the artifact rather than against the implementation.
    """
    from ml.dataset.labelling import NEVER_FAILS
    from ml.evaluation.pipeline import _onset_lookup

    index = build_index(dataset, Split.TEST)
    predictions = Predictions(
        split=Split.TEST,
        run_id="r",
        model="m",
        ends=index.ends,
        life=index.life,
        machine=index.machine,
        position=index.position,
        label=index.label,
        probability=np.zeros(len(index)),
        calibrated=np.zeros(len(index)),
    )

    recovered = _onset_lookup(predictions)

    for life in np.unique(index.life):
        actual = int(dataset.life_onset[life])
        if actual == NEVER_FAILS:
            assert int(recovered[life]) == NEVER_FAILS
        else:
            assert int(recovered[life]) == actual


def _write_artifact(dataset: Dataset, directory: Path) -> None:
    """Write the artifact files the evaluator reads."""
    from ml.dataset.artifacts import write_artifact

    write_artifact(directory, dataset)


def test_the_confusion_at_the_chosen_threshold_matches_the_summary(
    dataset: Dataset, tmp_path: Path
) -> None:
    """The report and the matrix must not disagree about the operating point."""
    index = build_index(dataset, Split.TEST)
    rng = np.random.default_rng(2)
    scores = rng.random(len(index))

    predictions = Predictions(
        split=Split.TEST,
        run_id="r",
        model="m",
        ends=index.ends,
        life=index.life,
        machine=index.machine,
        position=index.position,
        label=index.label,
        probability=scores,
        calibrated=scores,
    )
    path = write_predictions(tmp_path / "p.parquet", predictions)
    artifact = tmp_path / "artifact"
    _write_artifact(dataset, artifact)

    result = evaluate_run(predictions_path=path, artifact_dir=artifact)
    for item in result.bundle.evaluations:
        assert item.confusion is not None
        assert item.confusion.f1 == pytest.approx(
            metrics.confusion(
                predictions.label,
                predictions.calibrated,
                threshold=item.metrics["best_f1_threshold"],
            ).f1
        )
