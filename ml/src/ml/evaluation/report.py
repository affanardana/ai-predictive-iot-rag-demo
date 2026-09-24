"""Rendering the results, with the floor attached whether or not anyone asks.

The specification requires ROC-AUC; `ml.dataset.baseline` argues, correctly,
that at 4.6% prevalence it reads near 0.99 for almost anything. Both positions
are held, and this is where they are reconciled - not by choosing one, but by
making it impossible to publish the model's numbers without the clock baseline's
beside them, on the same rows, at the same thresholds.

That is a structural rule rather than an editorial one. `render` raises if it is
handed a model row with no floor row, so "the LSTM scored 0.994" cannot be
printed on its own even by someone who wants to. What gets printed is "the LSTM
scored 0.994 and so did a clock", which is the actual finding.

The same instinct runs through the rest of the table: precision is printed
beside predicted-positive rate, because a model that flags everything has
perfect recall; the per-machine distribution is printed beside the pooled
figure, because a mean over fifteen machines hides whether two of them carried
it; and the F1-optimal column is printed beside the product's own risk-band
edges, because those are the thresholds the product actually uses.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ml.dataset.splits import Split
from ml.evaluation.curves import Reliability
from ml.evaluation.errors import EvaluationError
from ml.evaluation.metrics import Confusion

#: The name the floor's row must carry for `render` to accept a bundle.
FLOOR_MODEL_NAME = "clock_baseline"

#: PRD section 9's band edges. Reported because they are the interface between
#: this model and everything built on it.
RISK_BAND_EDGES: tuple[float, ...] = (0.30, 0.60, 0.80)


@dataclass(frozen=True, slots=True)
class ModelEvaluation:
    """Every figure reported for one model on one split."""

    split: Split
    model: str
    metrics: dict[str, float]
    reliability: Reliability
    confusion: Confusion | None
    life_metrics: dict[str, float]
    per_machine: tuple[float, ...]
    bootstrap_low: float
    bootstrap_high: float
    lead_time: dict[str, float] | None


@dataclass(frozen=True, slots=True)
class Bundle:
    """A set of evaluations, and the provenance needed to attribute them."""

    run_id: str
    evaluations: tuple[ModelEvaluation, ...]
    floor_rows: dict[str, dict[str, float]]


def render(bundle: Bundle) -> str:
    """Format a bundle for a terminal.

    Raises:
        EvaluationError: if any split carries a model row without a floor row.
            This is the module's whole point, so it fails rather than warns.
    """
    splits = {evaluation.split for evaluation in bundle.evaluations}
    for split in splits:
        models = {
            evaluation.model for evaluation in bundle.evaluations if evaluation.split is split
        }
        if FLOOR_MODEL_NAME not in models:
            raise EvaluationError(
                f"The {split.value} split has model rows {sorted(models)} and no "
                f"'{FLOOR_MODEL_NAME}' row. A model's figures are not reportable "
                "without the clock baseline measured on the same rows, and this "
                "renderer will not print them."
            )

    lines = [
        f"run {bundle.run_id}",
        "=" * 78,
        "",
        "  Every figure below is measured on the same rows, for every model in",
        "  the table. The clock baseline sees only minutes since the life began",
        "  and no signals at all; it is the floor, and a model that does not",
        "  clear it by a wide margin has measured a clock.",
        "",
    ]

    for split in sorted(splits, key=lambda value: value.value):
        rows = [item for item in bundle.evaluations if item.split is split]
        lines += _render_split(split, rows)

    return "\n".join(lines)


def _render_split(split: Split, rows: Sequence[ModelEvaluation]) -> list[str]:
    """Render one split's table."""
    lines = [
        split.value,
        "-" * 78,
        f"  {'model':<18}{'prevalence':>12}{'avg precision':>15}{'roc auc':>10}"
        f"{'partial auc':>13}{'brier':>9}{'recall@p80':>12}",
    ]
    for item in rows:
        metrics = item.metrics
        lines.append(
            f"  {item.model:<18}{metrics.get('prevalence', 0.0):>11.2%}"
            f"{metrics.get('average_precision', 0.0):>15.3f}"
            f"{metrics.get('roc_auc', 0.0):>10.3f}"
            f"{metrics.get('partial_roc_auc_0.05', 0.0):>13.3f}"
            f"{item.reliability.brier_score:>9.3f}"
            f"{metrics.get('recall_at_precision_target', 0.0):>11.2%}"
        )

    lines += [
        "",
        "  'partial auc' is the ROC area over false-positive rates at or below",
        "  5%, normalised so an uninformative ranking scores 0.5. It is reported",
        "  beside 'roc auc' because at this prevalence the full AUC is close to",
        "  1 for almost any ranking, including the clock's.",
        "",
    ]

    for item in rows:
        lines += _render_thresholds(item)
        lines += _render_machines(item)
        lines += _render_life(item)
        if item.lead_time is not None:
            lines += _render_lead_time(item)
    return lines


def _render_thresholds(item: ModelEvaluation) -> list[str]:
    """Render the product's own operating points."""
    lines = [
        f"  {item.model} at the product's risk bands (PRD section 9)",
        f"    {'band edge':<12}{'precision':>11}{'recall':>9}{'f1':>8}{'flagged':>10}",
    ]
    for edge in RISK_BAND_EDGES:
        key = f"{edge:.2f}"
        lines.append(
            f"    {edge:<12.2f}"
            f"{item.metrics.get(f'precision_at_{key}', 0.0):>11.3f}"
            f"{item.metrics.get(f'recall_at_{key}', 0.0):>9.3f}"
            f"{item.metrics.get(f'f1_at_{key}', 0.0):>8.3f}"
            f"{item.metrics.get(f'predicted_positive_rate_at_{key}', 0.0):>9.2%}"
        )
    lines.append(
        f"    {'best f1':<12}"
        f"{item.metrics.get('best_f1_threshold', 0.0):>11.3f}"
        f"{'':>9}{item.metrics.get('best_f1', 0.0):>8.3f}"
    )
    lines.append("")
    return lines


def _render_machines(item: ModelEvaluation) -> list[str]:
    """Render the per-machine spread, which a pooled mean hides."""
    values = sorted(item.per_machine)
    if not values:
        return []
    quantiles = _five_number(values)
    return [
        f"  {item.model} per machine ({len(values)} machines)",
        f"    min {quantiles[0]:.3f}   q1 {quantiles[1]:.3f}   "
        f"median {quantiles[2]:.3f}   q3 {quantiles[3]:.3f}   max {quantiles[4]:.3f}",
        f"    cluster bootstrap 95%: {item.bootstrap_low:.3f} to "
        f"{item.bootstrap_high:.3f} - reported second, and weak: with this few",
        "    clusters the interval is lumpy, which is why the spread above leads.",
        "",
    ]


def _render_life(item: ModelEvaluation) -> list[str]:
    """Render the life-level view, where the near-duplicates stop mattering."""
    if not item.life_metrics:
        return []
    return [
        f"  {item.model} per life (fixed 60-window subsample, length-neutral)",
        f"    lives {item.life_metrics.get('lives', 0.0):.0f}   "
        f"failing {item.life_metrics.get('failing', 0.0):.0f}   "
        f"average precision {item.life_metrics.get('average_precision', 0.0):.3f}   "
        f"roc auc {item.life_metrics.get('roc_auc', 0.0):.3f}",
        "",
    ]


def _render_lead_time(item: ModelEvaluation) -> list[str]:
    """Render the four figures that make a lead time reportable."""
    lead = item.lead_time or {}
    threshold = lead.get("threshold", 0.0)
    consecutive = lead.get("consecutive", 5.0)
    return [
        f"  {item.model} lead time "
        f"(alarm: p >= {threshold:.2f} sustained {consecutive:.0f} windows)",
        f"    detected {lead.get('detection_rate', 0.0):.1%} of failing lives"
        f"   ({lead.get('detected', 0.0):.0f} of {lead.get('events', 0.0):.0f})",
        f"    lead time: median {lead.get('median_lead_time', 0.0):.0f} min"
        f"   iqr {lead.get('lead_time_q1', 0.0):.0f} to {lead.get('lead_time_q3', 0.0):.0f}"
        f"   min {lead.get('lead_time_min', 0.0):.0f}",
        f"    false alarms {lead.get('false_alarms_per_machine_day', 0.0):.2f} per machine-day"
        f"   abandoned crossings {lead.get('abandoned_crossings', 0.0):.0f}",
        "",
    ]


def _five_number(values: Sequence[float]) -> tuple[float, float, float, float, float]:
    """Return min, q1, median, q3, max of a sorted sequence."""
    import numpy as np

    low, first, middle, third, high = np.quantile(
        np.asarray(values, dtype=float), [0.0, 0.25, 0.5, 0.75, 1.0]
    )
    return float(low), float(first), float(middle), float(third), float(high)
