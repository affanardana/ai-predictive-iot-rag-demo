"""Describing the dataset, so its exit condition can be checked by eye.

`MASTERPLAN.md` §6 asks for a "reproducible training-ready dataset". Reproducible
is provable by tests; *training-ready* is a claim about numbers, and the only
honest way to make it is to print them. So this prints them:

* how many rows, lives and failure events there are, overall and per split;
* the positive rate, and how much of the data the post-onset tail accounts for;
* what the clock baseline scores, beside the prevalence it has to beat.

The last one is the report's most important line. It is deliberately not
flattering, and if it is ever high the honest response is to say so here rather
than to discover it in Phase 4 after the architecture has been chosen.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ml.dataset import baseline
from ml.dataset.artifacts import Dataset
from ml.dataset.features import FEATURE_COLUMNS
from ml.dataset.labelling import HORIZON_MINUTES, NEVER_FAILS
from ml.dataset.lives import MachinePlan
from ml.dataset.splits import Split

Int64Array = npt.NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class SplitSummary:
    """One split's share of the dataset."""

    split: Split
    machines: int
    lives: int
    rows: int
    trainable_rows: int
    positives: int
    failure_events: int

    @property
    def prevalence(self) -> float:
        """Return the positive rate over trainable rows."""
        return self.positives / self.trainable_rows if self.trainable_rows else 0.0

    @property
    def post_onset_share(self) -> float:
        """Return the fraction of rows excluded as already-failed."""
        return (self.rows - self.trainable_rows) / self.rows if self.rows else 0.0


@dataclass(frozen=True, slots=True)
class Report:
    """Everything worth printing about a dataset."""

    rows: int
    lives: int
    machines: int
    features: tuple[str, ...]
    splits: tuple[SplitSummary, ...]
    baseline_scores: dict[str, dict[str, float]]
    scenario_events: dict[str, int]

    def summary_for(self, split: Split) -> SplitSummary:
        """Return one split's summary.

        Raises:
            KeyError: if the split holds no machines.
        """
        return next(item for item in self.splits if item.split is split)

    def render(self) -> str:
        """Format the report for a terminal."""
        total_events = sum(item.failure_events for item in self.splits)
        lines = [
            "dataset",
            "=" * 72,
            f"  rows        {self.rows:>14,}",
            f"  lives       {self.lives:>14,}",
            f"  machines    {self.machines:>14,}",
            f"  features    {', '.join(self.features)}",
            "",
            "splits",
            "-" * 72,
            f"  {'split':<12}{'machines':>9}{'lives':>8}{'events':>8}"
            f"{'trainable':>12}{'positives':>11}{'prevalence':>12}",
        ]
        lines += [
            f"  {item.split.value:<12}{item.machines:>9}{item.lives:>8}"
            f"{item.failure_events:>8}{item.trainable_rows:>12,}"
            f"{item.positives:>11,}{item.prevalence:>11.2%}"
            for item in self.splits
        ]
        lines += [
            "",
            "  Prevalence is measured over trainable rows, at each split's own",
            "  natural rate. Only training is ever rebalanced, so the validation",
            "  and test figures stay comparable to deployment.",
            "",
        ]
        if self.lives:
            lines += [
                f"  {total_events:,} failure events across {self.lives:,} lives "
                f"({total_events / self.lives:.2f} per life).",
                "",
            ]
        lines += [
            "clock baseline",
            "-" * 72,
            "  A model given only minutes since the life began, and no signals.",
            "  This is the floor Phase 4 has to clear. The simulator's",
            "  degradation is a function of elapsed time, so a score close to",
            "  this one means a clock was measured rather than a machine.",
            "",
            f"  {'split':<12}{'prevalence':>12}{'average precision':>20}{'recall@p=0.8':>14}",
        ]
        lines += [
            f"  {name:<12}{scores['prevalence']:>11.2%}"
            f"{scores['average_precision']:>20.3f}"
            f"{scores['recall_at_target_precision']:>13.2%}"
            for name, scores in self.baseline_scores.items()
        ]

        lines += ["", "failure events by scenario", "-" * 72]
        lines += [
            f"  {scenario:<24}{count:>8,}"
            for scenario, count in sorted(self.scenario_events.items())
        ]

        return "\n".join(lines)


def build_report(dataset: Dataset, plans: Sequence[MachinePlan]) -> Report:
    """Compute every figure the report prints."""
    if dataset.features != len(FEATURE_COLUMNS):
        raise ValueError(
            f"the artifact carries {dataset.features} signals but this build "
            f"expects {len(FEATURE_COLUMNS)}. It was written by a different version."
        )

    life_split = np.array([dataset.splits[index] for index in dataset.life_machine], dtype=object)
    lengths = dataset.life_lengths()
    positive = (
        (dataset.minutes_to_onset > 0) & (dataset.minutes_to_onset <= HORIZON_MINUTES)
    ).astype(np.int64)
    trainable = dataset.trainable().astype(np.int64)
    has_onset = np.array(
        [int(value) != NEVER_FAILS for value in dataset.life_onset], dtype=np.int64
    )

    life_of_row = np.repeat(np.arange(dataset.lives, dtype=np.int64), lengths)
    positives_per_life = np.bincount(life_of_row, weights=positive, minlength=dataset.lives).astype(
        np.int64
    )
    trainable_per_life = np.bincount(
        life_of_row, weights=trainable, minlength=dataset.lives
    ).astype(np.int64)

    summaries = tuple(
        _summarise(
            split,
            lengths=lengths,
            positives=positives_per_life,
            trainable=trainable_per_life,
            has_onset=has_onset,
            life_split=life_split,
            split_count=sum(1 for value in dataset.splits if value is split),
        )
        for split in (Split.TRAIN, Split.VALIDATION, Split.TEST, Split.TEST_SHIFT)
        if any(value is split for value in dataset.splits)
    )

    scenarios = _scenario_by_life(plans, dataset)
    scenario_events: dict[str, int] = {}
    for index, scenario in enumerate(scenarios):
        if has_onset[index]:
            scenario_events[scenario] = scenario_events.get(scenario, 0) + 1

    hazard = baseline.fit(dataset)
    elapsed = baseline.elapsed_minutes(dataset)
    scores = hazard.score(elapsed)

    baseline_scores: dict[str, dict[str, float]] = {}
    for split in (Split.VALIDATION, Split.TEST, Split.TEST_SHIFT):
        if not any(value is split for value in dataset.splits):
            continue
        rows = baseline.split_rows(dataset, split)
        baseline_scores[split.value] = baseline.evaluate(positive[rows], scores[rows])

    return Report(
        rows=dataset.rows,
        lives=dataset.lives,
        machines=len(dataset.machine_ids),
        features=FEATURE_COLUMNS,
        splits=summaries,
        baseline_scores=baseline_scores,
        scenario_events=scenario_events,
    )


def _summarise(
    split: Split,
    *,
    lengths: Int64Array,
    positives: Int64Array,
    trainable: Int64Array,
    has_onset: Int64Array,
    life_split: npt.NDArray[np.object_],
    split_count: int,
) -> SplitSummary:
    """Aggregate one split's lives."""
    selected = np.array([value is split for value in life_split], dtype=bool)
    return SplitSummary(
        split=split,
        machines=split_count,
        lives=int(selected.sum()),
        rows=int(lengths[selected].sum()),
        trainable_rows=int(trainable[selected].sum()),
        positives=int(positives[selected].sum()),
        failure_events=int(has_onset[selected].sum()),
    )


def _scenario_by_life(plans: Sequence[MachinePlan], dataset: Dataset) -> list[str]:
    """Return each life's scenario."""
    flattened = [life.scenario.value for plan in plans for life in plan.lives]
    if len(flattened) != dataset.lives:
        raise ValueError(
            f"the plan holds {len(flattened)} lives and the artifact "
            f"{dataset.lives}; they were produced from different plans."
        )
    return flattened
