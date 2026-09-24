"""Command line entry point.

Three commands, in the order they are used:

    ml dataset generate    run the fleet, writing telemetry and truth
    ml dataset export      read those back, and write the training artifact
    ml dataset report      print what came out, and how hard it is

`export` is separate from `generate` deliberately. Labelling, splitting and
normalisation are decisions about *training*, and they can be revisited — a
different split, a different horizon — without regenerating six minutes of
telemetry. The two Parquet files are the interface between the halves.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path

from ml.dataset.artifacts import (
    build_dataset,
    life_table,
    read_artifact,
    write_artifact,
    write_life_table,
)
from ml.dataset.errors import DatasetError
from ml.dataset.generation import generate
from ml.dataset.lives import iter_machine_ids
from ml.dataset.plan import PLAN_FILENAME, document_for, read_plan, write_plan
from ml.dataset.report import build_report
from ml.evaluation.errors import EvaluationError
from ml.experiment.errors import ExperimentError
from simulator.domain.timestamps import utc_now

DEFAULT_DATASET_DIR = "data/dataset"
DEFAULT_ARTIFACT_DIR = "data/training"
DEFAULT_MODEL_DIR = "data/models"
DEFAULT_CONFIG = "ml/configs/lstm.json"

DEFAULT_MACHINE_COUNT = 100
DEFAULT_SHIFT_MACHINE_COUNT = 12
DEFAULT_DAYS = 30
DEFAULT_SEED = 20260923

EXIT_FAILURE = 1
EXIT_USAGE_ERROR = 2


def _parse_instant(value: str) -> datetime:
    """Parse an ISO 8601 instant, insisting on a timezone."""
    try:
        moment = datetime.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"'{value}' is not an ISO 8601 instant, e.g. 2026-01-01T00:00:00+00:00."
        ) from exc
    if moment.tzinfo is None:
        raise argparse.ArgumentTypeError(
            f"'{value}' has no timezone. Add an offset, e.g. 2026-01-01T00:00:00+00:00."
        )
    return moment


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        prog="ml",
        description="Build a training dataset from generated telemetry.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    dataset = commands.add_parser("dataset", help="Generate, export, or report.")
    actions = dataset.add_subparsers(dest="action", required=True)

    generate_parser = actions.add_parser("generate", help="Run the fleet and write telemetry.")
    generate_parser.add_argument("--machines", type=int, default=DEFAULT_MACHINE_COUNT)
    generate_parser.add_argument(
        "--shift-machines",
        type=int,
        default=DEFAULT_SHIFT_MACHINE_COUNT,
        help=(
            "Machines held out to run lives of a different length. Phase 4 uses "
            "them to tell a model that reads signals from one that reads a clock."
        ),
    )
    generate_parser.add_argument("--days", type=int, default=DEFAULT_DAYS)
    generate_parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    # `type=Path` on every directory flag. Without it the value arrives as a
    # string and fails at the first `/`, which is several frames away from
    # anything that names the flag.
    generate_parser.add_argument("--out", type=Path, default=Path(DEFAULT_DATASET_DIR))
    generate_parser.add_argument("--started-at", type=_parse_instant, default=None)

    export_parser = actions.add_parser("export", help="Write the training artifact.")
    export_parser.add_argument("--dataset", type=Path, default=Path(DEFAULT_DATASET_DIR))
    export_parser.add_argument("--out", type=Path, default=Path(DEFAULT_ARTIFACT_DIR))

    report_parser = actions.add_parser("report", help="Print what the dataset contains.")
    report_parser.add_argument("--dataset", type=Path, default=Path(DEFAULT_DATASET_DIR))
    report_parser.add_argument("--artifact", type=Path, default=Path(DEFAULT_ARTIFACT_DIR))

    model = commands.add_parser("model", help="Train, predict, evaluate, verify.")
    model_actions = model.add_subparsers(dest="action", required=True)

    train_parser = model_actions.add_parser("train", help="Train the LSTM.")
    train_parser.add_argument("--config", type=Path, default=Path(DEFAULT_CONFIG))
    train_parser.add_argument("--artifact", type=Path, default=Path(DEFAULT_ARTIFACT_DIR))
    train_parser.add_argument("--out", type=Path, default=None)
    train_parser.add_argument("--device", default=None)
    train_parser.add_argument("--resume", action="store_true")

    predict_parser = model_actions.add_parser("predict", help="Score a split.")
    predict_parser.add_argument("--run", type=Path, required=True)
    predict_parser.add_argument("--checkpoint", default="best.pt")
    predict_parser.add_argument("--artifact", type=Path, default=Path(DEFAULT_ARTIFACT_DIR))
    predict_parser.add_argument("--split", default="test")

    evaluate_parser = model_actions.add_parser("evaluate", help="Report the results.")
    evaluate_parser.add_argument("--predictions", type=Path, required=True)
    evaluate_parser.add_argument("--artifact", type=Path, default=Path(DEFAULT_ARTIFACT_DIR))
    evaluate_parser.add_argument("--out", type=Path, default=None)

    verify_parser = model_actions.add_parser("verify", help="Re-score a checkpoint.")
    verify_parser.add_argument("--run", type=Path, required=True)
    verify_parser.add_argument("--checkpoint", default="best.pt")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line. Returns the process exit code."""
    args = build_parser().parse_args(argv)
    try:
        return _dispatch(args)
    except (DatasetError, EvaluationError, ExperimentError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_FAILURE


def _dispatch(args: argparse.Namespace) -> int:
    """Route to the requested action."""
    if args.command == "model":
        return _model(args)
    if args.action == "generate":
        return _generate(args)
    if args.action == "export":
        return _export(args)
    return _report(args)


def _model(args: argparse.Namespace) -> int:
    """Route a model command.

    `ml.model` is imported inside the branches rather than at the top of the
    file, because it needs torch and this module does not. Importing it eagerly
    would make `ml dataset report` fail in an environment without the training
    extra — which is exactly the environment the evaluation half is built to
    work in.
    """
    if args.action == "evaluate":
        return _model_evaluate(args)

    try:
        from ml.model import pipeline
    except ImportError:
        print(
            "error: this command needs the training extra.\n"
            "       uv sync --locked --all-packages --extra train",
            file=sys.stderr,
        )
        return EXIT_FAILURE

    if args.action == "train":
        return _model_train(args, pipeline)
    if args.action == "predict":
        return _model_predict(args, pipeline)
    return _model_verify(args, pipeline)


def _model_train(args: argparse.Namespace, pipeline: object) -> int:
    """Train and report where the checkpoints went."""
    output = args.out or Path(DEFAULT_MODEL_DIR) / f"run-{_timestamp()}"
    result = pipeline.train_run(  # type: ignore[attr-defined]
        config_path=args.config,
        artifact_dir=args.artifact,
        output_dir=output,
        repository=Path.cwd(),
        resume=args.resume,
        device=args.device,
    )
    _announce(
        [
            ("epochs", str(len(result.epochs))),
            ("best epoch", str(result.best_epoch)),
            ("best life AP", f"{result.best_life_ap:.4f}"),
            ("stopped early", str(result.stopped_early)),
            ("checkpoints", str(output)),
        ]
    )
    return 0


def _model_predict(args: argparse.Namespace, pipeline: object) -> int:
    """Score a split from a checkpoint."""
    from ml.dataset.splits import Split

    path = pipeline.predict_run(  # type: ignore[attr-defined]
        checkpoint_path=args.run / args.checkpoint,
        artifact_dir=args.artifact,
        split=Split(args.split),
    )
    _announce([("predictions", str(path))])
    return 0


def _model_verify(args: argparse.Namespace, pipeline: object) -> int:
    """Re-score a checkpoint's golden batch and say whether it reproduces."""
    result = pipeline.verify_checkpoint(  # type: ignore[attr-defined]
        checkpoint_path=args.run / args.checkpoint
    )
    _announce(
        [
            ("rows", str(result.rows)),
            ("max deviation", f"{result.max_deviation:.3e}"),
            ("reproducible", "yes" if result.passed else "NO"),
        ]
    )
    return 0 if result.passed else EXIT_FAILURE


def _model_evaluate(args: argparse.Namespace) -> int:
    """Evaluate a prediction table. Needs no torch."""
    from ml.evaluation.pipeline import evaluate_run

    result = evaluate_run(
        predictions_path=args.predictions,
        artifact_dir=args.artifact,
        output_dir=args.out,
    )
    print(result.report)
    return 0


def _generate(args: argparse.Namespace) -> int:
    """Compose a fleet and run it."""
    if args.machines < 1:
        print("error: --machines must be at least 1", file=sys.stderr)
        return EXIT_USAGE_ERROR
    if args.days < 1:
        print("error: --days must be at least 1", file=sys.stderr)
        return EXIT_USAGE_ERROR

    main_machines = tuple(iter_machine_ids(args.machines))
    shift_machines = tuple(iter_machine_ids(args.shift_machines, first=args.machines + 1))
    document = document_for(
        seed=args.seed,
        started_at=args.started_at or utc_now(),
        main_machines=main_machines,
        shift_machines=shift_machines,
        timeline=timedelta(days=args.days),
    )
    plans = document.build()

    print(
        f"generating {document.machine_count} machines over {args.days} days "
        f"({sum(len(plan.lives) for plan in plans):,} lives)",
        file=sys.stderr,
    )
    results = generate(plans, output=args.out, progress=sys.stderr)
    write_plan(args.out / PLAN_FILENAME, document)

    _announce(
        [
            ("machines", f"{results.machines:,}"),
            ("lives", f"{results.lives:,}"),
            ("observations", f"{results.rows:,}"),
            ("telemetry", str(results.output)),
        ]
    )
    return 0


def _export(args: argparse.Namespace) -> int:
    """Label, split, normalise, and write the artifact."""
    dataset = build_dataset(args.dataset)
    write_artifact(args.out, dataset)

    plans = read_plan(args.dataset / PLAN_FILENAME).build()
    table_path = write_life_table(args.dataset, life_table(dataset, plans))

    _announce(
        [
            ("rows", f"{dataset.rows:,}"),
            ("lives", f"{dataset.lives:,}"),
            ("trainable rows", f"{int(dataset.trainable().sum()):,}"),
            ("normalisation from", f"{dataset.normalization.source_rows:,} rows"),
            ("artifact", str(args.out)),
            ("life table", str(table_path)),
        ]
    )
    return 0


def _report(args: argparse.Namespace) -> int:
    """Print the dataset's shape and its difficulty."""
    dataset = read_artifact(args.artifact)
    plans = read_plan(args.dataset / PLAN_FILENAME).build()
    print(build_report(dataset, plans).render())
    return 0


def _timestamp() -> str:
    """Return a sortable run stamp."""
    return datetime.now(UTC).strftime("%Y%m%d-%H%M%S")


def _announce(rows: Sequence[tuple[str, str]]) -> None:
    """Print an aligned summary."""
    width = max(len(label) for label, _ in rows)
    for label, value in rows:
        print(f"{label:<{width}}  {value}")


__all__ = ["build_parser", "main"]
