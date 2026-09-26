"""Command line entry point.

The dataset commands, in the order they are used:

    ml dataset generate    run the fleet, writing telemetry and truth
    ml dataset export      read those back, and write the training artifact
    ml dataset report      print what came out, and how hard it is

`export` is separate from `generate` deliberately. Labelling, splitting and
normalisation are decisions about *training*, and they can be revisited — a
different split, a different horizon — without regenerating six minutes of
telemetry. The two Parquet files are the interface between the halves.

The knowledge commands, which need the `knowledge` extra:

    ml knowledge extract   print what a PDF parses into, page and size included
    ml knowledge ingest    send the corpus to the API, which stores it
    ml knowledge evaluate  measure retrieval against the labelled questions

`extract` exists on its own because the parse is the fragile half. It prints
what the chunker will see, so a change in `pypdf` or in the corpus shows up as
readable output rather than as a retrieval score that moved.
"""

from __future__ import annotations

import argparse
import os
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
from ml.knowledge.corpus import CorpusEntry, read_corpus
from ml.knowledge.errors import KnowledgeError
from simulator.domain.timestamps import utc_now

DEFAULT_DATASET_DIR = "data/dataset"
DEFAULT_ARTIFACT_DIR = "data/training"
DEFAULT_MODEL_DIR = "data/models"
DEFAULT_CONFIG = "ml/configs/lstm.json"
DEFAULT_CORPUS = "knowledge/corpus.json"
DEFAULT_QUESTIONS = "knowledge/eval/questions.json"
#: Where the API listens on the development machine. Inside compose the ingest
#: container is pointed at `http://api:8000`, which is why this is a flag.
DEFAULT_API_URL = "http://localhost:8000"
#: The variable the token comes from, matching the API's own setting. Passed as
#: an environment variable rather than a flag so it stays out of shell history.
INGEST_TOKEN_VARIABLE = "INGEST_API_TOKEN"  # noqa: S105 - a variable name, not a secret

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

    knowledge = commands.add_parser("knowledge", help="Parse, ingest, or evaluate the corpus.")
    knowledge_actions = knowledge.add_subparsers(dest="action", required=True)

    extract_parser = knowledge_actions.add_parser(
        "extract", help="Print what a document parses into."
    )
    extract_parser.add_argument("--corpus", type=Path, default=Path(DEFAULT_CORPUS))
    extract_parser.add_argument(
        "--document",
        default=None,
        help="One document key from the manifest. Omitted means every document.",
    )

    ingest_parser = knowledge_actions.add_parser("ingest", help="Send the corpus to the API.")
    ingest_parser.add_argument("--corpus", type=Path, default=Path(DEFAULT_CORPUS))
    ingest_parser.add_argument("--api-url", default=DEFAULT_API_URL)
    ingest_parser.add_argument(
        "--activate",
        action="store_true",
        help="Make each ingested version the retrievable one.",
    )
    ingest_parser.add_argument(
        "--allow-replace",
        action="store_true",
        help="Rewrite a version whose text changed, instead of refusing it.",
    )

    evaluate_parser = knowledge_actions.add_parser(
        "evaluate", help="Measure retrieval against the labelled questions."
    )
    evaluate_parser.add_argument("--questions", type=Path, default=Path(DEFAULT_QUESTIONS))
    evaluate_parser.add_argument("--api-url", default=DEFAULT_API_URL)
    # Ten, not the five an answer would be given: recall@k for k above the
    # result limit is bounded by that limit rather than by retrieval, and the
    # first measured run reported recall@10 identical to recall@5 for exactly
    # that reason. Measuring wider than an answer needs is the point of
    # measuring.
    evaluate_parser.add_argument("--limit", type=int, default=10)
    evaluate_parser.add_argument("--out", type=Path, default=None)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line. Returns the process exit code."""
    args = build_parser().parse_args(argv)
    try:
        return _dispatch(args)
    except (DatasetError, EvaluationError, ExperimentError, KnowledgeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_FAILURE


def _dispatch(args: argparse.Namespace) -> int:
    """Route to the requested action."""
    if args.command == "model":
        return _model(args)
    if args.command == "knowledge":
        return _knowledge(args)
    if args.action == "generate":
        return _generate(args)
    if args.action == "export":
        return _export(args)
    return _report(args)


def _missing_knowledge_extra() -> int:
    """Report the missing extra, and the command that installs it."""
    print(
        "error: this command needs the knowledge extra.\n"
        "       uv sync --locked --all-packages --extra knowledge",
        file=sys.stderr,
    )
    return EXIT_FAILURE


def _knowledge(args: argparse.Namespace) -> int:
    """Route a knowledge command.

    Reading the manifest happens here; parsing and posting happen in the
    branches, because those need the `knowledge` extra -- `pypdf` for the first,
    `httpx` for the others -- and `ml dataset report` has to keep working
    without it. `ml.knowledge.corpus` is imported at the top of this file
    precisely because it is the one module in that package that needs nothing
    outside the standard library.
    """
    # `evaluate` reads no manifest: it measures a running stack, so it asks that
    # stack how large its corpus is. Dispatching before the read is what keeps a
    # subcommand without a `--corpus` flag from reaching for one.
    if args.action == "evaluate":
        return _knowledge_evaluate(args)

    entries = read_corpus(args.corpus)
    if args.action == "extract":
        return _knowledge_extract(args, entries)
    return _knowledge_ingest(args, entries)


def _knowledge_extract(
    args: argparse.Namespace,
    entries: Sequence[CorpusEntry],
) -> int:
    """Print what each document parses into, and at what sizes.

    The size histogram is the point. Heading detection is relative to a
    document's own body size, so a reader checking a parse needs to see the
    sizes rather than only the text.
    """
    from collections import Counter

    selected = [entry for entry in entries if args.document in (None, entry.document_key)]
    if not selected:
        print(f"error: no document with key '{args.document}'.", file=sys.stderr)
        return EXIT_USAGE_ERROR

    try:
        from ml.knowledge.extraction import extract_lines

        # `pypdf` is imported inside `extract_lines`, so the missing extra
        # surfaces here rather than at import time.
        parsed = [(entry, extract_lines(entry.path)) for entry in selected]
    except ImportError:
        return _missing_knowledge_extra()

    for entry, lines in parsed:
        sizes = Counter(round(line.font_size, 2) for line in lines)
        histogram = ", ".join(f"{size}pt x{count}" for size, count in sizes.most_common())
        print(f"\n=== {entry.title} — {entry.document_key} v{entry.version}")
        print(f"    {len(lines)} lines, sizes: {histogram}")
        for line in lines:
            print(f"{line.page:>3} {line.font_size:>6.2f} | {line.text}")
    return 0


def _knowledge_ingest(args: argparse.Namespace, entries: Sequence[CorpusEntry]) -> int:
    """Parse every document and send it to the API.

    Progress goes to stderr and the outcomes to stdout, so `ml knowledge ingest
    > outcomes.txt` still shows a reader what is happening.
    """
    try:
        import httpx

        from ml.knowledge.ingest import ingest_corpus
    except ImportError:
        return _missing_knowledge_extra()

    headers: dict[str, str] = {}
    token = os.environ.get(INGEST_TOKEN_VARIABLE)
    if token:
        headers["X-Ingest-Token"] = token
    elif args.api_url.startswith(("http://localhost", "http://127.")):
        # No token against a local API: `APP_ENV=local` is the one environment
        # where the API does not require one. Said out loud rather than assumed,
        # because the same command against a deployment fails without it.
        print("note: no token, addressing a local API.", file=sys.stderr)

    with httpx.Client(headers=headers) as client:
        outcomes = ingest_corpus(
            entries,
            client=client,
            api_url=args.api_url,
            activate=args.activate,
            allow_replace=args.allow_replace,
            progress=lambda message: print(message, file=sys.stderr),
        )

    for outcome in outcomes:
        print(outcome.render())
    passes = sum(outcome.chunk_count for outcome in outcomes)
    print(f"\n{len(outcomes)} documents, {passes} passages.")
    return 0


def _knowledge_evaluate(args: argparse.Namespace) -> int:
    """Measure retrieval with and without reranking, and report both."""
    import json
    from dataclasses import asdict

    try:
        import httpx

        from ml.knowledge import evaluation
    except ImportError:
        return _missing_knowledge_extra()

    questions = evaluation.read_questions(args.questions)

    with httpx.Client() as client:
        documents = evaluation.count_documents(client=client, api_url=args.api_url)
        print(f"{len(questions)} questions against {documents} documents.", file=sys.stderr)
        vector_answers = evaluation.ask(
            questions, client=client, api_url=args.api_url, rerank=False, limit=args.limit
        )
        reranked_answers = evaluation.ask(
            questions, client=client, api_url=args.api_url, rerank=True, limit=args.limit
        )

    without = evaluation.measure(vector_answers)
    with_rerank = evaluation.measure(reranked_answers)

    print(
        evaluation.render_comparison(
            without, with_rerank, documents=documents, result_limit=args.limit
        )
    )
    # The score summary goes underneath because the abstention row above cannot
    # be read without it: the two stages rank on different scales, and both are
    # compared against one threshold.
    print()
    print(
        evaluation.render_scores(
            evaluation.summarise_scores(vector_answers),
            evaluation.summarise_scores(reranked_answers),
        )
    )
    if args.out is not None:
        payload = {
            "questions": str(args.questions),
            "limit": args.limit,
            "documents": documents,
            "vector_only": evaluation.as_payload(without),
            "reranked": evaluation.as_payload(with_rerank),
            "scores": {
                "vector_only": asdict(evaluation.summarise_scores(vector_answers)),
                "reranked": asdict(evaluation.summarise_scores(reranked_answers)),
            },
        }
        args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {args.out}", file=sys.stderr)
    return 0


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
