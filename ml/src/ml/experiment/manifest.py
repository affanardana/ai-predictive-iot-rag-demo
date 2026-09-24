"""Reproducibility, written down beside the checkpoint.

The dataset gets this treatment already: `data/` is gitignored and the data is
rebuilt from a four-value `plan.json` rather than stored. A checkpoint gets the
same deal — not committed, but completely described by a small JSON file next to
it, so "which settings, on which data, at which commit" is answered by reading
rather than by remembering.

The fingerprint covers **every** artifact file including `normalization.json`,
where a silent change would otherwise leave no trace at all: the signals would
be identical, the labels would be identical, and every number would shift.
"""

from __future__ import annotations

import hashlib
import json
import platform
import shutil
import subprocess
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy

from ml.experiment.errors import ExperimentError

MANIFEST_FILENAME = "run.json"
MANIFEST_SCHEMA = 1


@dataclass(frozen=True, slots=True)
class Environment:
    """What the run actually executed on.

    Colab's preinstalled torch will not match the pinned range, so what actually
    ran is recorded rather than assumed. The lock is what a future reproduction
    targets; this is what happened, and the report says so when they differ.
    """

    python: str
    numpy: str
    torch: str | None
    device: str

    @classmethod
    def capture(cls, *, device: str = "unknown", torch_version: str | None = None) -> Environment:
        """Return the current environment.

        `torch_version` is passed in rather than detected here, and the reason is
        not stylistic: an import-linter contract forbids this package from
        importing torch at all, so that the evaluation and experiment cores stay
        runnable where the model cannot. Reading a version string is still an
        import, and an import added for a version string is how a boundary like
        this erodes. The torch layer is the one that knows.
        """
        numpy_version = numpy.__version__

        return cls(
            python=platform.python_version(),
            numpy=numpy_version,
            torch=torch_version,
            device=device,
        )


@dataclass(frozen=True, slots=True)
class RunManifest:
    """Everything needed to say what produced a checkpoint."""

    schema: int
    run_id: str
    created_at: str
    config_digest: str
    dataset_fingerprint: str
    dataset_directory: str
    git_sha: str | None
    git_dirty: bool | None
    environment: Environment

    @property
    def reproducible(self) -> bool:
        """Whether the run can be rebuilt from its own manifest."""
        return bool(self.config_digest and self.dataset_fingerprint)


def fingerprint(dataset_dir: Path) -> str:
    """Return a hash over every file in an artifact directory.

    Sorted by name, and the name is hashed with the contents — so a file moved
    or renamed changes the fingerprint, which a contents-only hash would miss.

    Raises:
        ExperimentError: if the directory holds no files, which would otherwise
            hash to a valid-looking constant.
    """
    if not dataset_dir.is_dir():
        raise ExperimentError(f"{dataset_dir} is not a directory.")

    digest = hashlib.sha256()
    files = sorted(path for path in dataset_dir.iterdir() if path.is_file())
    if not files:
        raise ExperimentError(f"{dataset_dir} holds no files to fingerprint.")

    for path in files:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def git_revision(directory: Path) -> tuple[str, bool] | None:
    """Return the current commit and whether the tree is dirty, if there is one.

    Returns `None` rather than raising when git is absent or this is not a
    checkout, which is the ordinary case for a Colab session that copied files
    in rather than cloning.
    """
    git = shutil.which("git")
    if git is None:
        return None

    def _run(*arguments: str) -> str:
        return subprocess.run(  # noqa: S603 - resolved path, fixed argument list
            [git, *arguments],
            cwd=directory,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout

    try:
        sha = _run("rev-parse", "HEAD").strip()
        status = _run("status", "--porcelain")
    except (OSError, subprocess.SubprocessError):
        return None
    return sha, bool(status.strip())


def build(
    *,
    run_id: str,
    config_digest: str,
    dataset_dir: Path,
    repository: Path,
    device: str,
    torch_version: str | None = None,
) -> RunManifest:
    """Assemble a manifest for a run about to start."""
    revision = git_revision(repository)
    return RunManifest(
        schema=MANIFEST_SCHEMA,
        run_id=run_id,
        created_at=datetime.now(UTC).isoformat(),
        config_digest=config_digest,
        dataset_fingerprint=fingerprint(dataset_dir),
        dataset_directory=str(dataset_dir),
        git_sha=revision[0] if revision else None,
        git_dirty=revision[1] if revision else None,
        environment=Environment.capture(device=device, torch_version=torch_version),
    )


def write_manifest(path: Path, manifest: RunManifest) -> Path:
    """Write a manifest as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(manifest)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def read_manifest(path: Path) -> RunManifest:
    """Read a manifest.

    Raises:
        ExperimentError: if the file is missing or from another schema.
    """
    if not path.exists():
        raise ExperimentError(f"{path} does not exist.")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != MANIFEST_SCHEMA:
        raise ExperimentError(
            f"{path} is manifest schema {payload.get('schema')}; this build reads "
            f"{MANIFEST_SCHEMA}."
        )
    payload["environment"] = Environment(**payload["environment"])
    return RunManifest(**payload)


def describe_mismatch(manifest: RunManifest, dataset_dir: Path) -> str | None:
    """Return a warning if the artifact no longer matches the manifest.

    A warning rather than an error, and returned rather than printed, so the
    caller decides. Verifying a checkpoint against a *different* dataset is
    exactly what should fail; running a new experiment against one is ordinary.
    """
    try:
        current = fingerprint(dataset_dir)
    except ExperimentError as exc:
        return str(exc)
    if current == manifest.dataset_fingerprint:
        return None
    return (
        f"the dataset at {dataset_dir} no longer matches the manifest "
        f"({current[:12]} against {manifest.dataset_fingerprint[:12]}); "
        "predictions from this checkpoint are not comparable to the recorded ones."
    )
