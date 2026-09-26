"""Where the checkpoint and the artifact are, read from the environment."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

CHECKPOINT_VARIABLE = "INFERENCE_CHECKPOINT"
ARTIFACT_VARIABLE = "INFERENCE_ARTIFACT_DIR"
EMBED_MODEL_VARIABLE = "INFERENCE_EMBED_MODEL"
EMBED_REVISION_VARIABLE = "INFERENCE_EMBED_REVISION"
RERANK_MODEL_VARIABLE = "INFERENCE_RERANK_MODEL"
RERANK_REVISION_VARIABLE = "INFERENCE_RERANK_REVISION"
MODEL_CACHE_VARIABLE = "INFERENCE_MODEL_CACHE"

#: The retrieval models. MiniLM rather than the BGE family `MASTERPLAN.md` §5
#: names: that section is a direction list, not one of PRD §25's constraints,
#: and these two are a fifth of the size on a one-core box.
DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"

#: Model revisions. Named explicitly rather than left to float, because an index
#: built with one revision and queried with another produces plausible-looking
#: nonsense with nothing logged anywhere. These defaults are the moving ref a
#: first build resolves; set the variables to the commit they resolved to, and
#: record the value once it has been observed on the server.
DEFAULT_REVISION = "main"


class ConfigurationError(RuntimeError):
    """Raised when the service cannot start."""


@dataclass(frozen=True, slots=True)
class Settings:
    """Everything the service needs to start."""

    checkpoint: Path
    artifact_dir: Path
    embed_model: str = DEFAULT_EMBED_MODEL
    embed_revision: str = DEFAULT_REVISION
    rerank_model: str = DEFAULT_RERANK_MODEL
    rerank_revision: str = DEFAULT_REVISION
    #: Where the weights are cached. Unset means the library's own default,
    #: which is a home directory -- fine locally, read-only in the container,
    #: so the image sets this to a writable path.
    model_cache: Path | None = None

    @classmethod
    def from_environment(cls) -> Settings:
        """Read the checkpoint and artifact paths from the environment.

        Raises:
            ConfigurationError: if either is unset, or the checkpoint is not
                there. A service that starts without a model and fails on the
                first request is worse than one that refuses to start.
        """
        checkpoint = os.environ.get(CHECKPOINT_VARIABLE)
        artifact_dir = os.environ.get(ARTIFACT_VARIABLE)
        if not checkpoint or not artifact_dir:
            raise ConfigurationError(
                f"Both {CHECKPOINT_VARIABLE} and {ARTIFACT_VARIABLE} must be set."
            )
        if not Path(checkpoint).exists():
            raise ConfigurationError(f"No checkpoint at '{checkpoint}'.")
        if not (Path(artifact_dir) / "normalization.json").exists():
            raise ConfigurationError(
                f"'{artifact_dir}' does not look like a training artifact; "
                "normalization.json is missing."
            )
        cache = os.environ.get(MODEL_CACHE_VARIABLE)
        return cls(
            checkpoint=Path(checkpoint),
            artifact_dir=Path(artifact_dir),
            embed_model=os.environ.get(EMBED_MODEL_VARIABLE, DEFAULT_EMBED_MODEL),
            embed_revision=os.environ.get(EMBED_REVISION_VARIABLE, DEFAULT_REVISION),
            rerank_model=os.environ.get(RERANK_MODEL_VARIABLE, DEFAULT_RERANK_MODEL),
            rerank_revision=os.environ.get(RERANK_REVISION_VARIABLE, DEFAULT_REVISION),
            model_cache=Path(cache) if cache else None,
        )

    @property
    def embed_model_id(self) -> str:
        """The embedding model's identity, as stored with every vector.

        Model name and revision together: the corpus records this string on
        every chunk, and retrieval only considers chunks that carry it.
        """
        return f"{self.embed_model}@{self.embed_revision}"
