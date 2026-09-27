"""Writing an answer from evidence, on this box, with no network.

The model is a GGUF — `llama.cpp` rather than transformers — and the reason is
the machine rather than the library. Measured here: the same weights cost about
their file size (940 MiB on disk, ~1.7 GiB resident including the KV cache and
the runtime) where a torch build of a 1.5B model would spend more than that on
the framework. `llama_cpp` is also imported inside the constructor, so a
deployment that has not configured a chat model pays nothing for this module
existing.

The parameters are measured rather than chosen: one thread because the box has
one core, 2048 context because the KV cache is what stays resident, and mmap
without mlock because file-backed weights can be reclaimed under pressure where
anonymous ones cannot — on a host with no swap that is the difference between a
slow answer and a killed container.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Protocol

from inference.settings import Settings

#: One thread. More than the core count adds context switching to a workload
#: that is memory-bandwidth-bound, which makes it slower rather than faster.
THREADS = 1

#: The batch the prompt is processed in. Bounds the compute buffer's peak, which
#: is anonymous memory and therefore unreclaimable.
BATCH = 256

#: Low, because the task is describing evidence rather than writing prose. The
#: measurement answered faithfully at this setting; raising it would buy variety
#: at the price of invention.
TEMPERATURE = 0.2


@dataclass(frozen=True, slots=True)
class Turn:
    """One message, as this service sees it.

    Its own type rather than the API's, because this is a service boundary:
    neither side should be able to change the other's shape by accident, the
    same argument `schemas.py` makes for the telemetry it shares with the API.
    """

    role: str
    content: str


class ChatModel(Protocol):
    """What the app needs from a language model, so tests can supply their own."""

    @property
    def model_id(self) -> str:
        """Identity of the loaded model, reported by the health endpoint."""
        ...

    def stream(self, messages: Sequence[Turn], *, max_tokens: int) -> Iterator[str]:
        """Yield the answer as it is written."""
        ...


class LlamaChat:
    """A GGUF loaded once at startup, answering through its chat template."""

    def __init__(self, settings: Settings) -> None:
        from llama_cpp import Llama

        if settings.chat_model is None:
            raise ValueError("LlamaChat needs a configured chat model.")

        self._llm = Llama(
            model_path=str(settings.chat_model),
            n_ctx=settings.chat_context,
            n_threads=THREADS,
            n_batch=BATCH,
            use_mmap=True,
            # Not locked: see the module docstring. The weights are reclaimable,
            # and pinning them on a host with no swap removes the only thing
            # standing between memory pressure and the OOM killer.
            use_mlock=False,
            verbose=False,
        )
        self._model_id = settings.chat_model_id

    @property
    def model_id(self) -> str:
        """Identity of the loaded model."""
        return self._model_id

    def stream(self, messages: Sequence[Turn], *, max_tokens: int) -> Iterator[str]:
        """Yield the answer a piece at a time.

        Through `create_chat_completion` rather than raw completion, so the
        model's own chat template is applied: an instruct model given a
        hand-rolled prompt is a model being used in a shape it was not trained
        for, and the failure looks like poor reasoning rather than a formatting
        mistake.
        """
        payload = [{"role": turn.role, "content": turn.content} for turn in messages]
        for chunk in self._llm.create_chat_completion(
            messages=payload,
            max_tokens=max_tokens,
            temperature=TEMPERATURE,
            stream=True,
        ):
            piece = chunk["choices"][0]["delta"].get("content")
            if piece:
                yield str(piece)
