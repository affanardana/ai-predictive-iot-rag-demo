"""Time a local GGUF on this machine, before the design depends on it.

Phase 10 puts a generative model on a box with one core, 3.9 GB of RAM and no
swap. Two numbers decide whether that is viable — decode speed and resident
memory — and both are properties of the host rather than of the code, so they
have to be measured rather than assumed. This is the measurement.

The parameters below are the ones the service will actually use, so the numbers
mean something: `n_threads=1` because there is one core and more threads only
add context switching, `n_ctx=2048` because the KV cache is what stays resident,
and `use_mmap=True` with `use_mlock=False` because file-backed weights can be
reclaimed under pressure where anonymous ones cannot — which, on a box with no
swap, is the difference between a slow answer and a killed container.

It runs in a throwaway container, because the model needs a llama.cpp build that
the repository does not otherwise depend on. Three things to mount and one
command to run, and `infra/compose/README.md` has the transcript to paste:

    image     python:3.12-slim
    mounts    <repo>/infra/compose/model      -> /model:ro
              <repo>/services/inference/scripts/bench_chat.py -> /bench.py:ro
    install   pip install "llama-cpp-python==0.3.35"
                --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
    run       python /bench.py

The pin matters: the CPU wheel index carries specific versions, and an unpinned
install falls back to building llama.cpp from source, which on this host is a
slow way to find out that a wheel existed.

`--model` defaults to the Qwen2.5-1.5B-Instruct Q4_K_M build the plan sizes
against; point it at another quant to compare them on the same box.

Lives outside `src/inference/` so it is not part of the installed package, and
is not in mypy's `files` — which is why every call here is written to be obvious
rather than clever. `ruff check .` still covers it.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

DEFAULT_MODEL = "/model/chat/qwen2.5-1.5b-instruct-q4_k_m.gguf"

#: The prompt the design expects to send: ~400 tokens of findings, one question.
#: Deliberately representative rather than minimal — prefill scales with it, and
#: prefill is half the wait.
PROMPT = "\n".join(
    [
        "You are a maintenance assistant for an industrial motor fleet.",
        "Use only the findings below. Do not state any number that is not in them.",
        "Do not describe a procedure that is not in a DOCUMENTED finding.",
        "If the findings do not answer the question, say so plainly.",
        "",
        "Findings:",
        "[1] OBSERVED   M003 was last seen at 2026-09-26 19:02 with vibration 1.42 mm/s.",
        "[2] INFERRED   M003 vibration fitted a change of +0.89 across the window.",
        "[3] PREDICTED  M003 failure probability 0.81 within 60 minutes.",
        "[4] DOCUMENTED Bearing Inspection SOP v1.4, section 3, page 1:"
        " Attach an accelerometer to the bearing housing.",
        "",
        "Question: M003 has increasing vibration. What should I inspect?",
        "Answer:",
    ]
)


@dataclass(frozen=True, slots=True)
class Timing:
    """One call's cost, split into the two halves a reader waits through."""

    prefill_seconds: float
    decode_seconds: float
    tokens: int
    text: str

    @property
    def tokens_per_second(self) -> float:
        """Decode speed, which is the number the design rests on."""
        return self.tokens / self.decode_seconds if self.decode_seconds > 0 else 0.0

    @property
    def total_seconds(self) -> float:
        """What a reader actually waits: prefill plus decode."""
        return self.prefill_seconds + self.decode_seconds


def measure(llm: object, prompt: str, max_tokens: int) -> Timing:
    """Time one call, streaming so the prefill and decode halves separate.

    Streamed because time-to-first-token *is* the prefill: a single blocking
    call reports one number for two costs, and it is the split that says which
    knob to turn.
    """
    start = time.perf_counter()
    first: float | None = None
    for _ in llm(prompt, max_tokens=max_tokens, stream=True):  # type: ignore[operator]
        if first is None:
            first = time.perf_counter()
    end = time.perf_counter()

    # The token count comes from the non-streamed call rather than from counting
    # chunks: llama.cpp streams a variable number of pieces per token, so
    # counting them would overstate the rate.
    result = llm(prompt, max_tokens=max_tokens)  # type: ignore[operator]
    usage = result["usage"]
    return Timing(
        prefill_seconds=(first - start) if first is not None else 0.0,
        decode_seconds=end - (first if first is not None else start),
        tokens=int(usage["completion_tokens"]),
        text=str(result["choices"][0]["text"]).strip(),
    )


def parse_args() -> argparse.Namespace:
    """Read the model path, the answer bound, and how many calls to time."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--model", type=Path, default=Path(DEFAULT_MODEL))
    parser.add_argument("--max-tokens", type=int, default=160)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--context", type=int, default=2048)
    return parser.parse_args()


def main() -> int:
    """Load the model, time it, and report what the box did."""
    args = parse_args()
    # Imported here rather than at module scope: `resource` is Unix-only, and
    # the script should be readable -- `--help` included -- anywhere.
    import resource

    from llama_cpp import Llama

    if not args.model.exists():
        print(f"error: no model at {args.model}")
        return 1

    print(f"model   {args.model.name} ({args.model.stat().st_size / 1024**2:.0f} MiB)")
    started = time.perf_counter()
    llm = Llama(
        model_path=str(args.model),
        n_ctx=args.context,
        n_threads=1,
        n_batch=256,
        use_mmap=True,
        use_mlock=False,
        verbose=False,
    )
    print(f"load    {time.perf_counter() - started:.1f} s")
    print()

    for run in range(1, args.runs + 1):
        timing = measure(llm, PROMPT, args.max_tokens)
        print(f"run {run}")
        print(f"  prefill        {timing.prefill_seconds:6.1f} s")
        print(f"  decode         {timing.decode_seconds:6.1f} s for {timing.tokens} tokens")
        print(f"  decode speed   {timing.tokens_per_second:6.1f} tokens/s")
        print(f"  total          {timing.total_seconds:6.1f} s")
        if run == 1:
            print()
            print("  answer:")
            print(f"  {timing.text}")
            print()

    print(f"peak RSS       {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:6.0f} MiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
