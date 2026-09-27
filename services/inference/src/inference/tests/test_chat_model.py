r"""The real weights, on the real box.

Marked `llm` and deselected by default, for the same reason the `torch` tests
are: it needs a 940 MB GGUF that is not in the repository and a
`llama-cpp-python` build that is not a dependency of the default environment.
The marker is named in `addopts` rather than merely declared, so a default run
reports these as deselected rather than appearing to have checked them.

**Where it runs: inside the inference image on the deployment host.** That is
the only place both the weights and the runtime exist, and the image already
carries this module:

    docker compose exec inference sh -c \\
      "pip install --quiet pytest && \\
       python -m pytest -m llm -q /app/src/inference/tests/test_chat_model.py"

The `pip install` is ad hoc and disappears with the container; the image carries
only what it needs in order to serve.

**Why it is worth having.** Everything else about the Copilot is tested against
a stub, and a stub can never fail the check the product's claim rests on -- it
answers with whatever the test wrote, so the grounding check is guaranteed to
pass. These two tests ask the real weights the questions a stub cannot: does it
load in the memory the container gives it, and is the number in its prose a
number that was in its evidence.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

pytest.importorskip("llama_cpp")

from inference.chat import LlamaChat, Turn
from inference.settings import Settings

pytestmark = pytest.mark.llm

#: Distinctive enough that finding it in the answer is not a coincidence: a
#: model writing from the evidence will use one of these, and a model making
#: things up has no reason to.
VIBRATION_THEN = "1.42"
VIBRATION_NOW = "2.31"
PROBABILITY = "0.81"

EVIDENCE = (
    "Observed: M003 vibration was 1.42 mm/s three hours ago and is 2.31 mm/s now.\n"
    "Predicted: failure probability within 60 minutes is 0.81 (risk band HIGH).\n"
    "Observed: temperature 61.2 C and rpm 1480, both steady."
)

QUESTION = "What is M003's failure probability, and what is its vibration now?"


@pytest.fixture(scope="module")
def model() -> Iterator[LlamaChat]:
    """The configured model, loaded once for the module.

    Loading costs about twenty seconds, which is why this is module-scoped: two
    loads would double the slowest thing this file does.
    """
    settings = Settings.from_environment()
    if settings.chat_model is None:
        pytest.skip("INFERENCE_CHAT_MODEL is not set, so there is no model to test.")
    yield LlamaChat(settings)


def test_the_model_loads_and_reports_which_file_it_was(model: LlamaChat) -> None:
    """The identity the API puts in every answer as `model_id`.

    It is the file's name, because these weights are placed by hand and the name
    the operator chose is the only identity there is -- so it is also the value
    `CHAT_MODEL_FILE` must agree with on both sides of the stack.
    """
    assert model.model_id.endswith(".gguf")
    assert "qwen" in model.model_id.lower()


def test_it_answers_with_a_number_that_was_in_its_evidence(model: LlamaChat) -> None:
    """The claim the whole phase rests on, at the smallest scale that tests it.

    Not an assertion about prose: the model may phrase this badly, and on a
    1.5B model it often will. What is asserted is that an answer exists and that
    a number in it came from the evidence rather than from the weights -- which
    is the difference between a Copilot and a plausible sentence.
    """
    turns = [
        Turn(role="system", content="Answer using only the evidence given."),
        Turn(role="user", content=f"{EVIDENCE}\n\n{QUESTION}"),
    ]

    answer = "".join(model.stream(turns, max_tokens=120))

    assert answer.strip(), "the model produced nothing at all"
    assert any(value in answer for value in (PROBABILITY, VIBRATION_NOW, VIBRATION_THEN)), (
        f"no evidence number appears in the answer: {answer!r}"
    )


def test_it_streams_rather_than_returning_one_piece(model: LlamaChat) -> None:
    """More than one chunk, because the whole page is built on that.

    A generator that yielded once at the end would satisfy every other test here
    and produce a twenty-second silence in the browser.
    """
    turns = [Turn(role="user", content=f"{EVIDENCE}\n\n{QUESTION}")]

    pieces = list(model.stream(turns, max_tokens=60))

    assert len(pieces) > 1
