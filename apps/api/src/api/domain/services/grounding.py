"""Checking that an answer's numbers came from the evidence.

A 1.5B model composes prose from the evidence it was handed, and sometimes it
writes a plausible number instead of the one it was given — which is precisely
the failure `PRD.md` section 16 exists to prevent and the reason every value the
Copilot states is supposed to be checkable.

This is the check. It does not rewrite the answer: a number that appears nowhere
in the evidence is a finding, and a finding that gets silently corrected is a
finding nobody ever sees. The response carries it, the log records it, and the
reader can see which claim to distrust.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

#: Numbers as a person writes them: an optional sign, digits, an optional
#: decimal part. Deliberately not a full number grammar — a false positive here
#: costs a line in a response, and a missed one costs a fabricated value.
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")

#: A finding reference, as the prompt asks the model to write them: `[1]`, `[2]`.
#: Removed before the numbers are read, because a marker is a pointer to a
#: finding and not a value — counting them would flag every correctly cited
#: answer as a fabrication.
_MARKER = re.compile(r"\[\d+\]")

#: How close two numbers must be to count as the same one, relative to the
#: evidence value. A model that writes "0.81" for an evidence value of "0.812"
#: rounded rather than invented, and calling that a fabrication would make the
#: check useless through noise.
TOLERANCE = 0.02


def ungrounded_numbers(answer: str, evidence: Sequence[str]) -> tuple[str, ...]:
    """Return the numbers in `answer` that appear nowhere in `evidence`.

    In the order they appear in the answer, deduplicated, and rendered as the
    model wrote them so a reader can find them in the text.
    """
    known = _numbers_in(" ".join(evidence))
    if not known:
        # Nothing numeric was supplied, so every number in the answer is the
        # model's own — including one it may have been told in the question.
        return ()

    ungrounded: list[str] = []
    for token in _NUMBER.findall(_MARKER.sub(" ", answer)):
        if token in ungrounded:
            continue
        if not _is_grounded(float(token), known):
            ungrounded.append(token)
    return tuple(ungrounded)


def _numbers_in(text: str) -> tuple[float, ...]:
    """Return every number in `text`, with its percentage and fraction forms.

    Percentages matter because the evidence and the answer routinely differ by a
    factor of a hundred: a probability of `0.81` is written "81%" by anyone
    describing it, and both spellings refer to the same measurement.
    """
    values: list[float] = []
    for token in _NUMBER.findall(text):
        value = float(token)
        values.extend((value, value * 100.0, value / 100.0))
    return tuple(values)


def _is_grounded(value: float, known: Sequence[float]) -> bool:
    """Whether `value` matches anything the model was given."""
    return any(
        abs(value - candidate) <= TOLERANCE * max(1.0, abs(candidate)) for candidate in known
    )
