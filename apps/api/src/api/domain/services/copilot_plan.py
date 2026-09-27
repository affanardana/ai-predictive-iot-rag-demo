"""Deciding which sources a question needs.

A rule rather than a model call, and that is the design decision this module
exists to make. A sub-2B model asked to choose among seven tools is right about
half the time, so a Copilot that depended on one would be wrong often and
unpredictably — and the fallback written to catch that would be this function
with an expensive prelude in front of it.

What the model does instead is write the answer from what these tools found.
That division is also the honest one for the product: choosing evidence is a
rule about the data, and composing sentences is what a language model is for.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from api.domain.errors import DomainValidationError
from api.domain.value_objects.copilot import CopilotPlan, CopilotTool
from api.domain.value_objects.machine_id import MACHINE_ID_PATTERN, MachineId
from api.domain.value_objects.time_window import TimeWindow

#: The window a question gets when it does not name one. Five hours is the
#: demonstration's window (`PRD.md` section 15 asks about "the last five
#: hours"), and it is long enough to contain a trend and short enough to
#: describe one.
DEFAULT_WINDOW = TimeWindow.FIVE_HOURS

#: Phrases that name a window, longest first so "last 24 hours" is not matched
#: by the "last hour" rule.
_WINDOW_PHRASES: tuple[tuple[tuple[str, ...], TimeWindow], ...] = (
    (("month", "30 days", "thirty days"), TimeWindow.THIRTY_DAYS),
    (("week", "7 days", "seven days"), TimeWindow.SEVEN_DAYS),
    (("day", "24 hours", "24h", "today"), TimeWindow.ONE_DAY),
    (("five hours", "5 hours", "5h"), TimeWindow.FIVE_HOURS),
    (("hour", "1h"), TimeWindow.ONE_HOUR),
)

#: Words that mean "read the maintenance documentation".
_DOCUMENTATION_WORDS = (
    "inspect",
    "sop",
    "procedure",
    "manual",
    "guide",
    "documentation",
    "according to",
    "checklist",
    "lubricat",
    "torque",
)

#: Words that mean "what has happened to this machine".
_HISTORY_WORDS = ("history", "previously", "how often", "past incidents", "before")

#: Words that mean "how bad is it, and what does the model say".
_RISK_WORDS = ("risk", "fail", "probability", "why", "condition", "deteriorat")

#: Words that mean "what has gone wrong".
_INCIDENT_WORDS = ("incident", "alert", "alarm", "raised")

#: `MachineId`'s own pattern with its anchors removed: that one validates a
#: whole string, and here an identifier arrives inside a sentence. Built from
#: the same pattern so the two cannot drift into accepting different shapes.
_ID_IN_TEXT = re.compile(MACHINE_ID_PATTERN.pattern.replace("^", "").replace("$", ""))


def plan_question(question: str) -> CopilotPlan:
    """Return the sources `question` needs.

    Raises:
        DomainValidationError: if the question is blank.
    """
    text = question.strip()
    if not text:
        raise DomainValidationError("A question must not be blank.")
    lowered = text.casefold()

    machine_id = _machine_id(text)
    window = _window(lowered)
    mentions_metrics = _mentions_a_signal(lowered)
    wants_documentation = any(word in lowered for word in _DOCUMENTATION_WORDS)

    tools: list[CopilotTool] = []
    if machine_id is not None:
        # Current state first: it is the cheapest read and it is what a reader
        # expects to see first in the activity trail.
        tools.append(CopilotTool.CURRENT_STATE)
        if (
            mentions_metrics
            or _mentions_a_window(lowered)
            or any(word in lowered for word in _RISK_WORDS)
        ):
            tools.extend([CopilotTool.TELEMETRY, CopilotTool.TREND])
        if any(word in lowered for word in _RISK_WORDS):
            tools.append(CopilotTool.PREDICTION)
        if any(word in lowered for word in _INCIDENT_WORDS) or _mentions_a_window(lowered):
            tools.append(CopilotTool.INCIDENTS)
        if any(word in lowered for word in _HISTORY_WORDS):
            tools.append(CopilotTool.HISTORY)

    if wants_documentation or machine_id is None:
        # A question with no machine is a documentation question by elimination:
        # the corpus is the only source that can answer it, and if it cannot,
        # the insufficiency verdict is the answer (PRD section 19).
        tools.append(CopilotTool.KNOWLEDGE)

    return CopilotPlan(
        question=text,
        tools=tuple(dict.fromkeys(tools)),
        machine_id=machine_id,
        window=window,
        search_query=text if CopilotTool.KNOWLEDGE in tools else None,
    )


def _machine_id(text: str) -> MachineId | None:
    """Return the first well-formed machine identifier in `text`.

    Well-formed rather than known: whether a machine is registered is a
    question for storage, and answering it here would put a directory lookup in
    a pure function. An identifier that turns out not to exist surfaces as the
    tool's own `MachineNotFoundError`, which is the honest place for it.
    """
    found = _ID_IN_TEXT.search(text.upper())
    return MachineId(found.group(0)) if found else None


def _window(lowered: str) -> TimeWindow:
    """Return the window the question names, or the default."""
    for phrases, window in _WINDOW_PHRASES:
        if any(phrase in lowered for phrase in phrases):
            return window
    return DEFAULT_WINDOW


def _mentions_a_window(lowered: str) -> bool:
    """Whether the question names a period of time."""
    return any(phrase in lowered for phrases, _ in _WINDOW_PHRASES for phrase in phrases)


def _mentions_a_signal(lowered: str) -> bool:
    """Whether the question names a measured signal."""
    return any(signal in lowered for signal in _SIGNAL_WORDS)


#: Spelled as a person writes them, not as the schema spells them.
_SIGNAL_WORDS: Sequence[str] = (
    "vibration",
    "temperature",
    "temp",
    "rpm",
    "speed",
    "current",
    "amp",
    "load",
    "voltage",
)
