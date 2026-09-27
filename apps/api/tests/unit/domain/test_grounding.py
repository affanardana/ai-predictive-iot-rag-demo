"""Checking an answer's numbers against the evidence it was given."""

from __future__ import annotations

from api.domain.services.grounding import ungrounded_numbers

EVIDENCE = [
    "M003 vibration was 1.42 and is now 2.31 mm/s.",
    "The model put the 60-minute failure probability at 0.81.",
]


def test_a_faithful_answer_is_clean() -> None:
    """Quoting the evidence verbatim is not a finding."""
    answer = "Vibration rose from 1.42 to 2.31 mm/s, and the failure probability is 0.81."

    assert ungrounded_numbers(answer, EVIDENCE) == ()


def test_a_fabricated_number_is_caught() -> None:
    """The failure the check exists for: a plausible number nobody supplied."""
    answer = "Vibration rose to 4.7 mm/s, which is above the 3.5 limit."

    assert set(ungrounded_numbers(answer, EVIDENCE)) == {"4.7", "3.5"}


def test_rounding_is_not_a_fabrication() -> None:
    """A model that writes 1.4 for 1.42 rounded rather than invented.

    A check that flagged this would be noise, and noise is what makes a check
    get switched off.
    """
    assert ungrounded_numbers("Vibration reached about 1.4 mm/s.", EVIDENCE) == ()


def test_a_percentage_form_is_the_same_number() -> None:
    """A probability of 0.81 is written "81%" by anyone describing it."""
    assert ungrounded_numbers("The failure probability is 81%.", EVIDENCE) == ()


def test_a_number_the_model_was_never_given_is_reported_once() -> None:
    """Deduplicated, so the response lists the value rather than every mention."""
    answer = "Risk is 0.55. I repeat, 0.55, and the horizon is 0.55 hours."

    assert ungrounded_numbers(answer, EVIDENCE) == ("0.55",)


def test_no_evidence_means_no_finding() -> None:
    """With nothing numeric supplied, the check cannot distinguish anything.

    Reporting every number as ungrounded would mark a refusal's own prose as
    fabricated, which is the one answer the system writes itself.
    """
    assert ungrounded_numbers("There is nothing recorded for this machine.", []) == ()
    assert ungrounded_numbers("Vibration is 1.42.", ["no numbers here"]) == ()


def test_the_sign_is_part_of_the_value() -> None:
    """A rise and a fall are different claims, not the same number."""
    assert ungrounded_numbers("Vibration changed by -0.42.", ["change -0.42"]) == ()
    assert ungrounded_numbers("Vibration changed by -0.42.", ["change 0.42"]) == ("-0.42",)
