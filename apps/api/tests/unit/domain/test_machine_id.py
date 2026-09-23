"""Machine identifier value object."""

from __future__ import annotations

import pytest

from api.domain.errors import DomainValidationError
from api.domain.value_objects.machine_id import MachineId


@pytest.mark.parametrize("value", ["M001", "M003", "PUMP012", "A999"])
def test_accepts_well_formed_identifiers(value: str) -> None:
    """Identifiers matching the documented pattern are accepted."""
    assert MachineId(value).value == value


@pytest.mark.parametrize(
    "value",
    [
        "",  # blank
        "m001",  # lowercase
        "M01",  # too few digits
        "M0001",  # too many digits
        "MOTOR1",  # one letter too many
        "M 003",  # embedded space
        "M-003",  # separator
        "003M",  # reversed
    ],
)
def test_rejects_malformed_identifiers(value: str) -> None:
    """Malformed identifiers fail at the boundary.

    Validating here means nothing downstream has to defend against a bad
    identifier, and the API returns a 422 rather than an empty result that
    looks like a machine with no data.
    """
    with pytest.raises(DomainValidationError):
        MachineId(value)


def test_stringifies_to_the_raw_identifier() -> None:
    """The identifier renders as its bare value."""
    assert str(MachineId("M003")) == "M003"


def test_compares_by_value() -> None:
    """Two identifiers with the same value are interchangeable."""
    assert MachineId("M003") == MachineId("M003")
    assert MachineId("M003") != MachineId("M004")
