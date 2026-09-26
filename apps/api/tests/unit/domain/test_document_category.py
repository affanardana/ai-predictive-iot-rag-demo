"""The maintenance document category set."""

from __future__ import annotations

from api.domain.value_objects.document_category import (
    PRD_DOCUMENT_CATEGORIES,
    DocumentCategory,
)


def test_the_prd_categories_are_all_present() -> None:
    """PRD section 17 names six categories the corpus must cover."""
    assert {category.value for category in PRD_DOCUMENT_CATEGORIES} == {
        "MOTOR_MAINTENANCE_MANUAL",
        "BEARING_INSPECTION",
        "OVERHEATING_TROUBLESHOOTING",
        "VIBRATION_DIAGNOSIS",
        "ELECTRICAL_SAFETY",
        "PREVENTIVE_MAINTENANCE_SCHEDULE",
    }


def test_the_prd_categories_are_a_subset_of_the_enum() -> None:
    """The named list and the enum cannot drift apart."""
    assert set(DocumentCategory) >= PRD_DOCUMENT_CATEGORIES


def test_there_is_no_catch_all_member() -> None:
    """An `OTHER` member would make the storage constraint meaningless.

    Documents nobody anticipated are exactly the ones needing classification,
    and funnelling them into one bucket would let a category filter quietly
    mean less than it says.
    """
    assert not {"OTHER", "UNKNOWN", "MISC"} & set(DocumentCategory)
