"""What kind of maintenance document this is.

PRD section 17 names six *initial* categories for the maintenance corpus. The
word "initial" is load-bearing: it is an opening set rather than a closed one,
and the corpus this project actually ships contains ten documents, four of which
match none of the six.

Those four are named here rather than funnelled into an `OTHER` member. An enum
whose escape hatch is `OTHER` makes the database CHECK constraint meaningless for
exactly the documents that most need classifying -- the ones nobody anticipated
-- and a free-text category would give up the constraint entirely, along with any
guarantee that a category filter means what it says.
"""

from __future__ import annotations

from enum import StrEnum


class DocumentCategory(StrEnum):
    """The kind of maintenance knowledge a document carries."""

    # --- PRD section 17's six named categories -----------------------------
    MOTOR_MAINTENANCE_MANUAL = "MOTOR_MAINTENANCE_MANUAL"
    BEARING_INSPECTION = "BEARING_INSPECTION"
    OVERHEATING_TROUBLESHOOTING = "OVERHEATING_TROUBLESHOOTING"
    VIBRATION_DIAGNOSIS = "VIBRATION_DIAGNOSIS"
    ELECTRICAL_SAFETY = "ELECTRICAL_SAFETY"
    PREVENTIVE_MAINTENANCE_SCHEDULE = "PREVENTIVE_MAINTENANCE_SCHEDULE"

    # --- Content the corpus holds that PRD section 17 does not name --------
    #
    # `Lubrication Procedure` is the one worth knowing about. It sits in
    # `dummy_pdfs/procedure/`, beside the bearing SOP, and it is a lubrication
    # procedure -- not a bearing inspection SOP, and not a motor maintenance
    # manual. Categorising it by its parent directory would have filed it under
    # the bearing SOPs and *looked correct*, which is the failure mode the
    # manifest exists to prevent.
    SENSOR_CALIBRATION = "SENSOR_CALIBRATION"
    LUBRICATION = "LUBRICATION"
    STATOR_WINDING_TEST = "STATOR_WINDING_TEST"
    INCIDENT_RESPONSE = "INCIDENT_RESPONSE"


#: The six categories PRD section 17 names by hand.
#:
#: Declared separately so a test can assert the ingested corpus populates every
#: one of them. Without that, the enum could quietly grow past the specification
#: and the fixture could drift below it, and neither would be visible from
#: reading either one alone.
PRD_DOCUMENT_CATEGORIES: frozenset[DocumentCategory] = frozenset(
    {
        DocumentCategory.MOTOR_MAINTENANCE_MANUAL,
        DocumentCategory.BEARING_INSPECTION,
        DocumentCategory.OVERHEATING_TROUBLESHOOTING,
        DocumentCategory.VIBRATION_DIAGNOSIS,
        DocumentCategory.ELECTRICAL_SAFETY,
        DocumentCategory.PREVENTIVE_MAINTENANCE_SCHEDULE,
    }
)
