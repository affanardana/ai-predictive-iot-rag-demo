"""Which columns a model may see, and why nothing else qualifies.

`telemetry.parquet` carries ten columns: four identifiers and six signals. The
obvious reading of a file like that is "drop the ids, keep the numbers". That is
the wrong shape of decision, because it means every column added to the
telemetry stream in future is a feature by default — and the two most dangerous
columns in this particular file are not numbers at all.

So the feature set is a **whitelist**, and a test asserts the training matrix's
columns equal it exactly. Adding a feature is then a deliberate act with a
review, which is the point.

The four identifiers, and what each would give away:

`event_id`
    Built as ``{session_id}-{machine_id}-{index:08d}``. The trailing digits are
    the tick's position within its life, and every degradation life ends at the
    onset of failure — so that position is a direct proxy for time-to-failure.
    A model handed this column would not learn physics; it would learn to read
    an eight-digit clock.

`session_id`
    Identifies the life. Every degradation life produces a positive example and
    no `NORMAL` life ever does, so membership alone predicts the class before a
    single signal is read.

`machine_id`
    A machine's nominal operating point, ambient temperature and susceptibility
    are all stable across its lives. Splitting by machine is what stops a model
    memorising those offsets — but only if the identifier is not itself an
    input, which would hand the offsets back.

`recorded_at`
    A timestamp. Any function of it — hour of day, minutes since the life
    began, row order — reconstructs elapsed time, and elapsed time is close to
    the answer for a simulator whose degradation is a deterministic function of
    it.

The identifiers are still carried through to the artifact, because Phase 4
needs them for grouped evaluation and per-event reporting. They are simply never
inputs.
"""

from __future__ import annotations

from simulator.domain.readings import SIGNAL_NAMES

#: The model's inputs, in the order the artifact stores them. Sourced from the
#: simulator rather than restated, so a signal added there cannot silently
#: diverge from the one here.
FEATURE_COLUMNS: tuple[str, ...] = SIGNAL_NAMES

#: The identifier columns present in `telemetry.parquet`. Named here so a test
#: can assert none of them reached the feature set, rather than relying on
#: someone noticing an extra column.
IDENTIFIER_COLUMNS: tuple[str, ...] = (
    "event_id",
    "machine_id",
    "recorded_at",
    "session_id",
)


def is_feature(column: str) -> bool:
    """Whether `column` may be used as a model input."""
    return column in FEATURE_COLUMNS
