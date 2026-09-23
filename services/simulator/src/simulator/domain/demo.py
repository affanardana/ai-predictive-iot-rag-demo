"""The canonical demonstration.

PRD §12 requires that a specific combination reproduce a recognisable
degradation sequence:

    Machine: M003
    Scenario: Bearing Degradation
    Mode: Demo

A demonstration that cannot be reproduced is not a demonstration, so the values
live here as named constants rather than as literals in the command line — a
test asserts that this exact combination produces the same series every time,
and a reviewer can point at one line to see what "demo mode" means.
"""

from __future__ import annotations

from datetime import timedelta

from simulator.domain.scenario import Scenario

#: Fixed machine from the PRD's demonstration script.
DEMO_MACHINE_ID = "M003"

#: The failure mode the demonstration shows.
DEMO_SCENARIO = Scenario.BEARING_DEGRADATION

#: A fixed seed. Arbitrary, but it must never change: altering it would change
#: every value the demonstration produces.
DEMO_SEED = 20_260_923

#: Enough simulated time for the degradation ramp to reach its steep section and
#: for the machine to cross into imminent failure.
DEMO_DURATION = timedelta(minutes=120)
