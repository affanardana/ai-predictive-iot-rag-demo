"""Seed derivation, and the reproducibility it exists to guarantee."""

from __future__ import annotations

import subprocess
import sys

from simulator.domain.seeding import derive_seed


def test_the_same_parts_always_give_the_same_seed() -> None:
    """The property everything else in the package depends on."""
    assert derive_seed("M003", 7) == derive_seed("M003", 7)


def test_different_parts_give_different_seeds() -> None:
    """Machines and ticks must not share a stream, or a fleet would be one machine.

    Every (machine, tick) pair must map to its own seed. Two machines sharing a
    stream would generate identical telemetry; two ticks sharing one would
    generate identical samples.
    """
    seeds = {
        derive_seed(1234, machine_id, index)
        for machine_id in ("M001", "M002", "M003")
        for index in range(50)
    }

    assert len(seeds) == 150


def test_parts_do_not_run_together() -> None:
    """`("ab", "c")` and `("a", "bc")` must not collide.

    The separator in the derivation is what prevents it. Without one, a session
    whose id ended in a digit and whose machine id began with one would share a
    stream with a different pair — a collision that would be near-impossible to
    diagnose from the symptom.
    """
    assert derive_seed("ab", "c") != derive_seed("a", "bc")


def test_the_seed_stable_across_processes() -> None:
    """The property `hash()` would fail, and the reason this module exists.

    Python randomises string hashing per process. A seed derived from `hash()`
    would therefore differ on every program start, making every dataset and
    every demonstration irreproducible — with no error to show for it, only
    results that quietly do not match.

    Proving this needs a fresh interpreter. An in-process assertion would pass
    whichever implementation was used, which is exactly why the subprocess is
    worth its cost.
    """
    script = "from simulator.domain.seeding import derive_seed; print(derive_seed('M003', 7))"
    runs = [
        _run_in_fresh_interpreter(script),
        _run_in_fresh_interpreter(script),
    ]

    assert runs[0] == runs[1]
    assert runs[0] == str(derive_seed("M003", 7))


def test_the_seed_is_wide_enough_to_avoid_collisions() -> None:
    """64 bits, as documented."""
    assert derive_seed("M003") >= 0
    assert derive_seed("M003").bit_length() > 60


def _run_in_fresh_interpreter(script: str) -> str:
    """Run `script` in a new Python process and return its stdout."""
    completed = subprocess.run(  # noqa: S603 - fixed script, no user input
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stdout.strip()
