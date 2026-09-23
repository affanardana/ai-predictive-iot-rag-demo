"""Deterministic seed derivation.

Reproducibility is a product requirement (`MASTERPLAN.md` §3.5), not a
convenience: a demonstration, a dataset, and an evaluation must all be
regenerable from a stated seed. Everything random in this package is therefore
seeded through `derive_seed`, and nothing else.
"""

from __future__ import annotations

import hashlib

#: 64 bits. Wide enough that collisions between derived streams are not a
#: practical concern at any scale this project generates.
_SEED_BYTES = 8


def derive_seed(*parts: object) -> int:
    """Return a stable integer seed derived from `parts`.

    Uses SHA-256 rather than the built-in `hash()`. Python randomises string
    hashing per process, so `hash("M003")` differs between runs and would give a
    different series every time the program started — precisely the property
    this module exists to prevent. The failure is invisible until someone tries
    to reproduce a demonstration, which is why it is worth the extra line.
    """
    material = ":".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:_SEED_BYTES], "big")
