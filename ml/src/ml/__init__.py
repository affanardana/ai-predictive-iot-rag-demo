"""Dataset engineering for the predictive maintenance platform.

Phase 3 of `MASTERPLAN.md`: turn the simulator's output into a labelled,
leakage-free, reproducible training set.

The split between this package and `simulator` follows the same rule as
everywhere else in this repository. `simulator` decides what a machine *does*;
this package decides what a model is *allowed to see* of it. The two are
separate deployables — this one runs on Colab or Modal — and an import-linter
contract keeps it that way.
"""

__all__: list[str] = []
