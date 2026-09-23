"""Composition root.

The only package permitted to know every layer at once. It selects concrete
adapters for the domain's ports and hands the wired result to the presentation
layer, which is why nothing else has to branch on which implementation it is
running against.
"""

from api.composition.container import Container, build_container, build_in_memory_container

__all__ = ["Container", "build_container", "build_in_memory_container"]
