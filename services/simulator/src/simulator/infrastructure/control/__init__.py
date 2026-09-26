"""The simulator's HTTP control surface.

Holds `server.py`, which imports FastAPI -- the `control` extra. Nothing imports
this package except `simulator.control`, which is itself reached only by running
it, so the default offline tier never touches FastAPI. See that module for why
the wiring sits there rather than here.
"""
