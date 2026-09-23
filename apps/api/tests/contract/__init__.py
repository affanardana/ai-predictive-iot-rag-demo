"""Repository contract suite.

One set of behavioural expectations, run against every adapter. This is what
makes the in-memory test double trustworthy: it cannot quietly diverge from the
SQL implementation, because both are held to the same assertions.
"""
