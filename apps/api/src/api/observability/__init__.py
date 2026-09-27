"""Metrics, for the two layers that record them.

A leaf package, sitting at the same rung as `api.request_context` and for the
same reason: infrastructure's adapters record a dependency failure and
presentation's route exposes the result, and neither layer may import the other.
Placed lowest, this can import nothing from `api` while staying importable by
everything above it.

**Nothing below this line knows what a metric is.** Neither `api.domain` nor
`api.application` imports this package or `prometheus_client` -- the domain
contract forbids the latter outright -- so the rules stay testable without a
registry and a metric can be added without touching a use case. Every recording
happens where the fact already exists: an adapter that saw a timeout, a route
that holds a finished answer, a probe that just failed.
"""
