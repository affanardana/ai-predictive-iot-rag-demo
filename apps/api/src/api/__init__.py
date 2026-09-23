"""AI Predictive Maintenance API.

Layering, from the inside out:

* ``api.domain`` -- entities, value objects, business rules, and the ports
  that infrastructure implements. Imports no framework.
* ``api.application`` -- use cases that orchestrate domain objects through
  those ports.
* ``api.infrastructure`` -- adapters: SQLAlchemy persistence, configuration,
  logging, system services.
* ``api.presentation`` -- FastAPI routers, schemas, and error translation.
* ``api.composition`` -- the composition root; the only package that knows
  every layer at once.

The dependency direction is enforced by the ``import-linter`` contracts in the
repository's root ``pyproject.toml``, not merely documented here.
"""

__version__ = "0.2.0"
