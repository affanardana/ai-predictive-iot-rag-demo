"""Failures that belong to the transport rather than to the domain.

The domain never raises these. Authentication is a property of how a request
arrived, not of any business rule, so adding it to `api.domain.errors` would
make the domain aware of a concern it does not have -- and the error catalog
there is typed `Mapping[type[DomainError], ErrorMapping]` precisely to keep that
boundary meaningful.

They live here instead so they can still be rendered through the shared error
envelope rather than FastAPI's default ``{"detail": ...}``, which has a
different shape from every other failure this API returns.
"""

from __future__ import annotations


class AuthenticationError(Exception):
    """The caller did not present the credential a route requires.

    Carries no detail deliberately: whether the token was absent or merely wrong
    is not something a caller should be able to probe for.
    """
