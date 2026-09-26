"""Model inference, as a service.

Kept out of `apps/api` so that torch does not enter the API's environment or its
container image. The API calls this over HTTP and owns everything the product
owns — the risk bands, the persistence, the domain rules.
"""

from __future__ import annotations

__all__: list[str] = []
