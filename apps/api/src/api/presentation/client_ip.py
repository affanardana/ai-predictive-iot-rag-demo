"""Which address to charge a request to.

Two hops stand between a visitor and this process, and both of them exist to
make this question harder than it looks:

* the shared Caddy on the host terminates TLS and forwards to the container's
  published loopback port;
* `uvicorn`'s own `ProxyHeadersMiddleware` sits inside the app.

Caddy sets `X-Forwarded-For` to the address it saw, and **ignores an inbound
one** unless `trusted_proxies` is configured -- so the value is Caddy's own
observation rather than a caller's claim. But the port is published through
Docker, so the socket peer this process sees is the **bridge gateway**
(`172.17.0.1` and friends), not `127.0.0.1`. That single fact is why the trust
decision cannot be a constant: uvicorn's default trusted set is
`127.0.0.1,::1`, which the gateway is not, so a limiter that believed nothing
would key every visitor on earth into one bucket -- and the first person to open
the dashboard would consume the budget for everyone after them.

The rule here is therefore explicit and is a deployment fact: believe the
forwarded chain only when the peer is a declared proxy, and then take the
**rightmost entry that is not itself a declared proxy**. Each hop appends the
address of whoever connected to it, so the rightmost non-proxy entry is the one
the nearest proxy wrote down itself; anything to its left arrived in a header
the client controlled. Reading it the conventional nginx way round -- leftmost
is the client -- would hand an attacker the key by forging one header.
"""

from __future__ import annotations

from ipaddress import IPv4Address, IPv4Network, IPv6Address, IPv6Network, ip_address, ip_network

from starlette.datastructures import Headers
from starlette.types import Scope

#: Either family's network, because a trusted-proxy list mixes them freely --
#: `127.0.0.1,::1,172.16.0.0/12` is the default and is all three.
Network = IPv4Network | IPv6Network

#: Used when there is neither a peer nor a parseable forwarded entry. A single
#: bucket for "cannot tell" is the safe direction: they share a budget rather
#: than each getting one.
UNKNOWN = "unknown"

#: Addresses whose forwarding headers are believed, and the one uvicorn itself
#: trusts for the same purpose. `172.16.0.0/12` is Docker's default address
#: pool; a host that overrides `default-address-pools` in `daemon.json` needs
#: this changed, and `README.md` says so beside the variable.
DEFAULT_TRUSTED_PROXIES = "127.0.0.1,::1,172.16.0.0/12"

#: The IPv6 prefix a client is charged for. A residential connection routinely
#: holds a whole /64, so keying on the full address would let anyone retire a
#: budget every few milliseconds by rotating the low bits -- a rate limiter that
#: does not limit. A /64 is still one subscriber.
_IPV6_PREFIX_LENGTH = 64


class TrustedProxies:
    """The networks whose forwarding headers are believed."""

    def __init__(self, networks: tuple[Network, ...]) -> None:
        self._networks = networks

    @classmethod
    def parse(cls, value: str) -> TrustedProxies:
        """Parse a comma-separated list of addresses and networks.

        Raises:
            ValueError: on an entry that is neither, so a typo fails at startup
                rather than silently trusting nobody.
        """
        networks = []
        for entry in value.split(","):
            stripped = entry.strip()
            if stripped:
                networks.append(ip_network(stripped, strict=False))
        return cls(tuple(networks))

    def contains(self, address: str) -> bool:
        """Whether `address` is one of the declared proxies."""
        parsed = _parse(address)
        if parsed is None:
            return False
        return any(_in_network(parsed, network) for network in self._networks)

    def __repr__(self) -> str:
        """Show the networks, for a test failure worth reading."""
        return f"TrustedProxies({[str(network) for network in self._networks]})"


def resolve_client_ip(scope: Scope, trusted: TrustedProxies) -> str:
    """Return the address this request is charged to."""
    # Annotated rather than inferred: `Scope` is a plain mapping, so the
    # address arrives as `Any` and returning it directly would switch the
    # return type off at this function's edge.
    peer = scope.get("client")
    peer_host: str | None = peer[0] if peer else None
    if not peer_host:
        return UNKNOWN
    if not trusted.contains(peer_host):
        # The peer is not a declared proxy, so it is the client, whatever it
        # wrote in a header. This is the branch that makes the limiter work: an
        # attacker cannot choose their own key by forging a header, and cannot
        # escape their own by the same trick.
        return peer_host

    for candidate in reversed(_forwarded_chain(scope)):
        parsed = _parse(candidate)
        if parsed is not None and not trusted.contains(candidate):
            return candidate
    # Every hop was a proxy, or nothing parsed. The peer is the honest answer.
    return peer_host


def bucket_key(address: str) -> str:
    """Return the grouping used for budgeting.

    IPv6 is grouped by its /64 and IPv4 is used whole. A /24 for IPv4 would put
    unrelated subscribers of one ISP in a single bucket, which is the failure
    this file exists to avoid -- in miniature, and on purpose.
    """
    parsed = _parse(address)
    if parsed is None:
        return UNKNOWN
    if isinstance(parsed, IPv6Address):
        network = ip_network(f"{parsed}/{_IPV6_PREFIX_LENGTH}", strict=False)
        return str(network)
    return str(parsed)


def _forwarded_chain(scope: Scope) -> list[str]:
    """Return every forwarded address, in the order the hops wrote them.

    `getlist`, not `get`: a chain that grew across two `X-Forwarded-For` headers
    arrives as two headers, and reading only the first would silently truncate
    the chain -- and pick the entry an attacker controls.
    """
    chain: list[str] = []
    for header in Headers(scope=scope).getlist("x-forwarded-for"):
        chain.extend(part.strip() for part in header.split(","))
    return [entry for entry in chain if entry]


def _in_network(address: IPv4Address | IPv6Address, network: Network) -> bool:
    """Whether `address` falls in `network`.

    Written out rather than `address in network`, because the two unions do not
    typecheck against each other. The version-mismatch case answers `False`
    here, which is what the stdlib does too -- a v4 address is not in a v6
    network -- but the explicit form is what a reader can check.
    """
    if isinstance(address, IPv4Address) and isinstance(network, IPv4Network):
        return address in network
    if isinstance(address, IPv6Address) and isinstance(network, IPv6Network):
        return address in network
    return False


def _parse(address: str) -> IPv4Address | IPv6Address | None:
    """Parse an address, unwrapping an IPv4-mapped IPv6 one.

    `::ffff:203.0.113.5` and `203.0.113.5` are the same client, and a
    dual-stack proxy may write either. Left wrapped, the v4 client would be
    grouped under `::ffff:0:0/64` together with every other v4 client, and
    `bucket_key` would return a /64 that means "all of IPv4".
    """
    try:
        parsed = ip_address(address)
    except ValueError:
        return None
    if isinstance(parsed, IPv6Address) and parsed.ipv4_mapped is not None:
        return parsed.ipv4_mapped
    return parsed
