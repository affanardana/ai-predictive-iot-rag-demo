"""Who a request is charged to.

The most valuable file in this tier, because every mistake it guards against is
silent: a limiter keyed on the wrong address either lets one caller spend
everyone's budget or hands an attacker a key they choose by forging a header.
Neither shows up as an error anywhere -- the symptom is a demonstration that
stops answering, with nothing in the logs to explain it.
"""

from __future__ import annotations

import pytest
from starlette.types import Scope

from api.presentation.client_ip import UNKNOWN, TrustedProxies, bucket_key, resolve_client_ip

#: The default from `Settings`, spelled out here rather than imported: if that
#: default changes, this file should be the thing that notices.
TRUSTED = "127.0.0.1,::1,172.16.0.0/12"


@pytest.fixture
def trusted() -> TrustedProxies:
    """The deployment's declared proxies."""
    return TrustedProxies.parse(TRUSTED)


def a_scope(peer: str | None, *forwarded: str) -> Scope:
    """Build a minimal HTTP scope with the given peer and forwarded headers."""
    headers = [(b"x-forwarded-for", value.encode()) for value in forwarded]
    return {
        "type": "http",
        "client": (peer, 1234) if peer is not None else None,
        "headers": headers,
    }


def test_the_default_trusts_the_docker_bridge() -> None:
    """The gateway, not loopback, is what the container actually sees.

    `compose.yaml` publishes the port through Docker, so the socket peer is the
    bridge address. A trusted set of `127.0.0.1,::1` -- which is uvicorn's own
    default -- would mean no forwarded header is ever believed and every visitor
    on earth shares one bucket. That failure has no symptom until the budget is
    gone, which is why it is asserted rather than reviewed.
    """
    proxies = TrustedProxies.parse(TRUSTED)

    assert proxies.contains("172.17.0.1")
    assert proxies.contains("172.18.0.1")


def test_an_untrusted_peer_is_the_client_whatever_it_claims(
    trusted: TrustedProxies,
) -> None:
    """A forged header is refused when the peer is not a declared proxy.

    This is the property that makes the limiter work at all: the caller cannot
    choose their own key, and cannot escape their own by the same trick.
    """
    scope = a_scope("203.0.113.9", "1.2.3.4")

    assert resolve_client_ip(scope, trusted) == "203.0.113.9"


def test_a_trusted_peer_has_its_forwarded_chain_believed(
    trusted: TrustedProxies,
) -> None:
    """Caddy's own observation, passed through."""
    scope = a_scope("172.17.0.1", "203.0.113.9")

    assert resolve_client_ip(scope, trusted) == "203.0.113.9"


def test_the_rightmost_non_proxy_entry_wins(trusted: TrustedProxies) -> None:
    """Not the leftmost, which is the entry a caller controls.

    Each hop appends the address of whoever connected to it, so the rightmost
    entry that is not itself a declared proxy is the one the nearest proxy wrote
    down itself. Reading it the conventional nginx way round would hand an
    attacker the key by forging one header.
    """
    scope = a_scope("172.17.0.1", "1.2.3.4, 203.0.113.9")

    assert resolve_client_ip(scope, trusted) == "203.0.113.9"


def test_a_chain_ending_in_a_proxy_falls_back_to_the_client(
    trusted: TrustedProxies,
) -> None:
    """A proxy inside the trusted range appended last is skipped, not returned."""
    scope = a_scope("172.17.0.1", "203.0.113.9, 172.18.0.5")

    assert resolve_client_ip(scope, trusted) == "203.0.113.9"


def test_every_forwarded_header_is_read(trusted: TrustedProxies) -> None:
    """`getlist`, not `get`.

    A chain that grew across two headers arrives as two, and reading only the
    first would truncate the chain and return the entry an attacker controls.
    """
    scope = a_scope("172.17.0.1", "1.2.3.4", "203.0.113.9")

    assert resolve_client_ip(scope, trusted) == "203.0.113.9"


def test_garbage_never_becomes_a_key(trusted: TrustedProxies) -> None:
    """An unparseable entry falls back to the peer rather than to the string."""
    scope = a_scope("172.17.0.1", "not-an-ip")

    assert resolve_client_ip(scope, trusted) == "172.17.0.1"


def test_no_peer_and_no_header_is_unknown() -> None:
    """One bucket for "cannot tell", which is the safe direction."""
    assert resolve_client_ip(a_scope(None), TrustedProxies.parse(TRUSTED)) == UNKNOWN


def test_ipv6_is_grouped_by_its_64(trusted: TrustedProxies) -> None:
    """A residential IPv6 host holds a whole /64 and can rotate the low bits.

    Keyed on the full address, one attacker retires a budget every few
    milliseconds: a rate limiter that does not limit.
    """
    first = bucket_key("2001:db8:1:2:3:4:5:6")
    second = bucket_key("2001:db8:1:2:ffff::9")

    assert first == second == "2001:db8:1:2::/64"


def test_ipv4_is_used_whole() -> None:
    """A /24 would put unrelated subscribers of one ISP in a single bucket."""
    assert bucket_key("203.0.113.9") == "203.0.113.9"
    assert bucket_key("203.0.113.10") != bucket_key("203.0.113.9")


def test_an_ipv4_mapped_address_is_unwrapped() -> None:
    """`::ffff:203.0.113.9` and `203.0.113.9` are the same client."""
    assert bucket_key("::ffff:203.0.113.9") == "203.0.113.9"


def test_a_malformed_address_lands_in_the_unknown_bucket() -> None:
    """Rather than in a bucket named after whatever was sent."""
    assert bucket_key("not-an-ip") == UNKNOWN


def test_an_invalid_trusted_entry_fails_at_startup() -> None:
    """A typo in the trusted list is a startup failure, not a silent no-op.

    The failure mode of trusting nobody is every visitor sharing one budget, and
    it would otherwise present as a demonstration that stops answering.
    """
    with pytest.raises(ValueError):
        TrustedProxies.parse("127.0.0.1,not-a-network")
