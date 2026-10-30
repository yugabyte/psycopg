"""
DB-24801 (per_txn variant): a connect that passes ``wait_for_dispatch``
just before a drain arms its barrier, and finishes its TCP connect after
the drain's last sweep of ``tracked_conns``, must not hand the app a conn
on the outgoing cluster.

Seen in stress test d67c6549 (Phase 2 converged_clean, 2026-10-30): the
barrier armed at mono 406.721, Phase 1 drained by 406.823, and seven
per-transaction connects then committed on A between 406.85 and 407.15
while Phase 2 held routing until 420.086.

The TCP connect is stubbed; the drain is simulated inside the stub, which
is exactly the window the race lives in.
"""

# Copyright (C) 2026 Yugabyte

from __future__ import annotations

import threading
import time

import pytest

from psycopg.yb.health import HealthResult
from psycopg.yb.registry import FailoverGroup

pytestmark = pytest.mark.yb_unit

DSN = "host=p1 load_balance_hosts=true yb.failover.secondaryClusterHosts=s1"


class _StubConn:
    _yb_uuid: str | None = None
    _yb_host: str | None = None
    _yb_cluster: str | None = None

    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _AsyncStubConn(_StubConn):
    async def close(self) -> None:  # type: ignore[override]
        self.closed = True


def _install_group(reg, fake_state) -> FailoverGroup:
    from psycopg.yb.circuit_breaker import AlwaysHealthyCircuitBreaker

    p = fake_state(("p1", "aws", "us-west", "us-west-1a", "primary"), uuid="P")
    s = fake_state(("s1", "aws", "us-east", "us-east-1a", "primary"), uuid="S")
    group = FailoverGroup(
        primary=p, secondary=s, lock=threading.Lock(),
        primary_status=HealthResult.HEALTHY,
        secondary_status=HealthResult.HEALTHY,
        cooldown_s=0,
        primary_circuit_breaker=AlwaysHealthyCircuitBreaker(),
        secondary_circuit_breaker=AlwaysHealthyCircuitBreaker(),
    )
    reg._clusters[p.uuid] = p
    reg._clusters[s.uuid] = s
    reg._failover_groups[p.uuid] = group
    return group


def _patch(monkeypatch, reg, group, connect_hook, aconnect_hook=None):
    from psycopg import AsyncConnection, Connection
    from psycopg.yb.policy.cluster_aware import ClusterAwarePolicy

    async def fake_aget(*a, **k):
        return group

    monkeypatch.setattr(reg, "aget_or_bootstrap_failover_group", fake_aget)
    monkeypatch.setattr(
        reg, "get_or_bootstrap_failover_group", lambda *a, **k: group
    )

    def fake_pick(self, state, attempted, ttl):
        for h, n in state.nodes.items():
            if h not in attempted:
                return n
        return None

    monkeypatch.setattr(ClusterAwarePolicy, "get_least_loaded_server", fake_pick)
    monkeypatch.setattr(
        Connection, "_connect_plain",
        classmethod(lambda cls, *a, **kw: connect_hook(kw["host"])),
    )
    if aconnect_hook is not None:
        async def fake_aconnect(cls, *a, **kw):
            return aconnect_hook(kw["host"])

        monkeypatch.setattr(
            AsyncConnection, "_aconnect_plain", classmethod(fake_aconnect)
        )


def _flip(group: FailoverGroup) -> None:
    """One complete drain: arm, flip routing to secondary, release."""
    group.pause_dispatch()
    group.force_primary_status(HealthResult.UNHEALTHY)
    group.resume_dispatch()


def test_full_drain_during_tcp_connect_reroutes_to_new_active(
    fresh_registry, fake_state, monkeypatch
):
    group = _install_group(fresh_registry, fake_state)
    made: list[tuple[str, _StubConn]] = []

    def connect(host):
        c = _StubConn()
        made.append((host, c))
        if len(made) == 1:
            _flip(group)  # the whole drain lands inside this TCP connect
        return c

    _patch(monkeypatch, fresh_registry, group, connect)
    from psycopg import Connection

    conn = Connection.connect(DSN)

    assert conn._yb_cluster == "secondary"
    assert conn._yb_uuid == "S"
    first_host, first = made[0]
    assert first_host == "p1"
    assert first.closed
    assert first not in group.primary.tracked_conns
    assert conn in group.secondary.tracked_conns


def test_barrier_still_armed_at_track_waits_then_reroutes(
    fresh_registry, fake_state, monkeypatch
):
    """Barrier armed mid-connect and still held (Phase 2 lag wait) when the
    conn is tracked: the retry must block until the flip, not reuse A."""
    group = _install_group(fresh_registry, fake_state)
    made: list[_StubConn] = []

    def connect(host):
        c = _StubConn()
        made.append(c)
        if len(made) == 1:
            group.pause_dispatch()

            def finish():
                time.sleep(0.2)
                group.force_primary_status(HealthResult.UNHEALTHY)
                group.resume_dispatch()

            threading.Thread(target=finish, daemon=True).start()
        return c

    _patch(monkeypatch, fresh_registry, group, connect)
    from psycopg import Connection

    t0 = time.monotonic()
    conn = Connection.connect(DSN)

    assert time.monotonic() - t0 >= 0.15
    assert conn._yb_cluster == "secondary"
    assert made[0].closed


def test_no_barrier_keeps_first_conn(fresh_registry, fake_state, monkeypatch):
    group = _install_group(fresh_registry, fake_state)
    made: list[_StubConn] = []

    def connect(host):
        c = _StubConn()
        made.append(c)
        return c

    _patch(monkeypatch, fresh_registry, group, connect)
    from psycopg import Connection

    conn = Connection.connect(DSN)

    assert len(made) == 1
    assert not made[0].closed
    assert conn._yb_cluster == "primary"
    assert conn in group.primary.tracked_conns


def test_barrier_armed_before_routing_read_is_not_a_false_retry(
    fresh_registry, fake_state, monkeypatch
):
    """A drain that completed entirely before this connect started must not
    trigger a retry: the generation is read after the barrier releases."""
    group = _install_group(fresh_registry, fake_state)
    _flip(group)
    made: list[_StubConn] = []

    def connect(host):
        c = _StubConn()
        made.append(c)
        return c

    _patch(monkeypatch, fresh_registry, group, connect)
    from psycopg import Connection

    conn = Connection.connect(DSN)

    assert len(made) == 1
    assert conn._yb_cluster == "secondary"


@pytest.mark.anyio
async def test_async_full_drain_during_tcp_connect_reroutes(
    fresh_registry, fake_state, monkeypatch
):
    group = _install_group(fresh_registry, fake_state)
    made: list[_AsyncStubConn] = []

    def aconnect(host):
        c = _AsyncStubConn()
        made.append(c)
        if len(made) == 1:
            _flip(group)
        return c

    _patch(monkeypatch, fresh_registry, group, lambda h: _StubConn(), aconnect)
    from psycopg import AsyncConnection

    conn = await AsyncConnection.connect(DSN)

    assert conn._yb_cluster == "secondary"
    assert made[0].closed
    assert made[0] not in group.primary.tracked_conns
