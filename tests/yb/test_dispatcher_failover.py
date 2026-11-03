"""
Unit tests for the dispatcher's xCluster failover routing (Phase 5).

These verify that ``Connection.connect`` / ``AsyncConnection.connect`` read
``FailoverGroup.status`` and route the connect to either ``group.primary``
or ``group.secondary`` accordingly. The tests stub out the actual TCP
connect via ``_connect_plain`` / ``_aconnect_plain`` so no real cluster is
needed.

Auto-tagged ``yb_unit`` via the filename — wait. Actually, the auto-marker
hook (``conftest.py`` lines 88–90) only matches filenames starting with
``test_params``, ``test_node``, ``test_policy``, ``test_registry``. Our
filename is ``test_dispatcher_failover.py``. We mark ``yb_unit`` explicitly.
"""

# Copyright (C) 2026 Yugabyte

from __future__ import annotations

import threading

import pytest

from psycopg.yb.health import HealthResult
from psycopg.yb.params import YBParams
from psycopg.yb.registry import ClusterRegistry, FailoverGroup


# This file's tests don't touch a real DB. Mark as yb_unit so they run with
# `pytest -m yb_unit`.
pytestmark = pytest.mark.yb_unit


# ----------------------------------------------------------------- helpers

class _StubConn:
    """A tiny stand-in for psycopg.Connection / AsyncConnection. Just
    enough to receive the tagging assignments the dispatcher does."""

    _yb_uuid: str | None = None
    _yb_host: str | None = None
    _yb_cluster: str | None = None


def _install_group(reg, fake_state, cooldown_s=0):
    """Install a FailoverGroup directly, bypassing bootstrap. Attaches
    ``AlwaysHealthyCircuitBreaker`` to both slots so the dispatcher's
    fail-fast check passes — these tests exercise routing, not health
    detection; the ``primary_status`` / ``secondary_status`` fields are
    driven by ``force_*_status`` directly."""
    from psycopg.yb.circuit_breaker import AlwaysHealthyCircuitBreaker
    p = fake_state(("p1", "aws", "us-west", "us-west-1a", "primary"), uuid="P")
    s = fake_state(("s1", "aws", "us-east", "us-east-1a", "primary"), uuid="S")
    group = FailoverGroup(
        primary=p, secondary=s, lock=threading.Lock(),
        primary_status=HealthResult.HEALTHY, secondary_status=HealthResult.HEALTHY, cooldown_s=cooldown_s,
        primary_circuit_breaker=AlwaysHealthyCircuitBreaker(),
        secondary_circuit_breaker=AlwaysHealthyCircuitBreaker(),
    )
    reg._clusters[p.uuid] = p
    reg._clusters[s.uuid] = s
    reg._failover_groups[p.uuid] = group
    return group


def _patch_dispatcher_internals(monkeypatch, fresh_registry, group):
    """Monkeypatch the registry + connect path so the dispatcher reaches
    `conn._yb_cluster = active_cluster` without doing real I/O.

    Returns the list to which routed-conn cluster tags are appended."""
    routed: list[str | None] = []

    # 1. bootstrap path — always return our installed group.
    async def fake_aget(yb_params, conninfo, kwargs):
        return group
    def fake_get(yb_params, conninfo, kwargs):
        return group
    monkeypatch.setattr(
        fresh_registry, "aget_or_bootstrap_failover_group", fake_aget
    )
    monkeypatch.setattr(
        fresh_registry, "get_or_bootstrap_failover_group", fake_get
    )

    # 2. refresh — no-op.
    async def fake_arefresh(state, interval_s):
        return None
    def fake_refresh(state, interval_s):
        return None
    monkeypatch.setattr(fresh_registry, "arefresh_if_stale", fake_arefresh)
    monkeypatch.setattr(fresh_registry, "refresh_if_stale", fake_refresh)

    # 3. Stub the actual TCP connect.
    from psycopg import AsyncConnection, Connection

    async def fake_aconnect_plain(cls, *a, **kw):
        return _StubConn()
    def fake_connect_plain(cls, *a, **kw):
        return _StubConn()
    monkeypatch.setattr(
        AsyncConnection, "_aconnect_plain", classmethod(fake_aconnect_plain)
    )
    monkeypatch.setattr(
        Connection, "_connect_plain", classmethod(fake_connect_plain)
    )

    # 4. Replace the policy's pick with a deterministic node from the active
    # state — the dispatcher passes `state` to `build_policy(...).get_least_loaded_server`.
    from psycopg.yb.node import NodeInfo
    from psycopg.yb.policy.cluster_aware import ClusterAwarePolicy

    def fake_pick(self, state, attempted, ttl):
        for h, n in state.nodes.items():
            if h not in attempted:
                return n
        return None
    monkeypatch.setattr(
        ClusterAwarePolicy, "get_least_loaded_server", fake_pick
    )

    return routed


# ----------------------------------------------------------------- sync dispatcher

def test_sync_dispatcher_routes_to_primary_when_healthy(
    fresh_registry, fake_state, monkeypatch
):
    group = _install_group(fresh_registry, fake_state)
    _patch_dispatcher_internals(monkeypatch, fresh_registry, group)

    from psycopg import Connection
    conn = Connection.connect(
        "host=p1 load_balance_hosts=true "
        "yb.failover.secondaryClusterHosts=s1"
    )
    assert conn._yb_cluster == "primary"
    assert conn._yb_uuid == "P"


def test_sync_dispatcher_routes_to_secondary_when_unhealthy(
    fresh_registry, fake_state, monkeypatch
):
    group = _install_group(fresh_registry, fake_state)
    group.force_primary_status(HealthResult.UNHEALTHY)
    _patch_dispatcher_internals(monkeypatch, fresh_registry, group)

    from psycopg import Connection
    conn = Connection.connect(
        "host=p1 load_balance_hosts=true "
        "yb.failover.secondaryClusterHosts=s1"
    )
    assert conn._yb_cluster == "secondary"
    assert conn._yb_uuid == "S"


def test_sync_dispatcher_no_xcluster_when_secondary_hosts_absent(
    fresh_registry, fake_state, monkeypatch
):
    """If xcluster_enabled is False, the dispatcher must NOT consult the
    FailoverGroup at all — conn._yb_cluster stays None."""
    # Force the existing single-cluster bootstrap path: install a state
    # directly, return it from `get_or_bootstrap`.
    p = fake_state(("p1", "aws", "us-west", "us-west-1a", "primary"), uuid="P")
    fresh_registry._clusters["P"] = p

    def fake_get(key, conninfo, kwargs):
        return p
    def fake_refresh(state, interval_s):
        return None
    monkeypatch.setattr(fresh_registry, "get_or_bootstrap", fake_get)
    monkeypatch.setattr(fresh_registry, "refresh_if_stale", fake_refresh)

    from psycopg import Connection
    from psycopg.yb.node import NodeInfo
    from psycopg.yb.policy.cluster_aware import ClusterAwarePolicy

    def fake_pick(self, state, attempted, ttl):
        for h, n in state.nodes.items():
            if h not in attempted:
                return n
        return None
    monkeypatch.setattr(
        ClusterAwarePolicy, "get_least_loaded_server", fake_pick
    )
    monkeypatch.setattr(
        Connection, "_connect_plain",
        classmethod(lambda cls, *a, **kw: _StubConn()),
    )

    conn = Connection.connect("host=p1 load_balance_hosts=true")
    assert conn._yb_cluster is None
    assert conn._yb_uuid == "P"


# ----------------------------------------------------------------- async dispatcher

@pytest.mark.anyio
async def test_async_dispatcher_routes_to_primary_when_healthy(
    fresh_registry, fake_state, monkeypatch
):
    group = _install_group(fresh_registry, fake_state)
    _patch_dispatcher_internals(monkeypatch, fresh_registry, group)

    from psycopg import AsyncConnection
    conn = await AsyncConnection.connect(
        "host=p1 load_balance_hosts=true "
        "yb.failover.secondaryClusterHosts=s1"
    )
    assert conn._yb_cluster == "primary"
    assert conn._yb_uuid == "P"


@pytest.mark.anyio
async def test_async_dispatcher_routes_to_secondary_when_unhealthy(
    fresh_registry, fake_state, monkeypatch
):
    group = _install_group(fresh_registry, fake_state)
    group.force_primary_status(HealthResult.UNHEALTHY)
    _patch_dispatcher_internals(monkeypatch, fresh_registry, group)

    from psycopg import AsyncConnection
    conn = await AsyncConnection.connect(
        "host=p1 load_balance_hosts=true "
        "yb.failover.secondaryClusterHosts=s1"
    )
    assert conn._yb_cluster == "secondary"
    assert conn._yb_uuid == "S"


# ----------------------------------------------------------------- flip mid-stream

def test_sync_dispatcher_flip_after_first_connect(
    fresh_registry, fake_state, monkeypatch
):
    """First connect → primary. Flip group to UNHEALTHY. Second connect →
    secondary. Verifies the dispatcher re-reads status on each connect."""
    group = _install_group(fresh_registry, fake_state)
    _patch_dispatcher_internals(monkeypatch, fresh_registry, group)

    from psycopg import Connection
    c1 = Connection.connect(
        "host=p1 load_balance_hosts=true "
        "yb.failover.secondaryClusterHosts=s1"
    )
    assert c1._yb_cluster == "primary"

    group.force_primary_status(HealthResult.UNHEALTHY)
    c2 = Connection.connect(
        "host=p1 load_balance_hosts=true "
        "yb.failover.secondaryClusterHosts=s1"
    )
    assert c2._yb_cluster == "secondary"

    group.force_primary_status(HealthResult.HEALTHY)
    c3 = Connection.connect(
        "host=p1 load_balance_hosts=true "
        "yb.failover.secondaryClusterHosts=s1"
    )
    assert c3._yb_cluster == "primary"
