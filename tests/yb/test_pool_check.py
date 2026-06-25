"""
Unit tests for ``psycopg.yb.pool.xcluster_check`` (Phase 6 of the xCluster
failover implementation plan).

These verify the pool's check callback raises ``OperationalError`` on
conns that belong to the inactive side of a FailoverGroup, and passes on
conns that match the active side. Metadata-only — no real DB.

Auto-tagged ``yb_unit`` via the filename — wait, no, ``test_pool_check``
doesn't match any of the auto-marker prefixes. We mark explicitly.
"""

# Copyright (C) 2026 Yugabyte

from __future__ import annotations

import threading

import pytest

import psycopg
from psycopg.yb.health import HealthResult
from psycopg.yb.pool import xcluster_check, xcluster_check_async
from psycopg.yb.registry import FailoverGroup


pytestmark = pytest.mark.yb_unit


class _FakeConn:
    """Tiny stand-in. Only ``_yb_uuid`` is read by the check."""
    def __init__(self, uuid: str | None) -> None:
        self._yb_uuid = uuid


def _install(reg, fake_state, status=HealthResult.HEALTHY):
    p = fake_state(("p1", "aws", "us-west", "us-west-1a", "primary"), uuid="P")
    s = fake_state(("s1", "aws", "us-east", "us-east-1a", "primary"), uuid="S")
    group = FailoverGroup(
        primary=p, secondary=s, lock=threading.Lock(), status=status,
    )
    reg._clusters[p.uuid] = p
    reg._clusters[s.uuid] = s
    reg._failover_groups[p.uuid] = group
    return group


# ----------------------------------------------------------------- sync

def test_check_passes_for_primary_conn_when_healthy(fresh_registry, fake_state):
    _install(fresh_registry, fake_state, status=HealthResult.HEALTHY)
    # No exception → conn kept.
    xcluster_check(_FakeConn("P"))


def test_check_raises_for_primary_conn_when_unhealthy(fresh_registry, fake_state):
    _install(fresh_registry, fake_state, status=HealthResult.UNHEALTHY)
    with pytest.raises(psycopg.OperationalError, match="xcluster_check"):
        xcluster_check(_FakeConn("P"))


def test_check_passes_for_secondary_conn_when_unhealthy(fresh_registry, fake_state):
    _install(fresh_registry, fake_state, status=HealthResult.UNHEALTHY)
    xcluster_check(_FakeConn("S"))


def test_check_raises_for_secondary_conn_when_healthy(fresh_registry, fake_state):
    """Symmetric to the above — after failback, secondary-tagged conns are
    no longer on the active cluster and must be evicted."""
    _install(fresh_registry, fake_state, status=HealthResult.HEALTHY)
    with pytest.raises(psycopg.OperationalError, match="xcluster_check"):
        xcluster_check(_FakeConn("S"))


def test_check_passes_for_non_smart_driver_conn(fresh_registry, fake_state):
    """Conn without ``_yb_uuid`` (pass-through libpq) must not trigger the
    check at all — even if a FailoverGroup is installed."""
    _install(fresh_registry, fake_state, status=HealthResult.UNHEALTHY)
    xcluster_check(_FakeConn(None))


def test_check_passes_for_smart_driver_conn_without_failover_group(fresh_registry):
    """Smart-driver conn (has _yb_uuid) but no FailoverGroup configured —
    e.g. user runs a single-cluster smart-driver setup. Check must pass."""
    # No FailoverGroup installed; just a bare ClusterRegistry instance.
    xcluster_check(_FakeConn("some-cluster-uuid"))


# ----------------------------------------------------------------- async

@pytest.mark.anyio
async def test_async_check_passes_for_primary_when_healthy(
    fresh_registry, fake_state
):
    _install(fresh_registry, fake_state, status=HealthResult.HEALTHY)
    await xcluster_check_async(_FakeConn("P"))


@pytest.mark.anyio
async def test_async_check_raises_for_primary_when_unhealthy(
    fresh_registry, fake_state
):
    _install(fresh_registry, fake_state, status=HealthResult.UNHEALTHY)
    with pytest.raises(psycopg.OperationalError, match="xcluster_check"):
        await xcluster_check_async(_FakeConn("P"))
