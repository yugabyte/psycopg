"""
Per-host connection counters are released at most once per conn
(review #18).

Two drifts this guards against:

* Decremented twice: the drain's pool branch finishes a conn and
  decrements; the finished conn stays in the pool, and the next
  ``xcluster_check`` on it decremented again.
* Never decremented: ``close()`` returned early on an already-closed
  conn (the server broke it, e.g. AdminShutdown during failover), so
  its count was never released.
"""

# Copyright (C) 2026 Yugabyte

from __future__ import annotations

import asyncio
import threading
import types

import pytest

import psycopg
from psycopg.connection import Connection
from psycopg.connection_async import AsyncConnection
from psycopg.pq import ConnStatus
from psycopg.yb.drain import _release_registry_count, _safe_close
from psycopg.yb.health import HealthResult
from psycopg.yb.pool import xcluster_check, xcluster_check_async
from psycopg.yb.registry import FailoverGroup


pytestmark = pytest.mark.yb_unit


def _pgconn(broken: bool = False) -> types.SimpleNamespace:
    pg = types.SimpleNamespace(
        status=ConnStatus.BAD if broken else ConnStatus.OK, socket=-1,
    )
    pg.finish = lambda: setattr(pg, "status", ConnStatus.BAD)
    return pg


class _SyncConn(Connection):
    def __init__(self, uuid, host, broken=False, pool=None):  # type: ignore[no-untyped-def]
        self.pgconn = _pgconn(broken)
        self.lock = threading.Lock()
        self._closed = False
        self._yb_uuid = uuid
        self._yb_host = host
        if pool is not None:
            self._pool = pool


class _AsyncConn(AsyncConnection):
    def __init__(self, uuid, host, broken=False):  # type: ignore[no-untyped-def]
        self.pgconn = _pgconn(broken)
        self.lock = asyncio.Lock()
        self._closed = False
        self._yb_uuid = uuid
        self._yb_host = host


@pytest.fixture
def group(fresh_registry, fake_state):
    """Both HEALTHY → routing on the primary; secondary conns are stale.
    Host s1 on the secondary starts with a count of 2."""
    p = fake_state(("p1", "aws", "us-west", "us-west-1a", "primary", 2), uuid="P")
    s = fake_state(("s1", "aws", "us-east", "us-east-1a", "primary", 2), uuid="S")
    g = FailoverGroup(
        primary=p, secondary=s, lock=threading.Lock(),
        primary_status=HealthResult.HEALTHY,
        secondary_status=HealthResult.HEALTHY,
    )
    fresh_registry._clusters[p.uuid] = p
    fresh_registry._clusters[s.uuid] = s
    fresh_registry._failover_groups[p.uuid] = g
    return g


def _count(group) -> int:
    return group.secondary.nodes["s1"].connection_count


def test_drain_pool_close_then_xcluster_check_decrements_once(group):
    c = _SyncConn("S", "s1", pool=object())
    assert _safe_close(c) is None
    assert _count(group) == 1
    with pytest.raises(psycopg.OperationalError, match="xcluster_check"):
        xcluster_check(c)
    assert _count(group) == 1


def test_async_drain_close_then_xcluster_check_decrements_once(group):
    c = _AsyncConn("S", "s1")
    assert _safe_close(c) is None
    assert _count(group) == 1
    with pytest.raises(psycopg.OperationalError, match="xcluster_check"):
        asyncio.run(xcluster_check_async(c))
    assert _count(group) == 1


def test_close_on_server_broken_conn_decrements(group):
    c = _SyncConn("S", "s1", broken=True)
    assert c.closed
    c.close()
    assert _count(group) == 1


def test_async_close_on_server_broken_conn_decrements(group):
    c = _AsyncConn("S", "s1", broken=True)
    asyncio.run(c.close())
    assert _count(group) == 1


def test_double_close_decrements_once(group):
    c = _SyncConn("S", "s1")
    c.close()
    c.close()
    assert _count(group) == 1


def test_async_double_close_decrements_once(group):
    c = _AsyncConn("S", "s1")

    async def main():
        await c.close()
        await c.close()

    asyncio.run(main())
    assert _count(group) == 1


def test_release_then_close_decrements_once(group):
    c = _SyncConn("S", "s1")
    _release_registry_count(c)
    c.close()
    assert _count(group) == 1


def test_untagged_conn_never_touches_counters(group):
    c = _SyncConn(None, None, broken=True)
    c.close()
    _release_registry_count(c)
    assert _count(group) == 2
