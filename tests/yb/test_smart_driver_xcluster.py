"""
Tier 2 xCluster integration tests against two REAL YugabyteDB clusters.

These exercise the full path end-to-end:

  * Real conninfo parsing on a real DSN
  * Real two-cluster bootstrap (two ``yb-ctl`` instances running on
    disjoint IP ranges with distinct ``universe_uuid``s)
  * Real TCP connect through the dispatcher
  * Real ``ConnectionPool`` / ``AsyncConnectionPool`` borrows
  * Real ``HealthProbe`` daemon thread (running but inert against the
    stub ``cluster_status_check``)

Failover is driven via ``FailoverGroup.force_status`` (from the test
itself or from a background thread, depending on the scenario) rather
than by stopping real cluster nodes. That matches what we have available
today — the real tracker-table-based detection logic is deferred to a
follow-on patch.

Why force_status instead of stopping nodes? Two reasons:
  1. The v1 stub ``cluster_status_check`` always returns HEALTHY, so
     stopping real nodes wouldn't flip the flag anyway.
  2. Driving failover via force_status lets the test control timing
     precisely — no waiting for refresh ticks to converge.

The file's name (``test_smart_driver_xcluster.py``) auto-tags it with
the ``yb`` marker via the conftest's ``startswith("test_smart_driver")``
rule.
"""

# Copyright (C) 2026 Yugabyte

from __future__ import annotations

import asyncio
import threading
import time

import pytest

import psycopg
from psycopg.yb.health import HealthResult
from psycopg.yb.pool import xcluster_check, xcluster_check_async
from psycopg.yb.registry import ClusterRegistry


# Number of connections opened per phase. 12 is the default the existing
# load-balancing integration tests use; it gives 4 conns per node on RF=3
# clusters for clean assertion arithmetic.
N_CONNS = 12

PRIMARY_HOSTS = "127.0.0.1,127.0.0.2,127.0.0.3"
SECONDARY_HOSTS = "127.0.0.4,127.0.0.5,127.0.0.6"

_XCLUSTER_DSN = (
    f"host={PRIMARY_HOSTS} port=5433 user=yugabyte dbname=yugabyte "
    f"load_balance_hosts=true "
    f"yb.failover.secondaryClusterHosts={SECONDARY_HOSTS}"
)


def _bootstrap_group_and_get(dsn: str):
    """Open one connection (which triggers FailoverGroup bootstrap) then
    return ``(group, primary_uuid)``. Used by tests to grab the group
    handle so they can call ``force_status`` on it."""
    conn = psycopg.connect(dsn)
    try:
        primary_uuid = conn._yb_uuid
        # ConnectionRegistry.get_failover_group(primary_uuid) — but the
        # conn might be on secondary if the test pre-flipped. Use
        # get_failover_group_by_uuid which searches both sides.
        group = ClusterRegistry.instance().get_failover_group_by_uuid(primary_uuid)
        assert group is not None, "FailoverGroup must be created at bootstrap"
        return group, group.primary.uuid
    finally:
        conn.close()


# --------------------------------------------------------------------- bootstrap

def test_bootstrap_creates_failover_group(yb_xcluster_clusters):
    """First connect with xCluster DSN must:
      * bootstrap a FailoverGroup
      * give primary and secondary distinct universe_uuids (proves the
        two clusters are genuinely independent, not the same physical one)
      * tag the conn as ``_yb_cluster = 'primary'`` while status is HEALTHY
    """
    conn = psycopg.connect(_XCLUSTER_DSN)
    try:
        assert conn._yb_cluster == "primary"
        group = ClusterRegistry.instance().get_failover_group_by_uuid(
            conn._yb_uuid
        )
        assert group is not None
        assert group.status == HealthResult.HEALTHY
        assert group.primary.uuid != group.secondary.uuid, (
            "Primary and secondary should report distinct universe_uuids — "
            "two separate clusters."
        )
        assert group.primary.uuid == conn._yb_uuid
    finally:
        conn.close()


# --------------------------------------------------------------------- routing

def test_dispatcher_routes_to_primary_when_healthy(yb_xcluster_clusters):
    """N connects with status=HEALTHY must all land on primary cluster
    nodes (127.0.0.1 / .2 / .3) and be tagged as 'primary'."""
    conns = []
    try:
        for _ in range(N_CONNS):
            conns.append(psycopg.connect(_XCLUSTER_DSN))
        for c in conns:
            assert c._yb_cluster == "primary", (
                f"expected primary, got {c._yb_cluster}; host={c._yb_host}"
            )
            assert c._yb_host in ("127.0.0.1", "127.0.0.2", "127.0.0.3")
    finally:
        for c in conns:
            c.close()


def test_dispatcher_routes_to_secondary_after_force_unhealthy(
    yb_xcluster_clusters,
):
    """After a sync `force_status(UNHEALTHY)`, the next N connects must
    all land on secondary cluster nodes (127.0.0.4 / .5 / .6) and be
    tagged as 'secondary'."""
    # Bootstrap and grab the group.
    group, primary_uuid = _bootstrap_group_and_get(_XCLUSTER_DSN)
    group.force_status(HealthResult.UNHEALTHY)

    conns = []
    try:
        for _ in range(N_CONNS):
            conns.append(psycopg.connect(_XCLUSTER_DSN))
        for c in conns:
            assert c._yb_cluster == "secondary", (
                f"expected secondary, got {c._yb_cluster}; host={c._yb_host}"
            )
            assert c._yb_host in ("127.0.0.4", "127.0.0.5", "127.0.0.6")
            assert c._yb_uuid == group.secondary.uuid
    finally:
        for c in conns:
            c.close()


def test_failback_routes_back_to_primary(yb_xcluster_clusters):
    """Set UNHEALTHY → route to secondary. Reset to HEALTHY (via
    `reset_failover_group`) → route to primary again."""
    group, primary_uuid = _bootstrap_group_and_get(_XCLUSTER_DSN)
    group.force_status(HealthResult.UNHEALTHY)

    c1 = psycopg.connect(_XCLUSTER_DSN)
    try:
        assert c1._yb_cluster == "secondary"
    finally:
        c1.close()

    assert ClusterRegistry.instance().reset_failover_group(primary_uuid) is True

    c2 = psycopg.connect(_XCLUSTER_DSN)
    try:
        assert c2._yb_cluster == "primary"
        assert c2._yb_host in ("127.0.0.1", "127.0.0.2", "127.0.0.3")
    finally:
        c2.close()


# --------------------------------------------------------------------- background-thread driven

def test_failover_via_background_thread(yb_xcluster_clusters):
    """A separate thread flips the primary to UNHEALTHY after a short
    delay. The test opens connections on a schedule that straddles the
    flip — connects BEFORE the flip should land on primary, connects
    AFTER on secondary. Mirrors the user's requested mental model:
    'a separate thread that will mark the primary cluster as unhealthy
    for the failover to trigger'."""
    group, _ = _bootstrap_group_and_get(_XCLUSTER_DSN)
    assert group.status == HealthResult.HEALTHY

    flip_time = [None]   # set when the flip actually fires
    stop_event = threading.Event()

    def flipper():
        # Wait ~0.4s then flip. Test code opens conns either side of this.
        if stop_event.wait(timeout=0.4):
            return
        group.force_status(HealthResult.UNHEALTHY)
        flip_time[0] = time.monotonic()

    t = threading.Thread(target=flipper, daemon=True)
    t.start()
    try:
        # Pre-flip: open a few conns immediately — must be primary.
        pre = [psycopg.connect(_XCLUSTER_DSN) for _ in range(3)]

        # Wait until the flipper has fired.
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and flip_time[0] is None:
            time.sleep(0.05)
        assert flip_time[0] is not None, "flipper did not fire within 2s"

        # Post-flip: open a few more — must be secondary.
        post = [psycopg.connect(_XCLUSTER_DSN) for _ in range(3)]

        for c in pre:
            assert c._yb_cluster == "primary", (
                f"pre-flip conn should be primary; got {c._yb_cluster}"
            )
        for c in post:
            assert c._yb_cluster == "secondary", (
                f"post-flip conn should be secondary; got {c._yb_cluster}"
            )
        for c in pre + post:
            c.close()
    finally:
        stop_event.set()
        t.join(timeout=2)


# --------------------------------------------------------------------- pool eviction

def test_pool_evicts_primary_conns_after_force_unhealthy(yb_xcluster_clusters):
    """A `ConnectionPool(check=xcluster_check)` opens conns on the
    primary cluster while status=HEALTHY. After force_status(UNHEALTHY),
    subsequent borrows hit the check callback which raises — the pool
    evicts the stale primary-tagged conn and replaces it with a fresh
    secondary-tagged one.

    We verify by counting how many evictions happened and where the
    replacement conns land."""
    psycopg_pool = pytest.importorskip("psycopg_pool")
    from psycopg_pool import ConnectionPool

    with ConnectionPool(
        _XCLUSTER_DSN, check=xcluster_check, min_size=4, max_size=4
    ) as pool:
        pool.wait()

        # Phase 1: borrow + return a conn. Status HEALTHY → primary.
        with pool.connection() as conn:
            primary_uuid = conn._yb_uuid
            assert conn._yb_cluster == "primary"

        # Grab the group and flip to UNHEALTHY.
        group = ClusterRegistry.instance().get_failover_group_by_uuid(primary_uuid)
        assert group is not None
        group.force_status(HealthResult.UNHEALTHY)

        # Phase 2: borrow several conns. Each borrow should fail the
        # xcluster_check (because the cached pool conn is tagged primary
        # but secondary is now active), the pool evicts and replaces.
        for _ in range(4):
            with pool.connection() as conn:
                assert conn._yb_cluster == "secondary", (
                    f"after flip, pool borrow must be secondary; "
                    f"got {conn._yb_cluster} (host={conn._yb_host})"
                )
                assert conn._yb_host in ("127.0.0.4", "127.0.0.5", "127.0.0.6")


# --------------------------------------------------------------------- async parity

@pytest.mark.anyio
async def test_async_dispatcher_routes_to_secondary_after_flip(
    yb_xcluster_clusters,
):
    """Async sibling of the sync routing test. Use `AsyncConnection.connect`
    and verify the same primary→secondary routing semantics."""
    # Bootstrap via sync connect to grab the group (the async path also
    # creates a FailoverGroup, but using sync here is simpler).
    group, primary_uuid = _bootstrap_group_and_get(_XCLUSTER_DSN)
    group.force_status(HealthResult.UNHEALTHY)

    conns = []
    try:
        for _ in range(N_CONNS):
            conns.append(await psycopg.AsyncConnection.connect(_XCLUSTER_DSN))
        for c in conns:
            assert c._yb_cluster == "secondary"
            assert c._yb_host in ("127.0.0.4", "127.0.0.5", "127.0.0.6")
    finally:
        for c in conns:
            await c.close()


@pytest.mark.anyio
async def test_async_pool_evicts_primary_conns_after_flip(yb_xcluster_clusters):
    """Async sibling of the pool-eviction test using ``AsyncConnectionPool``."""
    psycopg_pool = pytest.importorskip("psycopg_pool")
    from psycopg_pool import AsyncConnectionPool

    async with AsyncConnectionPool(
        _XCLUSTER_DSN, check=xcluster_check_async, min_size=4, max_size=4
    ) as pool:
        await pool.wait()

        async with pool.connection() as conn:
            primary_uuid = conn._yb_uuid
            assert conn._yb_cluster == "primary"

        group = ClusterRegistry.instance().get_failover_group_by_uuid(primary_uuid)
        group.force_status(HealthResult.UNHEALTHY)

        for _ in range(4):
            async with pool.connection() as conn:
                assert conn._yb_cluster == "secondary"
                assert conn._yb_host in ("127.0.0.4", "127.0.0.5", "127.0.0.6")
