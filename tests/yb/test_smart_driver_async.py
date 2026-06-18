"""
Async integration tests for the smart driver.

This file mirrors `test_smart_driver.py` test-for-test on the async surface,
plus a handful of async-only scenarios that have no sync analog (asyncio
cancellation, `asyncio.gather`, mixed sync+async sharing one registry).

The matrix coverage is intentional: every behavior the smart driver exposes
to a sync caller has to behave identically for an async caller. Drift
between the two paths has bitten us before — `connection.py` and
`connection_async.py` are mostly mirror images, but the failure modes around
cancellation, the event loop, and `await close()` are different enough that
each test has to run on both.
"""

# Copyright (C) 2026 Yugabyte

from __future__ import annotations

import asyncio
import time

import pytest

import psycopg


pytestmark = pytest.mark.anyio


# --------------------------------------------------------------------- pass-through paths


async def test_async_load_balance_false_is_passthrough(yb_cluster):
    """Async sibling of `test_load_balance_false_is_passthrough`."""
    from psycopg.yb.registry import ClusterRegistry
    registry = ClusterRegistry.instance()
    assert len(registry._clusters) == 0

    async with await psycopg.AsyncConnection.connect(
        yb_cluster + " load_balance_hosts=false"
    ) as conn:
        assert conn._yb_uuid is None

    assert len(registry._clusters) == 0


async def test_async_load_balance_absent_is_passthrough(yb_cluster):
    """Async sibling of `test_load_balance_absent_is_passthrough`."""
    async with await psycopg.AsyncConnection.connect(yb_cluster) as conn:
        assert conn._yb_uuid is None


# --------------------------------------------------------------------- basic async


async def test_async_smart_driver_distributes(yb_cluster, assert_balanced):
    """Open 24 async connections; ~8 per host on the 3-node cluster."""
    conns = []
    try:
        for _ in range(24):
            conns.append(
                await psycopg.AsyncConnection.connect(
                    yb_cluster + " load_balance_hosts=true"
                )
            )
        uuid = conns[0]._yb_uuid
        assert uuid is not None
        assert_balanced(uuid, {
            "127.0.0.1": 8, "127.0.0.2": 8, "127.0.0.3": 8,
        })
    finally:
        for c in conns:
            await c.close()


async def test_async_smart_driver_tags_connection_with_uuid_and_host(yb_cluster):
    """Async sibling of `test_smart_driver_tags_connection_with_uuid_and_host`."""
    async with await psycopg.AsyncConnection.connect(
        yb_cluster + " load_balance_hosts=true"
    ) as conn:
        assert conn._yb_uuid is not None
        assert conn._yb_host in ("127.0.0.1", "127.0.0.2", "127.0.0.3")


# --------------------------------------------------------------------- single-host bootstrap


async def test_async_single_host_bootstrap(yb_cluster, assert_balanced):
    """Async sibling of `test_single_host_bootstrap`."""
    dsn = "host=127.0.0.1 port=5433 user=yugabyte dbname=yugabyte load_balance_hosts=true"
    conns = []
    try:
        for _ in range(12):
            conns.append(await psycopg.AsyncConnection.connect(dsn))
        uuid = conns[0]._yb_uuid
        assert_balanced(uuid, {
            "127.0.0.1": 4,
            "127.0.0.2": 4,
            "127.0.0.3": 4,
        })
    finally:
        for c in conns:
            await c.close()


# --------------------------------------------------------------------- topology


async def test_async_topology_exact_match(yb_multi_zone_cluster, assert_balanced):
    """Async sibling of `test_topology_exact_match`. Multi-zone cluster
    (2 zoneA + 1 zoneB); `topology_keys=zoneA` must keep all traffic on
    the two zoneA nodes (6/6) and leave the zoneB node at 0."""
    dsn = (yb_multi_zone_cluster
           + " load_balance_hosts=true topology_keys=cloud1.datacenter1.zoneA")
    conns = []
    try:
        for _ in range(12):
            conns.append(await psycopg.AsyncConnection.connect(dsn))
        uuid = conns[0]._yb_uuid
        assert_balanced(uuid, {
            "127.0.0.1": 6,
            "127.0.0.2": 6,
            "127.0.0.3": 0,
        })
    finally:
        for c in conns:
            await c.close()


async def test_async_topology_wildcard_zone(yb_multi_zone_cluster, assert_balanced):
    """Async sibling of `test_topology_wildcard_zone`. Multi-zone cluster;
    `cloud1.datacenter1.*` matches both zoneA and zoneB → all three nodes
    eligible → 4/4/4."""
    dsn = (yb_multi_zone_cluster
           + " load_balance_hosts=true topology_keys=cloud1.datacenter1.*")
    conns = []
    try:
        for _ in range(12):
            conns.append(await psycopg.AsyncConnection.connect(dsn))
        uuid = conns[0]._yb_uuid
        assert_balanced(uuid, {
            "127.0.0.1": 4,
            "127.0.0.2": 4,
            "127.0.0.3": 4,
        })
    finally:
        for c in conns:
            await c.close()


async def test_async_topology_no_match_fails(yb_cluster):
    """Async sibling of `test_topology_no_match_fails`."""
    dsn = yb_cluster + " load_balance_hosts=true topology_keys=aws.us-west.us-west-1a"
    with pytest.raises(psycopg.OperationalError, match="no eligible"):
        await psycopg.AsyncConnection.connect(dsn)


async def test_async_topology_invalid_cloud_wildcard_rejected_at_parse(yb_cluster):
    """Async sibling of `test_topology_invalid_cloud_wildcard_rejected_at_parse`."""
    dsn = yb_cluster + " load_balance_hosts=true topology_keys=*.foo.bar"
    with pytest.raises(ValueError, match="cloud or region"):
        await psycopg.AsyncConnection.connect(dsn)


# --------------------------------------------------------------------- close / counter lifecycle


async def test_async_close_decrements_counter(yb_cluster):
    """`AsyncConnection.close()` must decrement just like the sync path."""
    from psycopg.yb.registry import ClusterRegistry
    registry = ClusterRegistry.instance()

    conn = await psycopg.AsyncConnection.connect(
        yb_cluster + " load_balance_hosts=true"
    )
    host = conn._yb_host
    uuid = conn._yb_uuid
    # On a fresh cluster only this connect lives on that host: counter == 1.
    assert registry.get_load(uuid, host) == 1

    await conn.close()
    assert registry.get_load(uuid, host) == 0


async def test_async_close_on_passthrough_doesnt_touch_registry(yb_cluster):
    """Async sibling of `test_close_on_passthrough_doesnt_touch_registry`."""
    from psycopg.yb.registry import ClusterRegistry

    sentinel = await psycopg.AsyncConnection.connect(
        yb_cluster + " load_balance_hosts=true"
    )
    uuid = sentinel._yb_uuid
    host = sentinel._yb_host
    baseline = ClusterRegistry.instance().get_load(uuid, host)

    plain = await psycopg.AsyncConnection.connect(yb_cluster)
    assert plain._yb_uuid is None
    await plain.close()

    assert ClusterRegistry.instance().get_load(uuid, host) == baseline
    await sentinel.close()


# --------------------------------------------------------------------- /rpcz cross-check


async def test_async_rpcz_vs_driver_counter_agreement(yb_cluster, assert_balanced):
    """Async sibling of `test_rpcz_vs_driver_counter_agreement`. Both sides
    exact, per host, no tolerance — `assert_balanced` accounts for the
    async control conn's +1 explicitly."""
    conns = [
        await psycopg.AsyncConnection.connect(yb_cluster + " load_balance_hosts=true")
        for _ in range(9)
    ]
    try:
        uuid = conns[0]._yb_uuid
        assert_balanced(uuid, {
            "127.0.0.1": 3,
            "127.0.0.2": 3,
            "127.0.0.3": 3,
        })
    finally:
        for c in conns:
            await c.close()


# --------------------------------------------------------------------- gather / concurrency


async def test_async_concurrent_connection_creation(yb_cluster, assert_balanced):
    """Async sibling of `test_concurrent_connection_creation`. 30 concurrent
    `AsyncConnection.connect` calls under `asyncio.gather` must distribute
    exactly 10/10/10 — the atomic pick+reserve under state.lock holds for
    the async path the same way it does for the sync path. Also covers the
    "many connects under gather don't leak counters" case (no separate
    sum-only test needed)."""
    async def open_one():
        return await psycopg.AsyncConnection.connect(
            yb_cluster + " load_balance_hosts=true"
        )

    conns = await asyncio.gather(*(open_one() for _ in range(30)))
    try:
        uuid = conns[0]._yb_uuid
        assert_balanced(uuid, {
            "127.0.0.1": 10,
            "127.0.0.2": 10,
            "127.0.0.3": 10,
        })
    finally:
        await asyncio.gather(*(c.close() for c in conns))


# --------------------------------------------------------------------- cancellation


async def test_async_cancellation_during_connect_no_leak(yb_cluster):
    """If an `asyncio.Task` running `AsyncConnection.connect` is cancelled
    before it completes, the per-host counter must not leak.

    We pre-warm the registry with one successful connect so we have a uuid +
    host to query, then fire-and-cancel several connects and assert the
    counters are unchanged.
    """
    from psycopg.yb.registry import ClusterRegistry
    registry = ClusterRegistry.instance()

    # Warm the registry.
    warm = await psycopg.AsyncConnection.connect(
        yb_cluster + " load_balance_hosts=true"
    )
    uuid = warm._yb_uuid
    baseline = {
        h: registry.get_load(uuid, h)
        for h in ("127.0.0.1", "127.0.0.2", "127.0.0.3")
    }

    async def slow_connect():
        return await psycopg.AsyncConnection.connect(
            yb_cluster + " load_balance_hosts=true"
        )

    # Spawn a task and cancel it before it finishes (very small wait).
    tasks = []
    for _ in range(5):
        t = asyncio.create_task(slow_connect())
        tasks.append(t)
    # Give them a tiny moment to start I/O, then cancel.
    await asyncio.sleep(0.001)
    for t in tasks:
        t.cancel()
    # Wait for them all to finish (cancelled or otherwise).
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Per-host check: for each host, the counter must equal baseline plus
    # exactly the number of OUR successful (non-cancelled) connects that
    # landed there. Cancelled tasks must not contribute even one.
    succeeded = [r for r in results if isinstance(r, psycopg.AsyncConnection)]
    per_host_added = {h: 0 for h in ("127.0.0.1", "127.0.0.2", "127.0.0.3")}
    for c in succeeded:
        per_host_added[c._yb_host] += 1
    after = {
        h: registry.get_load(uuid, h)
        for h in ("127.0.0.1", "127.0.0.2", "127.0.0.3")
    }
    for h, added in per_host_added.items():
        assert after[h] == baseline[h] + added, (
            f"{h}: counter is {after[h]}, expected baseline {baseline[h]} + "
            f"{added} successful connect(s) = {baseline[h] + added}. "
            f"Cancelled tasks must not contribute. "
            f"Full state: baseline={baseline}, after={after}, "
            f"succeeded_per_host={per_host_added}"
        )

    # Cleanup.
    for conn in succeeded:
        await conn.close()
    await warm.close()


# --------------------------------------------------------------------- mixed sync+async


async def test_mixed_sync_and_async_share_registry(yb_cluster):
    """A sync `Connection` and an async `AsyncConnection` to the same cluster
    must mutate the same per-host counters — the registry is process-global,
    not flavor-partitioned."""
    from psycopg.yb.registry import ClusterRegistry
    registry = ClusterRegistry.instance()

    # Open the async first (sets up the cluster state).
    aconn = await psycopg.AsyncConnection.connect(
        yb_cluster + " load_balance_hosts=true"
    )
    uuid = aconn._yb_uuid
    snapshot_after_async = {
        h: registry.get_load(uuid, h)
        for h in ("127.0.0.1", "127.0.0.2", "127.0.0.3")
    }

    # Now a sync connection to the same cluster — should see the existing
    # ClusterState (no second bootstrap) and bump the same counter map.
    sconn = psycopg.connect(yb_cluster + " load_balance_hosts=true")
    snapshot_after_both = {
        h: registry.get_load(uuid, h)
        for h in ("127.0.0.1", "127.0.0.2", "127.0.0.3")
    }
    # Sync conn's uuid must match — same cluster, same state object.
    assert sconn._yb_uuid == uuid

    # Per-host equality: the chosen host's counter went up by exactly 1, the
    # other two are unchanged. (The sync conn landed on exactly one host;
    # which one is non-deterministic, but the shape is exact.)
    chosen = sconn._yb_host
    for h in ("127.0.0.1", "127.0.0.2", "127.0.0.3"):
        expected = snapshot_after_async[h] + (1 if h == chosen else 0)
        assert snapshot_after_both[h] == expected, (
            f"{h}: counter is {snapshot_after_both[h]}, expected {expected}. "
            f"Sync conn landed on {chosen}; before={snapshot_after_async}, "
            f"after={snapshot_after_both}"
        )

    sconn.close()
    await aconn.close()
