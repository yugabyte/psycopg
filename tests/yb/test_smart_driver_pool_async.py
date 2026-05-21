"""
Async pool integration tests for the YugabyteDB smart driver.

Mirror of `test_smart_driver_pool.py` via `psycopg_pool.AsyncConnectionPool`.
Same scenarios — pass-through, basic balance, topology, borrow lifecycle,
close drain, growth under concurrent gather, mixed sync + async sharing
the registry — but running through the async pool path so we cover the
`AsyncConnection.connect` dispatcher branch identically.
"""

# Copyright (C) 2026 Yugabyte

from __future__ import annotations

import asyncio

import pytest

import psycopg

psycopg_pool = pytest.importorskip("psycopg_pool")
from psycopg_pool import AsyncConnectionPool  # noqa: E402


pytestmark = pytest.mark.anyio


# --------------------------------------------------------------------- pass-through


async def test_async_pool_load_balance_false_is_passthrough(yb_cluster):
    """Async sibling of `test_pool_load_balance_false_is_passthrough`."""
    from psycopg.yb.registry import ClusterRegistry
    registry = ClusterRegistry.instance()
    assert len(registry._clusters) == 0

    async with AsyncConnectionPool(
        yb_cluster + " load_balance_hosts=false",
        min_size=3, max_size=3,
    ) as pool:
        await pool.wait()
        async with pool.connection() as conn:
            assert conn._yb_uuid is None

    assert len(registry._clusters) == 0


# --------------------------------------------------------------------- basic balance


async def test_async_pool_smart_driver_distributes_evenly(yb_cluster, assert_balanced):
    """Async sibling of `test_pool_smart_driver_distributes_evenly`. 12-conn
    async pool over a 3-node cluster: exact 4/4/4 driver-side + /rpcz."""
    async with AsyncConnectionPool(
        yb_cluster + " load_balance_hosts=true",
        min_size=12, max_size=12,
    ) as pool:
        await pool.wait()
        async with pool.connection() as conn:
            uuid = conn._yb_uuid
            assert uuid is not None
            assert conn._yb_host in ("127.0.0.1", "127.0.0.2", "127.0.0.3")
        assert_balanced(uuid, {
            "127.0.0.1": 4,
            "127.0.0.2": 4,
            "127.0.0.3": 4,
        })


# --------------------------------------------------------------------- topology


async def test_async_pool_topology_exact_match(yb_multi_zone_cluster, assert_balanced):
    """Async sibling of `test_pool_topology_exact_match`. Multi-zone cluster
    (2 zoneA + 1 zoneB); `topology_keys=zoneA` must produce 6/6 across the
    zoneA nodes and 0 on the zoneB node, through the async pool path."""
    dsn = (yb_multi_zone_cluster
           + " load_balance_hosts=true topology_keys=cloud1.datacenter1.zoneA")
    async with AsyncConnectionPool(dsn, min_size=12, max_size=12) as pool:
        await pool.wait()
        async with pool.connection() as conn:
            uuid = conn._yb_uuid
        assert_balanced(uuid, {
            "127.0.0.1": 6,
            "127.0.0.2": 6,
            "127.0.0.3": 0,
        })


async def test_async_pool_topology_no_match_fails(yb_cluster):
    """Async sibling of `test_pool_topology_no_match_fails`."""
    dsn = yb_cluster + " load_balance_hosts=true topology_keys=aws.us-west.us-west-1a"
    pool = AsyncConnectionPool(dsn, min_size=1, max_size=1, open=False)
    with pytest.raises(Exception):
        await pool.open(wait=True, timeout=10)
    try:
        await pool.close()
    except Exception:
        pass


# --------------------------------------------------------------------- borrow + return


async def test_async_pool_borrow_does_not_change_counter(yb_cluster):
    """Async sibling of `test_pool_borrow_does_not_change_counter`."""
    from psycopg.yb.registry import ClusterRegistry
    registry = ClusterRegistry.instance()

    async with AsyncConnectionPool(
        yb_cluster + " load_balance_hosts=true",
        min_size=12, max_size=12,
    ) as pool:
        await pool.wait()
        async with pool.connection() as conn:
            uuid = conn._yb_uuid
        before = {
            h: registry.get_load(uuid, h)
            for h in ("127.0.0.1", "127.0.0.2", "127.0.0.3")
        }
        for _ in range(20):
            async with pool.connection() as conn:
                await conn.execute("SELECT 1")
        after = {
            h: registry.get_load(uuid, h)
            for h in ("127.0.0.1", "127.0.0.2", "127.0.0.3")
        }
        assert before == after, (
            f"borrow/return should not mutate counters; before={before}, "
            f"after={after}"
        )


# --------------------------------------------------------------------- close lifecycle


async def test_async_pool_close_drains_counter(yb_cluster):
    """Async sibling of `test_pool_close_drains_counter`."""
    from psycopg.yb.registry import ClusterRegistry
    registry = ClusterRegistry.instance()

    uuid: str | None = None
    async with AsyncConnectionPool(
        yb_cluster + " load_balance_hosts=true",
        min_size=9, max_size=9,
    ) as pool:
        await pool.wait()
        async with pool.connection() as conn:
            uuid = conn._yb_uuid
        mid = {
            h: registry.get_load(uuid, h)
            for h in ("127.0.0.1", "127.0.0.2", "127.0.0.3")
        }
        assert mid == {"127.0.0.1": 3, "127.0.0.2": 3, "127.0.0.3": 3}

    after = {
        h: registry.get_load(uuid, h)
        for h in ("127.0.0.1", "127.0.0.2", "127.0.0.3")
    }
    assert after == {"127.0.0.1": 0, "127.0.0.2": 0, "127.0.0.3": 0}, (
        f"pool close should drain all counters back to 0; got {after}"
    )


# --------------------------------------------------------------------- growth


async def test_async_pool_growth_distributes_via_smart_driver(
    yb_cluster, assert_balanced
):
    """Async sibling of `test_pool_growth_distributes_via_smart_driver`. Force
    the pool to grow from `min_size=3` to `max_size=12` via 12 concurrent
    `asyncio.gather` borrowers. End-state: exactly 4/4/4 across 3 nodes."""
    async def hold_one(pool, barrier_event, release_after_s=2.0):
        async with pool.connection() as conn:
            await barrier_event.wait()
            await asyncio.sleep(release_after_s)
            return conn._yb_host

    barrier = asyncio.Event()
    async with AsyncConnectionPool(
        yb_cluster + " load_balance_hosts=true",
        min_size=3, max_size=12,
    ) as pool:
        await pool.wait()
        tasks = [
            asyncio.create_task(hold_one(pool, barrier))
            for _ in range(12)
        ]
        # Let all 12 tasks check out a connection (forcing growth), then
        # release the barrier so they all return roughly together.
        await asyncio.sleep(0.5)
        barrier.set()
        await asyncio.gather(*tasks)

        async with pool.connection() as conn:
            uuid = conn._yb_uuid

        from psycopg.yb.registry import ClusterRegistry
        registry = ClusterRegistry.instance()
        total = sum(
            registry.get_load(uuid, h)
            for h in ("127.0.0.1", "127.0.0.2", "127.0.0.3")
        )
        assert total == 12
        assert_balanced(uuid, {
            "127.0.0.1": 4,
            "127.0.0.2": 4,
            "127.0.0.3": 4,
        })


# --------------------------------------------------------------------- mixed pool + direct


async def test_async_pool_and_direct_connect_share_registry(yb_cluster):
    """An async-pool-managed connection and a sync direct `psycopg.connect()`
    to the same cluster share one process-wide registry. The same coverage
    as `test_mixed_sync_and_async_share_registry` but using the async POOL
    as the first opener."""
    from psycopg.yb.registry import ClusterRegistry
    registry = ClusterRegistry.instance()

    async with AsyncConnectionPool(
        yb_cluster + " load_balance_hosts=true",
        min_size=3, max_size=3,
    ) as pool:
        await pool.wait()
        async with pool.connection() as conn:
            uuid = conn._yb_uuid
        snapshot_after_pool = {
            h: registry.get_load(uuid, h)
            for h in ("127.0.0.1", "127.0.0.2", "127.0.0.3")
        }
        assert sum(snapshot_after_pool.values()) == 3

        direct = psycopg.connect(yb_cluster + " load_balance_hosts=true")
        try:
            assert direct._yb_uuid == uuid
            chosen = direct._yb_host
            snapshot_after_both = {
                h: registry.get_load(uuid, h)
                for h in ("127.0.0.1", "127.0.0.2", "127.0.0.3")
            }
            for h in ("127.0.0.1", "127.0.0.2", "127.0.0.3"):
                expected = snapshot_after_pool[h] + (1 if h == chosen else 0)
                assert snapshot_after_both[h] == expected, (
                    f"{h}: counter is {snapshot_after_both[h]}, expected "
                    f"{expected}. Direct conn landed on {chosen}; before="
                    f"{snapshot_after_pool}, after={snapshot_after_both}"
                )
        finally:
            direct.close()
