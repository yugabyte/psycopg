"""
Async sibling of `test_smart_driver_pool_failover.py`. Same six scenarios
(cluster-aware × {stop, add}, topology-aware × {stop in, stop out, add in,
add out}) but running through `AsyncConnectionPool`.

See the sync file for the scenario walkthroughs and the expected-shape
derivations — the async variants are mechanical translations.
"""

# Copyright (C) 2026 Yugabyte

from __future__ import annotations

import asyncio
import time

import pytest

import psycopg

psycopg_pool = pytest.importorskip("psycopg_pool")
from psycopg_pool import AsyncConnectionPool  # noqa: E402


pytestmark = pytest.mark.anyio


# --------------------------------------------------------------------- cluster-aware × stop


async def test_async_pool_stopped_node_skipped_at_open(yb_cluster, yb_ctl):
    """Async sibling of `test_pool_stopped_node_skipped_at_open`."""
    yb_ctl.stop_node(2)
    try:
        async with AsyncConnectionPool(
            yb_cluster + " load_balance_hosts=true",
            min_size=10, max_size=10,
        ) as pool:
            await pool.wait()
            async with pool.connection() as conn:
                uuid = conn._yb_uuid
            from psycopg.yb.registry import ClusterRegistry
            registry = ClusterRegistry.instance()
            loads = {
                h: registry.get_load(uuid, h)
                for h in ("127.0.0.1", "127.0.0.2", "127.0.0.3")
            }
            assert loads == {"127.0.0.1": 5, "127.0.0.2": 0, "127.0.0.3": 5}, (
                f"survivors should split 10 conns 5/5; got {loads}"
            )
    finally:
        yb_ctl.start_node(2, placement_info="cloud1.datacenter1.rack1")
        await asyncio.sleep(2)


# --------------------------------------------------------------------- cluster-aware × add


async def test_async_pool_uniform_load_after_node_addition(yb_cluster, yb_ctl):
    """Async sibling of `test_pool_uniform_load_after_node_addition`."""
    from psycopg.yb.registry import ClusterRegistry
    from tests.yb import conftest as yb_conftest

    refresh_s = 3
    dsn = yb_cluster + f" load_balance_hosts=true yb_servers_refresh_interval={refresh_s}"
    added = False
    try:
        async with AsyncConnectionPool(dsn, min_size=9, max_size=9) as pool1:
            await pool1.wait()
            async with pool1.connection() as conn:
                uuid = conn._yb_uuid
            registry = ClusterRegistry.instance()
            for h in ("127.0.0.1", "127.0.0.2", "127.0.0.3"):
                assert registry.get_load(uuid, h) == 3, (
                    f"pool 1 should distribute 3/3/3; got "
                    f"{ {x: registry.get_load(uuid, x) for x in ('127.0.0.1','127.0.0.2','127.0.0.3')} }"
                )

            yb_ctl.add_node(placement_info="cloud1.datacenter1.rack1")
            added = True

            survivor_dsn = (
                "host=127.0.0.1 port=5433 user=yugabyte dbname=yugabyte"
            )
            deadline = time.monotonic() + 30
            seen_new = False
            while time.monotonic() < deadline:
                try:
                    async with await psycopg.AsyncConnection.connect(
                        survivor_dsn
                    ) as c:
                        cur = await c.execute("SELECT host FROM yb_servers()")
                        rows = await cur.fetchall()
                        if "127.0.0.4" in {r[0] for r in rows}:
                            seen_new = True
                            break
                except Exception:
                    pass
                await asyncio.sleep(1)
            if not seen_new:
                pytest.skip("yb_servers() did not show the added node within 30s")
            await asyncio.sleep(refresh_s + 1)

            async with AsyncConnectionPool(dsn, min_size=9, max_size=9) as pool2:
                await pool2.wait()
                loads = {
                    h: registry.get_load(uuid, h)
                    for h in ("127.0.0.1", "127.0.0.2", "127.0.0.3", "127.0.0.4")
                }
                assert sorted(loads.values()) == [4, 4, 5, 5], (
                    f"sorted shape should be [4, 4, 5, 5]; got "
                    f"{sorted(loads.values())} from {loads}"
                )
                assert sum(loads.values()) == 18
    finally:
        if added:
            try:
                yb_conftest._run_yb_ctl(["stop_node", "4"], timeout=60)
            except Exception:
                pass
            try:
                yb_conftest._run_yb_ctl(["remove_node", "4"], timeout=60)
            except Exception:
                pass
            await asyncio.sleep(5)


# --------------------------------------------------------------------- topology × stop (in topology)


async def test_async_pool_topology_stop_in_topology_zone(
    yb_multi_zone_cluster, yb_ctl
):
    """Async sibling of `test_pool_topology_stop_in_topology_zone`."""
    yb_ctl.stop_node(2)
    try:
        dsn = (yb_multi_zone_cluster
               + " load_balance_hosts=true topology_keys=cloud1.datacenter1.zoneA")
        async with AsyncConnectionPool(dsn, min_size=6, max_size=6) as pool:
            await pool.wait()
            async with pool.connection() as conn:
                uuid = conn._yb_uuid
            from psycopg.yb.registry import ClusterRegistry
            registry = ClusterRegistry.instance()
            loads = {
                h: registry.get_load(uuid, h)
                for h in ("127.0.0.1", "127.0.0.2", "127.0.0.3")
            }
            assert loads == {
                "127.0.0.1": 6,
                "127.0.0.2": 0,
                "127.0.0.3": 0,
            }, f"all 6 conns should land on .1; got {loads}"
    finally:
        yb_ctl.start_node(2, placement_info="cloud1.datacenter1.zoneA")
        await asyncio.sleep(2)


# --------------------------------------------------------------------- topology × stop (out of topology)


async def test_async_pool_topology_stop_out_of_topology_unaffected(
    yb_multi_zone_cluster, yb_ctl
):
    """Async sibling of `test_pool_topology_stop_out_of_topology_unaffected`."""
    yb_ctl.stop_node(3)
    try:
        dsn = (yb_multi_zone_cluster
               + " load_balance_hosts=true topology_keys=cloud1.datacenter1.zoneA")
        async with AsyncConnectionPool(dsn, min_size=6, max_size=6) as pool:
            await pool.wait()
            async with pool.connection() as conn:
                uuid = conn._yb_uuid
            from psycopg.yb.registry import ClusterRegistry
            registry = ClusterRegistry.instance()
            loads = {
                h: registry.get_load(uuid, h)
                for h in ("127.0.0.1", "127.0.0.2", "127.0.0.3")
            }
            assert loads == {"127.0.0.1": 3, "127.0.0.2": 3, "127.0.0.3": 0}, (
                f"stopped out-of-topology node must not perturb zoneA traffic; "
                f"got {loads}"
            )
    finally:
        yb_ctl.start_node(3, placement_info="cloud1.datacenter1.zoneB")
        await asyncio.sleep(2)


# --------------------------------------------------------------------- topology × add (in topology)


async def test_async_pool_topology_add_in_topology_zone_attracts_traffic(
    yb_multi_zone_cluster, yb_ctl
):
    """Async sibling of `test_pool_topology_add_in_topology_zone_attracts_traffic`."""
    from psycopg.yb.registry import ClusterRegistry
    from tests.yb import conftest as yb_conftest

    refresh_s = 3
    base_dsn = (yb_multi_zone_cluster
                + f" load_balance_hosts=true "
                f"topology_keys=cloud1.datacenter1.zoneA "
                f"yb_servers_refresh_interval={refresh_s}")
    added = False
    try:
        async with AsyncConnectionPool(base_dsn, min_size=1, max_size=1) as sentinel:
            await sentinel.wait()
            async with sentinel.connection() as c:
                uuid = c._yb_uuid

        yb_ctl.add_node(placement_info="cloud1.datacenter1.zoneA")
        added = True

        survivor_dsn = (
            "host=127.0.0.1 port=5433 user=yugabyte dbname=yugabyte"
        )
        deadline = time.monotonic() + 30
        seen_new = False
        while time.monotonic() < deadline:
            try:
                async with await psycopg.AsyncConnection.connect(
                    survivor_dsn
                ) as c:
                    cur = await c.execute("SELECT host FROM yb_servers()")
                    rows = await cur.fetchall()
                    if "127.0.0.4" in {r[0] for r in rows}:
                        seen_new = True
                        break
            except Exception:
                pass
            await asyncio.sleep(1)
        if not seen_new:
            pytest.skip("yb_servers() did not show the added node within 30s")
        await asyncio.sleep(refresh_s + 1)

        async with AsyncConnectionPool(base_dsn, min_size=9, max_size=9) as pool:
            await pool.wait()
            registry = ClusterRegistry.instance()
            loads = {
                h: registry.get_load(uuid, h)
                for h in ("127.0.0.1", "127.0.0.2", "127.0.0.3", "127.0.0.4")
            }
            assert loads == {
                "127.0.0.1": 3,
                "127.0.0.2": 3,
                "127.0.0.3": 0,
                "127.0.0.4": 3,
            }, f"new zoneA node should attract its share; got {loads}"
    finally:
        if added:
            try:
                yb_conftest._run_yb_ctl(["stop_node", "4"], timeout=60)
            except Exception:
                pass
            try:
                yb_conftest._run_yb_ctl(["remove_node", "4"], timeout=60)
            except Exception:
                pass
            await asyncio.sleep(5)


# --------------------------------------------------------------------- topology × add (out of topology)


async def test_async_pool_topology_add_out_of_topology_zone_no_traffic(
    yb_multi_zone_cluster, yb_ctl
):
    """Async sibling of `test_pool_topology_add_out_of_topology_zone_no_traffic`."""
    from psycopg.yb.registry import ClusterRegistry
    from tests.yb import conftest as yb_conftest

    refresh_s = 3
    base_dsn = (yb_multi_zone_cluster
                + f" load_balance_hosts=true "
                f"topology_keys=cloud1.datacenter1.zoneA "
                f"yb_servers_refresh_interval={refresh_s}")
    added = False
    try:
        async with AsyncConnectionPool(base_dsn, min_size=1, max_size=1) as sentinel:
            await sentinel.wait()
            async with sentinel.connection() as c:
                uuid = c._yb_uuid

        yb_ctl.add_node(placement_info="cloud1.datacenter1.zoneB")
        added = True

        survivor_dsn = (
            "host=127.0.0.1 port=5433 user=yugabyte dbname=yugabyte"
        )
        deadline = time.monotonic() + 30
        seen_new = False
        while time.monotonic() < deadline:
            try:
                async with await psycopg.AsyncConnection.connect(
                    survivor_dsn
                ) as c:
                    cur = await c.execute("SELECT host FROM yb_servers()")
                    rows = await cur.fetchall()
                    if "127.0.0.4" in {r[0] for r in rows}:
                        seen_new = True
                        break
            except Exception:
                pass
            await asyncio.sleep(1)
        if not seen_new:
            pytest.skip("yb_servers() did not show the added node within 30s")
        await asyncio.sleep(refresh_s + 1)

        async with AsyncConnectionPool(base_dsn, min_size=6, max_size=6) as pool:
            await pool.wait()
            registry = ClusterRegistry.instance()
            loads = {
                h: registry.get_load(uuid, h)
                for h in ("127.0.0.1", "127.0.0.2", "127.0.0.3", "127.0.0.4")
            }
            assert loads == {
                "127.0.0.1": 3,
                "127.0.0.2": 3,
                "127.0.0.3": 0,
                "127.0.0.4": 0,
            }, f"new out-of-topology node must not attract traffic; got {loads}"
    finally:
        if added:
            try:
                yb_conftest._run_yb_ctl(["stop_node", "4"], timeout=60)
            except Exception:
                pass
            try:
                yb_conftest._run_yb_ctl(["remove_node", "4"], timeout=60)
            except Exception:
                pass
            await asyncio.sleep(5)
