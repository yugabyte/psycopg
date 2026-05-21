"""
Pool failover tests for the YugabyteDB smart driver (sync surface).

The existing pool tests (`test_smart_driver_pool.py`) verify the steady-state
pool + smart-driver interaction: distribution, topology filtering on a clean
cluster, borrow/return semantics. These tests cover the cluster-state-change
scenarios on top of the pool — node stops, node additions — across both the
cluster-aware path and the topology-aware path.

Pattern: mutate cluster state first, then open the test pool. The pool's
initial connections go through the smart-driver dispatcher exactly once
each (no pre-existing conns to retain), so the resulting per-host
distribution directly reflects how the dispatcher reacts to the new
cluster shape.

Async siblings live in `test_smart_driver_pool_failover_async.py`.
"""

# Copyright (C) 2026 Yugabyte

from __future__ import annotations

import time

import pytest

import psycopg

psycopg_pool = pytest.importorskip("psycopg_pool")
from psycopg_pool import ConnectionPool  # noqa: E402


# --------------------------------------------------------------------- cluster-aware × stop


def test_pool_stopped_node_skipped_at_open(yb_cluster, yb_ctl):
    """Cluster-aware. Stop node 2 first. Then open a pool with 10 conns —
    the first conn that tries node 2 fails, `mark_failed` quarantines it
    for the TTL window, and the remaining 9 connects (plus the retry of
    the failed one) all land on node 1 or 3. Final per-host counts: zero
    on the stopped node, 10 conns split exactly 5/5 across the survivors."""
    yb_ctl.stop_node(2)
    try:
        with ConnectionPool(
            yb_cluster + " load_balance_hosts=true",
            min_size=10, max_size=10,
        ) as pool:
            pool.wait()
            with pool.connection() as conn:
                uuid = conn._yb_uuid
            from psycopg.yb.registry import ClusterRegistry
            registry = ClusterRegistry.instance()
            loads = {
                h: registry.get_load(uuid, h)
                for h in ("127.0.0.1", "127.0.0.2", "127.0.0.3")
            }
            assert loads["127.0.0.2"] == 0, (
                f"stopped node should be at 0; got {loads}"
            )
            assert loads == {"127.0.0.1": 5, "127.0.0.2": 0, "127.0.0.3": 5}, (
                f"survivors should split 10 conns 5/5; got {loads}"
            )
    finally:
        yb_ctl.start_node(2, placement_info="cloud1.datacenter1.rack1")
        time.sleep(2)


# --------------------------------------------------------------------- cluster-aware × add


def test_pool_uniform_load_after_node_addition(yb_cluster, yb_ctl):
    """Cluster-aware. Open pool 1 (9 conns) on the 3-node cluster → 3/3/3.
    Add node 4 in the same zone. Wait for refresh. Open pool 2 (9 conns)
    while pool 1's conns are still alive — pool 2 enters the picker with
    starting state [3, 3, 3, 0]. The picker's deterministic walk over 9
    more picks yields sorted shape [4, 4, 5, 5] (same derivation as
    `test_uniform_load_after_node_addition` for direct-connect)."""
    from psycopg.yb.registry import ClusterRegistry
    from tests.yb import conftest as yb_conftest

    refresh_s = 3
    dsn = yb_cluster + f" load_balance_hosts=true yb_servers_refresh_interval={refresh_s}"
    added = False
    try:
        with ConnectionPool(dsn, min_size=9, max_size=9) as pool1:
            pool1.wait()
            with pool1.connection() as conn:
                uuid = conn._yb_uuid
            registry = ClusterRegistry.instance()
            for h in ("127.0.0.1", "127.0.0.2", "127.0.0.3"):
                assert registry.get_load(uuid, h) == 3, (
                    f"pool 1 should distribute 3/3/3; got "
                    f"{ {x: registry.get_load(uuid, x) for x in ('127.0.0.1','127.0.0.2','127.0.0.3')} }"
                )

            # Add node 4.
            yb_ctl.add_node(placement_info="cloud1.datacenter1.rack1")
            added = True

            # Wait until yb_servers() shows .4, via a plain libpq connect.
            survivor_dsn = (
                "host=127.0.0.1 port=5433 user=yugabyte dbname=yugabyte"
            )
            deadline = time.monotonic() + 30
            seen_new = False
            while time.monotonic() < deadline:
                try:
                    with psycopg.connect(survivor_dsn) as c:
                        rows = c.execute(
                            "SELECT host FROM yb_servers()"
                        ).fetchall()
                        if "127.0.0.4" in {r[0] for r in rows}:
                            seen_new = True
                            break
                except Exception:
                    pass
                time.sleep(1)
            if not seen_new:
                pytest.skip("yb_servers() did not show the added node within 30s")
            # Sleep past refresh so the next pool's first connect triggers it.
            time.sleep(refresh_s + 1)

            # Open pool 2 — 9 more conns.
            with ConnectionPool(dsn, min_size=9, max_size=9) as pool2:
                pool2.wait()
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
            time.sleep(5)


# --------------------------------------------------------------------- topology × stop (in topology)


def test_pool_topology_stop_in_topology_zone(yb_multi_zone_cluster, yb_ctl):
    """Topology-aware. Multi-zone cluster (.1, .2 in zoneA; .3 in zoneB).
    Stop node 2 (a zoneA node) — leaves .1 as the only in-topology survivor.
    Open a pool with topology_keys=zoneA. All 6 pool conns must land on .1.
    The stopped .2 stays at 0 (mark_failed zeroed it on the first failed
    pick); the zoneB .3 stays at 0 (filtered out by topology, never
    considered)."""
    yb_ctl.stop_node(2)
    try:
        dsn = (yb_multi_zone_cluster
               + " load_balance_hosts=true topology_keys=cloud1.datacenter1.zoneA")
        with ConnectionPool(dsn, min_size=6, max_size=6) as pool:
            pool.wait()
            with pool.connection() as conn:
                uuid = conn._yb_uuid
            from psycopg.yb.registry import ClusterRegistry
            registry = ClusterRegistry.instance()
            loads = {
                h: registry.get_load(uuid, h)
                for h in ("127.0.0.1", "127.0.0.2", "127.0.0.3")
            }
            assert loads == {
                "127.0.0.1": 6,
                "127.0.0.2": 0,  # stopped, quarantined
                "127.0.0.3": 0,  # zoneB, out of topology
            }, f"all 6 conns should land on .1; got {loads}"
    finally:
        yb_ctl.start_node(2, placement_info="cloud1.datacenter1.zoneA")
        time.sleep(2)


# --------------------------------------------------------------------- topology × stop (out of topology)


def test_pool_topology_stop_out_of_topology_unaffected(
    yb_multi_zone_cluster, yb_ctl
):
    """Topology-aware. Stop node 3 (the zoneB node). With
    topology_keys=zoneA, the topology filter would never consider .3
    anyway — stopping it has no effect on pool distribution. 6 conns
    split exactly 3/3 across the two zoneA nodes."""
    yb_ctl.stop_node(3)
    try:
        dsn = (yb_multi_zone_cluster
               + " load_balance_hosts=true topology_keys=cloud1.datacenter1.zoneA")
        with ConnectionPool(dsn, min_size=6, max_size=6) as pool:
            pool.wait()
            with pool.connection() as conn:
                uuid = conn._yb_uuid
            from psycopg.yb.registry import ClusterRegistry
            registry = ClusterRegistry.instance()
            loads = {
                h: registry.get_load(uuid, h)
                for h in ("127.0.0.1", "127.0.0.2", "127.0.0.3")
            }
            assert loads == {
                "127.0.0.1": 3,
                "127.0.0.2": 3,
                "127.0.0.3": 0,  # stopped AND out of topology
            }, f"stopped out-of-topology node must not perturb zoneA traffic; got {loads}"
    finally:
        yb_ctl.start_node(3, placement_info="cloud1.datacenter1.zoneB")
        time.sleep(2)


# --------------------------------------------------------------------- topology × add (in topology)


def test_pool_topology_add_in_topology_zone_attracts_traffic(
    yb_multi_zone_cluster, yb_ctl
):
    """Topology-aware. Bootstrap a sentinel conn so the cluster state is
    populated, then add node 4 in zoneA. Open the test pool with 9 conns
    and topology_keys=zoneA — the picker considers .1, .2, .4 (all zoneA)
    and distributes evenly across the three: 3/3/3. zoneB (.3) stays at
    zero."""
    from psycopg.yb.registry import ClusterRegistry
    from tests.yb import conftest as yb_conftest

    refresh_s = 3
    base_dsn = (yb_multi_zone_cluster
                + f" load_balance_hosts=true "
                f"topology_keys=cloud1.datacenter1.zoneA "
                f"yb_servers_refresh_interval={refresh_s}")
    added = False
    try:
        # Bootstrap via a short-lived sentinel pool so the cluster state
        # is in the registry before we add the node.
        with ConnectionPool(base_dsn, min_size=1, max_size=1) as sentinel:
            sentinel.wait()
            with sentinel.connection() as c:
                uuid = c._yb_uuid

        yb_ctl.add_node(placement_info="cloud1.datacenter1.zoneA")
        added = True

        # Wait until .4 shows in yb_servers().
        survivor_dsn = (
            "host=127.0.0.1 port=5433 user=yugabyte dbname=yugabyte"
        )
        deadline = time.monotonic() + 30
        seen_new = False
        while time.monotonic() < deadline:
            try:
                with psycopg.connect(survivor_dsn) as c:
                    rows = c.execute("SELECT host FROM yb_servers()").fetchall()
                    if "127.0.0.4" in {r[0] for r in rows}:
                        seen_new = True
                        break
            except Exception:
                pass
            time.sleep(1)
        if not seen_new:
            pytest.skip("yb_servers() did not show the added node within 30s")
        time.sleep(refresh_s + 1)

        with ConnectionPool(base_dsn, min_size=9, max_size=9) as pool:
            pool.wait()
            registry = ClusterRegistry.instance()
            loads = {
                h: registry.get_load(uuid, h)
                for h in ("127.0.0.1", "127.0.0.2", "127.0.0.3", "127.0.0.4")
            }
            assert loads == {
                "127.0.0.1": 3,
                "127.0.0.2": 3,
                "127.0.0.3": 0,  # zoneB stays out
                "127.0.0.4": 3,  # newly-added zoneA gets its share
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
            time.sleep(5)


# --------------------------------------------------------------------- topology × add (out of topology)


def test_pool_topology_add_out_of_topology_zone_no_traffic(
    yb_multi_zone_cluster, yb_ctl
):
    """Topology-aware. Add node 4 in zoneB. Open the test pool with
    topology_keys=zoneA — the new node is out-of-topology and must
    receive zero traffic. 6 pool conns split 3/3 on .1/.2; both .3 and
    .4 stay at zero."""
    from psycopg.yb.registry import ClusterRegistry
    from tests.yb import conftest as yb_conftest

    refresh_s = 3
    base_dsn = (yb_multi_zone_cluster
                + f" load_balance_hosts=true "
                f"topology_keys=cloud1.datacenter1.zoneA "
                f"yb_servers_refresh_interval={refresh_s}")
    added = False
    try:
        with ConnectionPool(base_dsn, min_size=1, max_size=1) as sentinel:
            sentinel.wait()
            with sentinel.connection() as c:
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
                with psycopg.connect(survivor_dsn) as c:
                    rows = c.execute("SELECT host FROM yb_servers()").fetchall()
                    if "127.0.0.4" in {r[0] for r in rows}:
                        seen_new = True
                        break
            except Exception:
                pass
            time.sleep(1)
        if not seen_new:
            pytest.skip("yb_servers() did not show the added node within 30s")
        time.sleep(refresh_s + 1)

        with ConnectionPool(base_dsn, min_size=6, max_size=6) as pool:
            pool.wait()
            registry = ClusterRegistry.instance()
            loads = {
                h: registry.get_load(uuid, h)
                for h in ("127.0.0.1", "127.0.0.2", "127.0.0.3", "127.0.0.4")
            }
            assert loads == {
                "127.0.0.1": 3,
                "127.0.0.2": 3,
                "127.0.0.3": 0,
                "127.0.0.4": 0,  # newly-added zoneB stays out of topology
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
            time.sleep(5)
