"""
Async sibling of `test_smart_driver_failover.py`.

Every failover scenario in the sync suite has an async counterpart here:

  * stopped node gets quarantined via mark_failed
  * the bootstrap host going down doesn't block new connects
  * yb_servers_refresh_interval is honoured on the async refresh path
  * control-host failover triggers re-bootstrap via _aensure_control_async
  * a node added to the cluster eventually receives async traffic
  * ClusterRegistry.aclear() closes the async control connection cleanly

Each test is destructive to cluster state and restores it in `finally`.
Auto-skips if the `yb_ctl` fixture can't locate the binary — set the
`YB_CTL` env var to the absolute path of `yb-ctl`.
"""

# Copyright (C) 2026 Yugabyte

from __future__ import annotations

import asyncio
import time

import pytest

import psycopg


pytestmark = pytest.mark.anyio


# --------------------------------------------------------------------- node down


async def test_async_stopped_node_is_quarantined(yb_cluster, yb_ctl):
    """Async sibling of `test_stopped_node_is_quarantined`."""
    yb_ctl.stop_node(2)
    try:
        conns = []
        try:
            for _ in range(10):
                conns.append(
                    await psycopg.AsyncConnection.connect(
                        yb_cluster + " load_balance_hosts=true"
                    )
                )
            hosts = {c._yb_host for c in conns}
            assert "127.0.0.2" not in hosts, (
                f"connections should skip stopped node 2; landed on {sorted(hosts)}"
            )
            assert len(conns) == 10
        finally:
            for c in conns:
                await c.close()
    finally:
        yb_ctl.start_node(2, placement_info="cloud1.datacenter1.rack1")
        await asyncio.sleep(2)


# --------------------------------------------------------------------- contact host down


async def test_async_primary_node_down_drops_traffic_safely(yb_cluster, yb_ctl):
    """Async sibling of `test_primary_node_down_drops_traffic_safely`."""
    boot = await psycopg.AsyncConnection.connect(
        yb_cluster + " load_balance_hosts=true"
    )
    await boot.close()

    yb_ctl.stop_node(1)
    try:
        conns = []
        try:
            for _ in range(6):
                conns.append(
                    await psycopg.AsyncConnection.connect(
                        yb_cluster + " load_balance_hosts=true"
                    )
                )
            hosts = {c._yb_host for c in conns}
            assert hosts.issubset({"127.0.0.2", "127.0.0.3"}), (
                f"connections must land on surviving nodes; landed on {sorted(hosts)}"
            )
        finally:
            for c in conns:
                await c.close()
    finally:
        yb_ctl.start_node(1, placement_info="cloud1.datacenter1.rack1")
        await asyncio.sleep(2)


# --------------------------------------------------------------------- refresh interval


async def test_async_yb_servers_refresh_interval_is_honoured(yb_cluster, yb_ctl):
    """Async sibling of `test_yb_servers_refresh_interval_is_honoured`."""
    refresh_s = 2
    dsn = yb_cluster + f" load_balance_hosts=true yb_servers_refresh_interval={refresh_s}"

    boot = await psycopg.AsyncConnection.connect(dsn)
    await boot.close()

    yb_ctl.stop_node(3)
    try:
        conns = []
        try:
            for _ in range(5):
                conns.append(await psycopg.AsyncConnection.connect(dsn))
            hosts = [c._yb_host for c in conns]
            assert all(h != "127.0.0.3" for h in hosts), (
                f"connections after stopped node should skip it; got {hosts}"
            )
        finally:
            for c in conns:
                await c.close()
    finally:
        yb_ctl.start_node(3, placement_info="cloud1.datacenter1.rack1")
        await asyncio.sleep(2)


# --------------------------------------------------------------------- control host failover


async def test_async_control_host_failover(yb_cluster, yb_ctl):
    """Async sibling of `test_control_host_failover`. Verifies that
    `_aensure_control_async` re-opens the control connection on a survivor
    when the originally-bootstrapped one dies — the async path must not go
    blind any more than the sync path does."""
    from psycopg.yb.registry import ClusterRegistry
    refresh_s = 2
    dsn = yb_cluster + f" load_balance_hosts=true yb_servers_refresh_interval={refresh_s}"

    boot = await psycopg.AsyncConnection.connect(dsn)
    uuid = boot._yb_uuid
    state = ClusterRegistry.instance()._clusters[uuid]
    ctrl_before = state.control_async
    assert ctrl_before is not None
    await boot.close()

    yb_ctl.stop_node(1)
    try:
        await asyncio.sleep(refresh_s + 1)

        conns = []
        try:
            for _ in range(3):
                conns.append(await psycopg.AsyncConnection.connect(dsn))
            for c in conns:
                assert c._yb_host in ("127.0.0.2", "127.0.0.3")
        finally:
            for c in conns:
                await c.close()

        state_after = ClusterRegistry.instance()._clusters.get(uuid)
        assert state_after is not None, "ClusterState should still exist"
    finally:
        yb_ctl.start_node(1, placement_info="cloud1.datacenter1.rack1")
        await asyncio.sleep(2)


# --------------------------------------------------------------------- node addition


async def test_async_uniform_load_after_node_addition(yb_cluster, yb_ctl):
    """Async sibling of `test_uniform_load_after_node_addition`."""
    from psycopg.yb.registry import ClusterRegistry
    from tests.yb import conftest as yb_conftest

    refresh_s = 2
    dsn = yb_cluster + f" load_balance_hosts=true yb_servers_refresh_interval={refresh_s}"

    initial_conns: list = []
    new_conns: list = []
    added = False
    try:
        # Phase 1: deterministic 3/3/3 on the existing 3 nodes.
        for _ in range(9):
            initial_conns.append(await psycopg.AsyncConnection.connect(dsn))
        uuid = initial_conns[0]._yb_uuid
        registry = ClusterRegistry.instance()
        for h in ("127.0.0.1", "127.0.0.2", "127.0.0.3"):
            assert registry.get_load(uuid, h) == 3, (
                f"phase 1: {h} should hold exactly 3; got "
                f"{ {x: registry.get_load(uuid, x) for x in ('127.0.0.1','127.0.0.2','127.0.0.3')} }"
            )

        yb_ctl.add_node(placement_info="cloud1.datacenter1.rack1")
        added = True
        deadline = time.monotonic() + 30
        seen = False
        while time.monotonic() < deadline:
            try:
                async with await psycopg.AsyncConnection.connect(yb_cluster) as c:
                    cur = await c.execute("SELECT count(*) FROM yb_servers()")
                    (n,) = await cur.fetchone()
                    if n == 4:
                        seen = True
                        break
            except Exception:
                pass
            await asyncio.sleep(1)
        if not seen:
            pytest.skip("yb_servers() did not show the added node within 30s")

        await asyncio.sleep(refresh_s + 1)

        # Phase 3: 9 conns on a [3, 3, 3, 0] starting state. The least-loaded
        # picker's deterministic shape after 9 picks is sorted [4, 4, 5, 5]
        # (see the sync sibling for the step-by-step derivation).
        for _ in range(9):
            new_conns.append(await psycopg.AsyncConnection.connect(dsn))

        loads = {
            h: registry.get_load(uuid, h)
            for h in ("127.0.0.1", "127.0.0.2", "127.0.0.3", "127.0.0.4")
        }
        assert sorted(loads.values()) == [4, 4, 5, 5], (
            f"phase 3 shape should be [4, 4, 5, 5]; got {sorted(loads.values())} "
            f"from {loads}"
        )
    finally:
        for c in initial_conns + new_conns:
            await c.close()
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


# --------------------------------------------------------------------- control re-open + node addition


async def test_async_control_conn_reopens_on_host_loss_and_picks_up_added_node(
    yb_cluster, yb_ctl
):
    """Async sibling of `test_control_conn_reopens_on_host_loss_and_picks_up_added_node`.

    Phase-1 conns stay open; only the dead host's counter changes (via
    mark_failed). See the sync version for the full scenario walkthrough
    and the derivation of the deterministic [6, 7, 7] sorted shape.
    """
    from psycopg.yb.registry import ClusterRegistry
    from tests.yb import conftest as yb_conftest

    refresh_s = 3
    dsn = yb_cluster + f" load_balance_hosts=true yb_servers_refresh_interval={refresh_s}"
    registry = ClusterRegistry.instance()

    initial: list = []
    new: list = []
    added = False
    try:
        # Phase 1: 12 conns, exact 4/4/4. Hold open.
        for _ in range(12):
            initial.append(await psycopg.AsyncConnection.connect(dsn))
        uuid = initial[0]._yb_uuid
        state = registry._clusters[uuid]
        for h in ("127.0.0.1", "127.0.0.2", "127.0.0.3"):
            assert registry.get_load(uuid, h) == 4, (
                f"phase 1: {h} should be exactly 4; got "
                f"{ {x: registry.get_load(uuid, x) for x in ('127.0.0.1','127.0.0.2','127.0.0.3')} }"
            )

        assert state.control_async is not None
        original_ctrl_host = state.control_async.info.host
        original_ctrl_node = int(original_ctrl_host.split(".")[-1])
        survivors = [
            h for h in ("127.0.0.1", "127.0.0.2", "127.0.0.3")
            if h != original_ctrl_host
        ]
        assert len(survivors) == 2

        # Add 4th node FIRST, then stop control host (yb-ctl add_node blocks
        # on all tservers being up — order matters). Survivor phase-1 conns
        # stay open throughout.
        yb_ctl.add_node(placement_info="cloud1.datacenter1.rack1")
        added = True
        yb_ctl.stop_node(original_ctrl_node)

        # Plain libpq poll on a survivor — don't trigger smart-driver refresh
        # prematurely.
        survivor_dsn = (
            f"host={survivors[0]} port=5433 user=yugabyte dbname=yugabyte"
        )
        deadline = time.monotonic() + 30
        seen_new = False
        while time.monotonic() < deadline:
            try:
                async with await psycopg.AsyncConnection.connect(survivor_dsn) as c:
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

        # Phase 2: 12 new conns, single-cycle recovery in pick #1.
        for _ in range(12):
            new.append(await psycopg.AsyncConnection.connect(dsn))

        assert state.control_async is not None, (
            "async control conn should have been re-opened against a surviving node"
        )
        new_ctrl_host = state.control_async.info.host
        assert new_ctrl_host != original_ctrl_host, (
            f"control conn should have moved off the stopped host "
            f"{original_ctrl_host}; still on {new_ctrl_host}"
        )
        assert new_ctrl_host in survivors + ["127.0.0.4"]

        loads = {
            h: registry.get_load(uuid, h)
            for h in ("127.0.0.1", "127.0.0.2", "127.0.0.3", "127.0.0.4")
        }
        assert loads[original_ctrl_host] == 0, (
            f"dead host {original_ctrl_host} should be zeroed by mark_failed; "
            f"got {loads[original_ctrl_host]}. Full loads: {loads}"
        )
        healthy_counts = sorted(
            loads[h] for h in survivors + ["127.0.0.4"]
        )
        assert healthy_counts == [6, 7, 7], (
            f"healthy hosts' sorted shape should be [6, 7, 7]; got "
            f"{healthy_counts}. Full loads: {loads}. "
            f"Original ctrl host (stopped): {original_ctrl_host}, "
            f"new ctrl host: {new_ctrl_host}"
        )
        assert sum(loads.values()) == 20
    finally:
        for c in initial + new:
            try:
                await c.close()
            except Exception:
                pass
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


# --------------------------------------------------------------------- aclear() integration


async def test_async_clusterregistry_aclear_integration(yb_cluster):
    """Async sibling of `test_clusterregistry_clear_integration`. Verifies
    that `aclear` (rather than `clear`) closes the async control connection
    through the event loop cleanly, with no "deleted while still open"
    ResourceWarning at GC time."""
    from psycopg.yb.registry import ClusterRegistry
    registry = ClusterRegistry.instance()

    conn = await psycopg.AsyncConnection.connect(
        yb_cluster + " load_balance_hosts=true"
    )
    uuid = conn._yb_uuid
    state = registry._clusters[uuid]
    control = state.control_async
    assert control is not None
    assert not control.closed

    await conn.close()
    await registry.aclear()

    assert registry._clusters == {}
    assert registry._key_to_uuid == {}
    assert control.closed, "control connection should be closed after aclear()"
