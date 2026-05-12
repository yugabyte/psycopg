"""
Async sibling of `test_smart_driver_topology_failover.py`.

Every topology × cluster-state scenario in the sync suite has an async
counterpart here:

  * adding a node IN the bound topology zone attracts traffic
  * adding a node OUTSIDE the bound topology zone receives none
  * stopping a node IN topology redistributes within the zone (strict, no
    cluster-wide fallback)
  * stopping a node OUTSIDE topology has no effect on in-topology traffic
  * restarting an in-topology node re-enters rotation

Cluster shape (from the `yb_multi_zone_cluster` fixture): 2 nodes in zoneA
(127.0.0.1, 127.0.0.2), 1 in zoneB (127.0.0.3). Topology is bound to zoneA
throughout so the "in topology" / "outside topology" axis is unambiguous.
"""

# Copyright (C) 2026 Yugabyte

from __future__ import annotations

import asyncio

import pytest

import psycopg


pytestmark = pytest.mark.anyio


_REFRESH_S = 2
_ZONE_A_TOPOLOGY = "cloud1.datacenter1.zoneA"


def _zone_a_dsn(base_dsn: str, *, refresh_s: int = _REFRESH_S) -> str:
    return (
        f"{base_dsn} load_balance_hosts=true "
        f"topology_keys={_ZONE_A_TOPOLOGY} "
        f"yb_servers_refresh_interval={refresh_s}"
    )


# --------------------------------------------------------------------- ADD node


async def test_async_add_node_in_topology_zone_receives_traffic(
    yb_multi_zone_cluster, yb_ctl
):
    """Async sibling of `test_add_node_in_topology_zone_receives_traffic`."""
    from psycopg.yb.registry import ClusterRegistry
    dsn = _zone_a_dsn(yb_multi_zone_cluster)

    initial = []
    new: list = []
    try:
        for _ in range(6):
            initial.append(await psycopg.AsyncConnection.connect(dsn))
        uuid = initial[0]._yb_uuid
        registry = ClusterRegistry.instance()

        # Phase 1: no traffic in zoneB.
        assert registry.get_load(uuid, "127.0.0.3") == 0
        assert (registry.get_load(uuid, "127.0.0.1")
                + registry.get_load(uuid, "127.0.0.2")) == 6

        # Phase 2: add a 4th node in zoneA.
        yb_ctl.add_node(placement_info=_ZONE_A_TOPOLOGY)
        await asyncio.sleep(_REFRESH_S + 1)

        # Warm: triggers the refresh.
        warm = await psycopg.AsyncConnection.connect(dsn)
        await warm.close()

        # Phase 3: open 6 more, observe the deterministic per-host shape.
        # See sync sibling for the step-by-step least-loaded walk; final
        # in-topology counts are .1=4, .2=4, .4=4; zoneB (.3) stays at 0.
        for _ in range(6):
            new.append(await psycopg.AsyncConnection.connect(dsn))

        loads = {
            h: registry.get_load(uuid, h)
            for h in ("127.0.0.1", "127.0.0.2", "127.0.0.3", "127.0.0.4")
        }
        assert loads == {
            "127.0.0.1": 4,
            "127.0.0.2": 4,
            "127.0.0.3": 0,
            "127.0.0.4": 4,
        }, f"phase 3 per-host counts mismatch: {loads}"
    finally:
        for c in initial:
            await c.close()
        for c in new:
            await c.close()


async def test_async_add_node_outside_topology_zone_receives_no_traffic(
    yb_multi_zone_cluster, yb_ctl
):
    """Async sibling of `test_add_node_outside_topology_zone_receives_no_traffic`."""
    from psycopg.yb.registry import ClusterRegistry
    dsn = _zone_a_dsn(yb_multi_zone_cluster)

    boot = await psycopg.AsyncConnection.connect(dsn)
    uuid = boot._yb_uuid
    await boot.close()

    yb_ctl.add_node(placement_info="cloud1.datacenter1.zoneB")
    await asyncio.sleep(_REFRESH_S + 1)
    refresh = await psycopg.AsyncConnection.connect(dsn)
    await refresh.close()

    conns = []
    try:
        for _ in range(8):
            conns.append(await psycopg.AsyncConnection.connect(dsn))
        registry = ClusterRegistry.instance()
        for not_in_topology in ("127.0.0.3", "127.0.0.4"):
            assert registry.get_load(uuid, not_in_topology) == 0, (
                f"out-of-topology node {not_in_topology} got traffic; "
                + repr({
                    h: registry.get_load(uuid, h)
                    for h in ("127.0.0.1", "127.0.0.2", "127.0.0.3", "127.0.0.4")
                })
            )
        assert (registry.get_load(uuid, "127.0.0.1")
                + registry.get_load(uuid, "127.0.0.2")) == 8
    finally:
        for c in conns:
            await c.close()


# --------------------------------------------------------------------- STOP node


async def test_async_stop_node_in_topology_redistributes_within_topology(
    yb_multi_zone_cluster, yb_ctl
):
    """Async sibling of `test_stop_node_in_topology_redistributes_within_topology`."""
    from psycopg.yb.registry import ClusterRegistry
    dsn = _zone_a_dsn(yb_multi_zone_cluster)

    boot = await psycopg.AsyncConnection.connect(dsn)
    uuid = boot._yb_uuid
    await boot.close()

    yb_ctl.stop_node(2)

    conns = []
    try:
        for _ in range(6):
            conns.append(await psycopg.AsyncConnection.connect(dsn))
        registry = ClusterRegistry.instance()
        load_1 = registry.get_load(uuid, "127.0.0.1")
        load_2 = registry.get_load(uuid, "127.0.0.2")
        load_3 = registry.get_load(uuid, "127.0.0.3")
        assert load_2 == 0, f"stopped node got traffic: load_2={load_2}"
        assert load_3 == 0, (
            f"out-of-topology node got traffic: load_3={load_3} "
            f"(topology filter is strict in v1)"
        )
        assert load_1 == 6, f"all conns should be on 127.0.0.1; got {load_1}"
    finally:
        for c in conns:
            await c.close()


async def test_async_stop_node_outside_topology_does_not_affect_topology_traffic(
    yb_multi_zone_cluster, yb_ctl
):
    """Async sibling of `test_stop_node_outside_topology_does_not_affect_topology_traffic`."""
    from psycopg.yb.registry import ClusterRegistry
    dsn = _zone_a_dsn(yb_multi_zone_cluster)

    boot = await psycopg.AsyncConnection.connect(dsn)
    uuid = boot._yb_uuid
    await boot.close()

    yb_ctl.stop_node(3)

    conns = []
    try:
        for _ in range(6):
            conns.append(await psycopg.AsyncConnection.connect(dsn))
        registry = ClusterRegistry.instance()
        assert registry.get_load(uuid, "127.0.0.3") == 0
        total_zoneA = (registry.get_load(uuid, "127.0.0.1")
                       + registry.get_load(uuid, "127.0.0.2"))
        assert total_zoneA == 6
    finally:
        for c in conns:
            await c.close()


async def test_async_restart_node_in_topology_re_enters_rotation(
    yb_multi_zone_cluster, yb_ctl
):
    """Async sibling of `test_restart_node_in_topology_re_enters_rotation`."""
    from psycopg.yb.registry import ClusterRegistry
    dsn = _zone_a_dsn(yb_multi_zone_cluster)

    boot = await psycopg.AsyncConnection.connect(dsn)
    uuid = boot._yb_uuid
    await boot.close()

    yb_ctl.stop_node(2)

    pre_restart = []
    try:
        for _ in range(3):
            pre_restart.append(await psycopg.AsyncConnection.connect(dsn))
        registry = ClusterRegistry.instance()
        assert registry.get_load(uuid, "127.0.0.1") == 3
        assert registry.get_load(uuid, "127.0.0.2") == 0

        yb_ctl.start_node(2, placement_info=_ZONE_A_TOPOLOGY)
        await asyncio.sleep(max(_REFRESH_S, 5))

        post_restart = []
        try:
            for _ in range(6):
                post_restart.append(await psycopg.AsyncConnection.connect(dsn))
            # See sync sibling for the step-by-step least-loaded walk.
            # Sorted final shape on in-topology nodes: [4, 5]. One ends at 4,
            # the other at 5 (random permutation). zoneB stays at 0.
            loads = {
                h: registry.get_load(uuid, h)
                for h in ("127.0.0.1", "127.0.0.2", "127.0.0.3")
            }
            assert sorted(loads.values()) == [0, 4, 5], (
                f"sorted shape should be [0, 4, 5]; got {sorted(loads.values())} "
                f"from {loads}"
            )
            assert loads["127.0.0.3"] == 0
            assert {loads["127.0.0.1"], loads["127.0.0.2"]} == {4, 5}, (
                f"in-topology nodes should split [4, 5]; got "
                f"{loads['127.0.0.1']}, {loads['127.0.0.2']}"
            )
        finally:
            for c in post_restart:
                await c.close()
    finally:
        for c in pre_restart:
            await c.close()
