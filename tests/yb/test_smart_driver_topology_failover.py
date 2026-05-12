"""
Topology × node-state interaction tests.

These verify that topology-key binding interacts correctly with cluster
state changes:

  * adding a node IN the bound topology zone → it receives new traffic
  * adding a node OUTSIDE the bound topology zone → it stays at zero
  * stopping a node IN the bound topology zone → traffic redistributes
    within the remaining in-topology nodes
  * stopping a node OUTSIDE the bound topology zone → no impact on
    topology-bound traffic
  * restarting a previously-stopped node IN the topology zone → it
    re-enters rotation

Cluster shape (from `yb_multi_zone_cluster` fixture): 2 nodes in zoneA
(127.0.0.1, 127.0.0.2), 1 in zoneB (127.0.0.3). Tests typically bind
topology to zoneA so that:
  * "in topology" = 127.0.0.1 and 127.0.0.2
  * "outside topology" = 127.0.0.3
  * stopping one in-topology node still leaves one in-topology node
    eligible
"""

# Copyright (C) 2026 Yugabyte

from __future__ import annotations

import time

import pytest

import psycopg


# Refresh fast enough that tests don't need to wait minutes for the smart
# driver to pick up cluster changes.
_REFRESH_S = 2

# The topology binding used throughout this file.
_ZONE_A_TOPOLOGY = "cloud1.datacenter1.zoneA"

# Convenience: build the smart-driver DSN with this file's standard knobs.
def _zone_a_dsn(base_dsn: str, *, refresh_s: int = _REFRESH_S) -> str:
    return (
        f"{base_dsn} load_balance_hosts=true "
        f"topology_keys={_ZONE_A_TOPOLOGY} "
        f"yb_servers_refresh_interval={refresh_s}"
    )


# --------------------------------------------------------------------- ADD node


def test_add_node_in_topology_zone_receives_traffic(yb_multi_zone_cluster, yb_ctl):
    """Topology bound to zoneA. The fresh cluster has 2 nodes in zoneA. Add a
    3rd node in zoneA. After refresh, new connections distribute across all
    three in-topology nodes."""
    from psycopg.yb.registry import ClusterRegistry
    dsn = _zone_a_dsn(yb_multi_zone_cluster)

    # Open 6 connections — 3 to each zoneA host.
    initial = []
    try:
        for _ in range(6):
            initial.append(psycopg.connect(dsn))
        uuid = initial[0]._yb_uuid
        registry = ClusterRegistry.instance()

        # Phase 1 sanity: no traffic in zoneB (127.0.0.3).
        assert registry.get_load(uuid, "127.0.0.3") == 0
        assert registry.get_load(uuid, "127.0.0.1") + registry.get_load(uuid, "127.0.0.2") == 6

        # Phase 2: add a 4th node in zoneA.
        yb_ctl.add_node(placement_info=_ZONE_A_TOPOLOGY)
        # yb-ctl conventionally adds node 4 at 127.0.0.4.
        # Wait past the refresh window for the driver to see the new node.
        time.sleep(_REFRESH_S + 1)

        # Trigger a refresh by opening one connection (refresh fires on the
        # next connect after the interval).
        warm = psycopg.connect(dsn)
        # The warm conn may pick the new node (because it has count=0 and the
        # existing two have count=3 each).
        warm.close()
        # Decrement happens after close; account for it.

        # Phase 3: open 6 more connections.
        #
        # State entering phase 3 (after warm conn opened+closed): the warm
        # conn went to .4 (only host at 0), counter +1 then -1 on close, so
        # we're back at [.1=3, .2=3, .4=0] with zoneB (.3) at 0.
        #
        # Walking the in-topology least-loaded picker for 6 picks:
        #   pick 1: .4 (count=0) → [3, 3, 0, 1]
        #   pick 2: .4 → [3, 3, 0, 2]
        #   pick 3: .4 → [3, 3, 0, 3]
        #   pick 4: tied at 3 across {.1, .2, .4}, random → one gets +1
        #   pick 5: 2 of the 3 left at 3 → one gets +1
        #   pick 6: 1 left at 3 → +1
        # So each of {.1, .2, .4} receives exactly 1 from picks 4-6, in some
        # random order. Final counts: .1=4, .2=4, .4=4, .3=0. Per-host exact.
        new = []
        for _ in range(6):
            new.append(psycopg.connect(dsn))

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
        new.append(warm)  # not strictly needed but cleaner cleanup
    finally:
        for c in initial:
            c.close()
        try:
            for c in new:  # type: ignore[name-defined]
                c.close()
        except (NameError, AttributeError):
            pass
        # Cluster will be destroyed by the fixture teardown — no manual cleanup.


def test_add_node_outside_topology_zone_receives_no_traffic(
    yb_multi_zone_cluster, yb_ctl
):
    """Topology bound to zoneA. Add a new node in zoneB. The new node MUST
    receive zero traffic — connections stay in zoneA."""
    from psycopg.yb.registry import ClusterRegistry
    dsn = _zone_a_dsn(yb_multi_zone_cluster)

    # Bootstrap.
    boot = psycopg.connect(dsn)
    uuid = boot._yb_uuid
    boot.close()

    # Add a node in zoneB (outside topology).
    yb_ctl.add_node(placement_info="cloud1.datacenter1.zoneB")
    time.sleep(_REFRESH_S + 1)
    # Trigger refresh.
    psycopg.connect(dsn).close()

    # Open 8 connections — none should land in zoneB.
    conns = []
    try:
        for _ in range(8):
            conns.append(psycopg.connect(dsn))
        registry = ClusterRegistry.instance()
        for not_in_topology in ("127.0.0.3", "127.0.0.4"):
            assert registry.get_load(uuid, not_in_topology) == 0, (
                f"out-of-topology node {not_in_topology} got traffic; "
                + repr({
                    h: registry.get_load(uuid, h)
                    for h in ("127.0.0.1", "127.0.0.2", "127.0.0.3", "127.0.0.4")
                })
            )
        # All 8 conns are on the two zoneA nodes.
        assert (registry.get_load(uuid, "127.0.0.1")
                + registry.get_load(uuid, "127.0.0.2")) == 8
    finally:
        for c in conns:
            c.close()


# --------------------------------------------------------------------- STOP node


def test_stop_node_in_topology_redistributes_within_topology(
    yb_multi_zone_cluster, yb_ctl
):
    """Topology bound to zoneA (2 nodes). Stop one zoneA node. New connections
    should all land on the remaining zoneA node — and never spill to zoneB,
    because the topology filter is hard (no cluster-wide fallback in v1)."""
    from psycopg.yb.registry import ClusterRegistry
    dsn = _zone_a_dsn(yb_multi_zone_cluster)

    # Bootstrap.
    boot = psycopg.connect(dsn)
    uuid = boot._yb_uuid
    boot.close()

    # Stop one of the zoneA nodes (node 2 = 127.0.0.2).
    yb_ctl.stop_node(2)

    # Open new connections. They must:
    #   - not land on the stopped node
    #   - not land on the zoneB node (topology binding is strict)
    #   - all land on the remaining zoneA node (127.0.0.1)
    conns = []
    try:
        for _ in range(6):
            conns.append(psycopg.connect(dsn))
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
            c.close()


def test_stop_node_outside_topology_does_not_affect_topology_traffic(
    yb_multi_zone_cluster, yb_ctl
):
    """Topology bound to zoneA. Stop the zoneB node. Traffic in zoneA is
    unaffected — no detour to a now-unavailable node."""
    from psycopg.yb.registry import ClusterRegistry
    dsn = _zone_a_dsn(yb_multi_zone_cluster)

    # Bootstrap.
    boot = psycopg.connect(dsn)
    uuid = boot._yb_uuid
    boot.close()

    # Stop the zoneB node (node 3).
    yb_ctl.stop_node(3)

    # Traffic continues in zoneA.
    conns = []
    try:
        for _ in range(6):
            conns.append(psycopg.connect(dsn))
        registry = ClusterRegistry.instance()
        assert registry.get_load(uuid, "127.0.0.3") == 0, (
            "stopped (out-of-topology) node should remain at zero"
        )
        # Traffic balanced between the two in-topology zoneA nodes (3-3).
        total_zoneA = (registry.get_load(uuid, "127.0.0.1")
                       + registry.get_load(uuid, "127.0.0.2"))
        assert total_zoneA == 6
    finally:
        for c in conns:
            c.close()


def test_restart_node_in_topology_re_enters_rotation(
    yb_multi_zone_cluster, yb_ctl
):
    """Stop a zoneA node, then restart it. After refresh, the restarted node
    should re-enter the rotation and receive its share of new traffic."""
    from psycopg.yb.registry import ClusterRegistry
    dsn = _zone_a_dsn(yb_multi_zone_cluster)

    # Bootstrap.
    boot = psycopg.connect(dsn)
    uuid = boot._yb_uuid
    boot.close()

    # Stop node 2 (a zoneA node).
    yb_ctl.stop_node(2)

    # Open 3 conns; all go to node 1 because node 2 is dead and node 3 isn't
    # in the topology zone.
    pre_restart = []
    try:
        for _ in range(3):
            pre_restart.append(psycopg.connect(dsn))
        registry = ClusterRegistry.instance()
        assert registry.get_load(uuid, "127.0.0.1") == 3
        assert registry.get_load(uuid, "127.0.0.2") == 0

        # Restart node 2.
        yb_ctl.start_node(2, placement_info=_ZONE_A_TOPOLOGY)
        # Wait past the smart driver's refresh interval so it picks up the
        # change. yb_servers() may also need a moment to acknowledge.
        time.sleep(max(_REFRESH_S, 5))

        # Force a refresh by opening one connection past the interval, then
        # open more to verify node 2 is back in rotation. Node 2 has count=0
        # while node 1 has count=3, so the picker should heavily favour node 2.
        post_restart = []
        try:
            for _ in range(6):
                post_restart.append(psycopg.connect(dsn))
            # State entering this phase: [.1=3, .2=0 (re-eligible after TTL),
            # .3=0]. In-topology candidates are .1 and .2.
            #   pick 1: .2 (least) → [3, 1, 0]
            #   pick 2: .2 → [3, 2, 0]
            #   pick 3: .2 → [3, 3, 0]
            #   pick 4: tied at 3, random → +1 on one
            #   pick 5: 1 left at 3 → +1
            #   pick 6: tied at 4, random → +1 on one
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
                c.close()
    finally:
        for c in pre_restart:
            c.close()
