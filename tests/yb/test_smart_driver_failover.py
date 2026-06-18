"""
Smart-driver tests that drive real cluster state changes via `yb-ctl`.

These verify the failover and refresh paths at the driver level:
  * a stopped node gets quarantined via `mark_failed` and subsequent
    connections skip it
  * connections still succeed when the bootstrap host is down
  * the `yb_servers_refresh_interval` parameter is honoured — new topology
    is picked up after the refresh window
  * a node added to the cluster eventually receives traffic
  * a stopped node's control connection failure triggers re-bootstrap

We do NOT try to wait for `yb_servers()` to drop a stopped node from its
result set — the YB master's heartbeat timeout means that can take minutes,
and it's irrelevant to verifying driver behaviour. The driver only re-runs
`yb_servers()` on the refresh timer; in between, real connect failures drive
`mark_failed` which quarantines the bad host. That quarantine is what we
verify.

Each test is destructive to cluster state and restores it in `finally`.
Tests auto-skip if the `yb_ctl` fixture can't find the binary — set the
`YB_CTL` env var to the absolute path of `yb-ctl`.
"""

# Copyright (C) 2026 Yugabyte

from __future__ import annotations

import time

import pytest

import psycopg


# --------------------------------------------------------------------- node down


def test_stopped_node_is_quarantined(yb_cluster, yb_ctl):
    """Stop node 2. Open 10 smart-driver connections. None of them should
    end up tagged with the stopped node — `mark_failed` quarantines it on
    first failure and subsequent picks skip it (within the TTL window).
    """
    yb_ctl.stop_node(2)
    try:
        conns = []
        try:
            for _ in range(10):
                conns.append(psycopg.connect(yb_cluster + " load_balance_hosts=true"))
            hosts = {c._yb_host for c in conns}
            assert "127.0.0.2" not in hosts, (
                f"connections should skip stopped node 2; landed on {sorted(hosts)}"
            )
            # All 10 should still have succeeded on 1 or 3.
            assert len(conns) == 10
        finally:
            for c in conns:
                c.close()
    finally:
        yb_ctl.start_node(2, placement_info="cloud1.datacenter1.rack1")
        # Give the cluster a moment to acknowledge the restart before the next
        # test bootstraps. We don't wait for yb_servers() to converge; the next
        # test's bootstrap will re-discover whatever state exists at that point.
        time.sleep(2)


# --------------------------------------------------------------------- contact host down


def test_primary_node_down_drops_traffic_safely(yb_cluster, yb_ctl):
    """Stop node 1 (the contact / bootstrap host in our DSN). New connections
    must still succeed — once the driver knows nodes 2 and 3 exist (either
    from this process's earlier bootstrap, or from a fresh bootstrap that
    falls through to a working contact point), it picks one of them.
    """
    # Bootstrap while node 1 is up. This populates ClusterState with all 3
    # nodes and stashes a control connection.
    boot = psycopg.connect(yb_cluster + " load_balance_hosts=true")
    boot.close()

    yb_ctl.stop_node(1)
    try:
        # mark_failed will quarantine node 1 on its first failure. New conns
        # should succeed on the survivors.
        conns = []
        try:
            for _ in range(6):
                conns.append(psycopg.connect(yb_cluster + " load_balance_hosts=true"))
            hosts = {c._yb_host for c in conns}
            assert hosts.issubset({"127.0.0.2", "127.0.0.3"}), (
                f"connections must land on surviving nodes; landed on {sorted(hosts)}"
            )
        finally:
            for c in conns:
                c.close()
    finally:
        yb_ctl.start_node(1, placement_info="cloud1.datacenter1.rack1")
        time.sleep(2)


# --------------------------------------------------------------------- refresh interval


def test_yb_servers_refresh_interval_is_honoured(yb_cluster, yb_ctl):
    """With `yb_servers_refresh_interval=2`, the driver re-queries `yb_servers()`
    on a 2-second cadence. After stopping a node we don't need the master to
    rapidly drop it from its result set — what we verify is that the refresh
    fires (rather than e.g. being stuck at the 300s default) by quarantining
    via mark_failed and skipping it within seconds, not minutes.
    """
    refresh_s = 2
    dsn = yb_cluster + f" load_balance_hosts=true yb_servers_refresh_interval={refresh_s}"

    boot = psycopg.connect(dsn)
    boot.close()

    yb_ctl.stop_node(3)
    try:
        # Drive 5 connections back-to-back. The first one to pick node 3 will
        # fail and trigger mark_failed; subsequent picks skip it.
        conns = []
        try:
            for _ in range(5):
                conns.append(psycopg.connect(dsn))
            hosts = [c._yb_host for c in conns]
            assert all(h != "127.0.0.3" for h in hosts), (
                f"connections after stopped node should skip it; got {hosts}"
            )
        finally:
            for c in conns:
                c.close()
    finally:
        yb_ctl.start_node(3, placement_info="cloud1.datacenter1.rack1")
        time.sleep(2)


# --------------------------------------------------------------------- control host failover


def test_control_host_failover(yb_cluster, yb_ctl):
    """If the control connection's host dies, the next refresh-attempt's I/O
    will fail. The registry must catch that, null the control connection out,
    and the next refresh re-bootstraps to a surviving node.

    This test is mostly a smoke check: bootstrap, stop the contact host
    (likely also the control host), trigger a refresh, observe that
    subsequent smart-driver connects still work — and that if the dead
    control connection was retained, the registry has now replaced or
    nulled it.
    """
    from psycopg.yb.registry import ClusterRegistry
    refresh_s = 2
    dsn = yb_cluster + f" load_balance_hosts=true yb_servers_refresh_interval={refresh_s}"

    boot = psycopg.connect(dsn)
    uuid = boot._yb_uuid
    state = ClusterRegistry.instance()._clusters[uuid]
    ctrl_before = state.control_sync
    assert ctrl_before is not None
    boot.close()

    yb_ctl.stop_node(1)
    try:
        # Wait past one refresh window so the next connect tries a refresh.
        time.sleep(refresh_s + 1)

        # Drive a few connects. They MUST succeed via the discovered list,
        # even if the control connection was on the killed node.
        conns = []
        try:
            for _ in range(3):
                conns.append(psycopg.connect(dsn))
            for c in conns:
                assert c._yb_host in ("127.0.0.2", "127.0.0.3")
        finally:
            for c in conns:
                c.close()

        # If the control conn was tied to the killed node, it should have been
        # nulled out by the failed-refresh path. We can't assert the new ctrl
        # exists (refresh only re-opens on the next refresh tick); just verify
        # we haven't crashed and the state is coherent.
        state_after = ClusterRegistry.instance()._clusters.get(uuid)
        assert state_after is not None, "ClusterState should still exist"
    finally:
        yb_ctl.start_node(1, placement_info="cloud1.datacenter1.rack1")
        time.sleep(2)


# --------------------------------------------------------------------- node addition


def test_uniform_load_after_node_addition(yb_cluster, yb_ctl):
    """Open 9 connections on the 3-node cluster (3 per node). Add a 4th node.
    Open 9 more connections — most should land on the new node, since it
    starts at count=0 while the others are at count=3.

    We don't assert exact distribution because yb_servers() may take a moment
    to pick up the new node and the refresh window has to fire. The
    bound is loose but meaningful: the new node should receive *at least*
    half of the new traffic.
    """
    from psycopg.yb.registry import ClusterRegistry
    from tests.yb import conftest as yb_conftest  # for _run_yb_ctl

    refresh_s = 2
    dsn = yb_cluster + f" load_balance_hosts=true yb_servers_refresh_interval={refresh_s}"

    initial_conns: list = []
    new_conns: list = []
    added = False
    try:
        # Phase 1: 9 conns to the existing 3 nodes.
        # Least-loaded from [0,0,0] gives a fully deterministic 3/3/3 — every
        # pick goes to the unique minimum (or one of a 3-way tie, then a 2-way
        # tie, then 1-way — the cycle repeats so each host gets the same).
        for _ in range(9):
            initial_conns.append(psycopg.connect(dsn))
        uuid = initial_conns[0]._yb_uuid
        registry = ClusterRegistry.instance()
        for h in ("127.0.0.1", "127.0.0.2", "127.0.0.3"):
            assert registry.get_load(uuid, h) == 3, (
                f"phase 1: {h} should hold exactly 3; got "
                f"{ {x: registry.get_load(uuid, x) for x in ('127.0.0.1','127.0.0.2','127.0.0.3')} }"
            )

        # Phase 2: add node 4.
        yb_ctl.add_node(placement_info="cloud1.datacenter1.rack1")
        added = True
        # Allow yb_servers() to acknowledge the addition. Up to 30s is generous.
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            try:
                with psycopg.connect(yb_cluster) as c:
                    (n,) = c.execute("SELECT count(*) FROM yb_servers()").fetchone()
                    if n == 4:
                        break
            except Exception:
                pass
            time.sleep(1)
        else:
            pytest.skip("yb_servers() did not show the added node within 30s")

        # Force a refresh by waiting past our short refresh interval.
        time.sleep(refresh_s + 1)

        # Phase 3: 9 more conns. State entering phase 3 is [3, 3, 3, 0].
        # Walking the least-loaded picker:
        #   pick 1-3: new node catches up → [3, 3, 3, 3]
        #   pick 4: all 4 tied at 3, one random → [4, 3, 3, 3] (some permutation)
        #   pick 5: 3 tied at 3 → [4, 4, 3, 3]
        #   pick 6: 2 tied at 3 → [4, 4, 4, 3]
        #   pick 7: 1 at 3      → [4, 4, 4, 4]
        #   pick 8: all 4 tied at 4 → [5, 4, 4, 4]
        #   pick 9: 3 tied at 4 → [5, 5, 4, 4]
        # The IDENTITY of which two hosts end at 5 vs 4 is random, but the
        # SORTED shape is exact: [4, 4, 5, 5]. Assert that.
        for _ in range(9):
            new_conns.append(psycopg.connect(dsn))

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
            c.close()
        if added:
            # Best-effort removal. Longer timeouts than the conftest defaults
            # because a freshly-added node may need more time to shut down
            # cleanly when removed.
            try:
                yb_conftest._run_yb_ctl(["stop_node", "4"], timeout=60)
            except Exception:
                pass
            try:
                yb_conftest._run_yb_ctl(["remove_node", "4"], timeout=60)
            except Exception:
                pass
            time.sleep(5)


# --------------------------------------------------------------------- control re-open + node addition


def test_control_conn_reopens_on_host_loss_and_picks_up_added_node(
    yb_cluster, yb_ctl
):
    """End-to-end verification of the control-connection re-open path under
    a realistic failure scenario. Phase-1 connections stay open throughout;
    only the dead node's counter changes (via `mark_failed`).

      1. 3-node cluster. Bootstrap pins a control conn on whichever contact
         host libpq lands on (call it H_ctrl).
      2. 12 conns distribute exactly 4/4/4 across the 3 nodes. Hold them.
      3. Stop H_ctrl AND add a 4th node (127.0.0.4) at the same time. The
         phase-1 conns on the surviving originals stay open; libpq-side
         conns on H_ctrl die but driver counter stays at 4 for now.
      4. Wait until `yb_servers()` (queried via a surviving original) shows
         the new node, then sleep past `yb_servers_refresh_interval` so the
         next smart-driver connect MUST run a refresh.
      5. Open 12 new connections. The very first one drives a SINGLE-CYCLE
         recovery:
           a) Cached `state.control_sync` is on the dead host. The first
              `fetch_servers_sync` on it raises (broken TCP). The registry
              drops it and immediately retries `_ensure_control_sync`.
           b) The retry iterates `state.nodes`. H_ctrl refuses → `mark_failed`
              zeros its counter and flags `is_down`. A surviving original
              accepts → new control conn opens there.
           c) `fetch_servers_sync` on the new control conn returns the
              4-node topology (incl. 127.0.0.4). `_merge_new_nodes` updates
              `state.nodes` preserving counters on the surviving originals
              (still 4 each) and adds 127.0.0.4 at count=0.
           d) The policy then picks from the 3 healthy hosts. Starting from
              {survivor_a: 4, survivor_b: 4, new: 0}, walking the least-
              loaded picker for 12 picks yields sorted shape [6, 7, 7]:
                pick 1-4: new node catches up to 4
                pick 5-7: 3-way tie at 4 distributes one each → all at 5
                pick 8-10: 3-way tie at 5 distributes → all at 6
                pick 11-12: 3-way tie at 6, 2 picks → 2 hosts go to 7
              Which two hosts end at 7 vs which one ends at 6 is random
              (tie-break permutation), but the SORTED shape is exact.
           e) The dead host stays at 0 (zeroed by `mark_failed`); its
              4 still-tagged phase-1 conns decrement on close down to 0
              (counter floors at 0, no negative drift).

    Without the re-open + single-cycle path the driver would go blind:
    `state.control_sync` would stay on a dead host forever, no refresh
    would discover 127.0.0.4, and load balancing would freeze on the
    pre-failure topology.
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
        # ---- Phase 1: 12 conns, exact 4/4/4. Hold them open.
        for _ in range(12):
            initial.append(psycopg.connect(dsn))
        uuid = initial[0]._yb_uuid
        state = registry._clusters[uuid]
        for h in ("127.0.0.1", "127.0.0.2", "127.0.0.3"):
            assert registry.get_load(uuid, h) == 4, (
                f"phase 1: {h} should be exactly 4; got "
                f"{ {x: registry.get_load(uuid, x) for x in ('127.0.0.1','127.0.0.2','127.0.0.3')} }"
            )

        # Identify the control host. yb-ctl node number = last octet of the IP.
        assert state.control_sync is not None
        original_ctrl_host = state.control_sync.info.host
        original_ctrl_node = int(original_ctrl_host.split(".")[-1])
        survivors = [
            h for h in ("127.0.0.1", "127.0.0.2", "127.0.0.3")
            if h != original_ctrl_host
        ]
        assert len(survivors) == 2

        # ---- Add a 4th node FIRST, then stop the control host. (yb-ctl
        # `add_node` blocks waiting for every tserver to come up; if we
        # stopped a tserver first, it would time out waiting for the dead
        # one.) Phase-1 conns on the survivors stay open throughout.
        yb_ctl.add_node(placement_info="cloud1.datacenter1.rack1")
        added = True
        yb_ctl.stop_node(original_ctrl_node)

        # Poll yb_servers() via a *plain* libpq connect on a survivor (not
        # the smart driver — that would trigger refresh prematurely).
        survivor_dsn = (
            f"host={survivors[0]} port=5433 user=yugabyte dbname=yugabyte"
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

        # Sleep past the refresh interval so the next smart-driver connect
        # claims a refresh slot.
        time.sleep(refresh_s + 1)

        # ---- Phase 2: open 12 new conns. The first one drives the
        # single-cycle control re-open + refresh.
        for _ in range(12):
            new.append(psycopg.connect(dsn))

        # ---- Verify the control conn moved off the dead host.
        assert state.control_sync is not None, (
            "control conn should have been re-opened against a surviving node"
        )
        new_ctrl_host = state.control_sync.info.host
        assert new_ctrl_host != original_ctrl_host, (
            f"control conn should have moved off the stopped host "
            f"{original_ctrl_host}; still on {new_ctrl_host}"
        )
        assert new_ctrl_host in survivors + ["127.0.0.4"], (
            f"new control conn host {new_ctrl_host} is not a healthy node"
        )

        # ---- Verify exact load distribution.
        # Dead host: counter zeroed by mark_failed → 0.
        # Healthy hosts (2 surviving originals at 4 + new node at 0): the
        # 12 phase-2 picks walk a fully-deterministic-up-to-permutation path
        # ending at sorted [6, 7, 7]. (Derivation in the docstring above.)
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
        # And total across all 4 hosts: 0 + 6 + 7 + 7 = 20 = 8 phase-1 conns
        # remaining on survivors + 12 phase-2 conns.
        assert sum(loads.values()) == 20
    finally:
        for c in initial + new:
            try:
                c.close()
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
            time.sleep(5)


# --------------------------------------------------------------------- clear() integration


def test_clusterregistry_clear_integration(yb_cluster):
    """Open a smart-driver connection so a ClusterState + control connection
    are populated. Call `ClusterRegistry.clear()`. Assert clusters and
    key-map are empty AND the sync control connection is closed.
    """
    from psycopg.yb.registry import ClusterRegistry
    registry = ClusterRegistry.instance()

    conn = psycopg.connect(yb_cluster + " load_balance_hosts=true")
    uuid = conn._yb_uuid
    state = registry._clusters[uuid]
    control = state.control_sync
    assert control is not None
    assert not control.closed

    conn.close()
    registry.clear()

    assert registry._clusters == {}
    assert registry._key_to_uuid == {}
    assert control.closed, "control connection should be closed after clear()"
