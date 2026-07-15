"""
Tier 2 xCluster integration tests against the REAL ``TrackerTableCircuitBreaker``.

These exercise the full end-to-end loop:

  * Real two yb-ctl clusters (primary on 127.0.0.1-.3, secondary on .4-.6)
  * Real ``HealthProbe`` daemon thread running real tracker-table UPDATEs
    on the primary's control connection
  * Real ``TrackerTableCircuitBreaker`` observing real cluster failures
    via UPDATE failures, transitioning ``group.primary_status`` HEALTHY ↔ UNHEALTHY
  * Real dispatcher rerouting new connections to the secondary cluster on
    UNHEALTHY, back to primary on HEALTHY
  * Real ``ConnectionPool(check=xcluster_check)`` evicting stale conns

Failure injection is by ``yb-ctl stop_node`` / ``start_node`` against the
primary cluster's ``--data_dir``. The companion test file
``test_smart_driver_xcluster.py`` drives the same routing/pool plumbing
deterministically via ``FailoverGroup.force_status`` (fast, no node
stop/start needed); this file is the slower end-to-end counterpart.

Tunables baked into ``_CB_DSN``:

  * ``yb_servers_refresh_interval=3`` — short probe interval so tests
    don't need to wait minutes for the CB to tick.
  * ``yb.failover.maxUpdateFailuresAllowed=1`` — threshold of 2: rides
    out a single leader-election blip when one of three RF=3 nodes
    drops. Without this, a brief tablet-leader gap during the single-
    node-down test would spuriously trip the CB.
  * ``yb.failover.cooldownSecs=0`` — no cool-down between transitions
    so the test reads the most recent status.

Per-test cleanup: any test that calls ``stop_primary_node`` MUST restart
the node in a ``finally``. The session-scoped ``yb_xcluster_clusters``
fixture reuses the two clusters across all tests, so leaving a node down
would break subsequent tests.
"""

# Copyright (C) 2026 Yugabyte

from __future__ import annotations

import time
from collections import Counter

import pytest

import psycopg
from psycopg.yb.health import HealthResult
from psycopg.yb.registry import ClusterRegistry


PRIMARY_HOSTS = "127.0.0.1,127.0.0.2,127.0.0.3"
SECONDARY_HOSTS = "127.0.0.4,127.0.0.5,127.0.0.6"

PROBE_INTERVAL_S = 3
MAX_UPDATE_FAILURES_ALLOWED = 1
THRESHOLD = MAX_UPDATE_FAILURES_ALLOWED + 1   # consecutive ticks to flip

_CB_DSN = (
    f"host={PRIMARY_HOSTS} port=5433 user=yugabyte dbname=yugabyte "
    f"load_balance_hosts=true "
    f"yb.failover.secondaryClusterHosts={SECONDARY_HOSTS} "
    f"yb_servers_refresh_interval={PROBE_INTERVAL_S} "
    f"yb.failover.maxUpdateFailuresAllowed={MAX_UPDATE_FAILURES_ALLOWED} "
    f"yb.failover.cooldownSecs=0"
)


def _bootstrap_group():
    """Open one connection to trigger ``FailoverGroup`` bootstrap, then
    return the group handle so the test can observe ``group.primary_status``."""
    conn = psycopg.connect(_CB_DSN)
    try:
        group = ClusterRegistry.instance().get_failover_group_by_uuid(
            conn._yb_uuid
        )
        assert group is not None, "FailoverGroup must be created at bootstrap"
        return group
    finally:
        conn.close()


def _wait_for_status(
    group, expected: HealthResult, timeout_s: float
) -> bool:
    """Poll ``group.primary_status`` until it equals ``expected`` or timeout."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if group.primary_status == expected:
            return True
        time.sleep(0.2)
    return False


def _wait_for_primary_topology(
    group,
    expected_hosts: set[str],
    timeout_s: float = 60.0,
    stable_streak: int = 4,
) -> bool:
    """Poll until ``group.primary.nodes`` STABLY contains ``expected_hosts``.

    ``start_node`` returns once the tserver PROCESS is up, but the tserver
    can take several more seconds to re-register with the master. During
    that window ``yb_servers()`` returns a short list, and
    ``_merge_new_nodes`` wholesale-replaces ``state.nodes`` — dropping the
    still-registering host. The CB can flip back to HEALTHY (it only
    needs one tablet leader's UPDATE to succeed) while the dispatcher
    still sees a 2-node cluster.

    Worse: the topology can FLICKER. A one-shot check seeing all 3 hosts
    can be followed by a refresh that drops one again. So we require
    ``stable_streak`` consecutive observations of the full topology
    across separate refresh cycles before returning True. The refresh
    interval is 3s in our DSN, so 4 stable observations across forced
    refreshes (~4s apart) covers ~12s of consistent state — enough for
    yb_servers() to definitively reflect all restarted nodes.
    """
    deadline = time.monotonic() + timeout_s
    streak = 0
    while time.monotonic() < deadline:
        # Force a fresh refresh by opening (and discarding) one conn —
        # the dispatcher's `_refresh_if_stale` path picks up any newly
        # registered tserver.
        try:
            c = psycopg.connect(_CB_DSN + " connect_timeout=3")
            c.close()
        except Exception:
            pass
        with group.primary.lock:
            current = set(group.primary.nodes.keys())
        if current >= expected_hosts:
            streak += 1
            if streak >= stable_streak:
                return True
        else:
            streak = 0  # any miss resets — we want CONSECUTIVE stability
        # Slightly longer than the refresh interval (3s) so each
        # iteration sees a fresh refresh result.
        time.sleep(PROBE_INTERVAL_S + 0.5)
    return False


# How many connections to open per routing assertion. 12 gives us 4 conns
# per node when all 3 nodes of the active cluster are eligible, which is
# enough to catch "all conns go to one host" without making the tests slow.
N_CONNS = 12


def _open_n_connections(n: int = N_CONNS) -> list:
    """Open ``n`` independent connections against the test DSN. Caller is
    responsible for closing them; use ``_close_all`` in a finally."""
    return [psycopg.connect(_CB_DSN) for _ in range(n)]


def _close_all(conns) -> None:
    for c in conns:
        try:
            c.close()
        except Exception:
            pass


def _assert_distributed(
    conns,
    *,
    expected_cluster: str,
    eligible_hosts: set[str],
    min_per_host: int = 1,
) -> Counter:
    """Cross-check that ``conns`` (a) all landed on ``expected_cluster``,
    (b) only used hosts in ``eligible_hosts``, and (c) each eligible host
    received at least ``min_per_host`` conns.

    The third check is the "not just going to one node" assertion the user
    asked for. The smart driver's least-loaded policy increments under
    each cluster's per-cluster lock, so the distribution is deterministic
    in steady state — but we keep the threshold conservative
    (``min_per_host=1``) to absorb any control-conn pre-load on whichever
    host hosts the cluster's metadata refresher.

    Returns the per-host histogram so callers that want a tighter check
    (e.g. exact 4/4/4 balance) can assert on it directly.
    """
    histogram: Counter = Counter()
    for c in conns:
        assert c._yb_cluster == expected_cluster, (
            f"expected {expected_cluster}, got {c._yb_cluster}; "
            f"host={c._yb_host}"
        )
        assert c._yb_host in eligible_hosts, (
            f"conn landed on {c._yb_host}, expected one of "
            f"{sorted(eligible_hosts)}"
        )
        histogram[c._yb_host] += 1

    # Every eligible host must receive at least min_per_host conns. If a
    # host appears zero times, every conn went somewhere else — which is
    # exactly the "all to one node" failure mode this test guards against.
    missing = {h: histogram[h] for h in eligible_hosts if histogram[h] < min_per_host}
    assert not missing, (
        f"some eligible hosts received fewer than {min_per_host} conn(s); "
        f"shortfall={missing} full_histogram={dict(histogram)} "
        f"(load-balancing collapsed onto a subset)"
    )
    return histogram


# Detection latency budget. The dominant cost is NOT the probe interval —
# it's the YB internal RPC timeout per failed UPDATE: empirically, an
# UPDATE on a tracker-table tablet whose leader has lost quorum blocks
# for ~10s before raising "Perform RPC ... timed out after 10.000s".
# That's the timeout the tserver applies to its own Raft replication
# attempt; we observe it as the latency of each failed probe tick.
#
# Worst-case detection time for ``THRESHOLD`` consecutive failures:
#   * The first probe tick AFTER stop may already be in flight against a
#     transient-but-still-functional UPDATE → potentially counts as
#     success (~3s).
#   * THRESHOLD failed ticks at ~(10s RPC timeout + 3s probe interval)
#     each = THRESHOLD × 13s.
#   * Plus a few seconds of post-stop settling.
# 60s gives us comfortable headroom over the ~26-36s realistic worst case.
_RPC_TIMEOUT_BUDGET_S = 10
_TRIP_TIMEOUT_S = (THRESHOLD + 1) * (_RPC_TIMEOUT_BUDGET_S + PROBE_INTERVAL_S) + 10
# Recovery is faster because successful UPDATEs return in milliseconds —
# THRESHOLD ticks at ~PROBE_INTERVAL_S each, plus settling for the
# restarted tserver to fully reintegrate.
_RECOVER_TIMEOUT_S = PROBE_INTERVAL_S * (THRESHOLD + 3) + 20


# --------------------------------------------------------------------- single-node failure


def test_cb_stays_healthy_when_single_primary_node_down(
    yb_xcluster_clusters, yb_xcluster_ctl,
):
    """Single primary tserver down → CB MUST NOT trip.

    With RF=3 and ``maxUpdateFailuresAllowed=1`` (threshold=2), a single
    leader-election blip for tablets whose leader was on the dropped node
    is absorbed: the next tick succeeds, the failure counter resets, and
    the CB stays HEALTHY. This is the "single node failure should not
    trigger failover" assertion from the spec.
    """
    group = _bootstrap_group()
    # Let one probe tick run so the tracker table gets set up under healthy
    # conditions.
    time.sleep(PROBE_INTERVAL_S + 2)
    assert group.primary_status == HealthResult.HEALTHY

    yb_xcluster_ctl.stop_primary_node(2)
    started_back = False
    try:
        # Wait through enough probe ticks that if the CB were going to
        # trip, it would have. Threshold-many ticks + slack.
        time.sleep(PROBE_INTERVAL_S * (THRESHOLD + 2) + 3)
        assert group.primary_status == HealthResult.HEALTHY, (
            f"single-node-down should not trip CB; got {group.primary_status}"
        )

        # Open N conns. They must (a) all land on primary, (b) skip the
        # stopped node 2, and (c) distribute across the two surviving
        # primary hosts (.1 and .3) — NOT collapse onto a single host.
        conns = _open_n_connections()
        try:
            histogram = _assert_distributed(
                conns,
                expected_cluster="primary",
                eligible_hosts={"127.0.0.1", "127.0.0.3"},
            )
            assert "127.0.0.2" not in histogram, (
                f"stopped node must not receive traffic; histogram={dict(histogram)}"
            )
        finally:
            _close_all(conns)
    finally:
        yb_xcluster_ctl.recover_primary_cluster()


# --------------------------------------------------------------------- majority failure


def test_cb_trips_when_majority_primary_nodes_down(
    yb_xcluster_clusters, yb_xcluster_ctl,
):
    """Majority of primary tservers down → CB trips → failover to secondary.

    Stopping 2 of 3 RF=3 nodes leaves every tablet with one replica, which
    is a minority and cannot elect a leader. All writes (and our tracker
    UPDATE) fail. After THRESHOLD consecutive failures the CB flips to
    UNHEALTHY and the dispatcher routes new conns to the secondary.
    """
    group = _bootstrap_group()
    time.sleep(PROBE_INTERVAL_S + 2)
    assert group.primary_status == HealthResult.HEALTHY

    yb_xcluster_ctl.stop_primary_node(2)
    yb_xcluster_ctl.stop_primary_node(3)
    try:
        assert _wait_for_status(
            group, HealthResult.UNHEALTHY, _TRIP_TIMEOUT_S
        ), (
            f"CB did not trip within {_TRIP_TIMEOUT_S}s; "
            f"status={group.primary_status}"
        )

        # Open N conns. They must (a) all land on secondary, (b) only
        # use secondary hosts, and (c) distribute across ALL THREE
        # secondary nodes — confirming the secondary cluster's load
        # balancer is active and traffic isn't collapsed onto a single
        # secondary host.
        conns = _open_n_connections()
        try:
            _assert_distributed(
                conns,
                expected_cluster="secondary",
                eligible_hosts={"127.0.0.4", "127.0.0.5", "127.0.0.6"},
            )
        finally:
            _close_all(conns)
    finally:
        # restart_node (via recover_primary_cluster) handles the case
        # where node 1's postmaster also died via master-quorum-loss
        # cascade — start_node alone would skip a "still running" node.
        yb_xcluster_ctl.recover_primary_cluster()


# --------------------------------------------------------------------- all-primary-down


# Tight budget: this test models the operator-driven demo's failure mode
# (`yb-ctl --data_dir=~/yb-primary stop` kills every primary node,
# including the one holding the CB's long-lived control conn). The
# previous `majority_primary_nodes_down` test left node 1 alive, which
# kept the control socket healthy and exercised the FAST trip path
# (server-side `statement_timeout` fires on the failed UPDATE in ~5s).
# This test kills node 1 too, which on macOS exposes a slow TCP
# retransmit path that hangs the in-flight `SET` for ~9 minutes by
# default. The CB's wall-clock per-tick deadline is what bounds this;
# a 10s budget here forces that mechanism to actually fire.
_TRIP_ALL_DOWN_TIMEOUT_S = 10.0


def test_cb_trips_when_ALL_primary_nodes_down_within_10s(
    yb_xcluster_clusters, yb_xcluster_ctl,
):
    """ALL primary nodes down (including the CB's control-conn host) →
    CB must trip within 10 seconds.

    This is the operator-driven failure mode the demo triggers. With a
    healthy control socket the CB's server-side ``statement_timeout``
    catches a failed UPDATE in ~5s, but when the control-conn host
    itself dies the in-flight TCP send can sit in macOS' retransmit
    window for minutes — invisible to ``statement_timeout`` (server-
    side) and to ``keepalives_idle`` (only applies to idle sockets).
    The CB's per-tick wall-clock cap is the only thing that bounds
    this case; this test fails (times out) without it.
    """
    group = _bootstrap_group()
    time.sleep(PROBE_INTERVAL_S + 2)
    assert group.primary_status == HealthResult.HEALTHY

    # Stop EVERY primary node, including the one most likely hosting
    # the control conn (.1, the first contact host).
    yb_xcluster_ctl.stop_primary_node(1)
    yb_xcluster_ctl.stop_primary_node(2)
    yb_xcluster_ctl.stop_primary_node(3)
    try:
        t0 = time.monotonic()
        assert _wait_for_status(
            group, HealthResult.UNHEALTHY, _TRIP_ALL_DOWN_TIMEOUT_S
        ), (
            f"CB did not trip within {_TRIP_ALL_DOWN_TIMEOUT_S}s "
            f"after ALL primary nodes were killed; status={group.primary_status}. "
            f"This is the macOS-TCP-retransmit case — the CB needs a "
            f"per-tick wall-clock cap to bound it."
        )
        elapsed = time.monotonic() - t0
        # Sanity: we expect well under 10s on the happy path. Log it so
        # the test output shows the actual trip latency.
        assert elapsed < _TRIP_ALL_DOWN_TIMEOUT_S, (
            f"CB trip latency was {elapsed:.1f}s, over the 10s target"
        )

        # New conn must route to secondary — confirms the full failover
        # path works even when the entire primary cluster is gone.
        conns = _open_n_connections()
        try:
            _assert_distributed(
                conns,
                expected_cluster="secondary",
                eligible_hosts={"127.0.0.4", "127.0.0.5", "127.0.0.6"},
            )
        finally:
            _close_all(conns)
    finally:
        yb_xcluster_ctl.recover_primary_cluster()


# --------------------------------------------------------------------- failback


def test_cb_failback_after_majority_restored(
    yb_xcluster_clusters, yb_xcluster_ctl,
):
    """After a trip, restart the stopped primaries → CB recovers HEALTHY.

    Symmetric hysteresis: the same THRESHOLD that flipped us to UNHEALTHY
    also gates the recovery. Once THRESHOLD consecutive UPDATEs succeed
    against the restored primary, the CB flips HEALTHY and new conns go
    back to primary.
    """
    group = _bootstrap_group()
    time.sleep(PROBE_INTERVAL_S + 2)
    assert group.primary_status == HealthResult.HEALTHY

    yb_xcluster_ctl.stop_primary_node(2)
    yb_xcluster_ctl.stop_primary_node(3)
    started_back = {2: False, 3: False}
    try:
        assert _wait_for_status(
            group, HealthResult.UNHEALTHY, _TRIP_TIMEOUT_S
        ), f"CB did not trip; status={group.primary_status}"

        # Restart both nodes. start_node is slow — waits for the tserver
        # to register with the master — so this typically dominates the
        # test's wall-clock time.
        yb_xcluster_ctl.start_primary_node(2)
        started_back[2] = True
        yb_xcluster_ctl.start_primary_node(3)
        started_back[3] = True

        assert _wait_for_status(
            group, HealthResult.HEALTHY, _RECOVER_TIMEOUT_S
        ), (
            f"CB did not fail back within {_RECOVER_TIMEOUT_S}s; "
            f"status={group.primary_status}"
        )

        # The test body only restarted nodes 2+3; node 1's postmaster
        # may also be dead (master-quorum-loss cascade — stopping 2 of 3
        # masters kills .1's PG too). CB flips HEALTHY as soon as the
        # tracker UPDATE succeeds against ANY surviving primary, but
        # the distribution assertion below needs ALL three primaries
        # registered. recover_primary_cluster TCP-probes each host and
        # restart_nodes anything down, then waits for yb_servers() to
        # report 3 consistent rows from every host.
        yb_xcluster_ctl.recover_primary_cluster()

        # Final gate: dispatcher's view must include all 3 hosts.
        assert _wait_for_primary_topology(
            group, {"127.0.0.1", "127.0.0.2", "127.0.0.3"}
        ), f"primary topology never recovered to 3 nodes: {set(group.primary.nodes.keys())}"

        # Open N conns. They must (a) all land on primary, (b) only use
        # primary hosts, and (c) distribute across ALL THREE primary
        # nodes — confirming the restarted nodes are back in the load-
        # balancing pool, not just node 1 (which never went down).
        conns = _open_n_connections()
        try:
            _assert_distributed(
                conns,
                expected_cluster="primary",
                eligible_hosts={"127.0.0.1", "127.0.0.2", "127.0.0.3"},
            )
        finally:
            _close_all(conns)
    finally:
        yb_xcluster_ctl.recover_primary_cluster()


# --------------------------------------------------------------------- pool


def test_pool_failover_and_failback_with_real_cb(
    yb_xcluster_clusters, yb_xcluster_ctl,
):
    """Pool end-to-end: borrows reroute to secondary on real failure and
    flip back to primary on real recovery, driven entirely by the real CB
    (no ``force_status``).

    The ``xcluster_check`` pool-borrow callback raises on a stale-cluster
    conn so psycopg-pool evicts it and opens a replacement against the
    currently-active cluster.
    """
    pytest.importorskip("psycopg_pool")
    from psycopg_pool import ConnectionPool

    from psycopg.yb.pool import xcluster_check

    # Pool size 6 lets us drain six conns concurrently and observe their
    # host distribution. min_size==max_size==6 keeps the test simple
    # (no growth-races to reason about) while giving 2 conns per host on
    # a 3-node active cluster — enough that "all on one host" would
    # always violate the distribution assertion.
    POOL_SIZE = 6

    def _drain_pool_simultaneously() -> list:
        """Borrow every slot in the pool at once via a context-manager
        stack so all ``POOL_SIZE`` conns are held concurrently. Returns
        the borrowed conns (still inside the pool's borrow scope — the
        caller must close the contexts to return them)."""
        from contextlib import ExitStack
        stack = ExitStack()
        try:
            conns = [stack.enter_context(pool.connection()) for _ in range(POOL_SIZE)]
        except Exception:
            stack.close()
            raise
        return stack, conns

    # Bump the per-borrow timeout from the 30s default. Phase 2 drains
    # six idle primary-tagged conns sequentially through the eviction-
    # and-replace cycle (check raises → close → fresh secondary connect),
    # and each fresh secondary connect can take a few seconds while the
    # smart driver dispatches through the FailoverGroup; six × ~5 s flirts
    # with the default budget. 90 s gives comfortable headroom.
    with ConnectionPool(
        _CB_DSN,
        check=xcluster_check,
        min_size=POOL_SIZE,
        max_size=POOL_SIZE,
        timeout=90,
    ) as pool:
        pool.wait()

        # Phase 1: HEALTHY → all borrows land on primary AND spread
        # across all 3 primary nodes (.1/.2/.3).
        stack, conns = _drain_pool_simultaneously()
        try:
            primary_uuid = conns[0]._yb_uuid
            _assert_distributed(
                conns,
                expected_cluster="primary",
                eligible_hosts={"127.0.0.1", "127.0.0.2", "127.0.0.3"},
            )
        finally:
            stack.close()
        group = ClusterRegistry.instance().get_failover_group_by_uuid(
            primary_uuid
        )
        assert group is not None

        # Let one probe tick run to set up the tracker table cleanly.
        time.sleep(PROBE_INTERVAL_S + 2)

        yb_xcluster_ctl.stop_primary_node(2)
        yb_xcluster_ctl.stop_primary_node(3)
        started_back = {2: False, 3: False}
        try:
            assert _wait_for_status(
                group, HealthResult.UNHEALTHY, _TRIP_TIMEOUT_S
            ), f"CB did not trip; status={group.primary_status}"

            # Phase 2: UNHEALTHY → every pooled primary conn fails
            # ``xcluster_check`` on borrow, gets evicted, and the
            # replacement opens against secondary. Draining the whole
            # pool then asserts the replacements ALSO spread across all
            # 3 secondary nodes — not collapsed onto one.
            stack, conns = _drain_pool_simultaneously()
            try:
                _assert_distributed(
                    conns,
                    expected_cluster="secondary",
                    eligible_hosts={
                        "127.0.0.4", "127.0.0.5", "127.0.0.6"
                    },
                )
            finally:
                stack.close()

            yb_xcluster_ctl.start_primary_node(2)
            started_back[2] = True
            yb_xcluster_ctl.start_primary_node(3)
            started_back[3] = True

            assert _wait_for_status(
                group, HealthResult.HEALTHY, _RECOVER_TIMEOUT_S
            ), f"CB did not fail back; status={group.primary_status}"

            # Same caveat as the non-pool failback test: node 1's PG
            # may have died via master-quorum-loss cascade and the test
            # body only restarted .2/.3. recover_primary_cluster ensures
            # all three are up before the borrow-distribution assert.
            yb_xcluster_ctl.recover_primary_cluster()
            assert _wait_for_primary_topology(
                group, {"127.0.0.1", "127.0.0.2", "127.0.0.3"}
            ), f"primary topology never recovered: {set(group.primary.nodes.keys())}"

            # Phase 3: back to HEALTHY → pooled secondary conns get
            # evicted on borrow, replacements open on primary, and
            # again spread across all 3 primary nodes.
            stack, conns = _drain_pool_simultaneously()
            try:
                _assert_distributed(
                    conns,
                    expected_cluster="primary",
                    eligible_hosts={
                        "127.0.0.1", "127.0.0.2", "127.0.0.3"
                    },
                )
            finally:
                stack.close()
        finally:
            yb_xcluster_ctl.recover_primary_cluster()
