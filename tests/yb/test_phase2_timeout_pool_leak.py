"""
Integration-level repro tests for PR-H6 — the DB-corruption window
Shubham reported in failback_asym cycles.

Shubham's cycle 7 (Oct 23 2026) timeline:
  14:53:28   harness pauses B→A replication
  14:54:26   failback drain-wait starts; lag wait threshold=0.0ms,
             timeout=90s
  14:55:26.9 pool/2 INSERT 13 rows committed on B (60 s into the
             90 s lag wait — pool-conn leak during Phase 2)
  14:55:56.5 lag wait timeout after 90.0s — proceeding with flip
             (routing → A, even though B→A is behind)
  14:55:59   harness resumes B→A replication
  14:56:04.9 pool/2 DELETE WHERE txn_id = <the batch from 14:55:26.9>
             committed on A, rc=10 exp=13
  → index contains spurious row on BOTH universes, persists for 2+ hours

Root cause (two bugs compounding):

  1. Pool getconn during Phase 2 serves outgoing-cluster conns. The
     pool's `xcluster_check` uses `active_cluster(group)`, which reads
     the status flag. During Phase 2 the status hasn't flipped yet, so
     `active_cluster` returns the OUTGOING cluster, and conns on it
     pass the eviction check.

  2. Phase 2 timeout defaults to "proceed with flip to a known-stale
     target". The driver KNOWS lag didn't converge but flips anyway.

Fix (this PR):

  (a) `on_phase2_timeout="fail"` (opt-in; default is "proceed"): on a
      failback timeout, raise `Phase2TimeoutError`; the drain catches it
      and SKIPS the status change. Routing stays on the current side.
      Operator-visible `last_phase2_timeout_ts` stamped.

  (b) `_decide_eviction` during `dispatch_paused`: treat the current
      "active" cluster as OUTGOING, and evict any pool conn on it so
      the pool doesn't serve outgoing-side conns during the lag wait.

  (c) `close_returns=True` bypass: use `pgconn.finish()` directly in
      the xcluster_check so the pool's close-returns-to-pool loop
      doesn't put the stale conn right back in the queue.
"""

# Copyright (C) 2026 Yugabyte

from __future__ import annotations

import threading
import time
import types
from typing import Any

import pytest

from psycopg import errors as e
from psycopg.pq import TransactionStatus
from psycopg.yb.drain import Phase2TimeoutError, trigger_drain
from psycopg.yb.health import HealthResult
from psycopg.yb.pool import _decide_eviction, xcluster_check
from psycopg.yb.registry import ClusterRegistry, FailoverGroup
from psycopg.yb import NoViableClusterError


pytestmark = pytest.mark.yb_unit


# ===================================================================
# Shared harness
# ===================================================================


class _FakeYBAClient:
    """Lag-wait sticky-returns stub for the Phase 2 timeout path."""

    def __init__(self, lag_ms: float):
        self.lag_ms = lag_ms
        self.calls = 0

    def get_committed_lag_ms(self, uuid):
        self.calls += 1
        return self.lag_ms

    def get_committed_lag_sample(self, uuid):
        self.calls += 1
        return (self.lag_ms, time.time())


def _make_group(fake_state, *, yba_client=None, on_phase2_timeout="fail"):
    p = fake_state(("p1", "aws", "us-west", "us-west-1a", "primary"), uuid="PRIMARY-A")
    s = fake_state(("s1", "aws", "us-east", "us-east-1a", "primary"), uuid="SECONDARY-B")
    group = FailoverGroup(
        primary=p, secondary=s, lock=threading.Lock(),
        primary_status=HealthResult.HEALTHY,
        secondary_status=HealthResult.HEALTHY,
        cooldown_s=0,
        yba_client=yba_client,
        xcluster_config_uuid="xcc-a2b" if yba_client else "",
        failback_xcluster_config_uuid="xcc-b2a" if yba_client else "",
        threshold_replication_lag_ms=0,
        lag_wait_timeout_s=2,
        failback_lag_wait_timeout_s=2,
        on_phase2_timeout=on_phase2_timeout,
    )
    return group


# ===================================================================
# Part A — DB-corruption fix: Phase 2 timeout aborts the flip
# ===================================================================


def test_shubham_cycle7_failback_timeout_does_not_flip(
    fake_state, fresh_registry,
):
    """Replica of Shubham's cycle 7: failback lag never converges within
    the window. Under the old code the drain would flip to A despite
    knowing B→A replication was behind. The fix must NOT flip."""
    yba = _FakeYBAClient(lag_ms=5000.0)   # sticky above threshold
    group = _make_group(fake_state, yba_client=yba)
    group.primary_status = HealthResult.UNHEALTHY   # start on secondary
    group.secondary_status = HealthResult.HEALTHY

    # Failback trigger: primary CB flips to HEALTHY → drain fires.
    t0 = time.monotonic()
    trigger_drain(
        group,
        which_cluster="primary",
        new_status=HealthResult.HEALTHY,       # failback
        drain_timeout_s=0,
    )
    elapsed = time.monotonic() - t0

    assert 1.5 <= elapsed <= 3.5, f"expected ~2s lag wait, got {elapsed:.1f}s"
    # The critical assertion: the flip was ABORTED.
    assert group.primary_status is HealthResult.UNHEALTHY, (
        "Phase 2 timed out with lag=5000ms > threshold=0ms. The driver "
        "must NOT flip to primary when B→A replication is behind — that "
        "is the DB-corruption window Shubham observed in cycle 7"
    )
    assert group.secondary_status is HealthResult.HEALTHY
    assert group.last_phase2_timeout_ts > 0, (
        "operators + CBs need to see last_phase2_timeout_ts to detect "
        "degraded failover"
    )
    assert group.dispatch_paused is False, (
        "dispatch must unpause after an aborted drain — app can't hang"
    )


def test_default_policy_failback_timeout_proceeds(fake_state, fresh_registry):
    """Default is "proceed": a failback whose lag wait times out still
    flips back to primary (the stress harness expects failback to
    complete on timeout, e.g. `failback_asym`, `fault_in_drain`)."""
    yba = _FakeYBAClient(lag_ms=5000.0)
    p = fake_state(("p1", "aws", "us-west", "us-west-1a", "primary"), uuid="PRIMARY-A")
    s = fake_state(("s1", "aws", "us-east", "us-east-1a", "primary"), uuid="SECONDARY-B")
    group = FailoverGroup(
        primary=p, secondary=s, lock=threading.Lock(),
        primary_status=HealthResult.UNHEALTHY,
        secondary_status=HealthResult.HEALTHY,
        cooldown_s=0,
        yba_client=yba,
        xcluster_config_uuid="xcc-a2b",
        failback_xcluster_config_uuid="xcc-b2a",
        threshold_replication_lag_ms=0,
        lag_wait_timeout_s=2,
        failback_lag_wait_timeout_s=2,
    )
    assert group.on_phase2_timeout == "proceed"

    trigger_drain(
        group,
        which_cluster="primary",
        new_status=HealthResult.HEALTHY,
        drain_timeout_s=0,
    )

    assert group.primary_status is HealthResult.HEALTHY
    assert group.last_failback_complete_ts > 0
    assert group.dispatch_paused is False


def test_proceed_policy_preserves_pre_fix_behavior(fake_state, fresh_registry):
    """Operators who want aggressive failover can opt in via
    `onPhase2Timeout=proceed`. The driver still flips to the known-stale
    side — same as before this fix."""
    yba = _FakeYBAClient(lag_ms=5000.0)
    group = _make_group(fake_state, yba_client=yba, on_phase2_timeout="proceed")

    trigger_drain(
        group,
        which_cluster="primary",
        new_status=HealthResult.UNHEALTHY,
        drain_timeout_s=0,
    )
    # Pre-fix behavior: flips despite lag.
    assert group.primary_status is HealthResult.UNHEALTHY


def test_phase2_timeout_only_fires_on_timeout_not_on_fast_converge(
    fake_state, fresh_registry,
):
    """Regression — when lag converges before deadline, Phase 2 returns
    normally and the drain flips. Status transitions to UNHEALTHY and
    `last_phase2_timeout_ts` stays 0."""
    yba = _FakeYBAClient(lag_ms=0.0)   # already caught up
    group = _make_group(fake_state, yba_client=yba)

    trigger_drain(
        group,
        which_cluster="primary",
        new_status=HealthResult.UNHEALTHY,
        drain_timeout_s=0,
    )
    assert group.primary_status is HealthResult.UNHEALTHY
    assert group.last_phase2_timeout_ts == 0


def test_phase2_timeout_error_subclasses_exception():
    """`Phase2TimeoutError` should be a plain Exception subclass — the
    drain catches it internally and never raises to callers of
    `trigger_drain`."""
    assert issubclass(Phase2TimeoutError, Exception)


# ===================================================================
# Part B — Pool getconn during Phase 2 evicts outgoing-side conns
# ===================================================================


class _FakePoolConn:
    """Pool-tracked conn stub: carries `_yb_uuid`, `pgconn.finish()`,
    `_closed`. Shape matches psycopg Connection for `_decide_eviction`
    + `xcluster_check`."""

    def __init__(self, uuid):
        self._yb_uuid = uuid
        self._closed = False
        self._finish_count = 0

        def _finish():
            self._finish_count += 1
        self.pgconn = types.SimpleNamespace(finish=_finish)


def _install_group_in_registry(fresh_registry, group):
    fresh_registry._failover_groups[group.primary.uuid] = group


def test_pool_evicts_outgoing_conn_during_phase2_pause(fake_state, fresh_registry):
    """Shubham's cycle 7, Phase 2 window: dispatch_paused=True; status
    hasn't flipped yet (`active_cluster` returns PRIMARY-A, the outgoing
    side). A pool conn on PRIMARY-A MUST be evicted so the pool doesn't
    serve it to the app during the lag wait."""
    group = _make_group(fake_state)
    _install_group_in_registry(fresh_registry, group)
    group.dispatch_paused = True   # simulate drain in flight

    outgoing_conn = _FakePoolConn("PRIMARY-A")   # on the outgoing side
    should_evict, _, _, _ = _decide_eviction(outgoing_conn)
    assert should_evict is True, (
        "pool must evict outgoing-cluster conns during Phase 2 pause — "
        "otherwise the pool serves them to the app, which commits on "
        "the universe we are about to leave (Shubham's cycle 7)"
    )


def test_pool_evicts_incoming_conn_during_phase2_pause(fake_state, fresh_registry):
    """agent-K PR #18:94. During the Phase 2 pause, conns on the INCOMING
    side (SECONDARY-B, for a failover drain) MUST also be evicted. An
    idle inactive-side pool conn left over from a prior failover would
    otherwise be served to the app during the full Phase 2 lag wait,
    letting it commit on a target that hasn't caught up yet — exactly
    the DB-25008 index-corruption window. The pre-fix test asserted the
    opposite ("keep incoming conns") and locked that buggy behavior in.
    """
    group = _make_group(fake_state)
    _install_group_in_registry(fresh_registry, group)
    group.dispatch_paused = True

    incoming_conn = _FakePoolConn("SECONDARY-B")
    should_evict, _, _, _ = _decide_eviction(incoming_conn)
    assert should_evict is True, (
        "incoming-cluster conn survived the Phase 2 pause — this is the "
        "DB-25008 window. The pool would serve it to the app for the "
        "full lag_wait_timeout_s and the app would commit on a target "
        "that hasn't caught up yet. agent-K PR #18:94."
    )


def test_pool_evicts_stale_conn_outside_pause(fake_state, fresh_registry):
    """Regression — the standard "wrong-cluster eviction" path still
    fires outside the drain window."""
    group = _make_group(fake_state)
    _install_group_in_registry(fresh_registry, group)
    group.primary_status = HealthResult.UNHEALTHY   # routed to secondary
    group.dispatch_paused = False

    stale_conn = _FakePoolConn("PRIMARY-A")   # on the inactive cluster
    should_evict, _, _, _ = _decide_eviction(stale_conn)
    assert should_evict is True


# ===================================================================
# Part C — close_returns bypass: pgconn.finish() directly
# ===================================================================


def test_xcluster_check_bypasses_pool_close_returns_loop(
    fake_state, fresh_registry,
):
    """agent-K PR #10 finding: when `close_returns=True`, `conn.close()`
    is rerouted to `pool.putconn()`, which puts the stale conn right
    back in the pool. The fix calls `pgconn.finish()` directly to
    bypass the pool-aware close path."""
    group = _make_group(fake_state)
    _install_group_in_registry(fresh_registry, group)
    group.primary_status = HealthResult.UNHEALTHY   # route to secondary
    group.dispatch_paused = False

    stale_conn = _FakePoolConn("PRIMARY-A")

    # xcluster_check should: finish the pgconn, mark _closed, THEN raise.
    with pytest.raises(e.OperationalError):
        xcluster_check(stale_conn)
    assert stale_conn._finish_count == 1, (
        "pgconn.finish() not called — xcluster_check went through "
        "conn.close(), which with close_returns=True would route back "
        "to pool.putconn and NEVER evict (agent-K PR #10)"
    )
    assert stale_conn._closed is True


def test_xcluster_check_raises_no_viable_when_both_unhealthy(
    fake_state, fresh_registry,
):
    """`NoViableClusterError` should still bubble through the pool
    check (not be swallowed by the eviction path)."""
    group = _make_group(fake_state)
    _install_group_in_registry(fresh_registry, group)
    group.primary_status = HealthResult.UNHEALTHY
    group.secondary_status = HealthResult.UNHEALTHY   # both down
    group.dispatch_paused = False

    conn = _FakePoolConn("PRIMARY-A")
    with pytest.raises(NoViableClusterError):
        xcluster_check(conn)


# ===================================================================
# Full end-to-end: Shubham's observed invariant
# ===================================================================


def test_db_corruption_window_closed_end_to_end(fake_state, fresh_registry):
    """End-to-end repro of Shubham's cycle 7 invariant: after a
    failback where lag never converges within the window:

      (1) routing must NOT have flipped to A (otherwise the app writes
          to A while B→A is behind)
      (2) pool-side conns on the outgoing cluster must have been evicted
          during Phase 2 (not served to the app during the wait)
    """
    yba = _FakeYBAClient(lag_ms=5000.0)
    group = _make_group(fake_state, yba_client=yba)
    _install_group_in_registry(fresh_registry, group)
    group.primary_status = HealthResult.UNHEALTHY   # start on secondary
    group.secondary_status = HealthResult.HEALTHY

    # Observe pool eviction decisions DURING the drain, from a watcher
    # thread. (The drain thread toggles dispatch_paused while running.)
    outgoing_conn = _FakePoolConn("SECONDARY-B")   # currently active, soon-outgoing

    eviction_during_pause = {"observed": False}
    stop = threading.Event()

    def pool_watcher():
        while not stop.is_set():
            if group.dispatch_paused:
                should_evict, _, _, _ = _decide_eviction(outgoing_conn)
                if should_evict:
                    eviction_during_pause["observed"] = True
                    return
            time.sleep(0.01)

    watcher = threading.Thread(target=pool_watcher, daemon=True)
    watcher.start()

    try:
        trigger_drain(
            group,
            which_cluster="primary",
            new_status=HealthResult.HEALTHY,       # failback
            drain_timeout_s=0,
        )
    finally:
        stop.set()
        watcher.join(timeout=2.0)

    # (1) Flip aborted — routing stays on secondary.
    assert group.primary_status is HealthResult.UNHEALTHY, (
        "DB-corruption window: flip proceeded despite lag timeout"
    )
    # (2) Pool-side eviction observed during the pause window.
    assert eviction_during_pause["observed"] is True, (
        "pool watcher never saw an outgoing-side conn evicted during "
        "the Phase 2 pause — the leak that corrupted index in cycle 7"
    )
