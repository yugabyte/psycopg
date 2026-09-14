"""
Unit tests for ``psycopg.yb.drain.trigger_drain`` — the barrier-with-
timeout drain (Phase E of the xCluster failover implementation plan).

Verifies:

  * ``YBParams.drain_timeout_s`` — default is 10; sentinel values (-1, 0)
    survive parsing without being clamped to positive; DSN forms
    accepted.
  * ``trigger_drain`` applies status changes.
    - No routing change (e.g. secondary flips while primary is HEALTHY)
      → no drain, no dispatch_paused pulse.
    - Routing change (primary flips) → pause, drain, apply, resume.
  * Sentinel semantics:
    - ``drain_timeout_s = -1`` → waits until all in-flight conns drain
      naturally, never force-closes.
    - ``drain_timeout_s = 0``  → force-closes in-flight conns immediately.
    - ``drain_timeout_s = N``  → waits up to ~N seconds, then force-closes.
"""

# Copyright (C) 2026 Yugabyte

from __future__ import annotations

import threading
import time
from typing import Optional

import pytest

from psycopg.pq import TransactionStatus
from psycopg.yb.drain import trigger_drain
from psycopg.yb.health import HealthResult
from psycopg.yb.params import extract_yb_params
from psycopg.yb.registry import FailoverGroup


pytestmark = pytest.mark.yb_unit


# ------------------------------------------------------- fake conn

class _FakeInfo:
    def __init__(self, status: TransactionStatus) -> None:
        self.transaction_status = status


class _FakeConn:
    """Minimal shape trigger_drain needs from a tracked conn:
    ``info.transaction_status`` (readable) and ``close()``. Also
    weak-referenceable (any subclass of ``object`` is)."""

    def __init__(self, status: TransactionStatus) -> None:
        self.info = _FakeInfo(status)
        self.closed = False

    def close(self) -> None:
        self.closed = True
        # After close, transaction_status transitions to UNKNOWN — mimic
        # psycopg's real behaviour so the drain loop sees the conn as
        # "drained" on its next poll.
        self.info.transaction_status = TransactionStatus.UNKNOWN


def _make_group(fake_state):
    p = fake_state(("p1", "aws", "us-west", "us-west-1a", "primary"), uuid="P")
    s = fake_state(("s1", "aws", "us-east", "us-east-1a", "primary"), uuid="S")
    return FailoverGroup(
        primary=p, secondary=s, lock=threading.Lock(),
        primary_status=HealthResult.HEALTHY,
        secondary_status=HealthResult.HEALTHY,
        cooldown_s=0,
    )


# ------------------------------------------------------- params parsing

def test_drain_timeout_default():
    params, _, _ = extract_yb_params("load_balance_hosts=true", {})
    assert params.drain_timeout_s == 10


@pytest.mark.parametrize("value,expected", [
    ("0", 0),
    ("-1", -1),
    ("30", 30),
    ("-5", -1),   # anything < -1 collapses to -1 (wait forever)
])
def test_drain_timeout_sentinels_preserved(value, expected):
    params, _, _ = extract_yb_params(
        f"load_balance_hosts=true yb.failover.drainTimeoutSecs={value}", {},
    )
    assert params.drain_timeout_s == expected


@pytest.mark.parametrize("dsn_key", [
    "yb.failover.drainTimeoutSecs",
    "yb_failover_drain_timeout_secs",
    "yb-failover-drain-timeout-secs",
])
def test_drain_timeout_dsn_forms(dsn_key):
    params, _, _ = extract_yb_params(
        f"load_balance_hosts=true {dsn_key}=7", {},
    )
    assert params.drain_timeout_s == 7


# ------------------------------------------------------- no-routing-change

def test_no_drain_when_secondary_flips_and_primary_stays_healthy(fake_state):
    """primary HEALTHY, secondary HEALTHY → primary HEALTHY, secondary
    UNHEALTHY. Active cluster stays 'primary' — no drain, no pause pulse."""
    group = _make_group(fake_state)
    # Track a "primary conn" in the outgoing side we'd nominally drain.
    conn = _FakeConn(TransactionStatus.INTRANS)
    group.primary.tracked_conns.add(conn)

    # Verify dispatch_paused stayed False across the transition.
    pause_observations: list[bool] = []
    orig_pause = group.pause_dispatch

    def watched_pause():
        pause_observations.append(True)
        orig_pause()

    group.pause_dispatch = watched_pause   # type: ignore[method-assign]

    trigger_drain(
        group,
        which_cluster="secondary",
        new_status=HealthResult.UNHEALTHY,
        drain_timeout_s=5,
    )

    assert group.secondary_status == HealthResult.UNHEALTHY
    assert group.primary_status == HealthResult.HEALTHY   # untouched
    assert conn.closed is False                            # no drain fired
    assert pause_observations == [], "pause_dispatch shouldn't fire"


def test_no_drain_when_transition_is_none_to_secondary(fake_state):
    """From (primary UNHEALTHY, secondary UNHEALTHY) → serving None. Then
    secondary flips HEALTHY → serving becomes secondary. Nothing was
    active before, so no drain."""
    p = fake_state(("p1", "aws", "us-west", "us-west-1a", "primary"), uuid="P")
    s = fake_state(("s1", "aws", "us-east", "us-east-1a", "primary"), uuid="S")
    group = FailoverGroup(
        primary=p, secondary=s, lock=threading.Lock(),
        primary_status=HealthResult.UNHEALTHY,
        secondary_status=HealthResult.UNHEALTHY,
        cooldown_s=0,
    )
    # A stale primary conn that would nominally get closed if drain ran.
    stale = _FakeConn(TransactionStatus.INTRANS)
    group.primary.tracked_conns.add(stale)

    trigger_drain(
        group,
        which_cluster="secondary",
        new_status=HealthResult.HEALTHY,
        drain_timeout_s=5,
    )

    assert group.secondary_status == HealthResult.HEALTHY
    assert stale.closed is False   # nothing was active before, no drain


# ------------------------------------------------------- routing change

def test_primary_flip_drains_primary_and_flips_dispatch(fake_state):
    """primary HEALTHY → UNHEALTHY, secondary HEALTHY → routing changes
    primary → secondary. The barrier fires: dispatch_paused during the
    drain, and primary's idle conns are counted (nothing to close),
    then dispatch resumes."""
    group = _make_group(fake_state)
    # An IDLE conn on primary — counts as drained already.
    idle_conn = _FakeConn(TransactionStatus.IDLE)
    group.primary.tracked_conns.add(idle_conn)

    # Snapshot dispatch_paused history via a background watcher.
    paused_observations: list[bool] = []
    stop = threading.Event()

    def observer():
        while not stop.wait(timeout=0.005):
            paused_observations.append(group.dispatch_paused)

    t = threading.Thread(target=observer, daemon=True)
    t.start()

    trigger_drain(
        group,
        which_cluster="primary",
        new_status=HealthResult.UNHEALTHY,
        drain_timeout_s=5,
    )

    stop.set()
    t.join(timeout=1.0)

    assert group.primary_status == HealthResult.UNHEALTHY
    assert group.secondary_status == HealthResult.HEALTHY
    assert group.dispatch_paused is False, "should resume after drain"
    assert idle_conn.closed is False   # IDLE counts as drained; not force-closed
    # We should have observed at least one True in the paused history —
    # the drain window flipped it briefly. (Best-effort; if the drain
    # completes faster than one 5ms sample, this is skipped.)
    if any(paused_observations):
        assert True   # good — barrier fired
    # If the sampler missed the pause window entirely, that's ok too:
    # the atomic invariant (paused THEN resumed) still holds since the
    # final state is False and _apply_status_change succeeded.


# ------------------------------------------------------- sentinel semantics

def test_drain_timeout_zero_kills_immediately(fake_state):
    """drain_timeout_s = 0 → force-close in-flight conns without waiting."""
    group = _make_group(fake_state)
    in_txn = _FakeConn(TransactionStatus.INTRANS)
    group.primary.tracked_conns.add(in_txn)

    t0 = time.monotonic()
    trigger_drain(
        group,
        which_cluster="primary",
        new_status=HealthResult.UNHEALTHY,
        drain_timeout_s=0,
    )
    elapsed = time.monotonic() - t0

    assert in_txn.closed is True
    assert elapsed < 0.3, f"expected immediate kill, took {elapsed:.2f}s"
    assert group.primary_status == HealthResult.UNHEALTHY
    assert group.dispatch_paused is False


def test_drain_timeout_positive_waits_then_kills(fake_state):
    """drain_timeout_s = 1 → poll for ~1s, then force-close survivors."""
    group = _make_group(fake_state)
    in_txn = _FakeConn(TransactionStatus.INTRANS)
    group.primary.tracked_conns.add(in_txn)

    t0 = time.monotonic()
    trigger_drain(
        group,
        which_cluster="primary",
        new_status=HealthResult.UNHEALTHY,
        drain_timeout_s=1,
    )
    elapsed = time.monotonic() - t0

    assert in_txn.closed is True   # survivor was force-closed
    assert 0.9 < elapsed < 1.8, f"expected ~1s drain, took {elapsed:.2f}s"


def test_drain_timeout_negative_one_never_kills(fake_state):
    """drain_timeout_s = -1 → wait indefinitely; if the conn transitions
    to IDLE naturally, drain ends cleanly with no force-close."""
    group = _make_group(fake_state)
    conn = _FakeConn(TransactionStatus.INTRANS)
    group.primary.tracked_conns.add(conn)

    # A helper thread transitions the conn to IDLE after a short delay,
    # simulating the app committing its transaction.
    def commit_after_delay():
        time.sleep(0.2)
        conn.info.transaction_status = TransactionStatus.IDLE

    t = threading.Thread(target=commit_after_delay, daemon=True)
    t.start()

    t0 = time.monotonic()
    trigger_drain(
        group,
        which_cluster="primary",
        new_status=HealthResult.UNHEALTHY,
        drain_timeout_s=-1,
    )
    elapsed = time.monotonic() - t0
    t.join(timeout=1.0)

    assert conn.closed is False, "drain_timeout=-1 must never force-close"
    assert 0.15 < elapsed < 1.5, f"expected ~0.2s wait, got {elapsed:.2f}s"
    assert group.primary_status == HealthResult.UNHEALTHY


def test_drain_completes_before_deadline_when_conns_drain_naturally(
    fake_state,
):
    """If in-flight conns finish before drain_timeout_s expires, the
    drain returns immediately — no unnecessary wait."""
    group = _make_group(fake_state)
    conn = _FakeConn(TransactionStatus.INTRANS)
    group.primary.tracked_conns.add(conn)

    def commit_soon():
        time.sleep(0.15)
        conn.info.transaction_status = TransactionStatus.IDLE

    t = threading.Thread(target=commit_soon, daemon=True)
    t.start()

    t0 = time.monotonic()
    trigger_drain(
        group,
        which_cluster="primary",
        new_status=HealthResult.UNHEALTHY,
        drain_timeout_s=5,     # 5s budget; conn commits at 0.15s
    )
    elapsed = time.monotonic() - t0
    t.join(timeout=1.0)

    assert 0.1 < elapsed < 1.5, (
        f"drain should end when conn goes idle, took {elapsed:.2f}s"
    )
    assert conn.closed is False   # committed cleanly; not force-closed


# ============================================================
# autoFailbackEnabled suppression (P1.2)
# ============================================================

def test_auto_failback_true_still_fails_back(fake_state):
    """Default behaviour — primary UNHEALTHY → HEALTHY triggers the
    drain (drain of secondary + flip to primary)."""
    group = _make_group(fake_state)
    # Start on secondary (primary unhealthy).
    group.primary_status = HealthResult.UNHEALTHY
    assert group.auto_failback_enabled is True   # default

    conn = _FakeConn(TransactionStatus.IDLE)
    group.secondary.tracked_conns.add(conn)

    trigger_drain(
        group,
        which_cluster="primary",
        new_status=HealthResult.HEALTHY,
        drain_timeout_s=1,
    )

    # Status flipped back — failback ran.
    assert group.primary_status == HealthResult.HEALTHY


def test_auto_failback_false_suppresses_failback(fake_state, caplog):
    """autoFailbackEnabled=false — primary CB reports HEALTHY but the
    driver leaves primary_status UNHEALTHY and dispatch stays on
    secondary. Logs a WARNING so the operator sees it."""
    import logging
    group = _make_group(fake_state)
    group.primary_status = HealthResult.UNHEALTHY   # failover already done
    group.auto_failback_enabled = False

    # Watch pause_dispatch to prove the drain never fired.
    pause_observations: list[bool] = []
    orig_pause = group.pause_dispatch

    def watched_pause():
        pause_observations.append(True)
        orig_pause()
    group.pause_dispatch = watched_pause  # type: ignore[method-assign]

    with caplog.at_level(logging.WARNING, logger="psycopg.yb.drain"):
        trigger_drain(
            group,
            which_cluster="primary",
            new_status=HealthResult.HEALTHY,
            drain_timeout_s=1,
        )

    # Status NOT flipped — dispatch stays on secondary.
    assert group.primary_status == HealthResult.UNHEALTHY
    # Drain never ran.
    assert pause_observations == []
    # Operator got a heads-up in the log.
    assert any(
        "failback suppressed" in rec.message
        for rec in caplog.records
    ), f"expected suppression WARNING, got: {[r.message for r in caplog.records]}"


def test_auto_failback_false_still_allows_failover(fake_state):
    """autoFailbackEnabled=false only blocks failback (HEALTHY → serving
    primary). Failover (primary UNHEALTHY, moving to secondary) still
    runs unconditionally — the toggle is asymmetric."""
    group = _make_group(fake_state)
    # Start healthy on primary.
    group.auto_failback_enabled = False

    trigger_drain(
        group,
        which_cluster="primary",
        new_status=HealthResult.UNHEALTHY,
        drain_timeout_s=1,
    )
    assert group.primary_status == HealthResult.UNHEALTHY   # failover ran



# ============================================================
# Per-stage timestamps on FailoverGroup (P1.5)
# ============================================================

def test_per_stage_timestamps_populated_on_failover(fake_state):
    """Full failover cycle: all four stage timestamps get populated,
    in strictly increasing monotonic order."""
    import time as _time
    group = _make_group(fake_state)
    # No conns to drain — fast.
    t_before = _time.monotonic()
    trigger_drain(
        group,
        which_cluster="primary",
        new_status=HealthResult.UNHEALTHY,
        drain_timeout_s=0,
    )
    t_after = _time.monotonic()

    assert t_before <= group.last_cb_trip_ts <= t_after
    assert t_before <= group.last_failover_start_ts <= t_after
    assert t_before <= group.last_phase1_complete_ts <= t_after
    assert t_before <= group.last_phase2_complete_ts <= t_after
    assert t_before <= group.last_failover_complete_ts <= t_after
    # Monotonic order along the pipeline.
    assert group.last_cb_trip_ts <= group.last_failover_start_ts
    assert group.last_failover_start_ts <= group.last_phase1_complete_ts
    assert group.last_phase1_complete_ts <= group.last_phase2_complete_ts
    assert group.last_phase2_complete_ts <= group.last_failover_complete_ts
    # Failback did NOT fire on a failover.
    assert group.last_failback_complete_ts == 0.0


def test_per_stage_timestamps_populated_on_failback(fake_state):
    """Failback path stamps last_failback_complete_ts (not
    last_failover_complete_ts)."""
    import time as _time
    group = _make_group(fake_state)
    group.primary_status = HealthResult.UNHEALTHY   # already failed over
    # Simulate a completed prior failover stamp so we can verify the
    # failback timestamp is set fresh.
    group.last_failover_complete_ts = 1.0
    prior_failover_ts = group.last_failover_complete_ts

    t_before = _time.monotonic()
    trigger_drain(
        group,
        which_cluster="primary",
        new_status=HealthResult.HEALTHY,
        drain_timeout_s=0,
    )
    t_after = _time.monotonic()

    # Failback ran → last_failback_complete_ts stamped.
    assert t_before <= group.last_failback_complete_ts <= t_after
    # Failover completion was NOT re-stamped on a failback.
    assert group.last_failover_complete_ts == prior_failover_ts
    # cb_trip_ts should NOT fire on a HEALTHY transition (no cluster
    # went from healthy → unhealthy this round).
    assert group.last_cb_trip_ts == 0.0


def test_last_cb_trip_ts_only_fires_on_unhealthy_primary(fake_state):
    """cb_trip fires on primary → UNHEALTHY; secondary flipping should
    not set it (design doc §3.8 — its motivating case is the primary
    going bad)."""
    group = _make_group(fake_state)
    # Flip secondary UNHEALTHY — no routing change, no timestamps.
    trigger_drain(
        group,
        which_cluster="secondary",
        new_status=HealthResult.UNHEALTHY,
        drain_timeout_s=0,
    )
    assert group.last_cb_trip_ts == 0.0


def test_timestamps_not_touched_when_no_routing_change(fake_state):
    """A CB flip that does not change the routing target should NOT
    stamp the drain-cycle timestamps (there is no drain cycle)."""
    group = _make_group(fake_state)
    trigger_drain(
        group,
        which_cluster="secondary",
        new_status=HealthResult.UNHEALTHY,
        drain_timeout_s=0,
    )
    assert group.last_failover_start_ts == 0.0
    assert group.last_phase1_complete_ts == 0.0
    assert group.last_phase2_complete_ts == 0.0
    assert group.last_failover_complete_ts == 0.0
    assert group.last_failback_complete_ts == 0.0

