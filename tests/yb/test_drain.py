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


# ------------------------------------------------------- routing change

def test_primary_flip_drains_primary_and_flips_dispatch(fake_state):
    """primary HEALTHY → UNHEALTHY, secondary HEALTHY → routing changes
    primary → secondary. The barrier fires: dispatch_paused during the
    drain, every tracked conn on primary (in-flight AND idle) is
    closed at the flip boundary (DB-24801 — IDLE is a transient state
    between commits, so leaving an idle direct conn alive post-flip
    lets it commit on the outgoing universe on its next cursor call),
    then dispatch resumes."""
    group = _make_group(fake_state)
    # An IDLE conn on primary — closed at the flip boundary per DB-24801.
    idle_conn = _FakeConn(TransactionStatus.IDLE)
    group.primary.tracked_conns.add(idle_conn)

    # Verify dispatch_paused DID fire by wrapping pause_dispatch itself,
    # the same pattern the no-drain tests use. agent-K PR #8:208: a
    # background sampler racing a sub-millisecond drain can miss the
    # paused window entirely, so the previous "if any(obs): assert True"
    # was effectively a no-op — this test would pass even if the drain
    # stopped calling pause_dispatch.
    pause_fire_count = [0]
    resume_fire_count = [0]
    orig_pause = group.pause_dispatch
    orig_resume = group.resume_dispatch

    def watched_pause():
        pause_fire_count[0] += 1
        orig_pause()

    def watched_resume():
        resume_fire_count[0] += 1
        orig_resume()

    group.pause_dispatch = watched_pause   # type: ignore[method-assign]
    group.resume_dispatch = watched_resume  # type: ignore[method-assign]

    trigger_drain(
        group,
        which_cluster="primary",
        new_status=HealthResult.UNHEALTHY,
        drain_timeout_s=5,
    )

    assert group.primary_status == HealthResult.UNHEALTHY
    assert group.secondary_status == HealthResult.HEALTHY
    assert group.dispatch_paused is False, "should resume after drain"
    assert idle_conn.closed is True   # DB-24801 fix: closed at flip boundary
    # Barrier semantics: pause fired exactly once, resume fired exactly
    # once. Removes the race condition the old sampler-based assertion
    # had.
    assert pause_fire_count[0] == 1, (
        f"pause_dispatch should have fired exactly once during a "
        f"routing-change drain, fired {pause_fire_count[0]}×"
    )
    assert resume_fire_count[0] == 1, (
        f"resume_dispatch should have fired exactly once at drain end, "
        f"fired {resume_fire_count[0]}×"
    )


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


def test_drain_timeout_negative_one_waits_then_closes_idle_at_flip(fake_state):
    """drain_timeout_s = -1 → wait indefinitely for in-flight to drain;
    never force-close in-flight (that's what the -1 sentinel promises).
    But once in-flight is empty, close the conn at the flip boundary
    anyway (DB-24801 — leaving the now-idle conn alive would let it
    commit on the outgoing universe on its next cursor call)."""
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

    # DB-24801: idle survivors are closed at the flip boundary even
    # with -1, because the dual-write risk is independent of the
    # drain-timeout sentinel. The -1 promise is "never force-close
    # IN-FLIGHT", which we honour by waiting until it goes idle first.
    assert conn.closed is True
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
    assert conn.closed is True   # DB-24801 fix: closed at flip boundary
                                  # even though the txn drained naturally


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



# ------------------------------------------------------- no-viable transitions

def test_last_viable_cluster_going_down_opens_no_drain(fake_state, caplog):
    """Routing on secondary (primary already UNHEALTHY), then secondary
    goes UNHEALTHY → no viable cluster. There is nothing to drain
    towards, so no drain window opens (the harness models a drain as
    outgoing → incoming; an X → None drain never gets closed)."""
    import logging
    group = _make_group(fake_state)
    group.primary_status = HealthResult.UNHEALTHY
    conn = _FakeConn(TransactionStatus.INTRANS)
    group.secondary.tracked_conns.add(conn)
    pauses: list[bool] = []
    orig_pause = group.pause_dispatch
    group.pause_dispatch = lambda: (pauses.append(True), orig_pause())[1]  # type: ignore[method-assign]
    caplog.set_level(logging.INFO, logger="psycopg.yb.drain")

    trigger_drain(group, "secondary", HealthResult.UNHEALTHY, drain_timeout_s=0)

    assert group.secondary_status == HealthResult.UNHEALTHY
    assert pauses == []
    assert conn.closed is False
    assert not any("drain-wait start" in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize("recovering,other", [
    ("secondary", "primary"),
    ("primary", "secondary"),
])
def test_recovery_from_no_viable_drains_the_other_side(
    fake_state, caplog, recovering, other,
):
    """Both UNHEALTHY → one recovers. Routing changes None → recovering,
    so the driver must run a real drain against the other side (closing
    conns that outlived the outage there) and log the flip as a routing
    change, not "routing unchanged"."""
    import logging
    group = _make_group(fake_state)
    group.primary_status = HealthResult.UNHEALTHY
    group.secondary_status = HealthResult.UNHEALTHY
    survivor = _FakeConn(TransactionStatus.IDLE)
    getattr(group, other).tracked_conns.add(survivor)
    caplog.set_level(logging.INFO, logger="psycopg.yb.drain")

    trigger_drain(group, recovering, HealthResult.HEALTHY, drain_timeout_s=0)

    assert getattr(group, f"{recovering}_status") == HealthResult.HEALTHY
    assert survivor.closed is True
    assert group.dispatch_paused is False
    msgs = [r.getMessage() for r in caplog.records]
    other_uuid = getattr(group, other).uuid
    assert any(
        "drain-wait start" in m and f"outgoing={other_uuid}" in m for m in msgs
    ), msgs
    assert any("routing target flipped" in m for m in msgs), msgs
    assert not any("routing unchanged" in m for m in msgs), msgs
    if recovering == "primary":
        assert group.last_failback_complete_ts > 0
    else:
        assert group.last_failover_complete_ts > 0


def test_recovery_from_no_viable_skips_phase2(fake_state):
    """The other side wasn't serving during no-viable, so there is no
    lag to wait for: Phase 2 must not poll YBA on recovery."""
    group = _make_group(fake_state)
    group.primary_status = HealthResult.UNHEALTHY
    group.secondary_status = HealthResult.UNHEALTHY

    class _NoPollYBA:
        def get_committed_lag_ms(self, uuid):
            raise AssertionError("Phase 2 must be skipped on no-viable recovery")

        get_committed_lag_sample = get_committed_lag_ms

    group.yba_client = _NoPollYBA()
    group.xcluster_config_uuid = "xcc-a2b"
    group.failback_xcluster_config_uuid = "xcc-b2a"

    trigger_drain(group, "secondary", HealthResult.HEALTHY, drain_timeout_s=0)

    assert group.secondary_status == HealthResult.HEALTHY


# ------------------------------------------------------- close fence

def test_flip_waits_close_fence_after_closing_outgoing_conns(fake_state, monkeypatch):
    """two_leaders (a7932f37 cycle 1): the old leader's session was closed
    0.24s before the new leader acquired the advisory lock on the other
    universe, but its owner only noticed on its next 1s heartbeat. The
    drain holds dispatch for the close fence after closing conns."""
    from psycopg.yb import drain as drain_mod
    monkeypatch.setattr(drain_mod, "_CLOSE_FENCE_S", 1.0)
    group = _make_group(fake_state)
    conn = _FakeConn(TransactionStatus.IDLE)   # keep a ref: tracked_conns is weak
    group.primary.tracked_conns.add(conn)

    t0 = time.monotonic()
    trigger_drain(group, "primary", HealthResult.UNHEALTHY, drain_timeout_s=0)
    elapsed = time.monotonic() - t0

    assert conn.closed is True
    assert elapsed >= 0.95
    assert group.primary_status == HealthResult.UNHEALTHY
    assert group.dispatch_paused is False


def test_flip_skips_close_fence_when_nothing_was_closed(fake_state, monkeypatch):
    from psycopg.yb import drain as drain_mod
    monkeypatch.setattr(drain_mod, "_CLOSE_FENCE_S", 1.0)
    group = _make_group(fake_state)

    t0 = time.monotonic()
    trigger_drain(group, "primary", HealthResult.UNHEALTHY, drain_timeout_s=0)

    assert time.monotonic() - t0 < 0.5
