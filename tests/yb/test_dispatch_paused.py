"""
Unit tests for the dispatch-pause barrier and direct-connect weakref
tracking (Phase D of the xCluster failover implementation plan).

Verifies:

  * ``FailoverGroup.dispatch_paused`` + ``dispatch_paused_condition`` —
    pause blocks a ``wait_for_dispatch`` caller until resume flips it
    back and notifies.
  * ``FailoverGroup.pause_dispatch`` / ``resume_dispatch`` — the
    ergonomic API the drain sequence (Phase E) will call.
  * ``ClusterState.tracked_conns`` — a ``WeakSet`` that holds only weak
    references, doesn't prevent GC, and correctly deduplicates.
  * ``wait_for_dispatch`` timeout semantics — returns True when the
    pause clears, False on timeout.
"""

# Copyright (C) 2026 Yugabyte

from __future__ import annotations

import gc
import threading
import time
import weakref

import pytest

from psycopg.yb.health import HealthResult
from psycopg.yb.registry import ClusterState, FailoverGroup
from psycopg.yb.state import wait_for_dispatch


pytestmark = pytest.mark.yb_unit


def _make_group(fake_state):
    p = fake_state(("p1", "aws", "us-west", "us-west-1a", "primary"), uuid="P")
    s = fake_state(("s1", "aws", "us-east", "us-east-1a", "primary"), uuid="S")
    return FailoverGroup(
        primary=p, secondary=s, lock=threading.Lock(),
        primary_status=HealthResult.HEALTHY,
        secondary_status=HealthResult.HEALTHY,
        cooldown_s=0,
    )


# ---------------------------------------------------------- construction

def test_new_group_starts_unpaused(fake_state):
    group = _make_group(fake_state)
    assert group.dispatch_paused is False
    assert group.dispatch_paused_condition is not None


def test_post_init_binds_condition_to_lock(fake_state):
    """The Condition's underlying lock is the SAME object the rest of the
    group's state changes under — otherwise wait/notify wouldn't
    synchronise with `with group.lock:`."""
    group = _make_group(fake_state)
    # Behavioural test: acquire group.lock, then verify the condition
    # can't be re-acquired (it's held via the same lock).
    with group.lock:
        assert not group.dispatch_paused_condition.acquire(blocking=False)


# ---------------------------------------------------------- pause/resume

def test_pause_dispatch_flips_flag(fake_state):
    group = _make_group(fake_state)
    group.pause_dispatch()
    assert group.dispatch_paused is True


def test_resume_dispatch_flips_flag_and_notifies(fake_state):
    group = _make_group(fake_state)
    group.pause_dispatch()

    unblocked = threading.Event()

    def waiter():
        wait_for_dispatch(group)
        unblocked.set()

    t = threading.Thread(target=waiter, daemon=True)
    t.start()

    # Give the waiter time to enter the wait.
    time.sleep(0.1)
    assert not unblocked.is_set(), "waiter unblocked before resume"

    group.resume_dispatch()
    assert unblocked.wait(timeout=2.0), "waiter never unblocked"
    t.join(timeout=1.0)


# ---------------------------------------------------------- wait_for_dispatch

def test_wait_returns_immediately_when_not_paused(fake_state):
    group = _make_group(fake_state)
    t0 = time.monotonic()
    assert wait_for_dispatch(group, timeout_s=5.0) is True
    assert time.monotonic() - t0 < 0.05


def test_wait_returns_false_on_timeout(fake_state):
    group = _make_group(fake_state)
    group.pause_dispatch()

    t0 = time.monotonic()
    got_it = wait_for_dispatch(group, timeout_s=0.15)
    elapsed = time.monotonic() - t0
    assert got_it is False
    assert 0.10 < elapsed < 0.40, f"expected ~0.15s wait, got {elapsed}"


def test_wait_returns_true_after_late_resume(fake_state):
    group = _make_group(fake_state)
    group.pause_dispatch()

    # Schedule resume shortly after wait starts.
    def late_resume():
        time.sleep(0.1)
        group.resume_dispatch()

    t = threading.Thread(target=late_resume, daemon=True)
    t.start()

    got_it = wait_for_dispatch(group, timeout_s=2.0)
    assert got_it is True
    t.join(timeout=1.0)


# ---------------------------------------------------------- tracked_conns

def test_tracked_conns_holds_weakrefs_only(fake_state):
    group = _make_group(fake_state)
    state = group.primary

    class FakeConn:
        pass

    # Track a conn; verify it's discoverable.
    conn = FakeConn()
    state.tracked_conns.add(conn)
    assert conn in state.tracked_conns
    assert len(state.tracked_conns) == 1

    # Drop the strong ref and force GC — the WeakSet should evict.
    del conn
    gc.collect()
    assert len(state.tracked_conns) == 0


def test_tracked_conns_dedups(fake_state):
    """Adding the same conn twice is a no-op — WeakSet semantics."""
    group = _make_group(fake_state)
    state = group.primary

    class FakeConn:
        pass

    conn = FakeConn()
    state.tracked_conns.add(conn)
    state.tracked_conns.add(conn)
    assert len(state.tracked_conns) == 1


def test_pool_and_direct_tracked_conns_are_disjoint(fake_state):
    """Placeholder assertion — the pool-vs-direct policy is enforced by
    the dispatcher (only direct connects call state.tracked_conns.add).
    Here we simply document that a synthetic direct-connect goes in and
    a synthetic pool-borrow does NOT (because we don't add it)."""
    group = _make_group(fake_state)
    state = group.primary

    class DirectConn:
        pass

    class PooledConn:
        pass

    direct = DirectConn()
    pooled = PooledConn()   # never added — the pool holds this ref itself
    state.tracked_conns.add(direct)

    assert direct in state.tracked_conns
    assert pooled not in state.tracked_conns
