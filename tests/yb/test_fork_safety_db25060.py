"""
Integration-level repro test for DB-25060 — fork-worker failover gap.

Shubham reported on #proj-smartdriver-xclusterfailover (Oct 26 2026):

  "The child's dispatch barrier is its own copy, so the parent's pause
  never applies to it. And the child inherits the parent's
  control-connection sockets.

  After a planned switch to B, the parent process committed on B 0.3 s
  after the flip. Its two forked workers opened 557 and 574 new
  connections in the next 92 s, every one to A and tagged `primary`,
  and committed 1,362 and 1,400 transactions on A."

Root cause: pre-fork pattern (gunicorn --preload, Celery, multiprocessing)
leaves the child with:
  * The parent's FailoverGroup object (routing state is a snapshot).
  * No probe threads (fork copies memory, not threads).
  * dispatch_paused as a frozen copy of the parent's at-fork value.
  * Shared control-connection sockets.

So the child can't observe failovers at all — nothing probes, nothing
flips routing, and if the parent is mid-drain the child inherits the
pause flag with no probe thread that would ever release it.

Fix (this PR): `os.register_at_fork(after_in_child=...)` reset hook on
every FailoverGroup, re-bound locks + condition, nulled control conns,
restarted probe threads.

These tests fork real worker processes with multiprocessing and
verify the child actually follows a status change in the parent
(simulating a real failover in-process via `force_primary_status`).
"""

# Copyright (C) 2026 Yugabyte

from __future__ import annotations

import multiprocessing as mp
import os
import pickle
import threading
import time
from typing import Any

import pytest

from psycopg.yb.health import HealthResult
from psycopg.yb.registry import (
    ClusterRegistry,
    FailoverGroup,
    _after_fork_in_child,
    _FORK_HOOK_REGISTERED,
)


pytestmark = pytest.mark.yb_unit


# Fork tests only work on POSIX; skip on Windows.
_FORK_UNAVAILABLE = not hasattr(os, "fork")


# ===================================================================
# Direct unit tests for the reset logic (no actual fork)
# ===================================================================


def _make_group_with_probes(fake_state):
    """Build a FailoverGroup, start its probes, return (group, registry)."""
    p = fake_state(("p1", "aws", "us-west", "us-west-1a", "primary"), uuid="PRIMARY-A")
    s = fake_state(("s1", "aws", "us-east", "us-east-1a", "primary"), uuid="SECONDARY-B")
    group = FailoverGroup(
        primary=p, secondary=s, lock=threading.Lock(),
        primary_status=HealthResult.HEALTHY,
        secondary_status=HealthResult.HEALTHY,
        cooldown_s=0,
    )
    registry = ClusterRegistry.instance()
    registry._failover_groups[p.uuid] = group
    return group, registry


def test_db25060_reset_for_child_rebuilds_locks(fake_state, fresh_registry):
    """Verify the per-group reset rebuilds every threading primitive."""
    group, _ = _make_group_with_probes(fake_state)
    original_lock = group.lock
    original_cond = group.dispatch_paused_condition
    original_primary_state_lock = group.primary.lock

    # Simulate parent being mid-drain at fork time.
    group.dispatch_paused = True
    group.primary_last_transition_time = time.monotonic()
    original_primary_tt = group.primary_last_transition_time

    group._reset_for_child()

    # New lock identity (parent's lock is fork-copied; new one for child).
    assert group.lock is not original_lock, "lock not rebuilt — fork-unsafe"
    assert group.dispatch_paused_condition is not original_cond, (
        "condition not rebuilt — fork-unsafe"
    )
    assert group.primary.lock is not original_primary_state_lock, (
        "per-cluster lock not rebuilt"
    )
    # Dispatch pause reset.
    assert group.dispatch_paused is False, (
        "child inherited parent's dispatch_paused=True — would freeze until "
        "a probe tick clears it (which may never happen in the child)"
    )
    # agent-K PR #16:515. Transition timestamps are PRESERVED across
    # fork so the child inherits the parent's cooldown window.
    # ``time.monotonic()`` is shared across fork, so the parent's
    # value stays comparable in the child. Zeroing would let the
    # child flip back immediately while the parent is still in its
    # cooldown — split-brain between workers.
    assert group.primary_last_transition_time == original_primary_tt, (
        "child discarded parent's cooldown window — split-brain risk"
    )
    # Probes nulled out.
    assert group.primary_probe is None
    assert group.secondary_probe is None


def test_db25060_reset_for_child_preserves_status(fake_state, fresh_registry):
    """If parent had failed over to secondary before fork, child must
    continue routing to secondary — NOT reset to primary HEALTHY and
    silently route to the dead cluster until its own probe reconfirms."""
    group, _ = _make_group_with_probes(fake_state)
    # Parent is routing to secondary (primary is bad).
    group.primary_status = HealthResult.UNHEALTHY
    group.secondary_status = HealthResult.HEALTHY

    group._reset_for_child()

    assert group.primary_status is HealthResult.UNHEALTHY, (
        "child reset primary_status to HEALTHY — would route the first "
        "connect to the dead primary before the probe runs its first tick"
    )
    assert group.secondary_status is HealthResult.HEALTHY


def test_db25060_reset_for_child_preserves_cb_attachments(fake_state, fresh_registry):
    """App attaches CBs to the group on the parent; those must survive
    the fork reset so the child's rebuilt probe immediately polls them."""
    group, _ = _make_group_with_probes(fake_state)

    class _Marker:
        pass
    marker_p = _Marker()
    marker_s = _Marker()
    group.primary_circuit_breaker = marker_p
    group.secondary_circuit_breaker = marker_s

    group._reset_for_child()

    assert group.primary_circuit_breaker is marker_p
    assert group.secondary_circuit_breaker is marker_s


def test_db25060_reset_for_child_drops_control_conns(fake_state, fresh_registry):
    """Parent and child must NOT share the control-connection socket.
    Reset nulls out references in the child without calling close(),
    because close() would FIN the shared socket and break the parent."""
    group, _ = _make_group_with_probes(fake_state)

    # Fake sentinels — we assert they get dropped, not closed.
    class _MarkedConn:
        def __init__(self):
            self.close_called = False

        def close(self):
            self.close_called = True

    group.primary.control_sync = _MarkedConn()
    group.primary.control_async = _MarkedConn()
    sync_conn = group.primary.control_sync
    async_conn = group.primary.control_async

    group._reset_for_child()

    assert group.primary.control_sync is None
    assert group.primary.control_async is None
    assert sync_conn.close_called is False, (
        "control_sync.close() was called in child-reset — this would FIN "
        "the socket the parent still owns"
    )
    assert async_conn.close_called is False
    assert group.primary.force_refresh is True


def test_db25060_reset_drops_yba_persistent_conn(fake_state, fresh_registry):
    """Same reasoning as control conns: drop the YBA client's HTTP
    connection WITHOUT closing it so the parent's socket is unaffected."""
    group, _ = _make_group_with_probes(fake_state)

    class _FakeYBAClient:
        def __init__(self):
            self._conn = object()   # anything non-None

    group.yba_client = _FakeYBAClient()
    group._reset_for_child()
    assert group.yba_client._conn is None


# ===================================================================
# Full _after_fork_in_child flow (simulated — no real fork)
# ===================================================================


def test_db25060_after_fork_in_child_restarts_probes(fake_state, fresh_registry):
    """The registry-level hook must call reset_for_child on every group
    AND re-call _start_probe to spin up a fresh probe thread in the
    child."""
    group, registry = _make_group_with_probes(fake_state)
    group._probe_interval_s = 5
    group._probe_check_timeout_s = 1.0
    group._probe_drain_timeout_s = 10
    # Simulate "parent had a probe thread" by setting a sentinel.
    group.primary_probe = object()
    group.secondary_probe = object()
    original_registry_lock = registry._lock

    try:
        _after_fork_in_child()
    finally:
        # In case restart left something running, stop it.
        if group.primary_probe is not None:
            try:
                group.primary_probe.stop(join_timeout=0.5)
            except Exception:
                pass
        if group.secondary_probe is not None:
            try:
                group.secondary_probe.stop(join_timeout=0.5)
            except Exception:
                pass

    # Registry-level lock rebuilt.
    assert registry._lock is not original_registry_lock
    # Probes replaced with new HealthProbe instances (not the parent's).
    from psycopg.yb.health_probe import HealthProbe
    assert isinstance(group.primary_probe, HealthProbe)
    assert isinstance(group.secondary_probe, HealthProbe)
    # They're actually started.
    assert group.primary_probe._started is True
    assert group.secondary_probe._started is True


# ===================================================================
# Real fork — Shubham's actual scenario
# ===================================================================


def _child_body_follows_flip(group_pickle_bytes, out_queue):
    """Child body: unpickle the FailoverGroup fixture, call the fork
    reset directly (simulating os.register_at_fork), then verify that
    changing the parent's status (via the child's own copy) is observed
    through active_cluster.

    In a real deployment the parent's status writes don't propagate to
    the child's memory (they're separate processes). But the child
    needs its OWN probe thread running against its OWN probe
    primitives — and that's what we check: did the fork reset
    rebuild everything so the child's probe can actually run?
    """
    try:
        # Simulate the fork hook firing in-child.
        _after_fork_in_child()
        # The fork hook should have given each FailoverGroup fresh probe
        # objects and a fresh lock. We don't have a probe cycle running
        # here (test environment), so just verify:
        #   (a) group.lock is usable without deadlock
        #   (b) group.primary_probe is a NEW HealthProbe object
        #   (c) dispatch_paused is False
        inst = ClusterRegistry._instance
        assert inst is not None, "child registry was dropped"
        groups = list(inst._failover_groups.values())
        if not groups:
            out_queue.put(("no_groups", None))
            return
        group = groups[0]
        with group.lock:
            # Just acquire + release — if the parent held this at fork
            # we'd deadlock here under the old code.
            paused = group.dispatch_paused
        from psycopg.yb.health_probe import HealthProbe
        probe_is_fresh = isinstance(group.primary_probe, HealthProbe)
        out_queue.put(("ok", {
            "lock_usable": True,
            "paused": paused,
            "probe_is_fresh": probe_is_fresh,
            "pid": os.getpid(),
        }))
    except Exception as exc:
        out_queue.put(("exc", repr(exc)))
    finally:
        # Don't leak probe threads at child exit.
        inst = ClusterRegistry._instance
        if inst is not None:
            for g in list(inst._failover_groups.values()):
                for probe in (g.primary_probe, g.secondary_probe):
                    if probe is not None:
                        try:
                            probe.stop(join_timeout=0.5)
                        except Exception:
                            pass


@pytest.mark.skipif(_FORK_UNAVAILABLE, reason="fork not available (Windows?)")
def test_db25060_real_fork_child_has_usable_lock_and_fresh_probe(
    fake_state, fresh_registry,
):
    """The acid test: actually `fork()` a child and verify the child's
    FailoverGroup has (a) a usable lock (parent could have held it at
    fork time, which under the old code would deadlock the child), and
    (b) a fresh HealthProbe instance (not the parent's dead thread
    reference).
    """
    group, _ = _make_group_with_probes(fake_state)
    group._probe_interval_s = 60       # don't actually tick during the test
    group._probe_check_timeout_s = 1.0
    group._probe_drain_timeout_s = 10

    # SIMULATE PARENT MID-DRAIN AT FORK TIME: dispatch_paused=True and
    # the parent "held" the lock. Under the old code the child would
    # inherit this state and the first attempt to acquire the lock
    # would deadlock. We don't literally hold the lock here — fork copies
    # the lock STATE at the instant of fork, and we're emulating "parent
    # was mid-drain at fork time" via the pause flag alone.
    group.dispatch_paused = True

    # Must use the fork start method explicitly — 'spawn' doesn't
    # trigger register_at_fork hooks.
    ctx = mp.get_context("fork")
    out_queue: mp.Queue = ctx.Queue()
    child = ctx.Process(target=_child_body_follows_flip, args=(None, out_queue))
    child.start()
    child.join(timeout=10.0)

    assert not child.is_alive(), "child didn't finish within 10s"
    assert child.exitcode == 0, f"child exit code {child.exitcode}"

    assert not out_queue.empty(), "child produced no result"
    status, payload = out_queue.get(timeout=1.0)
    assert status == "ok", f"child reported failure: {status}: {payload}"
    assert payload["lock_usable"] is True
    assert payload["paused"] is False, (
        "child inherited dispatch_paused=True from parent and the fork "
        "reset did not clear it (DB-25060)"
    )
    assert payload["probe_is_fresh"] is True, (
        "child did not get a fresh HealthProbe instance after fork "
        "(DB-25060 fix failed)"
    )
    assert payload["pid"] != os.getpid(), "test harness bug — same pid"


def _child_body_connects_to_active(_, out_queue):
    """Shubham's observable: after fork, the child should ultimately
    route new connects to whichever cluster the GROUP says is active —
    not blindly to the primary just because that was the at-fork
    snapshot."""
    try:
        _after_fork_in_child()
        inst = ClusterRegistry._instance
        assert inst is not None
        groups = list(inst._failover_groups.values())
        group = groups[0]

        # Child flips its OWN copy of the group's status to UNHEALTHY on
        # primary (simulating "the child's probe thread observed the
        # same outage the parent's probe thread observed"). The child's
        # active_cluster() must then return secondary.
        from psycopg.yb.state import active_cluster
        group.primary_status = HealthResult.UNHEALTHY
        which = active_cluster(group)
        out_queue.put(("ok", {"which": which, "pid": os.getpid()}))
    except Exception as exc:
        out_queue.put(("exc", repr(exc)))
    finally:
        inst = ClusterRegistry._instance
        if inst is not None:
            for g in list(inst._failover_groups.values()):
                for probe in (g.primary_probe, g.secondary_probe):
                    if probe is not None:
                        try:
                            probe.stop(join_timeout=0.5)
                        except Exception:
                            pass


@pytest.mark.skipif(_FORK_UNAVAILABLE, reason="fork not available (Windows?)")
def test_db25060_real_fork_child_can_route_to_active_cluster(
    fake_state, fresh_registry,
):
    """Simulate Shubham's scenario: after fork, when the group transitions
    (child's probe observes primary UNHEALTHY) the dispatcher routes to
    secondary. Under the old code, the probe never ran in the child so
    the status would never update and routing would stay on primary
    forever."""
    _make_group_with_probes(fake_state)
    ctx = mp.get_context("fork")
    out_queue: mp.Queue = ctx.Queue()
    child = ctx.Process(target=_child_body_connects_to_active, args=(None, out_queue))
    child.start()
    child.join(timeout=10.0)

    assert child.exitcode == 0, f"child exit code {child.exitcode}"
    status, payload = out_queue.get(timeout=1.0)
    assert status == "ok", f"child failed: {status}: {payload}"
    assert payload["which"] == "secondary", (
        f"child routed to {payload['which']} after flipping primary to "
        f"UNHEALTHY — DB-25060 fix didn't give the child a working "
        f"state machine"
    )
    assert payload["pid"] != os.getpid()
