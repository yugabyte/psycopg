"""
DB-25060 — forked workers follow the parent's failover (fork_follow.py).

Stress fork_workers_flip (-r7, cycle 3): once each forked worker ran its
own probes, every process drained and flipped on its own schedule. The
parent's barrier did not stop the workers' writes (397 commits on the
outgoing side after it), workers flipped before the parent (420 commits
on the incoming side before the flip), and the extra drains showed up as
a false trip. The harness treats the forked application as one failover.

These tests drive the parent's drain and a child follower in one process:
two FailoverGroups (parent's and child's copies) sharing one mapping.
"""

# Copyright (C) 2026 Yugabyte

from __future__ import annotations

import os
import threading
import time
import types

import pytest

from psycopg.pq import TransactionStatus
from psycopg.yb import fork_follow
from psycopg.yb.drain import trigger_drain
from psycopg.yb.health import HealthResult
from psycopg.yb.registry import FailoverGroup


pytestmark = pytest.mark.yb_unit


class _FakeInfo:
    def __init__(self, status: TransactionStatus) -> None:
        self.transaction_status = status


class _FakeDirectConn:
    def __init__(self) -> None:
        self.info = _FakeInfo(TransactionStatus.IDLE)
        self.closed = False
        self.pgconn = types.SimpleNamespace(socket=-1)

    def begin(self) -> None:
        self.info.transaction_status = TransactionStatus.INTRANS

    def commit(self) -> None:
        self.info.transaction_status = TransactionStatus.IDLE

    def close(self) -> None:
        self.closed = True
        self.info.transaction_status = TransactionStatus.UNKNOWN


def _make_group(fake_state, tag: str) -> FailoverGroup:
    p = fake_state((f"p1{tag}", "aws", "us-west", "us-west-1a", "primary"), uuid="PRIMARY-A")
    s = fake_state((f"s1{tag}", "aws", "us-east", "us-east-1a", "primary"), uuid="SECONDARY-B")
    return FailoverGroup(
        primary=p, secondary=s, lock=threading.Lock(),
        primary_status=HealthResult.HEALTHY,
        secondary_status=HealthResult.HEALTHY,
        cooldown_s=0,
    )


@pytest.fixture
def pair(fake_state, monkeypatch):
    """Parent group (leader) + child group following it on a thread.
    Yields (parent, child, follow, controls)."""
    parent = _make_group(fake_state, "")
    child = _make_group(fake_state, "c")
    follow = fork_follow.SharedFollowState(parent.primary_status, parent.secondary_status)
    parent._follow = follow
    child._follow = follow

    alive = {"parent": True}
    monkeypatch.setattr(fork_follow, "_pid_alive", lambda pid: alive["parent"])
    gone = []

    slot = follow.reserve_slot()
    snap = follow.read()
    follow.claim_slot(slot, snap["gen"])
    t = threading.Thread(
        target=fork_follow._follow_loop,
        args=(child, follow, slot, snap, gone.append),
        daemon=True,
    )
    t.start()
    yield parent, child, follow, {"alive": alive, "gone": gone, "thread": t, "slot": slot}
    alive["parent"] = False
    t.join(timeout=2)


def _wait(cond, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if cond():
            return True
        time.sleep(0.01)
    return cond()


def test_parent_phase1_waits_for_child_in_flight_txn(pair):
    parent, child, follow, _ = pair
    conn = _FakeDirectConn()
    conn.begin()
    child.primary.tracked_conns.add(conn)

    threading.Timer(0.6, conn.commit).start()
    t0 = time.monotonic()
    trigger_drain(parent, "primary", HealthResult.UNHEALTHY, drain_timeout_s=5)
    elapsed = time.monotonic() - t0

    assert 0.5 <= elapsed < 4.0, elapsed
    assert conn.closed
    assert _wait(lambda: child.primary_status == HealthResult.UNHEALTHY)
    assert _wait(lambda: not child.dispatch_paused)


def test_child_conns_force_closed_at_parent_deadline(pair):
    parent, child, follow, _ = pair
    conn = _FakeDirectConn()
    conn.begin()                       # never commits
    child.primary.tracked_conns.add(conn)

    t0 = time.monotonic()
    trigger_drain(parent, "primary", HealthResult.UNHEALTHY, drain_timeout_s=1)
    elapsed = time.monotonic() - t0

    assert conn.closed
    assert elapsed < 1 + fork_follow._FOLLOW_POLL_S * 10 + 2.5
    assert _wait(lambda: child.primary_status == HealthResult.UNHEALTHY)


def test_child_pauses_with_parent_barrier(pair):
    parent, child, follow, _ = pair
    parent.pause_dispatch()
    follow.publish_pause(parent.barrier_generation, True)
    assert _wait(lambda: child.dispatch_paused)
    parent.resume_dispatch()
    follow.publish_resume(parent.primary_status, parent.secondary_status)
    assert _wait(lambda: not child.dispatch_paused)


def test_child_catches_up_on_missed_flip(pair):
    """A flip that started and finished between two follower polls still
    moves the child, and closes its conns on the cluster it leaves."""
    parent, child, follow, _ = pair
    conn = _FakeDirectConn()
    child.primary.tracked_conns.add(conn)
    follow.publish_statuses(HealthResult.UNHEALTHY, HealthResult.HEALTHY)

    assert _wait(lambda: child.primary_status == HealthResult.UNHEALTHY)
    assert conn.closed


def test_child_starts_own_probes_when_parent_gone(pair):
    parent, child, follow, ctl = pair
    ctl["alive"]["parent"] = False
    ctl["thread"].join(timeout=2)
    assert ctl["gone"] == [child]
    assert child._follow is None
    assert follow._read_slot(ctl["slot"])[0] == 0


def test_unacked_live_child_holds_phase1(fake_state):
    follow = fork_follow.SharedFollowState(HealthResult.HEALTHY, HealthResult.HEALTHY)
    slot = follow.reserve_slot()
    follow.claim_slot(slot, 0)
    # Barrier gen 1 armed; the child has not seen it yet.
    assert follow.children_in_flight(1) == 1
    follow.ack(slot, 1)
    assert follow.children_in_flight(1) == 0
    # A child that stopped heart-beating no longer counts.
    follow._write_slot(slot, acked_gen=0, heartbeat=time.monotonic() - 5)
    assert follow.children_in_flight(1) == 0


# ===================================================================
# Real fork through the registry hooks
# ===================================================================


def _child_follows(out_queue):
    from psycopg.yb.registry import ClusterRegistry

    try:
        # The registered at-fork hooks already ran (_before_fork in the
        # parent, _after_fork_in_child here).
        group = list(ClusterRegistry._instance._failover_groups.values())[0]
        followed = _wait(lambda: group.primary_status == HealthResult.UNHEALTHY, 5.0)
        out_queue.put(("ok", {
            "followed": followed,
            "probes": (group.primary_probe, group.secondary_probe) != (None, None),
            "follower": group._follower_thread is not None,
        }))
    except Exception as exc:
        out_queue.put(("exc", repr(exc)))


@pytest.mark.skipif(not hasattr(os, "fork"), reason="fork not available")
def test_real_fork_child_follows_parent_flip(fake_state, fresh_registry):
    import multiprocessing as mp

    from psycopg.yb.registry import ClusterRegistry, _register_fork_hook_once

    _register_fork_hook_once()
    group = _make_group(fake_state, "")
    ClusterRegistry.instance()._failover_groups[group.primary.uuid] = group
    group._probe_interval_s = 60
    group._probe_check_timeout_s = 1.0
    group._probe_drain_timeout_s = 10
    fork_follow.ensure_shared_state(group)

    ctx = mp.get_context("fork")
    out_queue = ctx.Queue()
    child = ctx.Process(target=_child_follows, args=(out_queue,))
    child.start()
    try:
        assert _wait(lambda: group._follow.live_children() == 1, 5.0)
        trigger_drain(group, "primary", HealthResult.UNHEALTHY, drain_timeout_s=1)
        status, payload = out_queue.get(timeout=10)
    finally:
        child.join(timeout=10)
        if child.is_alive():
            child.kill()

    assert status == "ok", payload
    assert payload == {"followed": True, "probes": False, "follower": True}
