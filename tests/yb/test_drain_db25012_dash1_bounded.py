"""
Integration-level repro tests for DB-25012 (PR-H7).

Shubham reported (#proj-smartdriver-xclusterfailover, Oct 23 2026):
``drainTimeoutSecs=-1`` the switch never completes while the application
keeps running — 3 of 3 drains that started with open transactions never
finished. Cycle 5 ran 433 s and committed 34,323 transactions on the
universe being left before the harness stopped the workload.

Root cause: the drain loop (``drain.py: _drain_and_apply``) exited only
when the in-flight-txn count hit zero. Persistent sockets loop
txn → IDLE → txn forever on the outgoing side; the barrier only holds
new ``connect()`` calls. Under steady load the "zero in-flight" instant
never arrives, so with ``-1`` the loop runs indefinitely.

Fix (PR-H7): close every IDLE tracked conn each poll of the drain loop.
Current in-flight txns are left alone (they complete at their own pace);
when a worker's current txn commits and the conn returns to IDLE, the
NEXT drain tick closes it. The worker's next cursor.execute raises
``OperationalError``, the retry opens a new conn on the active cluster.
Finite drain time (bounded by the longest in-flight txn), no new txns
on the outgoing side, no force-close ever needed for ``-1``.
"""

# Copyright (C) 2026 Yugabyte

from __future__ import annotations

import threading
import time
import types

import pytest

from psycopg.pq import TransactionStatus
from psycopg.yb.drain import (
    _close_idle_tracked_during_drain,
    trigger_drain,
)
from psycopg.yb.health import HealthResult
from psycopg.yb.registry import FailoverGroup


pytestmark = pytest.mark.yb_unit


# ===================================================================
# Minimal shared fakes — same shape as test_drain_qa_dual_write_fix.
# ===================================================================


class _FakeInfo:
    def __init__(self, status: TransactionStatus) -> None:
        self.transaction_status = status


class _FakeDirectConn:
    """IDLE at construction. ``begin()`` sets INTRANS, ``commit()`` sets
    IDLE, ``close()`` sets ``closed=True`` and status to UNKNOWN."""

    def __init__(self, cluster_uuid: str) -> None:
        self._yb_uuid = cluster_uuid
        self.info = _FakeInfo(TransactionStatus.IDLE)
        self.closed = False
        self.pgconn = types.SimpleNamespace(socket=-1)

    def begin(self) -> None:
        if self.closed:
            raise ConnectionError("closed")
        self.info.transaction_status = TransactionStatus.INTRANS

    def commit(self) -> None:
        if self.closed:
            raise ConnectionError("closed")
        self.info.transaction_status = TransactionStatus.IDLE

    def close(self) -> None:
        self.closed = True
        self.info.transaction_status = TransactionStatus.UNKNOWN


def _make_group(fake_state) -> FailoverGroup:
    p = fake_state(("p1", "aws", "us-west", "us-west-1a", "primary"), uuid="PRIMARY-A")
    s = fake_state(("s1", "aws", "us-east", "us-east-1a", "primary"), uuid="SECONDARY-B")
    return FailoverGroup(
        primary=p, secondary=s, lock=threading.Lock(),
        primary_status=HealthResult.HEALTHY,
        secondary_status=HealthResult.HEALTHY,
        cooldown_s=0,
    )


# ===================================================================
# Unit tests: _close_idle_tracked_during_drain
# ===================================================================


def test_idle_conn_is_closed_by_helper(fake_state):
    """A tracked conn in IDLE state must be closed by the helper. This
    is the single-conn equivalent of DB-25012: an IDLE socket on the
    outgoing universe must not survive the drain, because the app's
    next cursor.execute would start a fresh txn on that socket."""
    group = _make_group(fake_state)
    conn = _FakeDirectConn(group.primary.uuid)
    group.primary.tracked_conns.add(conn)

    closed = _close_idle_tracked_during_drain(
        group.primary.tracked_conns, group.primary.uuid,
    )

    assert closed == 1
    assert conn.closed is True


def test_in_flight_conn_is_NOT_closed_by_helper(fake_state):
    """A conn currently inside a transaction must be LEFT ALONE. The
    whole point of ``drainTimeoutSecs=-1`` is 'let in-flight txns
    finish without a deadline'. Closing a conn mid-txn would roll back
    an active commit and corrupt the application state."""
    group = _make_group(fake_state)
    conn = _FakeDirectConn(group.primary.uuid)
    conn.begin()                            # now INTRANS
    group.primary.tracked_conns.add(conn)

    closed = _close_idle_tracked_during_drain(
        group.primary.tracked_conns, group.primary.uuid,
    )

    assert closed == 0
    assert conn.closed is False
    assert conn.info.transaction_status == TransactionStatus.INTRANS


def test_already_closed_conn_is_skipped(fake_state):
    """A conn already closed from a prior tick must not be double-closed
    — the helper's return value must count only CONNS IT CLOSED this
    tick so the drain loop's log line is accurate."""
    group = _make_group(fake_state)
    conn = _FakeDirectConn(group.primary.uuid)
    conn.close()
    group.primary.tracked_conns.add(conn)

    closed = _close_idle_tracked_during_drain(
        group.primary.tracked_conns, group.primary.uuid,
    )

    assert closed == 0


def test_helper_handles_mixed_set(fake_state):
    """Mix of IDLE and INTRANS conns: only the IDLE ones get closed."""
    group = _make_group(fake_state)
    idle1 = _FakeDirectConn(group.primary.uuid)
    idle2 = _FakeDirectConn(group.primary.uuid)
    busy = _FakeDirectConn(group.primary.uuid)
    busy.begin()
    for c in (idle1, idle2, busy):
        group.primary.tracked_conns.add(c)

    closed = _close_idle_tracked_during_drain(
        group.primary.tracked_conns, group.primary.uuid,
    )

    assert closed == 2
    assert idle1.closed is True
    assert idle2.closed is True
    assert busy.closed is False


# ===================================================================
# End-to-end: drainTimeoutSecs=-1 under steady load must EXIT
# ===================================================================


def _steady_load_worker(
    conn_holder: list,
    outgoing_uuid: str,
    stop: threading.Event,
    commits_counter: list,
) -> None:
    """One persistent worker, the exact pattern Shubham measured:

      BEGIN → work → COMMIT → brief IDLE → BEGIN → ...

    The IDLE window is intentionally short (10 ms) so the drain loop
    almost always catches the worker IDLE on its first tick post-txn.
    The worker reconnects on close — mirrors the app-level
    "catch OperationalError, retry" pattern.
    """
    conn = conn_holder[0]
    while not stop.is_set():
        try:
            conn.begin()
            # Simulate the in-txn work window.
            if stop.wait(timeout=0.01):
                break
            conn.commit()
            commits_counter.append(1)
            # Short IDLE window between commits. The drain loop ticks
            # every 100 ms, so a 10 ms IDLE window is plenty of chance
            # to be caught IDLE.
            if stop.wait(timeout=0.01):
                break
        except ConnectionError:
            # Drain closed us — this is the H7 behavior. The real app
            # would open a new conn on the active cluster here. In this
            # unit test we just stop the worker — reconnection is
            # covered by the broader H1 harness.
            return


def test_dash1_drain_exits_under_steady_load_db25012(fake_state):
    """DB-25012 cycle 5 reduced: 3 persistent workers running a tight
    BEGIN/COMMIT loop with a 10 ms IDLE window. Trigger drain with
    ``drainTimeoutSecs=-1`` mid-workload.

    PRE-FIX: the drain loop would wait for the "zero in-flight" instant
    that never arrives under this load (the pre-fix test harness ran
    7 minutes before the harness killed it). POST-FIX: each tick closes
    IDLE conns; worker's next BEGIN raises; a few ticks later every
    worker's current txn has committed and been IDLE-closed → in-flight
    count hits zero → drain exits.

    Deadline for THIS test: 5 s. If the fix is reverted the drain
    runs forever and the test timer kills the process (visible as a
    hang in CI).
    """
    group = _make_group(fake_state)
    stop = threading.Event()
    commits_counter: list = []

    # Open 15 workers, each with its own tracked conn on PRIMARY. 15 is
    # the smallest count that makes "all workers happen to be IDLE at
    # the same drain-poll instant" vanishingly improbable, so the
    # pre-fix loop runs effectively forever. At 3 workers with a 50%
    # duty cycle, the probability of a simultaneous-IDLE instant is
    # ~12 % per tick — the pre-fix loop often exits by luck and the
    # test doesn't catch a wiring regression. Shubham's DB-25012
    # cycle 5 had 29 writers.
    workers = []
    conns = []
    for _ in range(15):
        conn = _FakeDirectConn(group.primary.uuid)
        group.primary.tracked_conns.add(conn)
        conns.append(conn)
        t = threading.Thread(
            target=_steady_load_worker,
            args=([conn], group.primary.uuid, stop, commits_counter),
            daemon=True,
        )
        t.start()
        workers.append(t)

    # Let the workers run so a few commits land before the drain.
    time.sleep(0.2)
    pre_drain_commits = len(commits_counter)

    # agent-K PR #19:260 — run trigger_drain in a background thread
    # with an explicit join timeout so the test fails fast if the
    # H7 fix is reverted. The pre-fix drain never returns under this
    # load (DB-25012 cycle 5: 433 s hang), and we don't want CI to
    # block on the whole suite's hard timeout — fail THIS test with
    # a clear "drain hung" assertion instead.
    t0 = time.monotonic()
    drain_done = threading.Event()

    def _run_drain():
        try:
            trigger_drain(
                group=group,
                which_cluster="primary",
                new_status=HealthResult.UNHEALTHY,
                drain_timeout_s=-1,
            )
        finally:
            drain_done.set()

    drain_thread = threading.Thread(
        target=_run_drain, name="drain-under-test", daemon=True,
    )
    drain_thread.start()
    drain_done.wait(timeout=10.0)
    drain_elapsed = time.monotonic() - t0

    assert drain_done.is_set(), (
        f"drain did not return within 10 s — DB-25012 regression. "
        f"The pre-fix drainTimeoutSecs=-1 loop blocks indefinitely "
        f"under steady persistent-sock load; this test asserts the "
        f"H7 fix (_close_idle_tracked_during_drain) is wired in."
    )

    stop.set()
    for t in workers:
        t.join(timeout=1.0)

    # Assertion 1: the drain actually finished (did not hang).
    # 5 s is generous — the real work is bounded by the ~10 ms
    # in-txn window, so a healthy fix completes in well under 1 s.
    assert drain_elapsed < 5.0, (
        f"drain with drainTimeoutSecs=-1 took {drain_elapsed:.1f} s under "
        f"steady load — DB-25012 says the pre-fix loop runs indefinitely"
    )

    # Assertion 2: every tracked conn on the outgoing side is closed —
    # no persistent socket survives into the post-flip world where it
    # could start a fresh txn on the old universe.
    not_closed = [c for c in conns if not c.closed]
    assert not not_closed, (
        f"{len(not_closed)}/{len(conns)} outgoing-cluster conns survived "
        f"the drain — they would commit the next txn on the old universe"
    )

    # Sanity: the workers were actually running (pre-drain commits
    # happened). Without this, a bug that nukes the workers at startup
    # would make the "no in-flight" check trivially true.
    assert pre_drain_commits > 0, (
        "workers did not commit anything before the drain — the test "
        "can't distinguish 'fix works' from 'workload never ran'"
    )


def test_dash1_drain_waits_for_in_flight_txn_to_finish(fake_state):
    """The ``-1`` promise is "let in-flight txns finish". A conn that
    is INTRANS at the barrier must NOT be closed mid-txn — its txn
    must be allowed to commit, THEN the next tick's IDLE close catches
    it. Verifies the "in-flight protected" invariant.
    """
    group = _make_group(fake_state)
    busy = _FakeDirectConn(group.primary.uuid)
    busy.begin()                            # INTRANS at barrier
    group.primary.tracked_conns.add(busy)

    # Thread: let the "txn" run for 150 ms (longer than one drain poll
    # tick of 100 ms), then commit. The drain's IDLE-close catches it
    # on the NEXT tick after commit.
    txn_done = threading.Event()

    def finish_txn_shortly():
        time.sleep(0.15)
        busy.commit()
        txn_done.set()

    t = threading.Thread(target=finish_txn_shortly, daemon=True)
    t.start()

    t0 = time.monotonic()
    trigger_drain(
        group=group,
        which_cluster="primary",
        new_status=HealthResult.UNHEALTHY,
        drain_timeout_s=-1,
    )
    drain_elapsed = time.monotonic() - t0

    t.join(timeout=1.0)

    # The drain must have waited for the txn (so it took ≥ ~150 ms)
    # but not forever (completed in under 2 s). And the conn must be
    # closed by the end.
    assert drain_elapsed >= 0.15, (
        f"drain finished in {drain_elapsed * 1000:.0f} ms — faster than "
        f"the in-flight txn (150 ms). The conn was closed mid-txn, "
        f"violating the ``-1`` 'let in-flight finish' promise"
    )
    assert drain_elapsed < 2.0, (
        f"drain took {drain_elapsed:.1f} s — far longer than the txn + "
        f"a few 100 ms ticks. Possible infinite loop regression"
    )
    assert txn_done.is_set(), "the in-flight txn never committed"
    assert busy.closed is True, (
        "the conn was NOT closed by the drain's IDLE-close pass. "
        "After the txn committed, the next tick should have caught it IDLE"
    )
