"""
Integration-level repro tests for the drain correctness fixes in PR-H1:

  * DB-24801 — persistent direct connections on the outgoing cluster
    kept committing for minutes after the routing flip because the
    drain spared IDLE conns. The fix closes every tracked conn (idle
    or in-flight) at the flip boundary.
  * DB-24960 — ``AsyncConnection.close`` is a coroutine; calling it
    from the sync drain executor returned an un-awaited coroutine and
    never actually closed the socket. The fix type-dispatches
    ``_safe_close`` to call ``pgconn.finish()`` for async conns.

These tests mirror the scenarios Shubham reported on
#proj-smartdriver-xclusterfailover (Oct 19 and Oct 22 2026) at
integration scope — multiple persistent workers running a BEGIN /
INSERT / COMMIT loop while a drain fires mid-workload. We assert the
exact observable Shubham measured: zero commits on the outgoing
cluster after the routing flip timestamp, and no
``RuntimeWarning: coroutine 'AsyncConnection.close' was never awaited``
in stderr.
"""

# Copyright (C) 2026 Yugabyte

from __future__ import annotations

import threading
import time
import types
import warnings
from dataclasses import dataclass, field
from typing import Any

import pytest

from psycopg.pq import TransactionStatus
from psycopg.yb.drain import _safe_close, trigger_drain
from psycopg.yb.health import HealthResult
from psycopg.yb.registry import FailoverGroup


pytestmark = pytest.mark.yb_unit


# ===================================================================
# Shared harness — mirrors the "worker commits to a cluster" pattern
# Shubham's test uses, but with FakeConn in place of a real pgconn.
# ===================================================================


@dataclass
class LedgerEntry:
    """One row in the per-worker commit ledger. `cluster_uuid` is the
    uuid of the cluster the worker's conn was bound to at the moment
    of commit — matches Shubham's ledger.jsonl `side=A|B` column."""
    worker_id: int
    monotonic: float
    cluster_uuid: str
    kind: str = "insert"


@dataclass
class Ledger:
    rows: list[LedgerEntry] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def record(self, worker_id: int, cluster_uuid: str, kind: str = "insert") -> None:
        with self.lock:
            self.rows.append(LedgerEntry(
                worker_id=worker_id,
                monotonic=time.monotonic(),
                cluster_uuid=cluster_uuid,
                kind=kind,
            ))

    def commits_after(self, boundary_mono: float, cluster_uuid: str) -> list[LedgerEntry]:
        with self.lock:
            return [
                r for r in self.rows
                if r.monotonic >= boundary_mono and r.cluster_uuid == cluster_uuid
            ]

    def by_worker(self, worker_id: int) -> list[LedgerEntry]:
        with self.lock:
            return [r for r in self.rows if r.worker_id == worker_id]


class _FakeInfo:
    def __init__(self, status: TransactionStatus) -> None:
        self.transaction_status = status


class _FakeDirectConn:
    """Minimal shape for the drain tracker: ``info.transaction_status``,
    ``close()``, ``closed``. Also carries ``_yb_uuid`` so the worker
    can read 'which cluster am I bound to' before every commit (the
    way a real conn implicitly commits to whichever host its TCP
    session points at)."""

    def __init__(self, cluster_uuid: str) -> None:
        self._yb_uuid = cluster_uuid
        self.info = _FakeInfo(TransactionStatus.IDLE)
        self.closed = False
        self.pgconn = types.SimpleNamespace(socket=-1)

    def begin(self) -> None:
        if self.closed:
            raise ConnectionError("conn is closed")
        self.info.transaction_status = TransactionStatus.INTRANS

    def commit(self) -> None:
        if self.closed:
            raise ConnectionError("conn is closed")
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


class _WorkerPool:
    """Spins up N background threads, each holding ONE persistent
    ``_FakeDirectConn`` bound to the active cluster at open-time.
    Each worker loops: BEGIN → sleep(write_interval) → COMMIT →
    record ledger entry → sleep(idle_interval). On any conn error
    (the drain just closed it), the worker reconnects to WHATEVER the
    active cluster is NOW — mirrors the real-world "catch
    OperationalError, retry" pattern.

    The idle_interval is critical: it is the time between COMMIT and
    the next BEGIN, which is the window Shubham observed as IDLE at
    the drain barrier. Making it longer than drain_timeout ensures a
    high probability that at least one worker is IDLE at the barrier.
    """

    def __init__(
        self,
        group: FailoverGroup,
        ledger: Ledger,
        n: int,
        active_uuid_fn,
        write_interval_s: float = 0.01,
        idle_interval_s: float = 0.3,
    ) -> None:
        self.group = group
        self.ledger = ledger
        self.n = n
        self.active_uuid_fn = active_uuid_fn
        self.write_interval_s = write_interval_s
        self.idle_interval_s = idle_interval_s
        self.stop = threading.Event()
        self.threads: list[threading.Thread] = []
        self.conns: list[_FakeDirectConn] = []
        self._lock = threading.Lock()

    def _open_conn(self) -> _FakeDirectConn:
        """Mirrors the real-world ``psycopg.connect`` → dispatcher
        picks the active cluster → conn is bound to that cluster's
        uuid. We register with the tracked_conns weakref set on the
        corresponding ClusterState just like the real registry does."""
        active_uuid = self.active_uuid_fn()
        state = (
            self.group.primary if active_uuid == self.group.primary.uuid
            else self.group.secondary
        )
        conn = _FakeDirectConn(active_uuid)
        state.tracked_conns.add(conn)
        return conn

    def _worker_loop(self, worker_id: int) -> None:
        conn = self._open_conn()
        with self._lock:
            self.conns.append(conn)
        try:
            while not self.stop.is_set():
                try:
                    conn.begin()
                    # Simulate the small window inside a transaction.
                    time.sleep(self.write_interval_s)
                    conn.commit()
                    self.ledger.record(worker_id, conn._yb_uuid)
                except ConnectionError:
                    # Drain closed us — reconnect to whatever is active now.
                    # Mirrors app-level "catch OperationalError, reopen" pattern.
                    conn = self._open_conn()
                    with self._lock:
                        self.conns.append(conn)
                    continue
                # IDLE window between commits — this is the state the
                # drain barrier used to spare under the old code.
                if self.stop.wait(timeout=self.idle_interval_s):
                    break
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def start(self) -> None:
        for i in range(self.n):
            t = threading.Thread(
                target=self._worker_loop, args=(i,),
                name=f"direct-worker-{i}", daemon=True,
            )
            t.start()
            self.threads.append(t)
        # Give workers time to open their first conn and enter the
        # first IDLE window so the drain fires against a realistic
        # mixed-state tracked set.
        time.sleep(0.1)

    def shutdown(self, timeout: float = 2.0) -> None:
        self.stop.set()
        for t in self.threads:
            t.join(timeout=timeout)


# ===================================================================
# DB-24801 — idle direct conns must not commit on outgoing after flip
# ===================================================================

def test_db24801_idle_direct_conns_closed_at_flip_no_dual_write(fake_state):
    """Shubham's cycle 1 reduced: 6 persistent direct workers running
    BEGIN/INSERT/COMMIT loops with a 0.3 s idle window between commits.
    Trigger drain mid-workload. Assert ZERO commits on PRIMARY-A after
    the routing flip timestamp, matching the design doc §3.3
    "no dual-write within a single application" promise.
    """
    group = _make_group(fake_state)
    ledger = Ledger()

    # The "active cluster" the dispatcher would hand to connect(). We
    # don't drive the real dispatcher here — we flip it ourselves via
    # trigger_drain's status change.
    def active_uuid_fn() -> str:
        return (
            group.primary.uuid if group.primary_status == HealthResult.HEALTHY
            else group.secondary.uuid
        )

    pool = _WorkerPool(
        group=group, ledger=ledger, n=6,
        active_uuid_fn=active_uuid_fn,
        write_interval_s=0.01, idle_interval_s=0.3,
    )
    pool.start()
    # Let the workload tick for a bit so the ledger has pre-flip rows.
    time.sleep(0.5)

    # Trigger drain → flip to secondary. drainTimeoutSecs=1 forces a
    # realistic deadline; most workers will be IDLE at the barrier
    # (idle_interval 0.3 s dominates write_interval 0.01 s).
    boundary = time.monotonic()
    trigger_drain(
        group=group,
        which_cluster="primary",
        new_status=HealthResult.UNHEALTHY,
        drain_timeout_s=1,
    )
    flip_time = time.monotonic()

    # Workers that got closed should have reconnected to the new
    # active cluster by now. Give them a bit of time to catch the
    # error and reopen, then check the post-flip ledger.
    time.sleep(0.5)
    pool.shutdown()

    # Assertion 1: every tracked conn on the outgoing cluster was
    # closed by the drain (DB-24801 — the IDLE ones used to be spared).
    outgoing_conns = [c for c in pool.conns if c._yb_uuid == group.primary.uuid]
    not_closed = [c for c in outgoing_conns if not c.closed]
    assert not not_closed, (
        f"{len(not_closed)}/{len(outgoing_conns)} outgoing-cluster "
        f"conns were left open after the drain — this is DB-24801. "
        f"The idle-sparing path leaked them to the next worker loop."
    )

    # Assertion 2: ZERO commits on the outgoing cluster after the flip.
    # This is the exact invariant Shubham's bank test checks.
    leaked = ledger.commits_after(flip_time, group.primary.uuid)
    assert not leaked, (
        f"{len(leaked)} commits landed on PRIMARY-A after the flip at "
        f"mono={flip_time:.3f}. Sample: "
        f"{[(r.worker_id, r.monotonic - flip_time) for r in leaked[:5]]}. "
        f"Design doc §3.3 promises no dual-write within a single app; "
        f"this is DB-24801."
    )

    # Sanity: the workload actually ran — if the test fakes anything
    # into no-op mode, we'd miss the real regression.
    assert len(ledger.rows) > 10, (
        f"workload only produced {len(ledger.rows)} commits — too few "
        f"to trust the no-dual-write assertion"
    )

    # Sanity: workers DID reconnect to the new cluster post-flip, so
    # we know the "close → reconnect" path works end-to-end.
    reconnected = [
        r for r in ledger.rows
        if r.monotonic > flip_time and r.cluster_uuid == group.secondary.uuid
    ]
    assert reconnected, (
        "no post-flip commits on SECONDARY-B — workers never "
        "reconnected, so the test can't distinguish 'leak fixed' from "
        "'workload stalled'"
    )


def test_db24801_pure_idle_at_barrier_still_closed(fake_state):
    """Edge case: every worker is 100% IDLE at the barrier (no in-flight
    transactions). Under the old code the drain's `in_txn` filter
    returned empty, the early-exit path fired, and survivors stayed
    at len=0 — ZERO conns were closed. All idle direct conns silently
    continued to the outgoing cluster.
    """
    group = _make_group(fake_state)
    # Six idle direct conns on primary — none in a transaction.
    conns = []
    for _ in range(6):
        c = _FakeDirectConn(group.primary.uuid)
        group.primary.tracked_conns.add(c)
        conns.append(c)

    trigger_drain(
        group=group,
        which_cluster="primary",
        new_status=HealthResult.UNHEALTHY,
        drain_timeout_s=1,
    )

    # DB-24801: every idle tracked conn must be closed at the flip
    # boundary. Old code would have left all 6 open.
    not_closed = [c for c in conns if not c.closed]
    assert not not_closed, (
        f"{len(not_closed)}/6 pure-idle direct conns were left open "
        f"— the drain's old early-exit-when-nothing-in-flight path "
        f"skipped the close step entirely (DB-24801)."
    )


def test_db24801_pool_conns_also_closed_defensively(fake_state):
    """The pool's ``xcluster_check`` evicts on next ``getconn``, so
    pool conns were safe under the old code. The fix closes them too
    at the drain boundary — redundant but harmless, and gives a
    uniform guarantee that no conn on the outgoing uuid is alive
    post-flip (ops don't have to reason about pool vs direct).
    """
    group = _make_group(fake_state)
    # Three pool-like conns (same shape as direct for our tracker).
    pool_conns = []
    for _ in range(3):
        c = _FakeDirectConn(group.primary.uuid)
        group.primary.tracked_conns.add(c)
        pool_conns.append(c)

    trigger_drain(
        group=group,
        which_cluster="primary",
        new_status=HealthResult.UNHEALTHY,
        drain_timeout_s=1,
    )

    not_closed = [c for c in pool_conns if not c.closed]
    assert not not_closed


# ===================================================================
# DB-24960 — AsyncConnection.close is a coroutine; the sync drain
# executor used to call c.close() without awaiting it, which returned
# an un-awaited coroutine and never actually closed the socket.
# ===================================================================


class _AsyncConnLike:
    """A stand-in for ``AsyncConnection`` that exposes the exact
    bug surface: ``close`` is an async function (so calling it from
    sync code returns a coroutine object), plus ``pgconn.finish()``
    and ``_closed`` the fix needs. The ``_safe_close`` type-dispatch
    uses ``isinstance(c, AsyncConnection)`` — we subclass
    ``AsyncConnection`` by bypassing ``__init__`` so isinstance
    returns True without needing a real libpq handle.

    ``info`` on the real class is a property backed by ``pgconn``,
    so we populate ``pgconn.transaction_status`` and let the real
    property wrap it. ``pgconn.status`` is set so ``__del__`` at GC
    time sees ``closed == True`` and skips its cleanup.
    """

    @classmethod
    def build(cls, tx_status: int = TransactionStatus.INTRANS):
        from psycopg.connection_async import AsyncConnection
        from psycopg.pq import ConnStatus
        c = object.__new__(AsyncConnection)
        c._closed = False
        c._yb_uuid = None
        c._yb_host = None
        # pgconn mock: records finish() calls, exposes the two fields
        # the drain reads (socket, transaction_status) and the one
        # __del__ reads (status).
        finish_calls = {"n": 0}

        def finish():
            finish_calls["n"] += 1
            c.pgconn.status = ConnStatus.BAD   # mark closed for __del__
        c.pgconn = types.SimpleNamespace(
            socket=-1,
            finish=finish,
            transaction_status=int(tx_status),
            status=ConnStatus.OK,
        )
        c._finish_calls = finish_calls
        return c


def test_db24960_safe_close_actually_finishes_async_conn():
    """Direct unit of the fix: ``_safe_close`` on an AsyncConnection
    must call ``pgconn.finish()`` and set ``_closed = True`` — not
    produce an un-awaited coroutine warning."""
    conn = _AsyncConnLike.build()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        exc = _safe_close(conn)

    assert exc is None, f"_safe_close raised: {exc!r}"
    assert conn._closed is True, "AsyncConnection _closed flag not set"
    assert conn._finish_calls["n"] == 1, (
        f"pgconn.finish() not called (count={conn._finish_calls['n']}); "
        f"the socket is still open — this is the DB-24960 symptom"
    )
    coroutine_warnings = [
        w for w in caught
        if issubclass(w.category, RuntimeWarning)
        and "coroutine" in str(w.message)
    ]
    assert not coroutine_warnings, (
        f"RuntimeWarning for un-awaited coroutine fired: "
        f"{[str(w.message) for w in coroutine_warnings]}. "
        f"This is the DB-24960 symptom — _safe_close called c.close() "
        f"instead of type-dispatching to pgconn.finish()."
    )


def test_db24960_full_drain_with_async_in_flight_closes_cleanly(fake_state):
    """End-to-end: an AsyncConnection with an in-flight transaction is
    tracked on the outgoing cluster at drain time. The drain must
    actually close it (not just log 'force-closed' and leave the
    socket open), and must not emit the ``AsyncConnection.close was
    never awaited`` warning that Shubham observed in 12 of 22 cycles."""
    group = _make_group(fake_state)
    # In-flight at the barrier — the exact scenario where the old
    # code's no-op close let the transaction commit on the old
    # universe after the flip.
    async_conn = _AsyncConnLike.build(tx_status=TransactionStatus.INTRANS)
    group.primary.tracked_conns.add(async_conn)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        trigger_drain(
            group=group,
            which_cluster="primary",
            new_status=HealthResult.UNHEALTHY,
            drain_timeout_s=0,   # immediate kill — exercises the
                                 # close path without a drain wait
        )

    # Must actually be closed, not just claimed.
    assert async_conn._closed is True, (
        "AsyncConnection was not actually closed by the drain — the "
        "sync executor's c.close() returned an un-awaited coroutine "
        "and never ran. This is DB-24960."
    )
    assert async_conn._finish_calls["n"] == 1, (
        "pgconn.finish() was not called — the socket is still open "
        "(DB-24960)"
    )
    coroutine_warnings = [
        w for w in caught
        if issubclass(w.category, RuntimeWarning)
        and "coroutine" in str(w.message).lower()
        and "close" in str(w.message).lower()
    ]
    assert not coroutine_warnings, (
        f"The 'AsyncConnection.close was never awaited' warning fired "
        f"— DB-24960 regression. Messages: "
        f"{[str(w.message) for w in coroutine_warnings]}"
    )


def test_db24960_mixed_sync_and_async_both_closed(fake_state):
    """Mixed workload — a sync direct conn + an async conn, both
    in-flight. Both must end up closed through the correct path:
    sync via c.close(), async via pgconn.finish()."""
    group = _make_group(fake_state)
    sync_conn = _FakeDirectConn(group.primary.uuid)
    sync_conn.info.transaction_status = TransactionStatus.INTRANS
    async_conn = _AsyncConnLike.build(tx_status=TransactionStatus.INTRANS)
    group.primary.tracked_conns.add(sync_conn)
    group.primary.tracked_conns.add(async_conn)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        trigger_drain(
            group=group,
            which_cluster="primary",
            new_status=HealthResult.UNHEALTHY,
            drain_timeout_s=0,
        )

    assert sync_conn.closed is True
    assert async_conn._closed is True
    assert async_conn._finish_calls["n"] == 1
    coroutine_warnings = [
        w for w in caught
        if issubclass(w.category, RuntimeWarning)
        and "coroutine" in str(w.message).lower()
    ]
    assert not coroutine_warnings
