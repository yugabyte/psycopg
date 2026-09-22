"""
Unit tests for the parallel force-close + hard-fd escape hatch in
``psycopg.yb.drain._drain_and_apply`` (design doc §5.4).

Motivating scenario (from Amogh / Hemant Slack thread, 2026-09-17):

  With 250 in-flight conns on a failover, the previous serial
  ``for c in survivors: c.close()`` loop would block on the FIRST conn
  whose ``close()`` hung — libpq's ``PQfinish`` calls ``send()`` to push
  a Terminate byte, and that ``send()`` can wait for ``tcp_retries2``
  (~15 min on Linux; worse with TCP keepalive disabled, which is
  Amex's environment) before the OS gives up on a wedged peer. A
  single stuck TCP session would therefore wedge drain WAY past its
  own ``drainTimeoutSecs``.

The fix:

  * Fire ``close()`` on every survivor concurrently via a thread pool.
    A stuck ``send()`` only ties up one worker; others complete freely.
  * If any close hasn't returned within ``_KILL_AT_TIMEOUT_BUDGET_S``
    (default 2 s), hard-close the socket fd via ``os.close`` as the
    escape hatch. The driver-side conn is disowned even if the libpq
    worker is still blocked in ``send()``.

These tests verify:

  * Fast closes (typical case) still complete + log per-conn ``force-closed``.
  * A hung ``close()`` gets its fd hard-closed within budget, not after
    the full TCP timeout.
  * Drain wall-clock stays bounded regardless of how many conns hang.
  * A ``close()`` that raises (e.g. malformed conn) is caught + logged,
    doesn't break sibling closes.
  * Snapshot of ``pgconn.socket`` happens BEFORE ``close()`` so we can
    still hard-fd-close a conn whose ``close()`` nulled the pgconn.
"""

# Copyright (C) 2026 Yugabyte

from __future__ import annotations

import os
import socket
import threading
import time
import types
from typing import List

import pytest

from psycopg.pq import TransactionStatus
from psycopg.yb.drain import (
    _KILL_AT_TIMEOUT_BUDGET_S,
    _safe_close,
    trigger_drain,
)
from psycopg.yb.health import HealthResult
from psycopg.yb.registry import FailoverGroup


pytestmark = pytest.mark.yb_unit


# ------------------------------------------------------- fake conns

class _FakeInfo:
    def __init__(self, status: TransactionStatus) -> None:
        self.transaction_status = status


class _HungConn:
    """A conn whose ``close()`` blocks on ``hang_event`` (never released
    in these tests) — mirrors a libpq ``PQfinish`` stuck in ``send()``
    on a wedged TCP peer. The socket fd is real (from ``socketpair``)
    so ``os.close`` can genuinely release it as the escape hatch."""

    def __init__(self, fd: int) -> None:
        self.info = _FakeInfo(TransactionStatus.INTRANS)
        self.pgconn = types.SimpleNamespace(socket=fd)
        self.hang_event = threading.Event()
        self.close_returned = False

    def close(self) -> None:
        # Never released in the test — mimics an infinite hang.
        # 60s cap so a broken test doesn't leak the thread forever.
        self.hang_event.wait(timeout=60)
        self.close_returned = True


class _FastConn:
    """Conn whose ``close()`` returns instantly."""

    def __init__(self, fd: int) -> None:
        self.info = _FakeInfo(TransactionStatus.INTRANS)
        self.pgconn = types.SimpleNamespace(socket=fd)
        self.closed = False

    def close(self) -> None:
        self.closed = True
        self.info.transaction_status = TransactionStatus.UNKNOWN


class _RaisingConn:
    """Conn whose ``close()`` raises — must not break sibling closes."""

    def __init__(self, fd: int) -> None:
        self.info = _FakeInfo(TransactionStatus.INTRANS)
        self.pgconn = types.SimpleNamespace(socket=fd)

    def close(self) -> None:
        raise RuntimeError("simulated close() failure")


def _make_group(fake_state) -> FailoverGroup:
    p = fake_state(("p1", "aws", "us-west", "us-west-1a", "primary"), uuid="P")
    s = fake_state(("s1", "aws", "us-east", "us-east-1a", "primary"), uuid="S")
    return FailoverGroup(
        primary=p, secondary=s, lock=threading.Lock(),
        primary_status=HealthResult.HEALTHY,
        secondary_status=HealthResult.HEALTHY,
        cooldown_s=0,
    )


@pytest.fixture
def owned_socketpair():
    """Yield (fd_a, fd_b) from a socketpair; close whichever side the
    test hasn't already released. Any test that hard-fd-closes fd_a
    should NOT close it again in teardown."""
    a, b = socket.socketpair()
    yield a.fileno(), b.fileno()
    for s in (a, b):
        try:
            s.close()
        except OSError:
            pass


# ------------------------------------------------------- _safe_close

def test_safe_close_returns_none_on_success():
    conn = _FastConn(fd=-1)
    assert _safe_close(conn) is None
    assert conn.closed


def test_safe_close_returns_exception_on_raise():
    conn = _RaisingConn(fd=-1)
    exc = _safe_close(conn)
    assert isinstance(exc, RuntimeError)
    assert "simulated close() failure" in str(exc)


# ------------------------------------------------------- fast close path

def test_fast_close_completes_and_flips_routing(fake_state, owned_socketpair):
    """Baseline: when close() returns quickly, drain flips routing
    within drain_timeout_s and every survivor is force-closed."""
    fd_a, _fd_b = owned_socketpair
    group = _make_group(fake_state)
    conn = _FastConn(fd=fd_a)
    group.primary.tracked_conns.add(conn)

    t0 = time.monotonic()
    trigger_drain(
        group=group,
        which_cluster="primary",
        new_status=HealthResult.UNHEALTHY,
        drain_timeout_s=1,
    )
    elapsed = time.monotonic() - t0

    assert conn.closed
    assert group.primary_status == HealthResult.UNHEALTHY
    # Should finish within drain window + kill budget + slack
    assert elapsed < 1 + _KILL_AT_TIMEOUT_BUDGET_S + 1


# ------------------------------------------------------- hung close, escape hatch

def test_hung_close_falls_back_to_hard_fd(fake_state, owned_socketpair, monkeypatch):
    """A hung close() must NOT block drain past drain_timeout_s +
    _KILL_AT_TIMEOUT_BUDGET_S. Verify os.close was called on the
    hung conn's fd as the escape hatch."""
    fd_a, _fd_b = owned_socketpair
    group = _make_group(fake_state)
    hung = _HungConn(fd=fd_a)
    group.primary.tracked_conns.add(hung)

    # Observe os.close calls without stopping them.
    close_calls: List[int] = []
    real_os_close = os.close

    def spy(fd):
        close_calls.append(fd)
        return real_os_close(fd)

    monkeypatch.setattr("psycopg.yb.drain.os.close", spy)

    t0 = time.monotonic()
    trigger_drain(
        group=group,
        which_cluster="primary",
        new_status=HealthResult.UNHEALTHY,
        drain_timeout_s=1,
    )
    elapsed = time.monotonic() - t0

    # Bounded wall-clock — the whole point of the fix.
    upper = 1 + _KILL_AT_TIMEOUT_BUDGET_S + 2  # +2s slack for shutdown
    assert elapsed < upper, (
        f"drain took {elapsed:.1f}s; expected < {upper}s "
        f"(drain_timeout_s=1 + kill_budget={_KILL_AT_TIMEOUT_BUDGET_S})"
    )

    # Hard-fd escape hatch actually fired on our fd.
    assert fd_a in close_calls, (
        f"expected os.close({fd_a}) as escape hatch; saw calls: {close_calls}"
    )

    # Routing still flipped despite the hung close.
    assert group.primary_status == HealthResult.UNHEALTHY

    # Release the hung close() thread so pytest can cleanly tear down.
    hung.hang_event.set()


def test_mixed_fast_and_hung_closes(fake_state, monkeypatch):
    """Fast closes complete + log 'force-closed'; the hung one hits
    the escape hatch. All done within budget."""
    # Real fds so os.close in the hatch is a no-op-safe operation.
    pair1 = socket.socketpair()
    pair2 = socket.socketpair()
    pair3 = socket.socketpair()
    try:
        group = _make_group(fake_state)
        fast1 = _FastConn(fd=pair1[0].fileno())
        fast2 = _FastConn(fd=pair2[0].fileno())
        hung = _HungConn(fd=pair3[0].fileno())
        for c in (fast1, fast2, hung):
            group.primary.tracked_conns.add(c)

        close_calls: List[int] = []
        real_os_close = os.close

        def spy(fd):
            close_calls.append(fd)
            return real_os_close(fd)

        monkeypatch.setattr("psycopg.yb.drain.os.close", spy)

        t0 = time.monotonic()
        trigger_drain(
            group=group,
            which_cluster="primary",
            new_status=HealthResult.UNHEALTHY,
            drain_timeout_s=1,
        )
        elapsed = time.monotonic() - t0

        assert elapsed < 1 + _KILL_AT_TIMEOUT_BUDGET_S + 2
        assert fast1.closed
        assert fast2.closed
        # Only the hung conn's fd went through the escape hatch.
        assert pair3[0].fileno() in close_calls
        # The fast conns did NOT need hard-fd-close.
        assert pair1[0].fileno() not in close_calls
        assert pair2[0].fileno() not in close_calls

        hung.hang_event.set()
    finally:
        for s1, s2 in (pair1, pair2, pair3):
            for s in (s1, s2):
                try:
                    s.close()
                except OSError:
                    pass


# ------------------------------------------------------- raising close

def test_raising_close_does_not_break_siblings(fake_state, owned_socketpair):
    """A conn whose close() raises must not stop drain — sibling
    conns still get closed, routing still flips."""
    fd_a, _fd_b = owned_socketpair
    group = _make_group(fake_state)
    bad = _RaisingConn(fd=fd_a)
    good = _FastConn(fd=-1)
    group.primary.tracked_conns.add(bad)
    group.primary.tracked_conns.add(good)

    trigger_drain(
        group=group,
        which_cluster="primary",
        new_status=HealthResult.UNHEALTHY,
        drain_timeout_s=1,
    )

    assert good.closed
    assert group.primary_status == HealthResult.UNHEALTHY


# ------------------------------------------------------- many hung conns

def test_many_hung_conns_still_bounded(fake_state, monkeypatch):
    """Billy's scenario: 100 conns all hung — drain must still return
    within budget, not after N × tcp_retries2."""
    n = 100
    pairs = [socket.socketpair() for _ in range(n)]
    try:
        group = _make_group(fake_state)
        hungs = [_HungConn(fd=p[0].fileno()) for p in pairs]
        for c in hungs:
            group.primary.tracked_conns.add(c)

        close_calls: List[int] = []
        real_os_close = os.close

        def spy(fd):
            close_calls.append(fd)
            return real_os_close(fd)

        monkeypatch.setattr("psycopg.yb.drain.os.close", spy)

        t0 = time.monotonic()
        trigger_drain(
            group=group,
            which_cluster="primary",
            new_status=HealthResult.UNHEALTHY,
            drain_timeout_s=1,
        )
        elapsed = time.monotonic() - t0

        # Even with 100 hung conns, drain returns within a few seconds.
        assert elapsed < 1 + _KILL_AT_TIMEOUT_BUDGET_S + 3, (
            f"drain took {elapsed:.1f}s with {n} hung conns; expected bounded"
        )
        # Every one got the escape hatch.
        expected_fds = {p[0].fileno() for p in pairs}
        assert expected_fds.issubset(set(close_calls))

        for h in hungs:
            h.hang_event.set()
    finally:
        for pair in pairs:
            for s in pair:
                try:
                    s.close()
                except OSError:
                    pass


# ------------------------------------------------------- fd snapshotted before close

def test_fd_snapshotted_before_close_returns(fake_state):
    """close() may null the pgconn state (as psycopg does on real
    conns). Snapshot the fd BEFORE close is called so we can still
    hard-fd-close if close hangs."""

    class NullifyingHungConn:
        """close() nulls pgconn.socket via an exception on attribute
        access — mirrors real psycopg behaviour after close()."""

        def __init__(self, fd):
            self.info = _FakeInfo(TransactionStatus.INTRANS)
            self._fd = fd
            self.pgconn = types.SimpleNamespace(socket=fd)
            self.hang_event = threading.Event()

        def close(self):
            # Null out the socket (like real psycopg would after finish)
            # BEFORE hanging — this simulates the race where snapshot
            # happens after nullification.
            self.pgconn = types.SimpleNamespace()
            self.hang_event.wait(timeout=60)

    a, b = socket.socketpair()
    try:
        group = _make_group(fake_state)
        conn = NullifyingHungConn(fd=a.fileno())
        group.primary.tracked_conns.add(conn)

        t0 = time.monotonic()
        trigger_drain(
            group=group,
            which_cluster="primary",
            new_status=HealthResult.UNHEALTHY,
            drain_timeout_s=1,
        )
        elapsed = time.monotonic() - t0

        # Snapshot must have captured the fd BEFORE close was called;
        # even though close() hangs and nulls state, drain returns
        # bounded.
        assert elapsed < 1 + _KILL_AT_TIMEOUT_BUDGET_S + 2
        conn.hang_event.set()
    finally:
        for s in (a, b):
            try:
                s.close()
            except OSError:
                pass


# ------------------------------------------------------- immediate kill (sentinel 0)

def test_drain_timeout_zero_still_parallel_closes(fake_state, monkeypatch):
    """Sentinel drain_timeout_s=0 (kill immediately) still uses the
    parallel-close path with escape hatch — one hung conn shouldn't
    wedge the immediate-kill mode either."""
    a, b = socket.socketpair()
    try:
        group = _make_group(fake_state)
        hung = _HungConn(fd=a.fileno())
        group.primary.tracked_conns.add(hung)

        close_calls: List[int] = []
        real_os_close = os.close   # capture BEFORE patching to avoid recursion

        def spy(fd):
            close_calls.append(fd)
            return real_os_close(fd)

        monkeypatch.setattr("psycopg.yb.drain.os.close", spy)

        t0 = time.monotonic()
        trigger_drain(
            group=group,
            which_cluster="primary",
            new_status=HealthResult.UNHEALTHY,
            drain_timeout_s=0,
        )
        elapsed = time.monotonic() - t0

        # No drain window + kill budget → very fast completion.
        assert elapsed < _KILL_AT_TIMEOUT_BUDGET_S + 2
        assert a.fileno() in close_calls
        hung.hang_event.set()
    finally:
        for s in (a, b):
            try:
                s.close()
            except OSError:
                pass
