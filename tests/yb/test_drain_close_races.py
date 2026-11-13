"""
Close races between the drain thread and a conn's owner (Amogh's review).

#21: the idle sweep read ``transaction_status`` before ``_safe_close``
took ``conn.lock``, so an owner that ran ``BEGIN`` in between had its
open transaction closed. ``_safe_close(only_if_idle=True)`` re-reads the
status under the lock and leaves the conn for the next tick.

#23: async conns were finished from the drain thread after a
``lock.locked()`` read; on the C build the loop could start a
``PQsendQuery`` between the read and ``PQfinish``. Conns opened through
the dispatcher record their loop, and the drain finishes them there.
"""

# Copyright (C) 2026 Yugabyte

from __future__ import annotations

import asyncio
import threading
import time
import types
import weakref

import pytest

from psycopg.pq import ConnStatus, TransactionStatus
from psycopg.yb import dispatch
from psycopg.yb.drain import (
    _CloseLockTimeoutError,
    _ConnNotIdleError,
    _close_idle_tracked_during_drain,
    _safe_close,
)


pytestmark = pytest.mark.yb_unit

IDLE = int(TransactionStatus.IDLE)
INTRANS = int(TransactionStatus.INTRANS)


# ------------------------------------------------------------- sync (#21)

class _SyncConn:
    """``transaction_status`` follows ``statuses`` read by read (the last
    value repeats), so a test can change it between the sweep's check
    and the close."""

    def __init__(self, *statuses: int) -> None:
        self.lock = threading.Lock()
        self.closed = False
        self.close_calls = 0
        self._statuses = list(statuses)
        self.pgconn = types.SimpleNamespace(socket=-1)

    @property
    def info(self):
        status = self._statuses.pop(0) if len(self._statuses) > 1 else self._statuses[0]
        return types.SimpleNamespace(transaction_status=status)

    def close(self) -> None:
        self.close_calls += 1
        self.closed = True


class _Tracked(weakref.WeakSet):
    pass


def test_idle_sweep_skips_a_conn_that_began_a_transaction():
    c = _SyncConn(IDLE, INTRANS)
    tracked = _Tracked([c])

    assert _close_idle_tracked_during_drain(tracked, "B") == 0
    assert c.close_calls == 0
    assert not c.lock.locked()


def test_idle_sweep_closes_a_conn_still_idle_under_the_lock():
    c = _SyncConn(IDLE)
    tracked = _Tracked([c])

    assert _close_idle_tracked_during_drain(tracked, "B") == 1
    assert c.close_calls == 1


def test_only_if_idle_leaves_an_open_transaction():
    c = _SyncConn(INTRANS)

    assert isinstance(_safe_close(c, only_if_idle=True), _ConnNotIdleError)
    assert c.close_calls == 0
    assert not c.lock.locked()


def test_kill_path_still_closes_an_open_transaction():
    c = _SyncConn(INTRANS)

    assert _safe_close(c) is None
    assert c.close_calls == 1


# ------------------------------------------------------------ async (#23)

class _FlagLock:
    def __init__(self, held: bool = False) -> None:
        self.held = held

    def locked(self) -> bool:
        return self.held


def _async_conn(lock, status: int = IDLE, loop=None):
    from psycopg.connection_async import AsyncConnection

    c = object.__new__(AsyncConnection)
    c._closed = False
    c._yb_uuid = None
    c._yb_host = None
    c.lock = lock
    c.finish_thread = None
    if loop is not None:
        c._yb_loop = loop

    def finish():
        c.finish_thread = threading.get_ident()
        c.pgconn.status = ConnStatus.BAD
    c.pgconn = types.SimpleNamespace(
        socket=-1, finish=finish, transaction_status=status, status=ConnStatus.OK,
    )
    return c


@pytest.fixture
def loop_thread():
    loop = asyncio.new_event_loop()
    t = threading.Thread(target=loop.run_forever, daemon=True)
    t.start()
    yield loop, t
    loop.call_soon_threadsafe(loop.stop)
    t.join(2)
    loop.close()


def test_async_close_runs_on_the_conns_loop(loop_thread):
    loop, t = loop_thread
    c = _async_conn(_FlagLock(), loop=loop)

    assert _safe_close(c) is None
    assert c.finish_thread == t.ident
    assert c._closed is True


def test_async_close_on_loop_waits_for_the_lock(loop_thread):
    loop, t = loop_thread
    lock = _FlagLock(held=True)
    c = _async_conn(lock, loop=loop)
    threading.Timer(0.2, lambda: setattr(lock, "held", False)).start()

    started = time.monotonic()
    assert _safe_close(c) is None
    assert time.monotonic() - started >= 0.2
    assert c.finish_thread == t.ident


def test_async_close_on_loop_escalates_when_lock_stays_held(loop_thread):
    loop, _ = loop_thread
    c = _async_conn(_FlagLock(held=True), loop=loop)

    assert isinstance(_safe_close(c, lock_timeout_s=0.1), _CloseLockTimeoutError)
    assert c.finish_thread is None
    c.pgconn.status = ConnStatus.BAD


def test_async_idle_sweep_on_loop_leaves_an_open_transaction(loop_thread):
    loop, _ = loop_thread
    c = _async_conn(_FlagLock(), status=INTRANS, loop=loop)

    exc = _safe_close(c, lock_timeout_s=0, only_if_idle=True)
    assert isinstance(exc, _ConnNotIdleError)
    assert c.finish_thread is None
    c.pgconn.status = ConnStatus.BAD


def test_async_close_escalates_when_the_loop_is_blocked(loop_thread):
    loop, _ = loop_thread
    c = _async_conn(_FlagLock(), loop=loop)
    unblock = threading.Event()
    loop.call_soon_threadsafe(unblock.wait, 2)

    exc = _safe_close(c, lock_timeout_s=0.1)
    unblock.set()
    time.sleep(0.05)   # let the loop drain the cancelled attempt

    assert isinstance(exc, _CloseLockTimeoutError)
    assert c.finish_thread is None, "a cancelled attempt must not finish the conn"
    c.pgconn.status = ConnStatus.BAD


def test_async_close_with_stopped_loop_closes_directly():
    loop = asyncio.new_event_loop()
    c = _async_conn(_FlagLock(), loop=loop)
    try:
        assert _safe_close(c) is None
        assert c.finish_thread == threading.get_ident()
    finally:
        loop.close()


def test_async_close_from_the_conns_own_loop_closes_inline():
    """An operator reset called from async code runs the drain on the
    conn's loop; waiting on that loop would stall the whole budget."""
    async def main():
        c = _async_conn(_FlagLock(), loop=asyncio.get_running_loop())
        started = time.monotonic()
        assert _safe_close(c) is None
        assert time.monotonic() - started < 0.5
        return c

    c = asyncio.run(main())
    assert c.finish_thread == threading.get_ident()


def test_atrack_or_reroute_records_the_loop(monkeypatch):
    monkeypatch.setattr(dispatch, "_track", lambda route, conn, host: False)
    route = types.SimpleNamespace(group=object())
    conn = types.SimpleNamespace()

    async def main():
        assert await dispatch.atrack_or_reroute(route, conn, "h1") is None
        return asyncio.get_running_loop()

    loop = asyncio.run(main())
    assert conn._yb_loop is loop
