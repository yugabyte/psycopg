"""
Force-close must not cut a COMMIT that is already on the wire.

r12 218881 cycle 3 (planned_signal_flip/drain_zero): an async_pool
INSERT on B sent COMMIT at mono 2510.464, the failback drain armed at
2510.479 and its kill path called ``pgconn.finish()`` on the
AsyncConnection while the COMMIT reply was still on its way. The app
saw "connection socket closed" (unknown outcome) after the end of
Phase 1, which the stress suite counts as outgoing_straddled_barrier.

Sync conns were already safe: ``_safe_close`` takes ``Connection.lock``
(DB-25090), so the close waits for the COMMIT to return. The fix gives
async conns the same guarantee by waiting, within the same budget,
until their ``asyncio.Lock`` is no longer held (read-only, from the
drain thread), and makes pool-borrowed sync conns take the lock before
``pgconn.finish()``.
"""

# Copyright (C) 2026 Yugabyte

from __future__ import annotations

import asyncio
import threading
import time
import types

import pytest

from psycopg.pq import ConnStatus, TransactionStatus
from psycopg.yb.drain import _CloseLockTimeoutError, _safe_close


pytestmark = pytest.mark.yb_unit


class _FlagLock:
    """Stands in for ``asyncio.Lock``: only ``locked()`` is read."""

    def __init__(self, held: bool) -> None:
        self.held = held

    def locked(self) -> bool:
        return self.held


def _async_conn(lock):
    from psycopg.connection_async import AsyncConnection

    c = object.__new__(AsyncConnection)
    c._closed = False
    c._yb_uuid = None
    c._yb_host = None
    c.lock = lock
    c.finished_at = None

    def finish():
        c.finished_at = time.monotonic()
        c.pgconn.status = ConnStatus.BAD
    c.pgconn = types.SimpleNamespace(
        socket=-1,
        finish=finish,
        transaction_status=int(TransactionStatus.ACTIVE),
        status=ConnStatus.OK,
    )
    return c


def _release_after(lock: _FlagLock, delay_s: float) -> list:
    released = []

    def run():
        time.sleep(delay_s)
        released.append(time.monotonic())
        lock.held = False
    threading.Thread(target=run, daemon=True).start()
    return released


def test_async_close_waits_for_commit_on_the_wire():
    lock = _FlagLock(held=True)
    conn = _async_conn(lock)
    released = _release_after(lock, 0.2)

    exc = _safe_close(conn)

    assert exc is None
    assert released, "close returned before the COMMIT finished"
    assert conn.finished_at >= released[0], (
        "pgconn.finish() ran while the COMMIT was still on the wire"
    )
    assert conn._closed is True


def test_async_close_escalates_when_lock_stays_held():
    conn = _async_conn(_FlagLock(held=True))

    started = time.monotonic()
    exc = _safe_close(conn, lock_timeout_s=0.1)

    assert isinstance(exc, _CloseLockTimeoutError)
    assert time.monotonic() - started < 1.0
    assert conn.finished_at is None, "finish() must be left to the hard-fd path"
    conn.pgconn.status = ConnStatus.BAD   # let __del__ skip its warning


def test_async_idle_close_does_not_wait():
    conn = _async_conn(_FlagLock(held=True))

    started = time.monotonic()
    exc = _safe_close(conn, lock_timeout_s=0)

    assert isinstance(exc, _CloseLockTimeoutError)
    assert time.monotonic() - started < 0.05
    assert conn.finished_at is None
    conn.pgconn.status = ConnStatus.BAD


def test_async_close_with_free_lock_finishes_at_once():
    conn = _async_conn(_FlagLock(held=False))

    assert _safe_close(conn) is None
    assert conn.finished_at is not None


def test_async_close_waits_on_a_real_asyncio_lock():
    """The lock held by a coroutine on another thread's event loop is
    only read, never acquired, from the drain thread."""
    lock = asyncio.Lock()
    conn = _async_conn(lock)
    holding = threading.Event()
    released = []

    async def commit_in_flight():
        async with lock:
            holding.set()
            await asyncio.sleep(0.2)
            released.append(time.monotonic())

    t = threading.Thread(target=lambda: asyncio.run(commit_in_flight()))
    t.start()
    assert holding.wait(2)

    exc = _safe_close(conn)
    t.join(2)

    assert exc is None
    assert released and conn.finished_at >= released[0]


class _PoolConn:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self._pool = object()
        self._yb_uuid = None
        self._yb_host = None
        self.finished_at = None
        self.pgconn = types.SimpleNamespace(socket=-1, finish=self._finish)

    def _finish(self) -> None:
        self.finished_at = time.monotonic()

    def close(self) -> None:   # pool's putconn; must not be used
        raise AssertionError("pool conn closed via close() (returns it to the pool)")


def test_pool_conn_close_waits_for_owner_lock():
    conn = _PoolConn()
    conn.lock.acquire()
    released = []

    def owner():
        time.sleep(0.2)
        released.append(time.monotonic())
        conn.lock.release()
    threading.Thread(target=owner, daemon=True).start()

    exc = _safe_close(conn)

    assert exc is None
    assert released and conn.finished_at >= released[0]
