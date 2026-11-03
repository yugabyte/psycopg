"""
Repro + regression tests for DB-25090 (PR-H9).

Shubham reported (#proj-smartdriver-xclusterfailover, Oct 26 2026): on
the C build, 2/19 drains in one run ended with the entire workload
process dying within ~1 s of ``drain complete: dispatch resumed`` on
a failback. In the same run, a failover force-close left one worker
with ``IndexError: list index out of range`` from inside psycopg and
another with ``consuming input failed: connection pointer is NULL``,
7 ms apart. Pure Python build: 0/5. First seen on 2.31.0.0-b523, run
39fd46d7.

Root cause: ``_safe_close`` called ``pgconn.finish()`` (async branch,
DB-24960) and ``c.close()`` (sync branch) from the drain thread
without serializing against the owner thread's libpq calls. On the C
build, freeing libpq state while a worker is inside ``consuming
input`` corrupts the result parser → NULL conn pointer, IndexError
from inside psycopg, or process death.

Fix: ``Connection.lock`` already serializes libpq calls on the owner
side (``cursor.py:112,127,168,324,342``). ``_safe_close`` now acquires
that lock with a 1.5 s budget before touching libpq. If the owner is
inside libpq and doesn't release in time, ``_safe_close`` returns
``_CloseLockTimeoutError`` and the kill-at-timeout caller escalates to
the existing hard-fd escape hatch — same path as a hung ``close()``.

Async conns don't get the treatment: their ``.lock`` is an
``asyncio.Lock`` and acquiring it from the drain thread would be a
cross-thread asyncio-lock call. The residual async finish-race is
tracked separately.
"""

# Copyright (C) 2026 Yugabyte

from __future__ import annotations

import threading
import time
import types

import pytest

from psycopg.yb.drain import (
    _SAFE_CLOSE_LOCK_BUDGET_S,
    _CloseLockTimeoutError,
    _safe_close,
)


pytestmark = pytest.mark.yb_unit


class _FakeSyncConn:
    """Minimal Connection shape for the drain path: ``.lock``
    (threading.Lock), ``.close()``, ``.closed``, ``.pgconn.socket``.
    The lock mirrors ``psycopg.Connection.lock`` which cursor
    operations acquire around every libpq call."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.closed = False
        self.close_called_at: float | None = None
        self.pgconn = types.SimpleNamespace(socket=-1)

    def close(self) -> None:
        self.close_called_at = time.monotonic()
        self.closed = True


# ===================================================================
# Happy path — lock free, close works
# ===================================================================


def test_safe_close_acquires_and_releases_lock():
    """When the owner isn't holding the lock, _safe_close acquires it,
    closes the conn, releases the lock. Returning None means success."""
    c = _FakeSyncConn()
    result = _safe_close(c)
    assert result is None
    assert c.closed is True
    # Lock must be released after close so subsequent ownership works.
    assert c.lock.acquire(blocking=False)
    c.lock.release()


# ===================================================================
# DB-25090 — owner-held lock races the drain
# ===================================================================


def test_safe_close_waits_for_owner_lock():
    """Owner thread holds the lock briefly (simulating "worker inside
    libpq consuming input"). _safe_close must wait for the lock, then
    close. Verifies the fix: pre-H9, close() ran without waiting →
    freed libpq state while the worker was still inside it (DB-25090
    on C build). Post-fix: close() waits."""
    c = _FakeSyncConn()
    c.lock.acquire()                       # owner enters libpq

    close_result: list = []

    def do_drain_close():
        close_result.append(_safe_close(c))

    drain_thread = threading.Thread(target=do_drain_close, daemon=True)
    drain_thread.start()

    # Give drain a moment to try to acquire. It must not have closed yet.
    time.sleep(0.1)
    assert c.closed is False, (
        "drain closed the conn while owner still held conn.lock — "
        "DB-25090 regression"
    )

    # Owner finishes its libpq call, releases the lock. Drain should
    # now acquire and close.
    c.lock.release()
    drain_thread.join(timeout=1.0)

    assert not drain_thread.is_alive(), "drain thread did not return"
    assert close_result == [None]
    assert c.closed is True


def test_safe_close_times_out_if_owner_never_releases():
    """If the owner holds the lock past the ``_SAFE_CLOSE_LOCK_BUDGET_S``
    budget (worker genuinely stuck in libpq — e.g. blocked on a dead
    TCP peer), _safe_close returns ``_CloseLockTimeoutError`` so the
    caller can escalate to the hard-fd escape hatch. Must NOT free
    libpq state in that case (would reproduce DB-25090)."""
    c = _FakeSyncConn()
    c.lock.acquire()                       # owner never releases

    try:
        t0 = time.monotonic()
        result = _safe_close(c)
        elapsed = time.monotonic() - t0

        assert isinstance(result, _CloseLockTimeoutError), (
            f"expected _CloseLockTimeoutError, got {type(result).__name__}: "
            f"{result!r}"
        )
        # Must respect the budget — not block indefinitely.
        assert elapsed >= _SAFE_CLOSE_LOCK_BUDGET_S - 0.1, (
            f"gave up after {elapsed:.2f} s — the budget is "
            f"{_SAFE_CLOSE_LOCK_BUDGET_S} s"
        )
        assert elapsed < _SAFE_CLOSE_LOCK_BUDGET_S + 0.5, (
            f"waited {elapsed:.2f} s — longer than the budget allows"
        )
        assert c.closed is False, (
            "close() must NOT run while owner still holds the lock — "
            "that's the DB-25090 crash path"
        )
    finally:
        c.lock.release()


def test_safe_close_releases_lock_on_close_exception():
    """If ``c.close()`` raises after we acquired the lock, we must
    still release it. Otherwise a bad close poisons the lock and the
    next tick deadlocks the owner thread."""
    c = _FakeSyncConn()

    def boom():
        raise RuntimeError("close exploded")

    c.close = boom
    result = _safe_close(c)

    assert isinstance(result, RuntimeError)
    # Lock must be free despite the close raising.
    assert c.lock.acquire(blocking=False), (
        "lock was not released after c.close() raised — poisoned lock "
        "deadlocks the next owner call"
    )
    c.lock.release()


# ===================================================================
# Backwards compatibility — mocks without .lock
# ===================================================================


def test_safe_close_tolerates_missing_lock():
    """Test fakes (and some pool shims) construct conn-like objects
    without a ``.lock``. The drain path must fall through to the direct
    close rather than raising AttributeError — pre-H1 tests still work."""
    c = types.SimpleNamespace(
        closed=False,
        close=lambda: setattr(c, "closed", True),
        pgconn=types.SimpleNamespace(socket=-1),
    )
    # No .lock attribute.
    result = _safe_close(c)
    assert result is None
    assert c.closed is True


# ===================================================================
# Async branch unaffected (lock is asyncio.Lock → NOT acquired here)
# ===================================================================


def test_safe_close_async_branch_unchanged():
    """AsyncConnection ships with an ``asyncio.Lock`` which is thread-
    hostile. The async branch must NOT try to acquire it; it should
    call ``pgconn.finish()`` directly, same as pre-H9."""
    import asyncio
    from psycopg.connection_async import AsyncConnection

    finished = {"called": False, "closed": False}

    class _FakeAsyncConn(AsyncConnection):
        """Subclass to pass the isinstance check; override __init__
        so we don't need a real event loop + connection."""

        def __init__(self):  # type: ignore[no-untyped-def]
            self.lock = asyncio.Lock()
            self._closed = False
            self.pgconn = types.SimpleNamespace(
                finish=lambda: finished.__setitem__("called", True),
                socket=-1,
            )

        @property
        def closed(self):
            return self._closed

    c = _FakeAsyncConn()
    result = _safe_close(c)

    assert result is None
    assert finished["called"] is True, "pgconn.finish() was not called"
    assert c._closed is True


# ===================================================================
# Budget sanity
# ===================================================================


def test_lock_budget_is_sized_against_kill_at_timeout_budget():
    """The lock-acquire budget must leave room inside the overall
    kill-at-timeout budget for the hard-fd escalation path. If
    ``_SAFE_CLOSE_LOCK_BUDGET_S >= _KILL_AT_TIMEOUT_BUDGET_S`` the
    outer futures.wait would time out before _safe_close returns the
    lock-timeout error → futures.done()==False → the hung-close
    branch would fire with a misleading reason instead of the
    DB-25090 branch. Keep an invariant for future refactors."""
    from psycopg.yb.drain import _KILL_AT_TIMEOUT_BUDGET_S

    assert _SAFE_CLOSE_LOCK_BUDGET_S < _KILL_AT_TIMEOUT_BUDGET_S, (
        f"_SAFE_CLOSE_LOCK_BUDGET_S ({_SAFE_CLOSE_LOCK_BUDGET_S}s) must "
        f"be strictly less than _KILL_AT_TIMEOUT_BUDGET_S "
        f"({_KILL_AT_TIMEOUT_BUDGET_S}s) so the lock-timeout branch is "
        f"reachable from the kill-at-timeout caller"
    )
