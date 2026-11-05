"""
Hold the secondary's connections during a failback that may be refused
(DB-25803).

Phase 1 of a drain closes every tracked connection on the outgoing side,
so that nothing commits there while Phase 2 waits for replication to
catch up. On a failback under ``yb.failback.onPhase2Timeout=fail`` the
outgoing side is the healthy secondary, and when the lag wait then fails
the failback is refused and routing stays there: closing its connections
cost the application a reconnect wave, every cooldown.

Such a drain holds those connections instead. A held connection's next
round-trip (a statement, a commit, a COPY: everything that goes through
``Connection.wait``) blocks until the drain settles the hold:

* released (failback refused): the round-trip goes ahead on the
  secondary, inside the transaction it was in, if any;
* closed (failback goes ahead): it raises ``OperationalError`` and the
  drain closes the connection, as without a hold.

A connection whose round-trip is already on the wire when the hold lands
can't be stopped this way. :func:`hold_conn` reports it as busy and the
drain closes it as before.

Forked workers (``fork_follow``) close their own connections as before.
"""

# Copyright (C) 2026 Yugabyte

from __future__ import annotations

import asyncio
import threading
from typing import Any

from .. import errors as e

RELEASE = "release"
CLOSE = "close"

# How often an async round-trip checks whether its hold was settled.
_ASYNC_POLL_S = 0.01

# Every round-trip on an xCluster connection takes one of these, briefly,
# to read its hold and count itself busy; striped so connections don't
# contend.
_STRIPES = tuple(threading.Lock() for _ in range(64))


def _lock_for(conn: Any) -> threading.Lock:
    return _STRIPES[id(conn) % len(_STRIPES)]


class Hold:
    """The hold one drain puts on the outgoing side's connections."""

    def __init__(self) -> None:
        self._cv = threading.Condition()
        self.verdict: str | None = None

    def wait(self) -> str:
        with self._cv:
            while self.verdict is None:
                self._cv.wait()
            return self.verdict

    def _settle(self, verdict: str) -> None:
        with self._cv:
            self.verdict = verdict
            self._cv.notify_all()


def hold_conn(conn: Any, hold: Hold) -> bool:
    """Put ``hold`` on ``conn``. Returns True if a round-trip is in
    progress on it right now: the hold only stops the next one."""
    with _lock_for(conn):
        if getattr(conn, "_yb_hold", None) is None:
            conn._yb_hold = hold
        return getattr(conn, "_yb_busy", 0) > 0


def release(hold: Hold, conns: list[Any]) -> None:
    """Lift ``hold``: blocked round-trips go ahead."""
    for c in conns:
        with _lock_for(c):
            if getattr(c, "_yb_hold", None) is hold:
                c._yb_hold = None
    hold._settle(RELEASE)


def close(hold: Hold) -> None:
    """Fail every blocked round-trip on ``hold``; the caller then closes
    the connections."""
    hold._settle(CLOSE)


def _closed_error() -> e.OperationalError:
    return e.OperationalError(
        "connection closed by an xCluster failback: routing is moving to "
        "the primary"
    )


def block_if_held(conn: Any) -> None:
    """Called by ``Connection.wait`` before a round-trip. Blocks while a
    drain holds ``conn``; otherwise counts the connection busy until
    :func:`done`."""
    while True:
        with _lock_for(conn):
            hold = getattr(conn, "_yb_hold", None)
            if hold is None:
                conn._yb_busy = getattr(conn, "_yb_busy", 0) + 1
                return
        if hold.wait() == CLOSE:
            raise _closed_error()


async def ablock_if_held(conn: Any) -> None:
    """Async version of :func:`block_if_held`: waits without blocking the loop."""
    while True:
        with _lock_for(conn):
            hold = getattr(conn, "_yb_hold", None)
            if hold is None:
                conn._yb_busy = getattr(conn, "_yb_busy", 0) + 1
                return
        while hold.verdict is None:
            await asyncio.sleep(_ASYNC_POLL_S)
        if hold.verdict == CLOSE:
            raise _closed_error()


def done(conn: Any) -> None:
    with _lock_for(conn):
        conn._yb_busy -= 1


def wait_settled(conn: Any) -> None:
    """Pool check: wait until the drain holding ``conn`` (if any) has
    decided, instead of evicting a connection it may give back."""
    hold = getattr(conn, "_yb_hold", None)
    if hold is not None:
        hold.wait()


async def await_settled(conn: Any) -> None:
    hold = getattr(conn, "_yb_hold", None)
    if hold is not None:
        while hold.verdict is None:
            await asyncio.sleep(_ASYNC_POLL_S)
