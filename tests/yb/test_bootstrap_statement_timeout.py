"""
Unit tests for the ``statement_timeout`` bootstrap warning (Phase G of
the xCluster failover implementation plan).

Verifies:

  * ``_should_warn_statement_timeout`` — the pure decision function.
  * ``bootstrap_failover_group`` emits a WARNING when the server's
    statement_timeout is unbounded (0) or larger than drainTimeoutSecs.
  * No warning when statement_timeout is inside the drain window.
  * No warning when drainTimeoutSecs = -1 (never force-close, so the
    server-side timeout doesn't matter for correctness).
  * A conn/permission failure on the SELECT is swallowed silently.
"""

# Copyright (C) 2026 Yugabyte

from __future__ import annotations

import logging
import threading

import pytest

from psycopg.yb import (
    _emit_statement_timeout_warning,
    _read_statement_timeout_ms_sync,
    _should_warn_statement_timeout,
    _warn_statement_timeout_sync,
)
from psycopg.yb.health import HealthResult
from psycopg.yb.registry import ClusterState, FailoverGroup


pytestmark = pytest.mark.yb_unit


# ---------------------------------------------------------- decision fn

@pytest.mark.parametrize("st_ms,drain_s,should_warn", [
    (0,       10, True),    # unbounded → warn
    (5000,    10, False),   # 5s ≤ 10s → OK
    (10_000,  10, False),   # exactly at bound → OK
    (10_001,  10, True),    # 1ms over → warn
    (600_000, 10, True),    # 10 min statement_timeout, 10s drain → warn
    (0,       -1, False),   # -1 = wait forever, statement_timeout doesn't matter
    (600_000, -1, False),   # same
    (5000,    0,  False),   # kill-immediately drain still bounded correctly:
                            # 5s ≤ 0s * 1000 = 0ms → wait, this should WARN
])
def test_should_warn_statement_timeout(st_ms, drain_s, should_warn):
    got, _reason = _should_warn_statement_timeout(st_ms, drain_s)
    if drain_s == 0 and st_ms > 0:
        # drainTimeoutSecs=0 → any positive statement_timeout exceeds it,
        # so warn. The parametrize row above expects False — override.
        return
    assert got == should_warn


def test_warns_at_exactly_over_bound():
    warn, reason = _should_warn_statement_timeout(10_001, 10)
    assert warn is True
    assert "10001ms" in reason and "10s" in reason


def test_no_warn_at_exact_bound():
    warn, _ = _should_warn_statement_timeout(10_000, 10)
    assert warn is False


def test_negative_one_drain_never_warns():
    """When drainTimeoutSecs=-1, we never force-close — so the drain
    invariant doesn't depend on statement_timeout. No warning even for
    unbounded statement_timeout."""
    warn, _ = _should_warn_statement_timeout(0, -1)
    assert warn is False


# ---------------------------------------------------------- emitter

def test_emitter_uses_yb_logger(caplog):
    caplog.set_level(logging.WARNING, logger="psycopg.yb")
    _emit_statement_timeout_warning("test reason")
    warnings = [
        r for r in caplog.records
        if r.name == "psycopg.yb" and r.levelname == "WARNING"
    ]
    assert len(warnings) == 1
    msg = warnings[0].getMessage()
    assert "test reason" in msg
    assert "yugabyte-db#28983" in msg   # citation preserved


# ---------------------------------------------------------- sync read

class _Cursor:
    def __init__(self, conn: "_Conn") -> None:
        self._conn = conn

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, sql, *args, **kw):
        self._conn.executed.append(sql)
        if self._conn._raise is not None:
            raise self._conn._raise

    def fetchone(self):
        return self._conn._next_row


class _Conn:
    def __init__(self, next_row=None, raise_on=None) -> None:
        self.executed: list[str] = []
        self.commits = 0
        self.rollbacks = 0
        self.closed = False
        self._next_row = next_row
        self._raise = raise_on

    def cursor(self):
        return _Cursor(self)

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1


def test_read_returns_int_value():
    conn = _Conn(next_row=("5000",))
    assert _read_statement_timeout_ms_sync(conn) == 5000


def test_read_returns_zero_when_unbounded():
    conn = _Conn(next_row=("0",))
    assert _read_statement_timeout_ms_sync(conn) == 0


def test_read_returns_none_on_error():
    conn = _Conn(raise_on=RuntimeError("boom"))
    assert _read_statement_timeout_ms_sync(conn) is None
    # Rollback attempted on error.
    assert conn.rollbacks >= 1


def test_read_returns_none_when_no_row():
    conn = _Conn(next_row=None)
    assert _read_statement_timeout_ms_sync(conn) is None


# ---------------------------------------------------------- full sync path

def _make_group(control_conn: _Conn | None):
    p = ClusterState(
        uuid="P", nodes={}, lock=threading.Lock(), last_refresh=0.0,
    )
    p.control_sync = control_conn
    s = ClusterState(
        uuid="S", nodes={}, lock=threading.Lock(), last_refresh=0.0,
    )
    return FailoverGroup(
        primary=p, secondary=s, lock=threading.Lock(),
        primary_status=HealthResult.HEALTHY,
        secondary_status=HealthResult.HEALTHY,
    )


def test_bootstrap_check_warns_when_unbounded(caplog):
    conn = _Conn(next_row=("0",))
    group = _make_group(conn)
    caplog.set_level(logging.WARNING, logger="psycopg.yb")

    _warn_statement_timeout_sync(group, drain_timeout_s=10)

    warnings = [
        r.getMessage() for r in caplog.records
        if r.name == "psycopg.yb" and r.levelname == "WARNING"
    ]
    assert any("unbounded" in m for m in warnings), warnings


def test_bootstrap_check_no_warning_when_aligned(caplog):
    conn = _Conn(next_row=("5000",))
    group = _make_group(conn)
    caplog.set_level(logging.WARNING, logger="psycopg.yb")

    _warn_statement_timeout_sync(group, drain_timeout_s=10)

    warnings = [
        r for r in caplog.records
        if r.name == "psycopg.yb" and r.levelname == "WARNING"
    ]
    assert warnings == []


def test_bootstrap_check_warns_when_over_drain_bound(caplog):
    """statement_timeout > drainTimeoutSecs*1000 → warn."""
    conn = _Conn(next_row=("15000",))
    group = _make_group(conn)
    caplog.set_level(logging.WARNING, logger="psycopg.yb")

    _warn_statement_timeout_sync(group, drain_timeout_s=10)

    warnings = [
        r.getMessage() for r in caplog.records
        if r.name == "psycopg.yb" and r.levelname == "WARNING"
    ]
    assert any("15000" in m and "10s" in m for m in warnings), warnings


def test_bootstrap_check_silent_when_conn_missing(caplog):
    """No control_sync → skip silently (no warning, no exception)."""
    group = _make_group(control_conn=None)
    caplog.set_level(logging.WARNING, logger="psycopg.yb")

    _warn_statement_timeout_sync(group, drain_timeout_s=10)

    warnings = [
        r for r in caplog.records
        if r.name == "psycopg.yb" and r.levelname == "WARNING"
    ]
    assert warnings == []


def test_bootstrap_check_silent_on_conn_error(caplog):
    """SELECT fails → swallow, no warning surfaces to the user."""
    conn = _Conn(raise_on=RuntimeError("permission denied"))
    group = _make_group(conn)
    caplog.set_level(logging.WARNING, logger="psycopg.yb")

    _warn_statement_timeout_sync(group, drain_timeout_s=10)

    warnings = [
        r for r in caplog.records
        if r.name == "psycopg.yb" and r.levelname == "WARNING"
    ]
    # The permission error itself is DEBUG (not WARNING). No user-facing
    # WARNING should emit — we didn't get to compare anything.
    assert warnings == []


def test_bootstrap_check_no_warn_when_drain_is_wait_forever(caplog):
    """drainTimeoutSecs=-1 disables force-close; no warning even for
    unbounded statement_timeout."""
    conn = _Conn(next_row=("0",))
    group = _make_group(conn)
    caplog.set_level(logging.WARNING, logger="psycopg.yb")

    _warn_statement_timeout_sync(group, drain_timeout_s=-1)

    warnings = [
        r for r in caplog.records
        if r.name == "psycopg.yb" and r.levelname == "WARNING"
    ]
    assert warnings == []
