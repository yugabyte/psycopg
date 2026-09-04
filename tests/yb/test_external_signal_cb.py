"""
Unit tests for the reference ``ExternalSignalCircuitBreaker`` sample
(``demo/samples/external_signal_cb.py`` — Phase F of the xCluster
failover implementation plan).

Verifies:

  * Empty result set → HEALTHY (no signal = no failover).
  * ``target_status`` = ``UNHEALTHY`` → UNHEALTHY. Case-insensitive.
  * Unrecognised ``target_status`` → HEALTHY + WARNING (fail-safe).
  * Missing table → CB creates it on the fly, returns HEALTHY.
  * Broken control conn → HEALTHY (this CB is a request channel, NOT a
    health detector; transient errors don't imply cluster failure).
  * ``which_cluster`` gates which side the CB reads from.
"""

# Copyright (C) 2026 Yugabyte

from __future__ import annotations

import logging
import threading

import pytest

from psycopg.yb.health import HealthResult
from psycopg.yb.registry import ClusterState, FailoverGroup

from demo.samples.external_signal_cb import ExternalSignalCircuitBreaker


pytestmark = pytest.mark.yb_unit


# ------------------------------------------------------------ mock conn

class _Cursor:
    def __init__(self, conn: "_Conn") -> None:
        self._conn = conn

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, sql, *args, **kw):
        self._conn.executed.append((sql, args))
        if self._conn._raise_on_execute is not None:
            exc = self._conn._raise_on_execute
            self._conn._raise_on_execute = None
            raise exc

    def fetchone(self):
        return self._conn._next_row


class _Conn:
    """Minimal Connection stand-in: cursor with execute + fetchone,
    commit, rollback, close.

    ``next_row`` is what the next fetchone() returns. Set to ``None`` to
    simulate an empty result set."""

    def __init__(
        self,
        next_row=None,
        raise_on_execute: Exception | None = None,
    ) -> None:
        self.executed: list = []
        self.commits = 0
        self.rollbacks = 0
        self.closed = False
        self._next_row = next_row
        self._raise_on_execute = raise_on_execute

    def cursor(self):
        return _Cursor(self)

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1


def _make_state(uuid: str, conn: _Conn | None = None):
    s = ClusterState(
        uuid=uuid, nodes={}, lock=threading.Lock(), last_refresh=0.0,
    )
    s.control_sync = conn
    return s


def _make_group(primary_conn=None, secondary_conn=None):
    return FailoverGroup(
        primary=_make_state("P", primary_conn),
        secondary=_make_state("S", secondary_conn),
        lock=threading.Lock(),
        primary_status=HealthResult.HEALTHY,
        secondary_status=HealthResult.HEALTHY,
    )


def _undefined_table_exc() -> Exception:
    """Build an exception whose sqlstate matches UndefinedTable (42P01),
    so ``_is_undefined_table`` recognises it."""
    exc = Exception("relation does not exist")
    diag = type("Diag", (), {"sqlstate": "42P01"})()
    exc.diag = diag
    return exc


# ------------------------------------------------------------ basic reads

def test_no_signal_returns_healthy():
    conn = _Conn(next_row=None)
    group = _make_group(primary_conn=conn)
    cb = ExternalSignalCircuitBreaker(which_cluster="primary")
    assert cb.check(group) == HealthResult.HEALTHY
    # Table was set up on first call (CREATE) plus the SELECT.
    kinds = [sql.split()[0] for sql, _ in conn.executed]
    assert "CREATE" in kinds
    assert "SELECT" in kinds


def test_unhealthy_signal_returns_unhealthy():
    conn = _Conn(next_row=("UNHEALTHY",))
    group = _make_group(primary_conn=conn)
    cb = ExternalSignalCircuitBreaker(which_cluster="primary")
    assert cb.check(group) == HealthResult.UNHEALTHY


def test_healthy_signal_returns_healthy():
    conn = _Conn(next_row=("HEALTHY",))
    group = _make_group(primary_conn=conn)
    cb = ExternalSignalCircuitBreaker(which_cluster="primary")
    assert cb.check(group) == HealthResult.HEALTHY


def test_signal_is_case_insensitive():
    """Operator can write 'unhealthy' or 'UnHealthy' — CB reads either."""
    conn = _Conn(next_row=("unhealthy",))
    group = _make_group(primary_conn=conn)
    cb = ExternalSignalCircuitBreaker(which_cluster="primary")
    assert cb.check(group) == HealthResult.UNHEALTHY


def test_signal_strips_whitespace():
    conn = _Conn(next_row=("  UNHEALTHY  ",))
    group = _make_group(primary_conn=conn)
    cb = ExternalSignalCircuitBreaker(which_cluster="primary")
    assert cb.check(group) == HealthResult.UNHEALTHY


# ------------------------------------------------------------ SELECT uses group_id

def test_select_uses_primary_uuid_as_group_id():
    conn = _Conn(next_row=None)
    group = _make_group(primary_conn=conn)
    cb = ExternalSignalCircuitBreaker(which_cluster="primary")
    cb.check(group)
    # Find the SELECT execution and verify its args.
    selects = [
        (sql, args) for sql, args in conn.executed if sql.startswith("SELECT")
    ]
    assert len(selects) == 1
    _, args = selects[0]
    # The CB calls ``cur.execute(SQL, (uuid,))``; our mock records
    # ``*args`` so what we see here is ``((uuid,),)``.
    assert args == ((group.primary.uuid,),)


# ------------------------------------------------------------ error paths

def test_unrecognised_status_returns_healthy_and_warns(caplog):
    conn = _Conn(next_row=("MAYBE",))
    group = _make_group(primary_conn=conn)
    cb = ExternalSignalCircuitBreaker(which_cluster="primary")

    caplog.set_level(logging.WARNING, logger="psycopg.yb.circuit_breaker")
    result = cb.check(group)
    assert result == HealthResult.HEALTHY
    warnings = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert any("unrecognised" in m.lower() for m in warnings), warnings


def test_missing_table_creates_it_and_returns_healthy():
    """First check: CREATE succeeds, SELECT raises UndefinedTable.
    Handler creates the table and returns HEALTHY. Next tick works."""
    # Sequence:
    #   1. CREATE (succeeds) — via _setup_table
    #   2. SELECT (raises UndefinedTable) — table dropped between setup and select
    #   3. Handler recovers: rollback, mark table setup pending, re-run _setup_table
    conn = _Conn(next_row=None)

    # Set the exception to fire on the SELECT (second execute).
    # We do this by patching execute directly to track which call it is.
    original_execute = _Cursor.execute
    call_count = {"n": 0}

    def sequenced_execute(self, sql, *args, **kw):
        call_count["n"] += 1
        if call_count["n"] == 2 and sql.startswith("SELECT"):
            raise _undefined_table_exc()
        original_execute(self, sql, *args, **kw)

    _Cursor.execute = sequenced_execute
    try:
        group = _make_group(primary_conn=conn)
        cb = ExternalSignalCircuitBreaker(which_cluster="primary")
        assert cb.check(group) == HealthResult.HEALTHY
        assert conn.rollbacks >= 1
        # The recovery path re-ran _setup_table (a second CREATE).
        creates = [sql for sql, _ in conn.executed if sql.startswith("CREATE")]
        assert len(creates) >= 1
    finally:
        _Cursor.execute = original_execute


def test_transient_sql_error_returns_healthy(caplog):
    """Any SQL error other than UndefinedTable → return HEALTHY, log
    WARNING. This CB is a request channel; transient errors don't imply
    the cluster is unusable (that's the tracker-table CB's job)."""
    conn = _Conn(
        next_row=None,
        raise_on_execute=RuntimeError("network flapped"),
    )
    group = _make_group(primary_conn=conn)
    cb = ExternalSignalCircuitBreaker(which_cluster="primary")

    caplog.set_level(logging.WARNING, logger="psycopg.yb.circuit_breaker")
    # First call: exception fires on the CREATE (first execute). Then
    # the handler eats it and returns HEALTHY.
    result = cb.check(group)
    assert result == HealthResult.HEALTHY
    # Rollback attempted (best-effort).
    assert conn.rollbacks >= 1


# ------------------------------------------------------------ which_cluster gating

def test_which_cluster_selects_correct_state():
    primary_conn = _Conn(next_row=("HEALTHY",))
    secondary_conn = _Conn(next_row=("UNHEALTHY",))
    group = _make_group(
        primary_conn=primary_conn, secondary_conn=secondary_conn,
    )

    primary_cb = ExternalSignalCircuitBreaker(which_cluster="primary")
    secondary_cb = ExternalSignalCircuitBreaker(which_cluster="secondary")

    assert primary_cb.check(group) == HealthResult.HEALTHY
    assert secondary_cb.check(group) == HealthResult.UNHEALTHY

    # Each CB read only its own conn.
    assert primary_conn.executed and not any(
        args == (group.primary.uuid,) for _, args in secondary_conn.executed[:1]
    )


def test_which_cluster_validation():
    with pytest.raises(ValueError, match="primary.*secondary"):
        ExternalSignalCircuitBreaker(which_cluster="tertiary")


# ------------------------------------------------------------ integration
# (still a unit test — mocks the conn but exercises the full delegation
#  path from psycopg.yb.health.check_primary_cluster.)

def test_installed_as_primary_cb_flows_through_delegation():
    from psycopg.yb.health import check_primary_cluster

    conn = _Conn(next_row=("UNHEALTHY",))
    group = _make_group(primary_conn=conn)
    group.primary_circuit_breaker = ExternalSignalCircuitBreaker(
        which_cluster="primary",
    )
    assert check_primary_cluster(group) == HealthResult.UNHEALTHY
