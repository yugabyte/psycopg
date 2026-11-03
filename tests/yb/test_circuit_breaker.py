"""
Unit tests for the reference ``TrackerTableCircuitBreaker`` sample.

The class lives at ``demo/samples/tracker_table_cb.py`` — the driver
no longer ships a default CB (Amogh's ask: applications choose their
own strategy). These tests verify the sample still behaves correctly
so it stays a good starting point for adopters.

Verify the tracker-table-based circuit breaker:

  * Creates the table on first call (with N tablets via SPLIT AT VALUES)
  * Seeds N rows (one per tablet) with ON CONFLICT DO NOTHING
  * Runs UPDATE on each subsequent tick
  * Counts consecutive failures and trips UNHEALTHY at the threshold
  * Resets to HEALTHY after the same number of consecutive successes
    (symmetric hysteresis)
  * Filters auth/TLS/permission/db-not-found exceptions out of the
    failure threshold
  * Recreates the table on UndefinedTable rather than counting it
  * Returns UNHEALTHY when ``group.primary`` is None (Phase 9.2 hook)

The conn is mocked — we never hit a real database here.

Marked ``yb_unit`` explicitly since the filename ``test_circuit_breaker``
doesn't match the auto-marker prefixes.
"""

# Copyright (C) 2026 Yugabyte

from __future__ import annotations

import threading

import pytest

from psycopg import errors as e
from psycopg.yb.circuit_breaker import AlwaysHealthyCircuitBreaker
from psycopg.yb.health import HealthResult
from psycopg.yb.registry import FailoverGroup

from demo.samples.tracker_table_cb import (
    TrackerTableCircuitBreaker,
    _create_table_sql,
    _insert_sql,
    _row_ids_for_tablets,
)


pytestmark = pytest.mark.yb_unit


# ----------------------------------------------------------------- helpers


class _Cursor:
    """Records every executed SQL string against the parent _Conn's
    counter. The exception trigger lives on the conn so it spans all
    cursors (matches real psycopg conn behaviour where the connection
    knows about an in-flight error)."""

    def __init__(self, conn: "_Conn") -> None:
        self._conn = conn
        self.rowcount = conn._rowcount

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, sql, *args, **kw):
        self._conn._executes += 1
        if (
            self._conn._raise_at is not None
            and self._conn._executes == self._conn._raise_at
        ):
            assert self._conn._raise_exc is not None
            raise self._conn._raise_exc
        self._conn.executed.append(sql)


class _Conn:
    """Stand-in for psycopg.Connection. Tracks executed SQL, commits,
    rollbacks. ``raise_at=N`` raises ``raise_exc`` on the Nth execute()
    across ALL cursors (count is shared)."""

    def __init__(
        self,
        raise_at: int | None = None,
        raise_exc: Exception | None = None,
        rowcount: int = 5,
    ) -> None:
        self.executed: list[str] = []
        self.commits = 0
        self.rollbacks = 0
        self.closed = False
        self._raise_at = raise_at
        self._raise_exc = raise_exc
        self._rowcount = rowcount
        self._executes = 0          # shared counter across cursors

    def cursor(self):
        return _Cursor(self)

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1


def _make_state(uuid="P"):
    """Synthetic ClusterState; only the attributes the CB actually reads."""
    from psycopg.yb.registry import ClusterState
    return ClusterState(
        uuid=uuid, nodes={}, lock=threading.Lock(), last_refresh=0.0,
    )


def _make_group(primary_conn=None, secondary_conn=None):
    """Synthetic FailoverGroup with control_sync pre-populated so the CB
    doesn't try to re-bootstrap."""
    primary = _make_state("P")
    secondary = _make_state("S")
    primary.control_sync = primary_conn
    secondary.control_sync = secondary_conn
    return FailoverGroup(
        primary=primary, secondary=secondary,
        lock=threading.Lock(), primary_status=HealthResult.HEALTHY, secondary_status=HealthResult.HEALTHY,
    )


def _make_sqlstate_exc(sqlstate: str, cls=Exception) -> Exception:
    """Build an exception with a ``.diag.sqlstate`` attribute that mimics
    psycopg's diagnostics structure."""
    exc = cls(f"sqlstate {sqlstate}")
    diag = type("Diag", (), {"sqlstate": sqlstate})()
    exc.diag = diag
    return exc


# ----------------------------------------------------------------- SQL helpers


def test_create_table_sql_default_tablets():
    sql = _create_table_sql(9)
    assert "CREATE TABLE IF NOT EXISTS yb_cluster_health_tracker" in sql
    # 9 tablets → 8 split points at multiples of 10
    assert "(10), (20), (30), (40), (50), (60), (70), (80)" in sql


def test_create_table_sql_single_tablet_no_splits():
    sql = _create_table_sql(1)
    assert "SPLIT AT VALUES" not in sql


def test_create_table_sql_clamps_to_minimum():
    sql = _create_table_sql(0)
    assert "SPLIT AT VALUES" not in sql


def test_row_ids_distribute_across_tablets():
    # For 9 tablets, ids should be at 5, 15, 25, ..., 85 — each in its own range.
    ids = _row_ids_for_tablets(9)
    assert ids == [5, 15, 25, 35, 45, 55, 65, 75, 85]


def test_insert_sql_on_conflict_do_nothing():
    sql = _insert_sql([5, 15, 25])
    assert "ON CONFLICT (id) DO NOTHING" in sql
    # The exact comma-spacing isn't important; what matters is each id
    # appears in its own (id, NOW()) tuple.
    for i in (5, 15, 25):
        assert f"({i}, NOW())" in sql


# ----------------------------------------------------------------- always-healthy stub

def test_always_healthy_cb():
    cb = AlwaysHealthyCircuitBreaker()
    group = _make_group(primary_conn=_Conn())
    assert cb.check(group) == HealthResult.HEALTHY


# ----------------------------------------------------------------- first-tick setup

def test_first_check_creates_table_and_seeds_rows():
    cb = TrackerTableCircuitBreaker(tracker_table_tablets=3)
    conn = _Conn()
    group = _make_group(primary_conn=conn)
    assert cb.check(group) == HealthResult.HEALTHY
    # CREATE TABLE + INSERT + UPDATE all ran.
    assert any("CREATE TABLE IF NOT EXISTS" in s for s in conn.executed)
    assert any("INSERT INTO yb_cluster_health_tracker" in s for s in conn.executed)
    assert any(s.startswith("UPDATE yb_cluster_health_tracker") for s in conn.executed)
    assert cb._table_setup_done is True


def test_subsequent_check_skips_table_setup():
    cb = TrackerTableCircuitBreaker(tracker_table_tablets=3)
    conn = _Conn()
    group = _make_group(primary_conn=conn)
    cb.check(group)
    # Second call should NOT re-create or re-INSERT — only UPDATE.
    conn.executed.clear()
    cb.check(group)
    assert all("CREATE TABLE" not in s for s in conn.executed)
    assert all("INSERT INTO" not in s for s in conn.executed)
    assert any(s.startswith("UPDATE") for s in conn.executed)


# ----------------------------------------------------------------- failure threshold

def test_single_failure_with_threshold_zero_trips_unhealthy():
    """max_update_failures_allowed=0 → first failure trips UNHEALTHY."""
    cb = TrackerTableCircuitBreaker(max_update_failures_allowed=0)
    # First call: setup works; second call: simulate UPDATE failure.
    setup_conn = _Conn()
    cb.check(_make_group(primary_conn=setup_conn))
    # Now an UPDATE failure on the next tick:
    # _Cursor counts execute() calls; UPDATE is the 1st execute on the
    # second tick (no table setup needed).
    fail_conn = _Conn(raise_at=1, raise_exc=RuntimeError("connection lost"))
    group = _make_group(primary_conn=fail_conn)
    # Re-use the same CB to preserve _table_setup_done = True.
    assert cb.check(group) == HealthResult.UNHEALTHY
    assert cb._consecutive_failures == 1


def test_threshold_3_requires_three_failures():
    """max_update_failures_allowed=2 → threshold is 3 consecutive failures."""
    cb = TrackerTableCircuitBreaker(max_update_failures_allowed=2)
    # First tick: successful setup.
    cb.check(_make_group(primary_conn=_Conn()))

    # Each subsequent failure call uses a fresh fail_conn since _Conn's
    # rollback path is per-instance.
    def fail_once() -> HealthResult:
        return cb.check(_make_group(
            primary_conn=_Conn(raise_at=1, raise_exc=RuntimeError("boom"))
        ))

    assert fail_once() == HealthResult.HEALTHY    # 1/3 — under threshold
    assert fail_once() == HealthResult.HEALTHY    # 2/3 — under threshold
    assert fail_once() == HealthResult.UNHEALTHY  # 3/3 — trips


def test_success_resets_failure_counter():
    cb = TrackerTableCircuitBreaker(max_update_failures_allowed=2)
    cb.check(_make_group(primary_conn=_Conn()))  # setup
    cb.check(_make_group(primary_conn=_Conn(raise_at=1, raise_exc=RuntimeError())))
    cb.check(_make_group(primary_conn=_Conn(raise_at=1, raise_exc=RuntimeError())))
    assert cb._consecutive_failures == 2
    cb.check(_make_group(primary_conn=_Conn()))   # success
    assert cb._consecutive_failures == 0


# ----------------------------------------------------------------- recovery (symmetric hysteresis)

def test_symmetric_recovery_requires_same_count_of_successes():
    """After UNHEALTHY (3 failures), need 3 successes before reporting HEALTHY."""
    cb = TrackerTableCircuitBreaker(max_update_failures_allowed=2)
    # Setup + trip to UNHEALTHY (3 failures).
    cb.check(_make_group(primary_conn=_Conn()))
    for _ in range(3):
        cb.check(_make_group(primary_conn=_Conn(raise_at=1, raise_exc=RuntimeError())))
    assert cb._last_reported == HealthResult.UNHEALTHY

    # First two successes don't flip yet (1/3, 2/3).
    assert cb.check(_make_group(primary_conn=_Conn())) == HealthResult.UNHEALTHY
    assert cb.check(_make_group(primary_conn=_Conn())) == HealthResult.UNHEALTHY
    # Third success → HEALTHY (3/3).
    assert cb.check(_make_group(primary_conn=_Conn())) == HealthResult.HEALTHY


# ----------------------------------------------------------------- exception filtering

@pytest.mark.parametrize("sqlstate", ["28000", "28P01", "42501", "3D000", "08006"])
def test_misconfig_sqlstates_do_not_count(sqlstate):
    """Auth / TLS / permission / db-not-found never count toward threshold."""
    cb = TrackerTableCircuitBreaker(max_update_failures_allowed=0)
    cb.check(_make_group(primary_conn=_Conn()))  # setup
    initial_failures = cb._consecutive_failures
    exc = _make_sqlstate_exc(sqlstate)
    result = cb.check(_make_group(primary_conn=_Conn(raise_at=1, raise_exc=exc)))
    assert result == HealthResult.HEALTHY    # last_reported preserved
    assert cb._consecutive_failures == initial_failures   # counter untouched


def test_undefined_table_recreates_and_reports_healthy():
    """SQLSTATE 42P01 → recreate table, return HEALTHY, don't count as failure.

    psycopg's ``e.UndefinedTable`` is a real exception whose ``diag`` is a
    read-only property; we can't easily construct one with a custom
    SQLSTATE from outside libpq. The CB's ``_is_undefined_table`` falls
    back to checking ``.diag.sqlstate == "42P01"``, so a synthetic
    exception with that attribute hits the same code path.
    """
    cb = TrackerTableCircuitBreaker(tracker_table_tablets=3)
    cb.check(_make_group(primary_conn=_Conn()))

    recover_conn = _Conn(
        raise_at=1, raise_exc=_make_sqlstate_exc("42P01"),
    )
    result = cb.check(_make_group(primary_conn=recover_conn))
    assert result == HealthResult.HEALTHY
    assert any("CREATE TABLE" in s for s in recover_conn.executed)
    assert cb._consecutive_failures == 0


# ----------------------------------------------------------------- no primary case

def test_check_returns_unhealthy_when_primary_is_none():
    """Phase 9.2 hook — if primary isn't bootstrapped, we have no control
    conn to run the check on. Report UNHEALTHY so dispatcher routes to
    secondary."""
    cb = TrackerTableCircuitBreaker()
    group = _make_group()
    # Synthesize the "no primary" condition by clearing it.
    object.__setattr__(group, "primary", None)
    assert cb.check(group) == HealthResult.UNHEALTHY


# ----------------------------------------------------------------- no control conn

def test_check_returns_unhealthy_when_control_conn_missing():
    cb = TrackerTableCircuitBreaker()
    group = _make_group(primary_conn=None)  # no control_sync
    # _ensure_control_sync will be called and return None because the
    # synthetic state has no nodes. CB should report UNHEALTHY.
    assert cb.check(group) == HealthResult.UNHEALTHY


def test_closed_control_conn_triggers_reopen_attempt(monkeypatch):
    """If control_sync is closed, the CB asks the registry to re-open it.
    Verify the registry helper is called."""
    cb = TrackerTableCircuitBreaker()
    closed_conn = _Conn()
    closed_conn.closed = True
    group = _make_group(primary_conn=closed_conn)

    called = []
    def fake_ensure(state):
        called.append(state)
        return None   # simulate reopen failure
    from psycopg.yb.registry import ClusterRegistry
    monkeypatch.setattr(
        ClusterRegistry.instance(), "_ensure_control_sync", fake_ensure
    )
    assert cb.check(group) == HealthResult.UNHEALTHY
    assert called == [group.primary]


# ----------------------------------------------------------------- rollback on failure

def test_failure_triggers_rollback_to_clear_transaction_state():
    """If UPDATE raises mid-transaction, the conn ends up in an aborted
    state until rolled back. CB must rollback so subsequent ticks don't
    fail with SQLSTATE 25P02."""
    cb = TrackerTableCircuitBreaker()
    cb.check(_make_group(primary_conn=_Conn()))  # setup
    fail_conn = _Conn(raise_at=1, raise_exc=RuntimeError("boom"))
    cb.check(_make_group(primary_conn=fail_conn))
    assert fail_conn.rollbacks >= 1


# --------------------------- delegation via health.check_{primary,secondary}_cluster

def test_check_primary_cluster_delegates_to_primary_cb():
    from psycopg.yb.health import check_primary_cluster
    group = _make_group(primary_conn=_Conn())
    group.primary_circuit_breaker = AlwaysHealthyCircuitBreaker()
    assert check_primary_cluster(group) == HealthResult.HEALTHY


def test_check_primary_cluster_falls_back_to_healthy_when_cb_is_none():
    """No primary CB attached → return HEALTHY (preserves old stub contract)."""
    from psycopg.yb.health import check_primary_cluster
    group = _make_group()
    group.primary_circuit_breaker = None
    assert check_primary_cluster(group) == HealthResult.HEALTHY


def test_check_secondary_cluster_delegates_to_secondary_cb():
    from psycopg.yb.health import check_secondary_cluster
    group = _make_group()
    group.secondary_circuit_breaker = AlwaysHealthyCircuitBreaker()
    assert check_secondary_cluster(group) == HealthResult.HEALTHY


def test_check_secondary_cluster_falls_back_to_healthy_when_cb_is_none():
    from psycopg.yb.health import check_secondary_cluster
    group = _make_group()
    group.secondary_circuit_breaker = None
    assert check_secondary_cluster(group) == HealthResult.HEALTHY
