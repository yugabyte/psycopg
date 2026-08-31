"""
Sample CircuitBreaker implementation: tracker-table probe.

This is a **reference implementation**, not driver-shipped code. The
psycopg-yugabytedb driver no longer ships a default CB — applications
choose their own health-detection strategy and attach it via
``bootstrap_failover_group``.

How this CB works (the tracker-table strategy):

  1. At first tick, create ``yb_cluster_health_tracker`` with N tablets
     (via ``SPLIT AT VALUES``). N is a constructor arg (default 9).
  2. Insert one row per tablet (row id chosen to land in that tablet's
     range; ``ON CONFLICT DO NOTHING`` makes it idempotent across apps).
  3. On each tick, run ``SELECT COUNT(*) FROM yb_cluster_health_tracker``
     — the aggregate forces YB to visit every tablet leader. Any
     unreachable leader fails the read.
  4. Count consecutive failures. After ``max_update_failures_allowed + 1``
     consecutive failures, return UNHEALTHY.
  5. Reset the failure counter after the same number of consecutive
     successes (symmetric hysteresis).

Usage:

    from psycopg.yb import bootstrap_failover_group
    from demo.samples.tracker_table_cb import TrackerTableCircuitBreaker

    group = bootstrap_failover_group(dsn)
    group.primary_circuit_breaker = TrackerTableCircuitBreaker(
        which_cluster="primary",
        tracker_table_tablets=9,
        max_update_failures_allowed=0,
    )
    group.secondary_circuit_breaker = TrackerTableCircuitBreaker(
        which_cluster="secondary",
        tracker_table_tablets=9,
        max_update_failures_allowed=0,
    )
    conn = psycopg.connect(dsn)

Exceptions explicitly NOT counted toward the failure threshold (these
indicate misconfiguration, not cluster unavailability):
  * ``InvalidPassword``, ``InvalidAuthorizationSpecification`` — auth
  * ``InsufficientPrivilege`` — the connecting user can't write
  * ``InvalidCatalogName`` — database doesn't exist
  * SSL/TLS handshake failures
  * ``UndefinedTable`` — tracker table was dropped externally; we
    re-create it and return HEALTHY (recovery, not failure)
"""

# Copyright (C) 2026 Yugabyte

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from psycopg import errors as e
from psycopg.yb.health import HealthResult

if TYPE_CHECKING:
    from psycopg.yb.registry import FailoverGroup

logger = logging.getLogger(__name__)


# Single coordinator UPDATE touches every row → routed to every tablet leader.
# If any leader is unreachable, the statement fails. That's exactly the
# signal we want.
# Bootstrap: single INSERT ... ON CONFLICT DO NOTHING seeds one row per
# tablet so every tablet has at least one row. Steady state health check:
# COUNT(*) forces YB to visit every tablet leader — if any leader is
# unreachable, the aggregate fails. A read is enough; we don't need to
# write on every tick (the bootstrap INSERT already exercised the write
# path once). This drops steady-state write pressure on the tracker table
# to zero.
_HEALTH_CHECK_SQL = (
    "SELECT COUNT(*) FROM yb_cluster_health_tracker"
)

# Bound the per-statement wait on the CB's control connection.
#
# Lower bound: must exceed the worst-case YB tablet-leader election
# window (Raft lease 2s + election ≈ 3s). Below that, stopping ONE
# node causes the UPDATE to time out mid-election → with threshold=1
# (max_update_failures_allowed=0) that would spuriously trip.
#
# Upper bound: also the detection latency for a REAL failure. When
# quorum is lost the UPDATE blocks until statement_timeout, then
# fails; that's the "2nd node stops → CB trips" wall-clock.
#
# 10 seconds: ~3× margin over the election window while keeping
# detection under ~10s.
_UPDATE_STATEMENT_TIMEOUT_MS = 10000


def _insert_sql(row_ids: list[int]) -> str:
    """Idempotent seed INSERT. Each id falls inside one of the tablet
    ranges defined by ``SPLIT AT VALUES``. ``ON CONFLICT DO NOTHING`` lets
    every app idempotently INSERT — first one wins."""
    rows = ",".join(f"({i}, NOW())" for i in row_ids)
    return (
        "INSERT INTO yb_cluster_health_tracker (id, last_updated) "
        f"VALUES {rows} ON CONFLICT (id) DO NOTHING"
    )


def _create_table_sql(tablets: int) -> str:
    """DDL for the tracker table. ``tablets`` controls how many tablets
    the table is sharded across."""
    if tablets < 1:
        tablets = 1
    splits = ", ".join(f"({i * 10})" for i in range(1, tablets))
    body = (
        "CREATE TABLE IF NOT EXISTS yb_cluster_health_tracker (\n"
        "    id INT,\n"
        "    last_updated TIMESTAMP,\n"
        "    PRIMARY KEY (id ASC)\n"
        ")"
    )
    if splits:
        body += f" SPLIT AT VALUES ({splits})"
    return body


def _row_ids_for_tablets(tablets: int) -> list[int]:
    """Pick one id per tablet so the INSERT seeds every tablet exactly
    once. Tablet boundaries are at multiples of 10; row id ``i*10 + 5``
    lands in the middle of the i'th tablet's range."""
    if tablets < 1:
        tablets = 1
    return [i * 10 + 5 for i in range(tablets)]


# SQLSTATE families that signal misconfiguration rather than cluster health.
# These never count toward the failure threshold — repeated misconfiguration
# bubbles up as WARNING logs instead.
_IGNORE_SQLSTATES = frozenset({
    "28000",  # InvalidAuthorizationSpecification
    "28P01",  # InvalidPassword
    "42501",  # InsufficientPrivilege
    "3D000",  # InvalidCatalogName (database does not exist)
    "08006",  # ConnectionFailure during SSL/TLS negotiation maps here
})


def _is_misconfig(exc: Exception) -> bool:
    diag = getattr(exc, "diag", None)
    if diag is None:
        return False
    code = getattr(diag, "sqlstate", None)
    if code is None:
        return False
    return code in _IGNORE_SQLSTATES


def _is_undefined_table(exc: Exception) -> bool:
    """True iff the exception is "tracker table doesn't exist" — i.e.
    ``42P01`` (UndefinedTable). SQLSTATE-based so tests can simulate."""
    if isinstance(exc, e.UndefinedTable):
        return True
    diag = getattr(exc, "diag", None)
    if diag is None:
        return False
    return getattr(diag, "sqlstate", None) == "42P01"


class TrackerTableCircuitBreaker:
    """Tracker-table health detector. See module docstring for the
    algorithm. Two instances per FailoverGroup (one per cluster).

    Instance state — only one ``check`` runs at a time per (group,
    which_cluster) pair (the probe is single-threaded), so no lock
    needed.
    """

    def __init__(
        self,
        which_cluster: str = "primary",
        tracker_table_tablets: int = 9,
        max_update_failures_allowed: int = 0,
    ) -> None:
        if which_cluster not in ("primary", "secondary"):
            raise ValueError(
                f"which_cluster must be 'primary' or 'secondary', "
                f"got {which_cluster!r}"
            )
        self.which_cluster = which_cluster
        self.tracker_table_tablets = max(1, tracker_table_tablets)
        self.max_update_failures_allowed = max(0, max_update_failures_allowed)
        # Both counters are reset on the OPPOSITE outcome — see _record_*.
        self._consecutive_failures = 0
        self._consecutive_successes = 0
        # Lazy table setup. The first UPDATE always sees `UndefinedTable`
        # if the table doesn't exist (or after a manual DROP), which
        # triggers re-creation; once we've seen one successful UPDATE we
        # know the table is there.
        self._table_setup_done = False
        # Last-known status this CB returned. Used to decide between
        # "report UNHEALTHY (we've crossed the threshold and haven't
        # recovered yet)" and "report HEALTHY (we're still under
        # threshold OR have recovered)".
        self._last_reported = HealthResult.HEALTHY
        # Wall-clock anchor for the current check() invocation. Reset at
        # the top of each check(); consulted by the trip/recovery log
        # lines so the operator sees CB detection latency inline.
        self._check_started_at: float = 0.0
        # ``time.monotonic()`` at the moment the CB last flipped a health
        # verdict. External observers (e.g. the demo's main thread that
        # watches ``group.*_status``) read this to compute how long the
        # drain window held after the CB's decision.
        self.last_trip_time: float = 0.0
        self.last_recovery_time: float = 0.0

    # ------------------------------------------------------------- public

    def _target_state(self, group: "FailoverGroup"):
        """Return the ``ClusterState`` this CB is assigned to poll."""
        return group.primary if self.which_cluster == "primary" else group.secondary

    def check(self, group: "FailoverGroup") -> HealthResult:
        state = self._target_state(group)
        if state is None:
            return HealthResult.UNHEALTHY

        # Anchor timestamp for this check() invocation. Reported in the
        # trip / recovery log lines so the operator can read the CB
        # detection latency directly off the log.
        self._check_started_at = time.monotonic()

        conn = state.control_sync
        if conn is None or conn.closed:
            # No usable control conn. Try to re-open via the registry
            # helper (iterates non-down candidates and marks-failed any
            # that refuse).
            from psycopg.yb.registry import ClusterRegistry
            conn = ClusterRegistry.instance()._ensure_control_sync(state)
            if conn is None:
                return self._record_failure(
                    f"no usable {self.which_cluster} control connection"
                )

        try:
            # Bound per-statement wait so quorum-loss is observed within
            # the timeout, not 1-2 minutes later.
            with conn.cursor() as cur:
                cur.execute(
                    f"SET statement_timeout = {_UPDATE_STATEMENT_TIMEOUT_MS}"
                )
            if not self._table_setup_done:
                self._setup_table(conn)
            with conn.cursor() as cur:
                cur.execute(_HEALTH_CHECK_SQL)
                row = cur.fetchone()
                # COUNT(*) always returns exactly one row; the count
                # itself is zero iff bootstrap's INSERT never ran. In
                # that case, re-seed so the next tick has something to
                # count (belt-and-braces — bootstrap should have done
                # this at first tick already).
                if row is not None and row[0] == 0:
                    logger.debug(
                        "tracker table has no rows; re-seeding "
                        "(%s_uuid=%s)", self.which_cluster, state.uuid,
                    )
                    self._seed_rows(conn)
            conn.commit()
        except Exception as exc:
            return self._handle_exception(group, conn, exc)

        return self._record_success()

    # ------------------------------------------------------------- table setup

    def _setup_table(self, conn) -> None:
        """Idempotent: CREATE TABLE IF NOT EXISTS + seed INSERTs."""
        with conn.cursor() as cur:
            cur.execute(_create_table_sql(self.tracker_table_tablets))
        self._seed_rows(conn)
        conn.commit()
        self._table_setup_done = True
        logger.info(
            "tracker table ready: tablets=%d", self.tracker_table_tablets
        )

    def _seed_rows(self, conn) -> None:
        row_ids = _row_ids_for_tablets(self.tracker_table_tablets)
        with conn.cursor() as cur:
            cur.execute(_insert_sql(row_ids))

    # ------------------------------------------------------------- decisions

    def _handle_exception(
        self, group: "FailoverGroup", conn, exc: Exception,
    ) -> HealthResult:
        """Decide whether ``exc`` counts as a failure, a recovery, or
        an ignored misconfiguration."""
        try:
            conn.rollback()
        except Exception:
            pass

        state = self._target_state(group)
        cluster_uuid = state.uuid if state is not None else "?"

        if _is_undefined_table(exc):
            # Table got dropped (or this is our first run on a fresh
            # database). Recreate; report HEALTHY (cluster is responsive).
            logger.info(
                "tracker table missing; recreating (%s_uuid=%s)",
                self.which_cluster, cluster_uuid,
            )
            try:
                self._table_setup_done = False
                self._setup_table(conn)
            except Exception:
                logger.warning(
                    "tracker table recreate failed; will retry next tick",
                    exc_info=True,
                )
                return self._record_failure("table recreate failed")
            return self._record_success()

        if _is_misconfig(exc):
            # Auth / TLS / permissions / db-doesn't-exist — config error,
            # not cluster failure. Log loud, don't count.
            logger.warning(
                "tracker-table health check failed due to misconfiguration "
                "(%s_uuid=%s); not counting toward failure threshold: %s",
                self.which_cluster, cluster_uuid, exc,
            )
            return self._last_reported

        return self._record_failure(f"health check failed: {exc}")

    def _record_success(self) -> HealthResult:
        self._consecutive_failures = 0
        self._consecutive_successes += 1
        if self._last_reported == HealthResult.UNHEALTHY:
            if self._consecutive_successes <= self.max_update_failures_allowed:
                logger.debug(
                    "%s recovering: %d/%d consecutive successes",
                    self.which_cluster,
                    self._consecutive_successes,
                    self.max_update_failures_allowed + 1,
                )
                return self._last_reported
            self.last_recovery_time = time.monotonic()
            logger.warning(
                "🟢 %s health check succeeded — CB recovering to HEALTHY",
                self.which_cluster,
            )
        self._last_reported = HealthResult.HEALTHY
        return HealthResult.HEALTHY

    def _record_failure(self, reason: str) -> HealthResult:
        self._consecutive_successes = 0
        self._consecutive_failures += 1
        threshold = self.max_update_failures_allowed + 1
        if self._consecutive_failures < threshold:
            logger.debug(
                "%s failure %d/%d (under threshold): %s",
                self.which_cluster, self._consecutive_failures, threshold, reason,
            )
            return self._last_reported
        if self._last_reported == HealthResult.HEALTHY:
            self.last_trip_time = time.monotonic()
            logger.warning(
                "🔴 %s health check failed — CB tripping to UNHEALTHY",
                self.which_cluster,
            )
        self._last_reported = HealthResult.UNHEALTHY
        return HealthResult.UNHEALTHY
