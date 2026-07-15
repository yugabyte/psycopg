"""
Circuit-breaker implementations for xCluster health detection.

The probe thread (``HealthProbe``) calls ``check_primary_cluster`` and
``check_secondary_cluster`` on each tick — one per cluster. Each function
delegates to the matching CB slot (``group.primary_circuit_breaker`` /
``group.secondary_circuit_breaker``), which is an instance of one of the
classes in this module. Keeping the
implementation in a swap-pluggable class (per Amogh's request — "circuit
breaker will be separate classes so we can always modify or replace
those with different logic if required") lets us:

  * Ship a real tracker-table-based detector now
    (``TrackerTableCircuitBreaker``)
  * Override with a different strategy from tests (any class implementing
    the ``CircuitBreaker`` protocol works)
  * Replace the implementation entirely without touching the dispatcher,
    the registry, or the probe thread

The tracker-table strategy is documented in the spec ("Circuit Breaker
(CB) — How it'll work"):

  1. At first probe, create ``yb_cluster_health_tracker`` with N tablets
     (via SPLIT AT VALUES). N is set by ``yb.failover.trackerTableTablets``
     (default 9).
  2. Insert one row per tablet (one row id chosen to fall in each tablet's
     range; ``ON CONFLICT DO NOTHING`` makes this idempotent across apps).
  3. On each subsequent tick, ``UPDATE yb_cluster_health_tracker SET
     last_updated = NOW()`` — touches all rows, which YB routes across
     every tablet leader. A leader being unreachable fails the UPDATE.
  4. Count consecutive failures. After ``maxUpdateFailuresAllowed + 1``
     consecutive failures, return UNHEALTHY.
  5. Reset the failure counter after the same number of consecutive
     successes — symmetric hysteresis. Amogh in the call: "whatever
     criteria we have for lower from primary to secondary, we should
     have the same criteria from secondary to primary."

Exceptions explicitly NOT counted toward the failure threshold (these
indicate misconfiguration, not cluster unavailability):

  * ``InvalidPassword``, ``InvalidAuthorizationSpecification`` —
    auth failures
  * SSL/TLS handshake failures
  * ``InsufficientPrivilege`` — the connecting user can't write
  * ``InvalidCatalogName`` — database doesn't exist
  * ``UndefinedTable`` — the tracker table was dropped externally; we
    re-create it on the next tick and return HEALTHY (this is recovery,
    not failure)
"""

# Copyright (C) 2026 Yugabyte

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Protocol

from .. import errors as e
from .health import HealthResult

if TYPE_CHECKING:
    from .registry import FailoverGroup

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------- protocol

class CircuitBreaker(Protocol):
    """The contract every circuit-breaker implementation must satisfy.

    A single instance is owned by one ``FailoverGroup``. The probe thread
    calls ``check(group)`` on each tick; the return value flows back into
    ``HealthProbe._maybe_apply`` which writes the new status (subject to
    cool-down) under ``group.lock``.

    Implementations MUST be thread-safe with respect to the probe thread
    only — the probe is the sole caller. But the implementation may
    maintain mutable state across calls (e.g. failure counters), so don't
    create one instance and share it across multiple FailoverGroups.
    """

    def check(self, group: "FailoverGroup") -> HealthResult: ...


# ----------------------------------------------------------------- stub

class AlwaysHealthyCircuitBreaker:
    """Original stub from Phase 3. Kept for tests that explicitly install
    it, and as the documented "what does an inert CB look like" example.

    Most tests don't need to install this — they monkeypatch
    ``check_primary_cluster`` / ``check_secondary_cluster`` directly. But
    the ``AlwaysHealthyCircuitBreaker``
    is useful when a test wants the *delegation* path to still run.
    """

    def check(self, group: "FailoverGroup") -> HealthResult:
        return HealthResult.HEALTHY


# ----------------------------------------------------------------- tracker table

# Single coordinator UPDATE touches every row → routed to every tablet leader.
# If any leader is unreachable, the statement fails. That's exactly the
# signal we want.
_UPDATE_SQL = "UPDATE yb_cluster_health_tracker SET last_updated = NOW()"

# Bound the per-statement wait on the CB's control connection. The probe
# reuses one long-lived conn across ticks; on that conn, a tracker-table
# UPDATE during quorum loss can block 1-2 minutes while the tserver
# retries unreachable tablet leaders (vs ~10s on a fresh conn that
# rediscovers the topology). 5 seconds is long enough to absorb normal
# leader-election blips on a healthy cluster and short enough that
# detection latency stays bounded by the spec's design target.
_UPDATE_STATEMENT_TIMEOUT_MS = 5000

# Each id falls inside one of the tablet ranges defined by `SPLIT AT VALUES`.
# `ON CONFLICT DO NOTHING` lets every app idempotently INSERT — first one
# wins, the rest are no-ops.
def _insert_sql(row_ids: list[int]) -> str:
    rows = ",".join(f"({i}, NOW())" for i in row_ids)
    return (
        "INSERT INTO yb_cluster_health_tracker (id, last_updated) "
        f"VALUES {rows} ON CONFLICT (id) DO NOTHING"
    )


def _create_table_sql(tablets: int) -> str:
    """DDL for the tracker table. `tablets` controls how many tablets the
    table is sharded across — `SPLIT AT VALUES ((10), (20), …)` creates
    `tablets` ranges so writes distribute across `tablets` tablet leaders.
    """
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
    once. Tablet boundaries are at multiples of 10; row id `i*10 + 5`
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
    """True iff the exception's SQLSTATE indicates a config error rather
    than a cluster availability problem. The dispatcher's auth/TLS
    handling already excludes these from the smart driver's failed-host
    counter; we mirror the same policy here."""
    diag = getattr(exc, "diag", None)
    if diag is None:
        return False
    code = getattr(diag, "sqlstate", None)
    if code is None:
        return False
    return code in _IGNORE_SQLSTATES


def _is_undefined_table(exc: Exception) -> bool:
    """True iff the exception is "tracker table doesn't exist" — i.e.
    ``42P01`` (UndefinedTable). We use SQLSTATE rather than `isinstance`
    so tests can simulate the condition without importing the exact
    psycopg exception class hierarchy."""
    if isinstance(exc, e.UndefinedTable):
        return True
    diag = getattr(exc, "diag", None)
    if diag is None:
        return False
    return getattr(diag, "sqlstate", None) == "42P01"


# ----------------------------------------------------------------- concrete


class TrackerTableCircuitBreaker:
    """Default xCluster health detector. Periodically updates a tracker
    table on the assigned cluster's control connection; UNHEALTHY iff
    ``max_update_failures_allowed + 1`` consecutive UPDATEs fail.

    Symmetric hysteresis: HEALTHY → UNHEALTHY requires N consecutive
    failures; UNHEALTHY → HEALTHY requires N consecutive successes.

    ``which_cluster`` selects which side of the FailoverGroup this
    instance probes — ``"primary"`` or ``"secondary"``. A group holds
    two instances (one per cluster) so both clusters' health is polled
    on every probe tick.

    Instance state — only one ``check`` runs at a time per (group,
    which_cluster) pair (the probe is single-threaded), so no lock
    needed.

    See module docstring for the full spec reference.
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

    # ------------------------------------------------------------- public

    def _target_state(self, group: "FailoverGroup"):
        """Return the ``ClusterState`` this CB is assigned to poll."""
        return group.primary if self.which_cluster == "primary" else group.secondary

    def check(self, group: "FailoverGroup") -> HealthResult:
        """Run the tracker-table UPDATE on the assigned cluster's control
        connection. Returns the new health status."""
        state = self._target_state(group)
        # If the assigned cluster isn't bootstrapped yet (e.g. it was
        # down at app startup), we can't run the check. Report UNHEALTHY.
        # Phase 9.2 will add a bootstrap retry here.
        if state is None:
            return HealthResult.UNHEALTHY

        conn = state.control_sync
        if conn is None or conn.closed:
            # No usable control conn. Try to re-open via the existing
            # registry helper (which iterates non-down candidates and
            # marks-failed any that refuse).
            from .registry import ClusterRegistry
            conn = ClusterRegistry.instance()._ensure_control_sync(state)
            if conn is None:
                return self._record_failure(
                    f"no usable {self.which_cluster} control connection"
                )

        try:
            # Bound per-statement wait so quorum-loss is observed
            # within the timeout, not 1-2 minutes later. SET (session-
            # scoped) is idempotent — cheap to re-issue every tick, and
            # it survives any later conn replacement transparently
            # because we set it again on the next acquire.
            with conn.cursor() as cur:
                cur.execute(
                    f"SET statement_timeout = {_UPDATE_STATEMENT_TIMEOUT_MS}"
                )
            if not self._table_setup_done:
                self._setup_table(conn)
            with conn.cursor() as cur:
                cur.execute(_UPDATE_SQL)
                # If the UPDATE hit no rows, the table exists but the
                # seed INSERT never ran — re-seed and re-check next tick.
                if cur.rowcount == 0:
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
        """Idempotent: CREATE TABLE IF NOT EXISTS + seed INSERTs. Runs
        once per CB instance unless ``_table_setup_done`` is cleared by
        an UndefinedTable error."""
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
        self, group: "FailoverGroup", conn, exc: Exception
    ) -> HealthResult:
        """Decide whether ``exc`` counts as a failure, a recovery, or
        an ignored misconfiguration."""
        # Rollback any half-applied state on the conn so subsequent ticks
        # don't see "transaction aborted" SQLSTATE 25P02 on every query.
        try:
            conn.rollback()
        except Exception:
            pass

        state = self._target_state(group)
        cluster_uuid = state.uuid if state is not None else "?"

        if _is_undefined_table(exc):
            # Table got dropped (or this is our first run on a fresh
            # database). Try to recreate; report HEALTHY for this tick
            # since the cluster itself is responsive.
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
                "tracker-table UPDATE failed due to misconfiguration "
                "(%s_uuid=%s); not counting toward failure threshold: %s",
                self.which_cluster, cluster_uuid, exc,
            )
            # Don't touch counters — return current reported status.
            return self._last_reported

        # Anything else — connection-level, quorum-loss, server-side
        # error — counts as a real failure.
        return self._record_failure(f"UPDATE failed: {exc}")

    def _record_success(self) -> HealthResult:
        self._consecutive_failures = 0
        self._consecutive_successes += 1
        # Symmetric hysteresis: we only RETURN healthy after the same
        # number of successes that would have triggered UNHEALTHY in the
        # other direction. Until we've seen enough successes, keep
        # reporting the previous state.
        if self._last_reported == HealthResult.UNHEALTHY:
            if self._consecutive_successes <= self.max_update_failures_allowed:
                logger.debug(
                    "%s recovering: %d/%d consecutive successes",
                    self.which_cluster,
                    self._consecutive_successes,
                    self.max_update_failures_allowed + 1,
                )
                return self._last_reported
            logger.info(
                "%s recovered: %d consecutive successes",
                self.which_cluster, self._consecutive_successes,
            )
        self._last_reported = HealthResult.HEALTHY
        return HealthResult.HEALTHY

    def _record_failure(self, reason: str) -> HealthResult:
        self._consecutive_successes = 0
        self._consecutive_failures += 1
        # Threshold is `max_update_failures_allowed + 1`: with the
        # default 0 a single failure trips the breaker; with 2 it takes
        # three consecutive failures.
        threshold = self.max_update_failures_allowed + 1
        if self._consecutive_failures < threshold:
            logger.debug(
                "%s failure %d/%d (under threshold): %s",
                self.which_cluster, self._consecutive_failures, threshold, reason,
            )
            return self._last_reported
        if self._last_reported == HealthResult.HEALTHY:
            logger.warning(
                "%s unhealthy: %d consecutive failures crossed "
                "threshold (last reason: %s)",
                self.which_cluster, self._consecutive_failures, reason,
            )
        self._last_reported = HealthResult.UNHEALTHY
        return HealthResult.UNHEALTHY


# ----------------------------------------------------------------- external signal

# Signal-table schema — operator writes the desired ``target_status``
# ('HEALTHY' or 'UNHEALTHY') for a group. The CB reads the latest row
# on every tick and returns the corresponding HealthResult.
_SIGNAL_TABLE_DDL = (
    "CREATE TABLE IF NOT EXISTS yb_failover_signals (\n"
    "    group_id     TEXT      NOT NULL,\n"
    "    target_status TEXT     NOT NULL,\n"
    "    reason       TEXT,\n"
    "    ts           TIMESTAMP NOT NULL DEFAULT NOW(),\n"
    "    PRIMARY KEY (group_id, ts DESC)\n"
    ")"
)

# Latest signal for a specific group, ordered by ts DESC.
_SIGNAL_SELECT_SQL = (
    "SELECT target_status FROM yb_failover_signals "
    "WHERE group_id = %s ORDER BY ts DESC LIMIT 1"
)


class ExternalSignalCircuitBreaker:
    """Operator-controlled failover via a SQL signal table.

    Reads the latest row from ``yb_failover_signals`` for
    ``group.primary.uuid`` on every probe tick. The operator writes a
    row to trigger a failover / failback:

    .. code-block:: sql

        INSERT INTO yb_failover_signals (group_id, target_status, reason)
        VALUES ('<primary-uuid>', 'UNHEALTHY', 'planned maintenance');

    ``target_status`` is case-insensitive and must be one of
    ``HEALTHY`` / ``UNHEALTHY``. No row means "no signal, return
    HEALTHY" — this is the default state for fresh deployments.

    Two instances per ``FailoverGroup`` (one per cluster). Each reads
    from its own cluster's ``control_sync`` connection. Operators who
    want the signal to be observed by both clusters should either write
    to both, or rely on xCluster bidirectional replication to carry the
    row across.

    Failure modes and their handling:

      * No usable control connection — return HEALTHY (no signal ⇒
        no reason to fail over). Logged at WARNING.
      * Table missing (``UndefinedTable``) — create it on the fly and
        return HEALTHY. Same recovery pattern as
        ``TrackerTableCircuitBreaker``.
      * Any other SQL error — return HEALTHY and log WARNING. The
        signal CB is a **request** channel, not a health-detection
        channel: transient DB errors don't imply the cluster is unusable
        (that's what the tracker-table CB is for).
    """

    def __init__(self, which_cluster: str = "primary") -> None:
        if which_cluster not in ("primary", "secondary"):
            raise ValueError(
                f"which_cluster must be 'primary' or 'secondary', "
                f"got {which_cluster!r}"
            )
        self.which_cluster = which_cluster
        self._table_setup_done = False

    def _target_state(self, group: "FailoverGroup"):
        return group.primary if self.which_cluster == "primary" else group.secondary

    def check(self, group: "FailoverGroup") -> HealthResult:
        state = self._target_state(group)
        if state is None:
            return HealthResult.HEALTHY

        conn = state.control_sync
        if conn is None or conn.closed:
            from .registry import ClusterRegistry
            conn = ClusterRegistry.instance()._ensure_control_sync(state)
            if conn is None:
                logger.warning(
                    "ExternalSignalCircuitBreaker: no usable control "
                    "connection for %s cluster; assuming HEALTHY "
                    "(primary_uuid=%s)",
                    self.which_cluster, group.primary.uuid,
                )
                return HealthResult.HEALTHY

        try:
            if not self._table_setup_done:
                self._setup_table(conn)
            with conn.cursor() as cur:
                cur.execute(_SIGNAL_SELECT_SQL, (group.primary.uuid,))
                row = cur.fetchone()
            conn.commit()
        except Exception as exc:
            return self._handle_exception(group, conn, exc)

        if row is None:
            return HealthResult.HEALTHY
        value = str(row[0]).strip().upper()
        try:
            return HealthResult[value]
        except KeyError:
            logger.warning(
                "ExternalSignalCircuitBreaker: unrecognised target_status "
                "%r (expected HEALTHY or UNHEALTHY); assuming HEALTHY "
                "(primary_uuid=%s)",
                value, group.primary.uuid,
            )
            return HealthResult.HEALTHY

    def _setup_table(self, conn) -> None:
        """Idempotent: CREATE TABLE IF NOT EXISTS. Runs once per CB
        instance unless ``_table_setup_done`` is cleared by an
        UndefinedTable error."""
        with conn.cursor() as cur:
            cur.execute(_SIGNAL_TABLE_DDL)
        conn.commit()
        self._table_setup_done = True
        logger.info(
            "ExternalSignalCircuitBreaker: signal table ready on %s cluster",
            self.which_cluster,
        )

    def _handle_exception(
        self, group: "FailoverGroup", conn, exc: Exception,
    ) -> HealthResult:
        # Rollback any half-applied state so subsequent ticks don't see
        # SQLSTATE 25P02 (transaction aborted) on every query.
        try:
            conn.rollback()
        except Exception:
            pass

        if _is_undefined_table(exc):
            logger.info(
                "ExternalSignalCircuitBreaker: signal table missing; "
                "creating (primary_uuid=%s)", group.primary.uuid,
            )
            try:
                self._table_setup_done = False
                self._setup_table(conn)
            except Exception:
                logger.warning(
                    "signal table create failed; will retry next tick",
                    exc_info=True,
                )
            return HealthResult.HEALTHY

        # Any other error: log and default to HEALTHY. This CB is a
        # request channel, not a health detector — transient errors
        # here should NOT trigger a failover.
        logger.warning(
            "ExternalSignalCircuitBreaker: signal check failed on %s "
            "cluster; assuming HEALTHY (primary_uuid=%s): %s",
            self.which_cluster, group.primary.uuid, exc,
        )
        return HealthResult.HEALTHY
