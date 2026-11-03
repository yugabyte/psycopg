"""
Sample CircuitBreaker implementation: external SQL signal.

This is a **reference implementation**, not driver-shipped code. The
psycopg-yugabytedb driver no longer ships a default CB — applications
choose their own strategy and attach it via ``bootstrap_failover_group``.

This CB is the operator-controlled failover path: an operator writes
a row to a well-known signal table, the CB reads it on the next probe
tick, and the driver flips accordingly. Useful for planned
maintenance / drill-runs / integration tests where you want to trip
failover on demand rather than wait for a real cluster failure.

Signal-table schema (auto-created on first tick):

    CREATE TABLE yb_failover_signals (
        group_id      TEXT      NOT NULL,
        which_cluster TEXT      NOT NULL,   -- 'primary' | 'secondary'
        target_status TEXT      NOT NULL,   -- 'HEALTHY' | 'UNHEALTHY'
        reason        TEXT,
        ts            TIMESTAMP NOT NULL DEFAULT NOW(),
        PRIMARY KEY (group_id, which_cluster, ts DESC)
    );

Operator triggers a failover-to-secondary by marking primary UNHEALTHY:

    INSERT INTO yb_failover_signals
        (group_id, which_cluster, target_status, reason)
    VALUES
        ('<primary-uuid>', 'primary', 'UNHEALTHY', 'planned maintenance');

Failback-to-primary is the mirror: mark secondary UNHEALTHY (so the
dispatcher stops routing to it) and clear the primary signal, or write
primary HEALTHY and secondary HEALTHY once both clusters recover.

No row for a (group_id, which_cluster) pair means "no signal for that
slot, return HEALTHY" — the default state for fresh deployments.

Two instances per ``FailoverGroup`` (one per cluster). Each reads from
its own cluster's ``control_sync`` connection AND filters on its own
``which_cluster`` so a signal meant for one slot doesn't trip the
other. Operators who want the signal to be observed by both clusters
in a bidirectional-xCluster setup can rely on replication to carry
the row across — the ``which_cluster`` column keeps the filter correct
on either side.

Usage:

    from psycopg.yb import bootstrap_failover_group
    from demo.samples.external_signal_cb import ExternalSignalCircuitBreaker

    group = bootstrap_failover_group(dsn)
    group.primary_circuit_breaker = ExternalSignalCircuitBreaker(
        which_cluster="primary",
    )
    group.secondary_circuit_breaker = ExternalSignalCircuitBreaker(
        which_cluster="secondary",
    )
    conn = psycopg.connect(dsn)

Failure modes and their handling:

  * No usable control connection — return HEALTHY (no signal ⇒ no
    reason to fail over). Logged at WARNING.
  * Table missing (``UndefinedTable``) — create it on the fly and
    return HEALTHY.
  * Any other SQL error — return HEALTHY and log WARNING. The signal
    CB is a **request** channel, not a health-detection channel.
"""

# Copyright (C) 2026 Yugabyte

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from psycopg import errors as e
from psycopg.yb.health import HealthResult

if TYPE_CHECKING:
    from psycopg.yb.registry import FailoverGroup

logger = logging.getLogger(__name__)


_SIGNAL_TABLE_DDL = (
    "CREATE TABLE IF NOT EXISTS yb_failover_signals (\n"
    "    group_id      TEXT      NOT NULL,\n"
    # Which CB slot this signal is for: 'primary' or 'secondary'. A
    # signal written for 'primary' is read only by the primary slot's
    # CB — otherwise both CBs would see every row and e.g. a single
    # "primary UNHEALTHY" signal would also trip the secondary slot,
    # leaving both clusters UNHEALTHY (NoViableClusterError on connect).
    # agent-K PR #12:33.
    "    which_cluster TEXT      NOT NULL,\n"
    "    target_status TEXT      NOT NULL,\n"
    "    reason        TEXT,\n"
    "    ts            TIMESTAMP NOT NULL DEFAULT NOW(),\n"
    "    PRIMARY KEY (group_id, which_cluster, ts DESC)\n"
    ")"
)

_SIGNAL_SELECT_SQL = (
    "SELECT target_status FROM yb_failover_signals "
    "WHERE group_id = %s AND which_cluster = %s "
    "ORDER BY ts DESC LIMIT 1"
)


def _is_undefined_table(exc: Exception) -> bool:
    """True iff the exception is "signal table doesn't exist" — i.e.
    SQLSTATE ``42P01`` (UndefinedTable)."""
    if isinstance(exc, e.UndefinedTable):
        return True
    diag = getattr(exc, "diag", None)
    if diag is None:
        return False
    return getattr(diag, "sqlstate", None) == "42P01"


class ExternalSignalCircuitBreaker:
    """Operator-controlled failover CB. See module docstring for the
    signal-table schema and semantics."""

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
            from psycopg.yb.registry import ClusterRegistry
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
                cur.execute(
                    _SIGNAL_SELECT_SQL,
                    (group.primary.uuid, self.which_cluster),
                )
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
        """Idempotent: CREATE TABLE IF NOT EXISTS."""
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

        logger.warning(
            "ExternalSignalCircuitBreaker: signal check failed on %s "
            "cluster; assuming HEALTHY (primary_uuid=%s): %s",
            self.which_cluster, group.primary.uuid, exc,
        )
        return HealthResult.HEALTHY
