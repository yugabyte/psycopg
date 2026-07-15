"""
YugabyteDB smart-driver subpackage.

All smart-driver behaviour (cluster discovery, connection-count tracking,
least-loaded picking, topology filtering) lives under this subpackage. Edits
to upstream files (connection_async.py, _conninfo_attempts_async.py) only
call into here — they don't carry smart-driver logic themselves.

This keeps the merge-conflict surface small on each upstream rebase.

Logging
-------

Every smart-driver module uses ``logging.getLogger(__name__)``, so the
loggers form a tree rooted at ``psycopg.yb``:

  * ``psycopg.yb.registry``           — bootstrap, refresh, counters, control conn,
                                         FailoverGroup pairing + operator API
  * ``psycopg.yb.discovery``          — yb_servers() query
  * ``psycopg.yb.policy.base``        — eligibility filter
  * ``psycopg.yb.policy.cluster_aware``  — least-loaded pick, tie-break
  * ``psycopg.yb.policy.topology_aware`` — placement filter
  * ``psycopg.yb.health``             — check_primary_cluster / check_secondary_cluster
                                         delegating helpers + HealthResult enum
  * ``psycopg.yb.health_probe``       — per-cluster background probe threads
                                         (INFO on transitions, DEBUG per tick,
                                         WARNING on probe exceptions or check
                                         timeouts)
  * ``psycopg.yb.circuit_breaker``    — pluggable CB implementations
                                         (TrackerTable, AlwaysHealthy,
                                         ExternalSignal)
  * ``psycopg.yb.pool``               — pool ``check=`` callback that evicts
                                         conns belonging to the inactive cluster
                                         (raises NoViableClusterError when both
                                         CBs are UNHEALTHY)
  * ``psycopg.yb.dispatcher``         — connect-time branch on
                                         ``active_cluster(group)`` to route to
                                         primary or secondary cluster
  * ``psycopg.yb.state``              — ``active_cluster(group)`` state machine
                                         combining primary + secondary CB
                                         statuses into the routing decision
  * ``psycopg.yb.drain``              — barrier-with-timeout drain sequence
                                         (INFO on drain start/complete,
                                         WARNING on force-close survivors)
  * ``psycopg.yb``                    — bootstrap-time warnings (e.g. server
                                         ``statement_timeout`` misalignment)

Tune verbosity per subsystem, or set the parent ``psycopg.yb`` once:

    import logging
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("psycopg.yb").setLevel(logging.DEBUG)

The subpackage also registers a custom ``TRACE`` level (numeric 5, below
``DEBUG``). Use it via the standard ``logger.log(TRACE, …)`` API, or via
the ``logger.trace(…)`` helper we install on ``logging.Logger``:

    from psycopg.yb import TRACE
    logging.getLogger("psycopg.yb").setLevel(TRACE)

Level usage convention inside the subpackage:

  * WARNING — driver had to give up (all candidates refused, …)
  * INFO    — lifecycle events worth seeing by default (cluster bootstrap,
              host quarantined, control conn moved, topology change observed)
  * DEBUG   — per-operation breadcrumbs (each pick, each refresh result,
              each control-conn open/close)
  * TRACE   — very verbose (every counter mutation, every filtered candidate)
"""

# Copyright (C) 2026 Yugabyte

import logging as _logging

from ..errors import OperationalError as _OperationalError


# Custom log level finer-grained than DEBUG. Standard library levels are
# CRITICAL=50, ERROR=40, WARNING=30, INFO=20, DEBUG=10, NOTSET=0; we slot
# in at 5. Re-registering the name is a no-op if it's already present.
TRACE = 5
_logging.addLevelName(TRACE, "TRACE")


class NoViableClusterError(_OperationalError):
    """Raised when both primary and secondary circuit breakers report
    UNHEALTHY — no cluster is a viable target for new connections.

    Design doc §4 state-machine row: (primary UNHEALTHY, secondary UNHEALTHY)
    → serving None. Callers (dispatcher, pool.check) translate that into
    this exception. Existing conns are not affected — this only refuses
    new opens. Subclass of ``OperationalError`` so applications' generic
    "retry on OperationalError" handling picks it up.
    """


# Install `logger.trace(...)` as a convenience so call sites read like the
# other levels. We only install if no other library beat us to it, so we
# don't clobber a third-party `trace` method that might use it differently.
if not hasattr(_logging.Logger, "trace"):
    def _trace(self, message, *args, **kwargs):  # type: ignore[no-redef]
        if self.isEnabledFor(TRACE):
            self._log(TRACE, message, args, **kwargs)
    _logging.Logger.trace = _trace  # type: ignore[attr-defined]

# NOTE: _logging is retained module-level — the Phase G bootstrap warning
# helper further down uses it. Keep the leading `_` so `from psycopg.yb
# import *` doesn't leak it.


# --------------------------------------------------------------------- public
# xCluster failover bootstrap helpers.
#
# Applications typically get a FailoverGroup as a side-effect of the first
# ``psycopg.connect(DSN)``. That works, but hides the group inside a user
# connection that has to be opened + closed just to reach the registry.
# The helpers below let the caller bootstrap explicitly at startup — the
# canonical use case being "install a custom CircuitBreaker before any
# application traffic starts flowing through the driver."


def bootstrap_failover_group(conninfo: str = "", **kwargs):
    """Bootstrap the xCluster ``FailoverGroup`` for ``conninfo``.

    Opens one control connection under the hood (owned by the group,
    never returned to the caller) and starts the background probe.
    Returns the ``FailoverGroup`` so the caller can, e.g., attach a
    custom ``circuit_breaker`` before any application connections use it.

    Subsequent ``psycopg.connect`` calls with a DSN that resolves to the
    same ``ClusterKey`` (same host set, port, dbname, user) reuse this
    same group — including whatever ``circuit_breaker`` you installed.

    Also emits a WARNING at bootstrap if the server's ``statement_timeout``
    is unbounded or larger than ``drainTimeoutSecs`` — because socket
    close doesn't cancel the server-side query on YugabyteDB today
    (see design doc §3.4 + issues #28983, #29379).

    Raises ``ValueError`` if the DSN doesn't opt into xCluster
    (``yb.failover.secondaryClusterHosts`` must be set, and
    ``load_balance_hosts`` must be enabled — same conditions
    ``psycopg.connect`` would use to decide xCluster is active).
    """
    from .params import extract_yb_params
    from .registry import ClusterRegistry

    yb_params, cleaned_conninfo, cleaned_kwargs = extract_yb_params(
        conninfo, kwargs
    )
    if not yb_params.xcluster_enabled:
        raise ValueError(
            "bootstrap_failover_group requires an xCluster DSN: "
            "yb.failover.secondaryClusterHosts must be set AND "
            "load_balance_hosts must be true"
        )
    group = ClusterRegistry.instance().get_or_bootstrap_failover_group(
        yb_params, cleaned_conninfo, cleaned_kwargs
    )
    _warn_statement_timeout_sync(group, yb_params.drain_timeout_s)
    return group


async def abootstrap_failover_group(conninfo: str = "", **kwargs):
    """Async sibling of :func:`bootstrap_failover_group`. Uses the async
    bootstrap path (``aget_or_bootstrap_failover_group``) so the control
    connection is opened via ``AsyncConnection``. Behaviour, guarantees,
    and error conditions match the sync form."""
    from .params import extract_yb_params
    from .registry import ClusterRegistry

    yb_params, cleaned_conninfo, cleaned_kwargs = extract_yb_params(
        conninfo, kwargs
    )
    if not yb_params.xcluster_enabled:
        raise ValueError(
            "abootstrap_failover_group requires an xCluster DSN: "
            "yb.failover.secondaryClusterHosts must be set AND "
            "load_balance_hosts must be true"
        )
    group = await ClusterRegistry.instance().aget_or_bootstrap_failover_group(
        yb_params, cleaned_conninfo, cleaned_kwargs
    )
    await _awarn_statement_timeout_async(group, yb_params.drain_timeout_s)
    return group


# --- statement_timeout coordination (Phase G / design doc §3.4)
#
# YugabyteDB does not cancel a running query when its client socket
# closes today (see yugabyte-db#28983, #29379). Consequence: the drain
# force-close at drainTimeoutSecs deadline does NOT actually stop
# server-side execution — the backend keeps running until it finishes
# or `statement_timeout` fires. For the drain to hold the invariant in
# §3, `statement_timeout` must be no larger than the drain window.
#
# The driver does NOT modify the setting (§7.3 Q7: option (b) — warn,
# don't enforce). It reads once at bootstrap and logs a WARNING when
# alignment is off. Ops are expected to set `statement_timeout` at DSN
# or role level.

_bootstrap_logger = _logging.getLogger("psycopg.yb")


def _should_warn_statement_timeout(
    statement_timeout_ms: int, drain_timeout_s: int
) -> "tuple[bool, str]":
    """Return ``(should_warn, reason)`` based on the alignment between
    the server's ``statement_timeout`` and the driver's
    ``drainTimeoutSecs``.

    Rules:
      * drainTimeoutSecs == -1 (wait forever): no warning — we never
        force-close, so statement_timeout is decoupled from the drain.
      * statement_timeout == 0 (unbounded): WARN — a long query could
        outlive the drain window on any positive drainTimeoutSecs.
      * statement_timeout > drainTimeoutSecs*1000: WARN — same reason.
      * Otherwise: no warning.
    """
    if drain_timeout_s == -1:
        return False, ""
    if statement_timeout_ms == 0:
        return (
            True,
            "statement_timeout is unbounded (0), so any in-flight query "
            f"could outlive drainTimeoutSecs={drain_timeout_s}s.",
        )
    drain_ms = drain_timeout_s * 1000
    if statement_timeout_ms > drain_ms:
        return (
            True,
            f"statement_timeout={statement_timeout_ms}ms exceeds "
            f"drainTimeoutSecs={drain_timeout_s}s ({drain_ms}ms).",
        )
    return False, ""


def _emit_statement_timeout_warning(reason: str) -> None:
    _bootstrap_logger.warning(
        "xCluster: %s Server-side query cancellation on socket close "
        "does not fire on YugabyteDB (see yugabyte-db#28983, #29379). "
        "Set statement_timeout at the DSN or role level to a value "
        "≤ drainTimeoutSecs*1000 (ms) so any in-flight query is "
        "bounded by the drain window.",
        reason,
    )


def _read_statement_timeout_ms_sync(conn) -> "int | None":
    """SELECT statement_timeout from pg_settings on the given sync conn.
    Returns the value in milliseconds (0 = unbounded), or ``None`` if
    the read failed (permission, disconnect, etc.). Best-effort: never
    raises."""
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT setting::int FROM pg_settings "
                "WHERE name = 'statement_timeout'"
            )
            row = cur.fetchone()
        conn.commit()
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass
        _bootstrap_logger.debug(
            "statement_timeout read failed; skipping bootstrap warning",
            exc_info=True,
        )
        return None
    if row is None:
        return None
    try:
        return int(row[0])
    except (TypeError, ValueError):
        return None


def _warn_statement_timeout_sync(group, drain_timeout_s: int) -> None:
    """Sync check called at the tail of ``bootstrap_failover_group``."""
    conn = group.primary.control_sync
    if conn is None or conn.closed:
        _bootstrap_logger.debug(
            "no sync control conn available; skipping statement_timeout warn"
        )
        return
    st_ms = _read_statement_timeout_ms_sync(conn)
    if st_ms is None:
        return
    warn, reason = _should_warn_statement_timeout(st_ms, drain_timeout_s)
    if warn:
        _emit_statement_timeout_warning(reason)


async def _awarn_statement_timeout_async(group, drain_timeout_s: int) -> None:
    """Async check called at the tail of ``abootstrap_failover_group``.
    Uses ``group.primary.control_async`` if available; otherwise falls
    through to the sync path if a sync control conn also exists."""
    aconn = group.primary.control_async
    if aconn is not None and not aconn.closed:
        try:
            async with aconn.cursor() as cur:
                await cur.execute(
                    "SELECT setting::int FROM pg_settings "
                    "WHERE name = 'statement_timeout'"
                )
                row = await cur.fetchone()
            await aconn.commit()
        except Exception:
            try:
                await aconn.rollback()
            except Exception:
                pass
            _bootstrap_logger.debug(
                "statement_timeout read failed (async); skipping warn",
                exc_info=True,
            )
            return
        if row is None:
            return
        try:
            st_ms = int(row[0])
        except (TypeError, ValueError):
            return
        warn, reason = _should_warn_statement_timeout(st_ms, drain_timeout_s)
        if warn:
            _emit_statement_timeout_warning(reason)
        return
    # No async control conn — try sync fallback.
    _warn_statement_timeout_sync(group, drain_timeout_s)


__all__ = [
    "TRACE",
    "NoViableClusterError",
    "bootstrap_failover_group",
    "abootstrap_failover_group",
]
