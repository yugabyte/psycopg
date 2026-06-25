"""
Pool integration helpers for xCluster failover.

``ConnectionPool`` / ``AsyncConnectionPool`` from ``psycopg-pool`` accept a
``check=`` callback that runs on every borrow. The contract (verified in
``psycopg_pool/pool.py`` ``_check_connection``): **raise to evict**, return
normally to keep. ``CLIENT_EXCEPTIONS = Exception``, so any exception class
works.

``xcluster_check`` looks at the conn's ``_yb_cluster`` / ``_yb_uuid`` tags
and the current ``FailoverGroup.status``. If the conn belongs to a cluster
that's no longer the active one, raise ``OperationalError`` so the pool
evicts it. The pool then opens a fresh conn via the dispatcher, which routes
it to the new active cluster.

This check is **metadata-only** — it does not issue a SQL query, does not
touch the wire, and runs in constant time. The detection-window trade-off
this implies is documented in design doc §9: between probe ticks (or
before the probe has accumulated `maxUpdateFailuresAllowed + 1` failures),
``group.status`` may still be HEALTHY even if the cluster is already
unreachable. During that window the pool will hand the app a stale conn
whose first operation will fail. **Apps using this check MUST set query
timeouts and expect transient operation failures during the detection
window.** Tuning failover speed is a matter of lowering
``yb-servers-refresh-interval`` and/or ``yb.failover.maxUpdateFailuresAllowed``,
not adding wire probes here.

See:
  * /tmp/xcluster_failover_design.html §9
  * Functional spec — "With a pool" subsection
"""

# Copyright (C) 2026 Yugabyte

from __future__ import annotations

import logging

from .. import errors as e
from .health import HealthResult
from .registry import ClusterRegistry

logger = logging.getLogger(__name__)


def _decide_eviction(conn) -> tuple[bool, str, str]:
    """Pure-metadata decision: should ``conn`` be evicted?

    Returns ``(should_evict, active_uuid, active_label)``. ``active_uuid``
    and ``active_label`` are only meaningful when ``should_evict`` is True
    (they describe the cluster the conn SHOULD have come from). When
    ``should_evict`` is False, both are empty strings.
    """
    uuid = getattr(conn, "_yb_uuid", None)
    if uuid is None:
        return False, "", ""
    group = ClusterRegistry.instance().get_failover_group_by_uuid(uuid)
    if group is None:
        return False, "", ""
    with group.lock:
        active_state = (
            group.secondary
            if group.status == HealthResult.UNHEALTHY
            else group.primary
        )
        active_uuid = active_state.uuid
        active_label = (
            "secondary" if active_state is group.secondary else "primary"
        )
    return uuid != active_uuid, active_uuid, active_label


def xcluster_check(conn) -> None:
    """``psycopg-pool`` ``check=`` callback. Pure metadata; no SQL.

    When ``conn`` belongs to a cluster that is no longer the active side
    of its ``FailoverGroup``, this function:

      1. **Closes the conn first.** psycopg-pool's check-loop puts a
         conn back into the pool on every raised exception EXCEPT when
         ``conn.pgconn.transaction_status == UNKNOWN`` (which it
         interprets as "conn is broken; replace via AddConnection").
         Closing the conn sets transaction_status to UNKNOWN, so the
         pool replaces it. Without this close, the check loop would
         loop forever evicting and re-adding the same stale conn.
      2. **Raises ``OperationalError``.** This is what tells the pool
         the conn shouldn't be served to the current borrow request.

    Non-smart-driver conns (no ``_yb_uuid``) and conns whose primary
    cluster has no FailoverGroup configured always pass through.
    """
    should_evict, active_uuid, active_label = _decide_eviction(conn)
    if not should_evict:
        return
    logger.debug(
        "xcluster_check evicting conn: conn_uuid=%s, active=%s (%s)",
        conn._yb_uuid, active_uuid, active_label,
    )
    # Close BEFORE raising — see docstring for why.
    try:
        conn.close()
    except Exception:
        # If close itself fails (e.g. conn was already closed),
        # transaction_status is still UNKNOWN, so the pool will still
        # replace it. Swallow.
        pass
    raise e.OperationalError(
        f"xcluster_check: connection belongs to cluster {conn._yb_uuid!r}, "
        f"but the active cluster is now {active_label} ({active_uuid!r})"
    )


async def xcluster_check_async(conn) -> None:
    """Async sibling of :func:`xcluster_check`, for ``AsyncConnectionPool``.

    Same shape as the sync version: closes the conn (so the pool's check
    loop treats it as broken and triggers ``AddConnection``) and then
    raises. The close uses ``await conn.close()`` here so the wrapper's
    event-loop bookkeeping happens cleanly.
    """
    should_evict, active_uuid, active_label = _decide_eviction(conn)
    if not should_evict:
        return
    logger.debug(
        "xcluster_check_async evicting conn: conn_uuid=%s, active=%s (%s)",
        conn._yb_uuid, active_uuid, active_label,
    )
    try:
        await conn.close()
    except Exception:
        pass
    raise e.OperationalError(
        f"xcluster_check: connection belongs to cluster {conn._yb_uuid!r}, "
        f"but the active cluster is now {active_label} ({active_uuid!r})"
    )
