"""
Pool integration helpers for xCluster failover.

``ConnectionPool`` / ``AsyncConnectionPool`` from ``psycopg-pool`` accept a
``check=`` callback that runs on every borrow. The contract (verified in
``psycopg_pool/pool.py`` ``_check_connection``): **raise to evict**, return
normally to keep. ``CLIENT_EXCEPTIONS = Exception``, so any exception class
works.

``xcluster_check`` looks at the conn's ``_yb_uuid`` tag and the current
``active_cluster(group)`` from the state module. If the conn belongs to a
cluster that is no longer the active one — including the "no viable cluster"
case where both CBs report UNHEALTHY — raise so the pool evicts it. The
pool then opens a fresh conn via the dispatcher (which either routes to the
new active cluster or itself raises ``NoViableClusterError``).

This check is **metadata-only** — it does not issue a SQL query, does not
touch the wire, and runs in constant time. The detection-window trade-off
this implies is documented in design doc §9: between probe ticks (or
before the probe has accumulated `maxUpdateFailuresAllowed + 1` failures),
the per-cluster status may still be HEALTHY even if the cluster is already
unreachable. During that window the pool will hand the app a stale conn
whose first operation will fail. **Apps using this check MUST set query
timeouts and expect transient operation failures during the detection
window.** Tuning failover speed is a matter of lowering
``yb-servers-refresh-interval`` and/or ``yb.failover.maxUpdateFailuresAllowed``,
not adding wire probes here.

**NoViableCluster distinguishability** (agent-K PR #10:81):
``psycopg_pool`` catches every exception raised from the ``check=``
callback (``CLIENT_EXCEPTIONS = Exception`` in pool.py:35), discards the
conn, backs off and retries until the pool's own timeout fires. A
``getconn()`` caller therefore sees ``psycopg_pool.PoolTimeout`` whether
the cluster was transiently stale or permanently in the no-viable
state, with no API-level way to tell them apart. We cannot change this
inside ``xcluster_check`` without either (a) making
``NoViableClusterError`` a ``BaseException`` subclass — a breaking
change for every app that catches ``OperationalError`` — or (b) adding
a subclass hook into psycopg_pool, which is a separate package. Two
affordances for the operator:

  * We stamp ``group.last_no_viable_ts`` the moment xcluster_check
    observes the no-viable state. Applications that catch
    ``PoolTimeout`` can inspect this timestamp (vs their own
    ``time.monotonic()`` from before the borrow) to distinguish
    "both CBs UNHEALTHY" from "cluster transient".
  * ``assert_viable_or_raise(group)`` below is the recommended pattern:
    call it immediately BEFORE ``pool.connection()`` to fail-fast with
    ``NoViableClusterError`` instead of waiting for the pool to time
    out when both CBs are already UNHEALTHY. Zero latency when both
    CBs are HEALTHY.

See:
  * docs/xcluster_failover_design.html §4 (dual-CB state machine)
  * docs/xcluster_failover_design.html §9
"""

# Copyright (C) 2026 Yugabyte

from __future__ import annotations

import logging

from .. import errors as e
from . import NoViableClusterError
from .registry import ClusterRegistry
from .state import active_cluster

logger = logging.getLogger(__name__)


def _decide_eviction(conn) -> tuple[bool, str, str, bool]:
    """Pure-metadata decision: should ``conn`` be evicted?

    Returns ``(should_evict, active_uuid, active_label, no_viable)``:
      * ``should_evict`` — True iff the conn should be dropped from the pool
      * ``active_uuid`` / ``active_label`` — describe the cluster the conn
        SHOULD have come from; empty strings when should_evict is False or
        when no_viable is True
      * ``no_viable`` — True iff both CBs report UNHEALTHY (§4 state
        machine row: no viable target). Callers translate this into
        ``NoViableClusterError`` instead of the plain routing error.

    DB-corruption fix: during ``group.dispatch_paused`` (drain in
    progress) the "active" cluster is the OUTGOING side (status hasn't
    flipped yet; the flip happens at the end of Phase 2). Pool conns on
    the outgoing cluster must be evicted during that window — not
    served to the app — otherwise they commit on the universe we're
    about to leave. Shubham's cycle 7 traced the index-corruption window
    back to a pooled conn that committed on B 60 s into the failback
    lag wait, then got asked to delete the same rows on A after routing
    flipped.
    """
    uuid = getattr(conn, "_yb_uuid", None)
    if uuid is None:
        return False, "", "", False
    group = ClusterRegistry.instance().get_failover_group_by_uuid(uuid)
    if group is None:
        return False, "", "", False
    which = active_cluster(group)
    if which is None:
        # Both CBs UNHEALTHY. Every conn is stale — evict regardless of
        # which cluster it came from — and signal to the caller.
        return True, "", "", True
    active_state = group.primary if which == "primary" else group.secondary
    # Normal (non-paused) path: evict iff the conn is on a different
    # cluster than the one the state machine routes to.
    if not getattr(group, "dispatch_paused", False):
        return uuid != active_state.uuid, active_state.uuid, which, False
    # Dispatch is paused — drain is in flight. agent-K PR #18:94
    # revisit: evict EVERY pool conn, not just outgoing-side ones.
    # Phase 2 is a lag wait on the INCOMING target; serving an idle
    # pool conn on the incoming side during that window means the app
    # commits on a target that hasn't caught up yet — exactly the
    # DB-25008 index-corruption window. Idle inactive-side conns are
    # common because they're only evicted on checkout; one left over
    # from a prior failover would get served through the full Phase 2
    # lag wait of the next drain. Any post-barrier replacement conn
    # will be routed via ``active_cluster`` once dispatch resumes.
    return True, active_state.uuid, which, False


def xcluster_check(conn) -> None:
    """``psycopg-pool`` ``check=`` callback. Pure metadata; no SQL.

    Evicts ``conn`` when it belongs to a cluster that is no longer the
    active side of its ``FailoverGroup``. In the "no viable target" state
    (both CBs UNHEALTHY), raises :class:`NoViableClusterError` — a
    subclass of ``OperationalError`` — so the pool can propagate a
    distinct error to the borrow caller.

    Closes the conn BEFORE raising so psycopg-pool's check-loop treats it
    as broken (transaction_status = UNKNOWN → the pool calls
    ``AddConnection`` for a replacement). Without this close, the check
    loop would evict-and-re-add the same stale conn forever.

    Non-smart-driver conns (no ``_yb_uuid``) and conns whose primary
    cluster has no FailoverGroup configured always pass through.
    """
    should_evict, active_uuid, active_label, no_viable = _decide_eviction(conn)
    if not should_evict:
        return
    logger.debug(
        "xcluster_check evicting conn: conn_uuid=%s, active=%s (%s), no_viable=%s",
        conn._yb_uuid, active_uuid or "—", active_label or "—", no_viable,
    )
    # Close BEFORE raising — see docstring for why.
    #
    # agent-K finding on PR #10: when the pool uses close_returns=True,
    # Connection.close() is rerouted to pool.putconn() which puts the
    # stale conn back in the pool. The check then raises, the pool
    # catches it and calls _putconn again — the stale conn ends up
    # queued twice and NEVER actually evicted. Call pgconn.finish()
    # directly to bypass the pool-aware close path.
    try:
        conn.pgconn.finish()
    except Exception:
        pass
    try:
        conn._closed = True
    except Exception:
        pass
    # agent-K PR #18:130 — pgconn.finish() bypasses Connection.close()
    # which normally calls ClusterRegistry.decrement(). Without an
    # explicit decrement the per-host counter leaks every time the
    # pool evicts a stale-cluster conn, which happens on every pool
    # check during a drain. Over many failovers the dispatcher's
    # least-loaded policy would starve the heaviest-leaked host.
    from .drain import _release_registry_count
    _release_registry_count(conn)
    if no_viable:
        _stamp_no_viable(conn)
        raise NoViableClusterError(
            f"xcluster_check: connection belongs to cluster {conn._yb_uuid!r}, "
            "but both primary and secondary clusters are UNHEALTHY — no "
            "viable target for new connections"
        )
    raise e.OperationalError(
        f"xcluster_check: connection belongs to cluster {conn._yb_uuid!r}, "
        f"but the active cluster is now {active_label} ({active_uuid!r})"
    )


def _stamp_no_viable(conn) -> None:
    """Record ``time.monotonic()`` on the conn's group as
    ``last_no_viable_ts`` so operators can distinguish a ``PoolTimeout``
    caused by no-viable from one caused by transient stale conns
    (agent-K PR #10:81)."""
    import time
    uuid = getattr(conn, "_yb_uuid", None)
    if not uuid:
        return
    group = ClusterRegistry.instance().get_failover_group_by_uuid(uuid)
    if group is None:
        return
    with group.lock:
        group.last_no_viable_ts = time.monotonic()


def assert_viable_or_raise(group) -> None:
    """Raise :class:`NoViableClusterError` immediately if both CBs on
    ``group`` report UNHEALTHY. Zero-cost no-op when at least one side is
    HEALTHY. Call this right before ``pool.connection()`` to fail-fast
    instead of waiting for the pool's ``check=`` loop to time out —
    ``psycopg_pool`` catches NoViableClusterError inside its loop and
    the borrow caller otherwise only sees ``PoolTimeout`` after the
    full pool timeout elapses (agent-K PR #10:81).

    Usage pattern::

        from psycopg.yb.pool import xcluster_check, assert_viable_or_raise

        pool = ConnectionPool(dsn, check=xcluster_check, ...)
        ...
        assert_viable_or_raise(failover_group)   # fail-fast if both UNHEALTHY
        with pool.connection() as conn:
            ...
    """
    import time
    if active_cluster(group) is not None:
        return
    with group.lock:
        group.last_no_viable_ts = time.monotonic()
    raise NoViableClusterError(
        f"xCluster: both primary and secondary clusters report UNHEALTHY — "
        f"no viable target for new connections "
        f"(primary_uuid={group.primary.uuid})"
    )


async def xcluster_check_async(conn) -> None:
    """Async sibling of :func:`xcluster_check`, for ``AsyncConnectionPool``.

    Same shape as the sync version: closes the conn (so the pool's check
    loop treats it as broken and triggers ``AddConnection``) and then
    raises. The close uses ``await conn.close()`` here so the wrapper's
    event-loop bookkeeping happens cleanly.
    """
    should_evict, active_uuid, active_label, no_viable = _decide_eviction(conn)
    if not should_evict:
        return
    logger.debug(
        "xcluster_check_async evicting conn: conn_uuid=%s, active=%s (%s), "
        "no_viable=%s",
        conn._yb_uuid, active_uuid or "—", active_label or "—", no_viable,
    )
    # agent-K finding: `await conn.close()` on an AsyncConnection
    # whose `_pool` was set by `_getconn_unchecked` routes through
    # `pool.putconn()` with close_returns=True, putting the stale
    # conn right back into the pool. Call pgconn.finish() directly
    # to bypass. Same workaround as the sync xcluster_check above
    # and as PR-H1's async-close fix in drain._safe_close.
    try:
        conn.pgconn.finish()
    except Exception:
        pass
    try:
        conn._closed = True
    except Exception:
        pass
    # agent-K PR #18:172 — mirror sync branch: decrement the per-host
    # registry counter manually since pgconn.finish() bypasses
    # Connection.close()'s registry teardown.
    from .drain import _release_registry_count
    _release_registry_count(conn)
    if no_viable:
        _stamp_no_viable(conn)
        raise NoViableClusterError(
            f"xcluster_check: connection belongs to cluster {conn._yb_uuid!r}, "
            "but both primary and secondary clusters are UNHEALTHY — no "
            "viable target for new connections"
        )
    raise e.OperationalError(
        f"xcluster_check: connection belongs to cluster {conn._yb_uuid!r}, "
        f"but the active cluster is now {active_label} ({active_uuid!r})"
    )
