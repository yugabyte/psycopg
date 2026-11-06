"""
Cluster routing for the smart-driver ``connect`` path.

``Connection.connect`` / ``AsyncConnection.connect`` call into this module
so that the upstream connection files only carry a few YB lines:

* :func:`resolve` / :func:`aresolve` run before the per-host loop and
  return a :class:`Route`: the ``ClusterState`` to pick nodes from, the
  kwargs to connect with and, in xCluster mode, the ``FailoverGroup``,
  the active cluster name and the dispatch-barrier generation.
* :func:`track_or_reroute` / :func:`atrack_or_reroute` run after a node
  connect succeeds. In xCluster mode they register the conn for the drain
  and, if a drain armed while the TCP connect was in progress, close the
  conn and return a fresh :class:`Route` so the caller connects again.

Single-cluster mode (``xcluster_enabled`` False, the common case) bootstraps
the cluster by ``ClusterKey`` and keeps the lazy ``yb_servers()`` refresh,
since there is no probe thread.

xCluster mode:

* ``cleaned_kwargs`` gets the xCluster socket defaults (keepalives,
  ``tcp_user_timeout``) for keys the DSN does not set.
* The ``FailoverGroup`` is bootstrapped. The driver ships without a default
  circuit breaker: the app must attach both slots (``bootstrap_failover_group``
  plus explicit assignment) before any connect, so a missing slot raises
  ``MissingCircuitBreakerError`` instead of routing on a status flag no CB
  writes to.
* Barrier: if a drain sequence has paused dispatch, wait until it releases.
  There is no timeout in the direct-connect path; the wait is bounded by
  the Phase 1 drain plus the Phase 2 lag wait. With ``drainTimeoutSecs=-1``
  the Phase 1 bound is the longest in-flight transaction plus one drain-poll
  tick (DB-25012); with a finite ``drainTimeoutSecs`` it is that value. The
  async path must use the async wait: the sync one blocks the event-loop
  thread for the whole drain window, stalling every other coroutine
  (DB-24838 observed a 129.93 s stall on one failback).
* The wait returns the barrier generation read in the same lock section
  that saw the pause clear. Reading it in a second step would miss a drain
  armed in between (Review #10).
* The active cluster comes from the dual-CB state machine
  (:func:`psycopg.yb.state.active_cluster`); both UNHEALTHY raises
  ``NoViableClusterError``.
* The lazy on-demand topology refresh is skipped: per-cluster probe threads
  refresh ``yb_servers()`` on each tick (design doc §4.1) and are
  authoritative for topology in xCluster mode.

After connect (xCluster only): the conn is added to ``state.tracked_conns``
under ``state.lock`` (the drain iterates a snapshot taken under the same
lock), so the drain can find in-flight transactions on the outgoing
cluster (direct-connect weakref tracking, design doc §3.1). Then, if
:func:`psycopg.yb.state.barrier_crossed` reports that a drain armed after
the generation was read, the drain may already have swept
``tracked_conns`` and this conn would write to the outgoing cluster
(DB-24801): it is untracked, closed, and the caller routes again once the
barrier releases.

``psycopg.yb.state`` helpers are looked up at call time, not bound at
import, so they can be patched on the module.
"""

# Copyright (C) 2026 Yugabyte

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Literal
from dataclasses import dataclass

if TYPE_CHECKING:
    from ..abc import ConnDict
    from .params import YBParams
    from .registry import ClusterRegistry, ClusterState, FailoverGroup
    from ..connection import Connection
    from ..connection_async import AsyncConnection

# Same logger as the connect path in connection.py / connection_async.py.
logger = logging.getLogger("psycopg.yb.dispatcher")


@dataclass
class Route:
    """Where a smart-driver connect should go."""

    #: Cluster to pick nodes from.
    state: ClusterState
    #: Kwargs for the per-node connect (xCluster socket defaults applied).
    kwargs: ConnDict
    #: The failover group; None outside xCluster mode.
    group: FailoverGroup | None = None
    #: "primary" / "secondary" in xCluster mode, else None.
    cluster: Literal["primary", "secondary"] | None = None
    #: Dispatch-barrier generation read when the route was chosen.
    generation: int = 0


def _require_circuit_breakers(group: FailoverGroup) -> None:
    from . import MissingCircuitBreakerError

    if group.primary_circuit_breaker is None or group.secondary_circuit_breaker is None:
        raise MissingCircuitBreakerError(
            "Configuration error: 'yb.failover.secondaryClusterHosts' "
            "requires circuit breaker class to be configured. Exiting."
        )


def _route_active(
    group: FailoverGroup, kwargs: ConnDict, generation: int
) -> Route:
    from . import NoViableClusterError
    from . import state as _state

    which = _state.active_cluster(group)
    if which is None:
        raise NoViableClusterError(
            f"xCluster: both primary and secondary clusters report "
            f"UNHEALTHY — no viable target for new connections "
            f"(primary_uuid={group.primary.uuid})"
        )
    state = group.secondary if which == "secondary" else group.primary
    return Route(state, kwargs, group, which, generation)


def _log_active(route: Route) -> None:
    assert route.group is not None
    logger.debug(
        "xcluster active cluster: %s (primary_uuid=%s)",
        route.cluster,
        route.group.primary.uuid,
    )


def _track(route: Route, conn: Any, host: str) -> bool:
    """Track ``conn``; return True if it must be dropped and re-routed."""
    from . import state as _state

    assert route.group is not None
    state = route.state
    with state.lock:
        state.tracked_conns.add(conn)
    if not _state.barrier_crossed(route.group, route.generation):
        return False
    logger.info(
        "xCluster: drain barrier armed while connecting to %s "
        "(uuid=%s); closing and re-routing",
        host,
        state.uuid,
    )
    with state.lock:
        state.tracked_conns.discard(conn)
    return True


def resolve(
    registry: ClusterRegistry,
    yb_params: YBParams,
    conninfo: str,
    kwargs: ConnDict,
) -> Route:
    """Pick the cluster for a new connection. See the module docstring."""
    from .registry import ClusterKey
    from ..conninfo import conninfo_to_dict

    if not yb_params.xcluster_enabled:
        # ClusterKey wants a dict; parse the cleaned conninfo once for it.
        key = ClusterKey.from_params(conninfo_to_dict(conninfo, **kwargs))
        state = registry.get_or_bootstrap(key, conninfo, kwargs)
        registry.refresh_if_stale(state, yb_params.refresh_interval_s)
        return Route(state, kwargs)

    from . import state as _state
    from .params import with_xcluster_socket_defaults

    kwargs = with_xcluster_socket_defaults(conninfo, kwargs)
    group = registry.get_or_bootstrap_failover_group(yb_params, conninfo, kwargs)
    _require_circuit_breakers(group)
    generation = _state.wait_for_dispatch_generation(group)
    route = _route_active(group, kwargs, generation)
    _log_active(route)
    return route


async def aresolve(
    registry: ClusterRegistry,
    yb_params: YBParams,
    conninfo: str,
    kwargs: ConnDict,
) -> Route:
    """Async version of :func:`resolve`; the barrier wait yields to the loop."""
    from .registry import ClusterKey
    from ..conninfo import conninfo_to_dict

    if not yb_params.xcluster_enabled:
        # ClusterKey wants a dict; parse the cleaned conninfo once for it.
        key = ClusterKey.from_params(conninfo_to_dict(conninfo, **kwargs))
        state = await registry.aget_or_bootstrap(key, conninfo, kwargs)
        await registry.arefresh_if_stale(state, yb_params.refresh_interval_s)
        return Route(state, kwargs)

    from . import state as _state
    from .params import with_xcluster_socket_defaults

    kwargs = with_xcluster_socket_defaults(conninfo, kwargs)
    group = await registry.aget_or_bootstrap_failover_group(
        yb_params, conninfo, kwargs
    )
    _require_circuit_breakers(group)
    generation = await _state.wait_for_dispatch_generation_async(group)
    route = _route_active(group, kwargs, generation)
    _log_active(route)
    return route


def track_or_reroute(route: Route, conn: Connection[Any], host: str) -> Route | None:
    """Register a freshly connected ``conn`` with its cluster's drain.

    Return None if the conn can be handed to the caller. Return a new
    :class:`Route` if a drain armed while connecting: ``conn`` has been
    closed and the caller must connect again on the new route. No-op
    outside xCluster mode.
    """
    from . import state as _state

    if route.group is None or not _track(route, conn, host):
        return None
    conn.close()
    generation = _state.wait_for_dispatch_generation(route.group)
    return _route_active(route.group, route.kwargs, generation)


async def atrack_or_reroute(
    route: Route, conn: AsyncConnection[Any], host: str
) -> Route | None:
    """Async version of :func:`track_or_reroute`."""
    import asyncio

    from . import state as _state

    if route.group is None:
        return None
    # The drain closes the conn on this loop (_safe_close), so the close
    # can't interleave with a libpq call the loop is making on it.
    conn._yb_loop = asyncio.get_running_loop()  # type: ignore[attr-defined]
    if not _track(route, conn, host):
        return None
    await conn.close()
    generation = await _state.wait_for_dispatch_generation_async(route.group)
    return _route_active(route.group, route.kwargs, generation)
