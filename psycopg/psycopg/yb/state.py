"""
xCluster active-cluster state machine.

``FailoverGroup`` holds two circuit breakers — one per cluster — that each
report HEALTHY or UNHEALTHY on every probe tick. ``active_cluster(group)``
combines those two signals into the routing decision the dispatcher and
pool.check both consult:

  primary   secondary   →   serving
  --------  ---------       -------
  HEALTHY   any             primary
  UNHEALTHY HEALTHY         secondary
  UNHEALTHY UNHEALTHY       None (no viable target — caller raises
                                  NoViableClusterError)

Reads both statuses under ``group.lock`` so the answer is atomic across a
concurrent probe write. Callers MUST NOT hold the lock across a network
I/O — see design doc §4.
"""

# Copyright (C) 2026 Yugabyte

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from .health import HealthResult

if TYPE_CHECKING:
    from .registry import FailoverGroup


def active_cluster(group: "FailoverGroup") -> Literal["primary", "secondary"] | None:
    """Return which cluster the dispatcher should route new connections to,
    based on the current primary and secondary CB statuses.

    Returns ``None`` iff BOTH clusters are UNHEALTHY. Callers translate that
    into ``NoViableClusterError``; there is no "best-effort try primary
    anyway" fallback in v1 (see design doc §7.3 Q2).
    """
    with group.lock:
        primary_healthy = group.primary_status == HealthResult.HEALTHY
        secondary_healthy = group.secondary_status == HealthResult.HEALTHY
    if primary_healthy:
        return "primary"
    if secondary_healthy:
        return "secondary"
    return None


def active_state(group: "FailoverGroup"):
    """Convenience — return the ``ClusterState`` object matching
    ``active_cluster(group)``, or ``None`` if no viable target."""
    which = active_cluster(group)
    if which is None:
        return None
    return group.primary if which == "primary" else group.secondary


def wait_for_dispatch(
    group: "FailoverGroup", timeout_s: "float | None" = None,
) -> bool:
    """Block until ``group.dispatch_paused`` clears.

    The dispatcher calls this immediately before ``active_cluster`` so
    new connects are held during a drain sequence (Phase E). The wait
    releases ``group.lock`` while sleeping and re-acquires on notify —
    the drain thread flips the flag and calls ``notify_all``.

    Returns ``True`` if the pause cleared (or was never set), ``False``
    if ``timeout_s`` elapsed with the pause still held. ``timeout_s=None``
    means wait indefinitely.
    """
    with group.lock:
        if not group.dispatch_paused:
            return True
        assert group.dispatch_paused_condition is not None
        return group.dispatch_paused_condition.wait_for(
            lambda: not group.dispatch_paused,
            timeout=timeout_s,
        )
