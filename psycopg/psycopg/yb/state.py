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

    SYNC path only. Callers on an asyncio event loop must use
    ``wait_for_dispatch_async`` instead — the ``threading.Condition.
    wait_for`` here would block the loop OS thread for the entire
    drain window (DB-24838 part 1).
    """
    with group.lock:
        if not group.dispatch_paused:
            return True
        assert group.dispatch_paused_condition is not None
        return group.dispatch_paused_condition.wait_for(
            lambda: not group.dispatch_paused,
            timeout=timeout_s,
        )


async def wait_for_dispatch_async(
    group: "FailoverGroup", timeout_s: "float | None" = None,
) -> bool:
    """Async sibling of :func:`wait_for_dispatch` (DB-24838 part 1).

    The sync variant blocks the calling OS thread via
    ``threading.Condition.wait_for``; on an asyncio event loop that
    means the whole loop stops — handlers, timers, pool timeouts —
    for the entire drain window (Shubham measured 129.93 s in one
    failback cycle). This variant yields to the event loop instead.

    Mechanism: under ``group.lock`` we check ``dispatch_paused``; if
    already clear, return immediately. Otherwise create a per-call
    ``asyncio.Event`` and register ``(running_loop, event)`` with
    ``group.async_dispatch_waiters``. ``resume_dispatch`` on the probe
    thread fans out to every registered waiter via
    ``loop.call_soon_threadsafe(event.set)``, so the Event is set from
    the waiter's own loop and the ``await event.wait()`` returns in
    that loop's natural step. Multiple concurrent event loops (common
    with per-worker loops) each wake up on their own thread.

    Returns ``True`` if the pause cleared (or was never set), ``False``
    if ``timeout_s`` elapsed with the pause still held. ``timeout_s=None``
    means wait indefinitely.
    """
    import asyncio

    loop = asyncio.get_running_loop()
    deadline: "float | None" = None
    if timeout_s is not None:
        deadline = loop.time() + timeout_s
    while True:
        with group.lock:
            if not group.dispatch_paused:
                # Mirrors wait_for_dispatch's sync `wait_for(lambda:
                # not dispatch_paused, ...)` recheck — a wake must
                # observe the predicate True, not merely that the
                # event fired (agent-K PR #14:126). Resume → pause
                # → resume races can set + clear the event between
                # the wait_for and the rebind; re-reading
                # dispatch_paused under the lock is the
                # authoritative check.
                return True
            event = asyncio.Event()
            waiter = (loop, event)
            group.async_dispatch_waiters.append(waiter)
        try:
            remaining: "float | None" = None
            if deadline is not None:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    return False
            if remaining is None:
                await event.wait()
            else:
                try:
                    await asyncio.wait_for(event.wait(), timeout=remaining)
                except asyncio.TimeoutError:
                    return False
        finally:
            # Remove ourselves if still present (not yet fired). If the
            # resume already cleared the list this is a no-op.
            with group.lock:
                try:
                    group.async_dispatch_waiters.remove(waiter)
                except ValueError:
                    pass
        # Loop around: re-check dispatch_paused under the lock. If it
        # was re-set (pause → resume → pause race) we re-register and
        # keep waiting; if it's clear now, we return True.
