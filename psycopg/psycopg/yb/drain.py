"""
Barrier-with-timeout drain for xCluster failover (design doc §3).

``trigger_drain`` is the entry point every probe transition goes through.
It composes:

  1. **Pause dispatch.** Under ``group.lock``, set
     ``group.dispatch_paused = True``. New connect calls block on the
     Condition until step 4 releases them.
  2. **Poll in-flight transactions on the OUTGOING cluster.** Walk
     ``outgoing.tracked_conns`` (weakref-backed) and count conns with
     ``conn.info.transaction_status`` not in ``(IDLE, UNKNOWN)`` — those
     are "in a transaction" and cannot be safely severed.
  3. **Close every tracked conn on the outgoing cluster at the flip
     boundary (parallel with hard-fd escape hatch).** The poll loop
     exits as soon as all in-flight transactions complete, so natural
     drain stays the fast path; whatever conns remain in the tracked
     set at that point (idle + any still-in-flight if we hit the
     deadline) are force-closed in a single parallel pass. Closing
     idle conns too is required by design doc §3.3 ("no dual-write
     within a single application") — IDLE is a transient state
     between two commits on a persistent direct conn, and leaving one
     alive after the routing flip means the next cursor call commits
     on the outgoing universe (DB-24801). ``AsyncConnection.close``
     is a coroutine and the sync executor cannot ``await`` it, so
     ``_safe_close`` type-dispatches and calls ``pgconn.finish()``
     for async conns (DB-24960). Close is bounded by
     ``_KILL_AT_TIMEOUT_BUDGET_S`` (default 2 s); any conn whose
     close hasn't returned within the budget gets its socket fd
     hard-closed via ``os.close`` — a stuck TCP ``send()`` in libpq's
     ``PQfinish`` could otherwise block for ``tcp_retries2`` (~15 min
     on Linux, worse with TCP keepalive disabled). Bounds total drain
     wall-clock at ``drain_timeout_s + _KILL_AT_TIMEOUT_BUDGET_S``
     regardless of TCP state. The application sees
     ``psycopg.OperationalError`` on next use of a closed conn and
     treats it like any transient disconnect. Sentinels on
     ``drain_timeout_s``:
       * ``-1`` — wait indefinitely for in-flight to drain; never
         force-close in-flight ones. Idle ones are still closed at
         the flip boundary (natural-drain path).
       * ``0`` — close immediately; no drain window.
       * ``N > 0`` — poll for N seconds, then close every survivor.
  4. **Apply the status change and resume dispatch.** Under
     ``group.lock``, write the per-cluster status field + transition
     timestamp, then set ``dispatch_paused = False`` and notify all
     waiters.

The dispatcher-transition helper (``_maybe_apply`` in the probe) calls
this for every eligible per-cluster CB transition — but the drain only
performs the barrier + close when the routing target actually changes.
For "same active cluster" transitions (e.g. secondary flipping while
primary is HEALTHY), the drain is a no-op wrapper around the status
write.

See design doc §3.2 for the pseudocode.
"""

# Copyright (C) 2026 Yugabyte

from __future__ import annotations

import concurrent.futures
import logging
import os
import time
import weakref
from typing import TYPE_CHECKING

from . import TRACE
from .health import HealthResult

if TYPE_CHECKING:
    from .registry import ClusterState, FailoverGroup

logger = logging.getLogger(__name__)


class Phase2TimeoutError(Exception):
    """Raised by a FAILBACK Phase 2 lag wait when it times out and the
    group's ``on_phase2_timeout`` is ``"fail"`` (opt-in; the default
    is ``"proceed"``).

    Caught by ``_drain_and_apply``, which then SKIPS the status
    transition (does NOT flip routing) and resumes dispatch.
    """

# Minimum time between closing the outgoing side's conns and resuming
# dispatch onto the incoming side (see the close fence in
# _drain_and_apply). Covers an owner that polls its conn once a second.
_CLOSE_FENCE_S = 1.0

# Wall-clock budget for the parallel force-close phase — separate from
# drainTimeoutSecs (which is the grace window for natural drain).
# Rationale: libpq's PQfinish calls send() to push a Terminate byte
# before closing the socket. On a wedged TCP session (peer unresponsive
# or partitioned, keepalive disabled) that send() can block until the
# OS's tcp_retries2 gives up — ~15 min on Linux default. A serial
# for-loop of c.close() would therefore hang drain past its own
# deadline. This budget bounds the parallel close phase; anything still
# stuck at the end gets its fd hard-closed via os.close.
_KILL_AT_TIMEOUT_BUDGET_S = 2.0


def _release_registry_count(c) -> None:
    """Mirror what ``Connection.close`` does for the registry's per-host
    counter. The async and pool branches below bypass ``Connection.close``
    (DB-24960 and the pool's ``close_returns=True`` reroute) by calling
    ``pgconn.finish()`` directly — doing so skips the normal close path
    that calls ``ClusterRegistry.decrement``, leaving the per-host
    ``connection_count`` permanently high after every drain. The
    dispatcher's least-loaded policy then starves that host for the
    rest of the process lifetime (agent-K PR #13:111). Call this
    manually on any close path that doesn't flow through
    ``Connection.close``.
    """
    uuid = getattr(c, "_yb_uuid", None)
    host = getattr(c, "_yb_host", None)
    if not uuid or not host:
        # Not a smart-driver conn (control conn, pool bypass, test
        # fake). The registry didn't increment for it, nothing to do.
        return
    try:
        from .registry import ClusterRegistry
        ClusterRegistry.instance().decrement(uuid, host)
    except Exception:
        # Registry decrement is best-effort — a stale uuid / torn-down
        # state must not block the actual close path.
        pass


# Budget for acquiring the per-conn lock in ``_safe_close`` before
# escalating to the hard-fd escape hatch. Covers two separate races:
#
#   * On the C build, the owner thread holds ``Connection.lock``
#     around each libpq call (cursor.py:112,127,168,324,342). Closing
#     the conn without acquiring that lock frees libpq state while
#     the owner is still inside ``consuming input`` → segfault,
#     IndexError-from-inside-psycopg, or NULL conn pointer. Shubham
#     observed all three in DB-25090 (2 / 19 drains on C, 0 / 5 pure
#     Python in the same run).
#   * The owner may be blocked on a network read inside libpq. The
#     lock will stay held until the OS unblocks it. If drain times
#     out here it falls back to the existing hard-fd escape hatch,
#     same as a stuck ``close()``.
#
# 1.5 s leaves ~500 ms of the ``_KILL_AT_TIMEOUT_BUDGET_S`` budget for
# the actual close + hard-fd escalation. For the IDLE-close path
# (H7) the lock is almost always free and the acquire returns
# instantly.
_SAFE_CLOSE_LOCK_BUDGET_S = 1.5


class _CloseLockTimeoutError(Exception):
    """``_safe_close`` could not acquire the connection's lock within
    ``_SAFE_CLOSE_LOCK_BUDGET_S`` — owner is inside a libpq call that
    didn't return. Caller escalates to the hard-fd escape hatch.
    DB-25090 (C-build thread-safety).
    """


def _safe_close(c, lock_timeout_s: "float | None" = None) -> "Exception | None":
    """Call close on ``c``; return the exception if it raises, else
    ``None``. Isolates a raising close so it doesn't blow up the
    executor's worker thread — the caller tracks per-conn outcomes
    and logs each individually.

    Dispatches on connection type (DB-24960): ``AsyncConnection.close``
    is a coroutine, so calling it from this sync executor returns an
    un-awaited coroutine object and never actually closes the socket.
    Logs would still report ``force-closed`` while the TCP session
    stayed alive and the in-flight transaction would commit on the
    outgoing universe after the routing flip. For async conns we
    force-finish the libpq handle directly instead — the same
    workaround ``ClusterRegistry.clear()`` already uses for the async
    control conn. The socket is a libpq object; it doesn't need the
    event loop to be severed. We release the per-host registry
    counter manually on this path (agent-K PR #13:111) because
    ``pgconn.finish()`` doesn't walk the normal ``Connection.close``
    teardown.

    Thread safety (DB-25090): sync conns ship with a ``Connection.lock``
    that cursor operations already acquire around every libpq call
    (``cursor.py: with self._conn.lock``). Acquire it here with a
    bounded wait before touching libpq, so we never free pgconn state
    while a worker thread is still inside ``consuming input`` or
    equivalent. On the C build the race corrupts libpq → NULL conn
    pointer, IndexError from inside psycopg, or process death. On
    pure Python it corrupts result handling (less dramatic but still
    an incorrect error for the worker). If the lock stays held past
    the budget, return ``_CloseLockTimeoutError`` so the caller
    escalates to the existing hard-fd escape hatch.

    Async conns don't get this treatment: their ``.lock`` is an
    ``asyncio.Lock`` and acquiring it from the drain thread would be
    a cross-thread asyncio-lock call (undefined behavior). The
    residual async finish-race is tracked separately.
    """
    try:
        # Lazy import — ``..connection_async`` pulls ``.yb.state`` which
        # pulls back into this subpackage; top-level import would cycle.
        from ..connection_async import AsyncConnection
        if isinstance(c, AsyncConnection):
            try:
                c.pgconn.finish()
            finally:
                c._closed = True
                _release_registry_count(c)
            return None
        # Pool-borrowed conn: ``psycopg_pool.ConnectionPool`` patches
        # ``Connection.close`` with ``close_returns=True`` so that
        # ``c.close()`` reroutes to ``pool.putconn(c)`` — the conn
        # ends up back in the idle pool, NOT closed. The next
        # ``pool.getconn()`` hands the same stale-cluster socket to
        # another worker and the dual-write window re-opens. Bypass
        # by calling ``pgconn.finish()`` directly for any conn whose
        # ``_pool`` is set (agent-K PR #13:370). Same pattern H6
        # uses in ``xcluster_check``.
        if getattr(c, "_pool", None) is not None:
            try:
                c.pgconn.finish()
            finally:
                _release_registry_count(c)
            return None
        # Sync branch — acquire the per-conn lock first (DB-25090).
        # Missing lock on a mock / test shim: fall through to the
        # direct close path (what we did pre-fix). Real Connection
        # instances always have one.
        lock = getattr(c, "lock", None)
        if lock is None:
            c.close()
            return None
        # Lock-acquire budget: lock_timeout_s=None means use default
        # (_SAFE_CLOSE_LOCK_BUDGET_S); lock_timeout_s=0 means
        # non-blocking (agent-K PR #21:146 — the idle-close path
        # should NOT stall 1.5 s per conn whose owner is in a long
        # libpq call like LISTEN/NOTIFY; it just skips and retries
        # the next tick).
        budget = (
            lock_timeout_s
            if lock_timeout_s is not None
            else _SAFE_CLOSE_LOCK_BUDGET_S
        )
        if budget <= 0:
            if not lock.acquire(blocking=False):
                return _CloseLockTimeoutError(
                    "conn.lock held and non-blocking acquire failed; "
                    "will retry next tick"
                )
        else:
            if not lock.acquire(timeout=budget):
                return _CloseLockTimeoutError(
                    f"could not acquire conn.lock within "
                    f"{budget}s — owner thread is "
                    f"inside libpq; escalate to hard-fd close (DB-25090)"
                )
        try:
            c.close()
        finally:
            lock.release()
        return None
    except Exception as exc:
        return exc

# Transaction-status values that count as "in flight" (not drained). See
# psycopg.pq.TransactionStatus:
#   IDLE=0, ACTIVE=1, INTRANS=2, INERROR=3, UNKNOWN=4
# Design doc §3.2: IDLE and UNKNOWN count as drained.
_IN_FLIGHT_TX_STATUSES = frozenset((1, 2, 3))   # ACTIVE, INTRANS, INERROR


def _in_flight_conns(tracked: "weakref.WeakSet") -> list:
    """Return the list of currently-in-transaction conns from a tracked
    set. Iterates over a snapshot so mid-iteration GC evictions don't
    trip us up. Silently skips conns whose ``.info.transaction_status``
    read raises (broken conn, closed mid-check, etc.)."""
    result = []
    for c in list(tracked):
        try:
            status = c.info.transaction_status
        except Exception:
            continue
        # `status` may be an IntEnum or a raw int depending on psycopg
        # version — comparing via `int(...)` is safe for both.
        try:
            status_int = int(status)
        except (TypeError, ValueError):
            continue
        if status_int in _IN_FLIGHT_TX_STATUSES:
            result.append(c)
    return result


def _close_idle_tracked_during_drain(
    tracked: "weakref.WeakSet", outgoing_uuid: str,
) -> int:
    """Close every tracked direct conn that is currently IDLE (not in a
    transaction) and not yet closed. Returns the number of conns closed.

    Called from the drain poll loop while ``dispatch_paused=True`` (barrier
    held) to prevent persistent sockets from starting a FRESH transaction
    on the outgoing universe between two commits. The current txn (if any)
    is left alone — the loop will see it as in-flight on the next tick and
    only close it once it commits. Without this, with
    ``drainTimeoutSecs=-1`` the loop waits for a "zero in-flight" instant
    that never arrives under steady load — persistent conns loop
    txn→IDLE→txn forever on the outgoing side (DB-25012).

    The close is sync here (no thread pool) because an IDLE conn has no
    in-flight libpq state — ``pgconn.finish()`` returns in microseconds.
    Uses ``_safe_close`` for sync/async dispatch (DB-24960, PR-H1). Pool
    conns are strong-referenced by the pool and are not in ``tracked`` —
    they're evicted by ``xcluster_check`` on the next ``getconn``, not
    here.
    """
    closed = 0
    for c in list(tracked):
        try:
            if getattr(c, "closed", False):
                continue
            status = c.info.transaction_status
            if int(status) in _IN_FLIGHT_TX_STATUSES:
                continue
        except Exception:
            # Broken conn / stale weakref / in the middle of a status
            # read race — treat as "do nothing", the next tick retries.
            continue
        # agent-K PR #21:146 — use non-blocking lock acquire here.
        # An IDLE conn whose owner is in a long libpq call (e.g.
        # LISTEN/NOTIFY) would otherwise stall the drain poll by
        # 1.5 s per such conn. Skip + retry next tick instead.
        exc = _safe_close(c, lock_timeout_s=0)
        if exc is None:
            closed += 1
        elif isinstance(exc, _CloseLockTimeoutError):
            # Owner's holding the lock — not stuck, just long-running.
            # Next poll will retry.
            logger.log(
                TRACE,
                "drain: idle-close skipped on %s (lock held, retry next tick)",
                outgoing_uuid,
            )
        else:
            # Rare — _safe_close on an IDLE conn mostly just calls
            # pgconn.finish(). If it raised, log at debug (not warning)
            # because the next tick's _in_flight_conns check will treat
            # a half-closed conn as not-in-flight and the drain still
            # progresses.
            logger.debug(
                "drain: idle-close on tracked conn failed on uuid=%s: %s",
                outgoing_uuid, exc,
            )
    return closed


def _compute_active(primary_healthy: bool, secondary_healthy: bool):
    """State-machine reducer — matches ``state.active_cluster``. Kept
    here for the transition-plan preview logic; a duplicate is
    acceptable given how tight both are."""
    if primary_healthy:
        return "primary"
    if secondary_healthy:
        return "secondary"
    return None


def trigger_drain(
    group: "FailoverGroup",
    which_cluster: str,
    new_status: HealthResult,
    drain_timeout_s: int,
) -> None:
    """Apply a per-cluster CB status transition, running the barrier
    drain against the outgoing cluster iff the routing target actually
    changes.

    Called from the probe thread — see health_probe.HealthProbe._maybe_apply.
    """
    if which_cluster not in ("primary", "secondary"):
        raise ValueError(
            f"which_cluster must be 'primary' or 'secondary', "
            f"got {which_cluster!r}"
        )

    # Snapshot both statuses under the lock so the transition plan uses
    # a consistent view.
    with group.lock:
        pri_now = group.primary_status
        sec_now = group.secondary_status

    before_active = _compute_active(
        pri_now == HealthResult.HEALTHY,
        sec_now == HealthResult.HEALTHY,
    )
    if which_cluster == "primary":
        after_active = _compute_active(
            new_status == HealthResult.HEALTHY,
            sec_now == HealthResult.HEALTHY,
        )
    else:
        after_active = _compute_active(
            pri_now == HealthResult.HEALTHY,
            new_status == HealthResult.HEALTHY,
        )

    if before_active == after_active:
        # No routing change — apply the status without disturbing
        # dispatch. This covers e.g. "secondary flips while primary is
        # HEALTHY".
        _apply_status_change(group, which_cluster, new_status)
        return

    if after_active is None:
        # Last viable cluster went UNHEALTHY. There is no target to
        # drain towards, so don't open a drain window: dispatch starts
        # raising NoViableClusterError and conns on the dead side fail
        # on their own. Any that survive are closed by the drain that
        # runs when a cluster comes back (below).
        logger.warning(
            "xCluster %s → %s leaves no viable cluster (primary_uuid=%s); "
            "routing suspended until a cluster recovers",
            which_cluster, new_status.value, group.primary.uuid,
        )
        _apply_status_change(group, which_cluster, new_status)
        return

    if before_active is None:
        # Recovery from no-viable. Routing changes (None → after_active),
        # so run a real drain against the OTHER side: it closes anything
        # still open there (e.g. conns that outlived a probe-only
        # outage) before dispatch resumes onto after_active. Phase 2 is
        # skipped — the other side was not serving, so there is nothing
        # in flight to replicate.
        outgoing = group.secondary if after_active == "primary" else group.primary
        now_start = time.monotonic()
        with group.lock:
            group.last_failover_start_ts = now_start
        _drain_and_apply(
            group=group,
            which_cluster=which_cluster,
            new_status=new_status,
            outgoing=outgoing,
            drain_timeout_s=drain_timeout_s,
            is_failback=after_active == "primary",
            skip_phase2=True,
        )
        return

    # Auto-failback gate. When the operator has set
    # ``yb.failover.autoFailbackEnabled=false``, block the driver from
    # returning traffic to the primary automatically — even though the
    # primary CB now reports HEALTHY. The routing target must be moving
    # away from secondary (i.e., a genuine failback in the making) for
    # this gate to fire; failover directions (primary → secondary) are
    # never gated.
    is_failback = (
        before_active == "secondary"
        and after_active == "primary"
    )
    if is_failback and not group.auto_failback_enabled:
        logger.warning(
            "xCluster failback suppressed: %s CB reported HEALTHY but "
            "autoFailbackEnabled=false; staying on secondary "
            "(primary_uuid=%s). Operator must trigger failback manually.",
            which_cluster, group.primary.uuid,
        )
        # Do NOT update primary_status. Leaving it UNHEALTHY keeps
        # active_cluster() = "secondary". The CB will keep reporting
        # HEALTHY on each tick; each tick will hit this branch and log
        # again. Cool-down bounds the log rate.
        return

    # Stamp per-stage timestamps at their boundaries (design doc §3.8).
    # last_cb_trip_ts fires on UNHEALTHY-direction transitions (the CB
    # just observed the outgoing cluster as bad); last_failover_start_ts
    # is set on any routing-changing transition — even a failback
    # counts as a "failover" for timing purposes (the drain barrier
    # runs symmetrically).
    now_start = time.monotonic()
    with group.lock:
        if new_status == HealthResult.UNHEALTHY and which_cluster == "primary":
            group.last_cb_trip_ts = now_start
        group.last_failover_start_ts = now_start

    outgoing = group.primary if before_active == "primary" else group.secondary
    _drain_and_apply(
        group=group,
        which_cluster=which_cluster,
        new_status=new_status,
        outgoing=outgoing,
        drain_timeout_s=drain_timeout_s,
        is_failback=is_failback,
    )


def _apply_status_change(
    group: "FailoverGroup",
    which_cluster: str,
    new_status: HealthResult,
    after_drain: bool = False,
) -> None:
    """Update the per-cluster status field under ``group.lock`` — atomic
    with reads by the dispatcher's ``active_cluster``.

    ``after_drain=False`` (the default) means this is a no-op transition
    that didn't change the routing target (e.g. secondary flipped while
    primary stayed HEALTHY). ``after_drain=True`` means it's the tail of
    a full drain sequence and the routing target DID change; the log
    line differentiates so operators aren't confused by "no drain
    needed" appearing after a real drain."""
    now = time.monotonic()
    status_attr = f"{which_cluster}_status"
    transition_attr = f"{which_cluster}_last_transition_time"
    with group.lock:
        previous = getattr(group, status_attr)
        setattr(group, status_attr, new_status)
        setattr(group, transition_attr, now)
    if after_drain:
        logger.info(
            "xCluster %s status transition: %s → %s "
            "(primary_uuid=%s, routing target flipped)",
            which_cluster,
            previous.value if hasattr(previous, "value") else previous,
            new_status.value,
            group.primary.uuid,
        )
    else:
        logger.info(
            "xCluster %s status transition: %s → %s "
            "(primary_uuid=%s, no drain needed — routing unchanged)",
            which_cluster,
            previous.value if hasattr(previous, "value") else previous,
            new_status.value,
            group.primary.uuid,
        )


def _drain_and_apply(
    group: "FailoverGroup",
    which_cluster: str,
    new_status: HealthResult,
    outgoing: "ClusterState",
    drain_timeout_s: int,
    is_failback: bool = False,
    skip_phase2: bool = False,
) -> None:
    """Full barrier-drain-flip sequence. See module docstring.

    ``skip_phase2`` is set for recovery from no-viable, where the
    outgoing side was not serving and there is no lag to wait for."""
    # Snapshot the initial tracked-conn count for the log line — the
    # weakref set may shrink under GC during the drain.
    initial_tracked = len(outgoing.tracked_conns)
    initial_in_flight = len(_in_flight_conns(outgoing.tracked_conns))
    initial_drained = initial_tracked - initial_in_flight
    logger.info(
        "xCluster drain-wait start (barrier armed, deadline in %ss): "
        "outgoing=%s → new=%s (primary_uuid=%s, initial_tracked=%d, "
        "in_flight=%d, already_idle_or_closed=%d)",
        drain_timeout_s, outgoing.uuid, new_status.value,
        group.primary.uuid, initial_tracked, initial_in_flight,
        initial_drained,
    )
    if initial_in_flight == 0:
        logger.info(
            "xCluster drain: no in-flight txns at start — Phase 1 will "
            "exit early (per design doc §3.2 pseudocode). Common cause: "
            "server-side AdminShutdown killed conns before CB detected."
        )

    # 1. Pause dispatch (new opens block on the Condition).
    group.pause_dispatch()

    try:
        # 2. Compute deadline from the sentinel.
        if drain_timeout_s == 0:
            deadline = time.monotonic()          # kill immediately
        elif drain_timeout_s == -1:
            deadline = float("inf")              # wait forever
        else:
            deadline = time.monotonic() + drain_timeout_s

        # 3. Poll until either all in-flight transactions drain OR the
        # deadline hits. The loop exit is driven by in-flight count
        # alone so natural drain remains the fast path.
        #
        # Each tick ALSO closes any tracked direct conn that happens to
        # be IDLE at the moment of the check — stops the persistent-
        # socket reuse pattern (txn commits → conn goes IDLE → app's next
        # cursor.execute starts a new txn on the same socket, still
        # routed to the outgoing universe) that made `drainTimeoutSecs=-1`
        # loop forever under steady load (DB-25012). The current txn on
        # a persistent conn is left alone; it will be seen in-flight and
        # closed the tick after it commits. See
        # `_close_idle_tracked_during_drain`.
        drained_naturally = False
        total_idle_closed = 0
        while True:
            idle_closed = _close_idle_tracked_during_drain(
                outgoing.tracked_conns, outgoing.uuid,
            )
            total_idle_closed += idle_closed
            in_txn = _in_flight_conns(outgoing.tracked_conns)
            if not in_txn:
                drained_naturally = True
                logger.debug(
                    "drain: all in-flight txns completed on %s "
                    "(tracked=%d, idle_closed_this_drain=%d)",
                    outgoing.uuid, len(outgoing.tracked_conns),
                    total_idle_closed,
                )
                break
            now = time.monotonic()
            if now >= deadline:
                break
            # Sleep in short slices so `drain_timeout_s == -1` still
            # returns promptly if all conns drain. 100 ms matches the
            # design doc §3.2 pseudocode.
            time.sleep(0.1)

        # Fix for DB-24801: close EVERY tracked conn on the outgoing
        # cluster at the deadline, not just the in-flight ones. The
        # design doc §3.3 promises "no dual-write within a single
        # application"; sparing IDLE conns violates that because IDLE
        # is a transient state between two commits — whether a
        # persistent direct conn happens to be IDLE at the barrier
        # instant is pure timing luck, and the next moment it goes
        # right back into INTRANS and commits on the OUTGOING cluster.
        # The app's next cursor call on a closed conn raises
        # OperationalError and reconnects to the active cluster. Pool
        # conns are already evicted by ``xcluster_check`` on next
        # ``getconn``; closing them here is redundant but harmless.
        survivors = [
            c for c in list(outgoing.tracked_conns)
            if c is not None and not getattr(c, "closed", False)
        ]

        # 4. Kill at timeout (step 3 in design doc §3). Force-close
        # any conns still in-flight at the deadline. Only reached with
        # a finite deadline; the -1 branch loops until in_txn is empty.
        #
        # Concurrency: fire close() on every survivor in parallel via
        # a bounded thread pool. A serial for-loop of c.close() blocks
        # on the FIRST hung TCP session for ~tcp_retries2 (up to ~15
        # min on Linux, worse with keepalive=0), which would wedge
        # drain past its own deadline. For any close() that doesn't
        # return within _KILL_AT_TIMEOUT_BUDGET_S, escape via raw
        # os.close on the socket fd — the libpq send() may still be
        # blocked upstream, but from the driver's perspective the conn
        # is disowned and drain proceeds. Bounds total wall-clock at
        # drain_timeout_s + _KILL_AT_TIMEOUT_BUDGET_S regardless of
        # TCP state.
        if survivors:
            if drained_naturally:
                # All in-flight transactions completed before the
                # deadline; the surviving conns are IDLE tracked conns
                # we're closing at the flip boundary to prevent
                # post-flip dual-write (DB-24801).
                logger.info(
                    "xCluster drain: closing %d tracked conn(s) on %s "
                    "at flip boundary (all in-flight txns drained "
                    "naturally; parallel, budget=%ss)",
                    len(survivors), outgoing.uuid,
                    _KILL_AT_TIMEOUT_BUDGET_S,
                )
            else:
                logger.warning(
                    "xCluster drain-wait deadline (%ss elapsed) — "
                    "force-closing %d survivor(s) on %s "
                    "(parallel, budget=%ss)",
                    drain_timeout_s, len(survivors), outgoing.uuid,
                    _KILL_AT_TIMEOUT_BUDGET_S,
                )
            # Snapshot fds BEFORE close() nulls the pgconn state —
            # we need them for the hard-fd escape hatch.
            snapshot = []
            for c in survivors:
                try:
                    fd = c.pgconn.socket
                    conn_id = f"conn=fd{fd}"
                except Exception:
                    fd, conn_id = None, f"conn={id(c):x}"
                snapshot.append((c, fd, conn_id))

            hung = 0
            # NB: intentionally NOT using `with ThreadPoolExecutor()` —
            # its __exit__ calls shutdown(wait=True), which would block
            # on hung workers and re-introduce the very problem we're
            # solving. Manual instantiation + explicit shutdown(wait=
            # False) in finally lets a stuck close() worker remain
            # detached in the background; the driver-side conn is
            # already disowned via os.close of its fd (see below), so
            # the hang has no downstream effect. When the OS eventually
            # gives up on the wedged send() (tcp_retries2), the worker
            # unwinds naturally.
            ex = concurrent.futures.ThreadPoolExecutor(
                max_workers=min(len(survivors), 32),
                thread_name_prefix="yb-kill",
            )
            try:
                fut_by_conn = {
                    ex.submit(_safe_close, c): (c, fd, conn_id)
                    for c, fd, conn_id in snapshot
                }
                concurrent.futures.wait(
                    list(fut_by_conn),
                    timeout=_KILL_AT_TIMEOUT_BUDGET_S,
                )
                for fut, (c, fd, conn_id) in fut_by_conn.items():
                    needs_hard_close = False
                    # agent-K PR #21:672 blocker. The hard-fd escape
                    # hatch is only safe when `_closed=True` has been
                    # set on the conn — then the driver has disowned it
                    # and the owner thread won't touch it again. On the
                    # hung-close branch, `_safe_close` has already set
                    # `_closed=True` before `c.close()` started blocking.
                    # On the lock-timeout branch (DB-25090), `_closed`
                    # is still False and the owner is still inside
                    # libpq holding a PGconn whose fd we are about to
                    # free; Linux will give the next socket the lowest
                    # free fd (likely this one), the owner's next libpq
                    # read/write hits a different conn → cross-conn
                    # protocol corruption. On that branch use
                    # `shutdown(SHUT_RDWR)` instead: it unblocks the
                    # owner's blocked read/write but keeps the fd
                    # number reserved so no reuse race, and the
                    # owner's eventual close() finishes teardown
                    # cleanly (including ClusterRegistry.decrement).
                    hard_close_mode = "close"   # vs "shutdown"
                    hard_close_reason = ""
                    if fut.done():
                        exc = fut.result()
                        if exc is None:
                            # agent-K PR #13:391 — this block runs both
                            # on the natural-drain path (closing IDLE
                            # tracked conns at the flip boundary, no
                            # txns were force-closed) and on the
                            # deadline-hit path. Split the log so an
                            # operator doesn't see "force-closed in
                            # flight" when nothing was in flight.
                            if drained_naturally:
                                logger.info(
                                    "flip-boundary close: %s closed "
                                    "(was IDLE on %s)",
                                    conn_id, outgoing.uuid,
                                )
                            else:
                                logger.info(
                                    "kill-at-timeout: %s force-closed "
                                    "(was in-flight on %s)",
                                    conn_id, outgoing.uuid,
                                )
                        elif isinstance(exc, _CloseLockTimeoutError):
                            # Owner thread is inside libpq and didn't
                            # release conn.lock in time (DB-25090). The
                            # conn is still alive on the outgoing side
                            # from the driver's point of view, which
                            # means the owner can still issue the next
                            # cursor call on this socket post-flip and
                            # dual-write. Use the shutdown-not-close
                            # variant so we unblock the owner's libpq
                            # call without racing fd reuse (agent-K
                            # PR #21:672).
                            needs_hard_close = True
                            hard_close_mode = "shutdown"
                            hard_close_reason = (
                                f"lock held past "
                                f"{_SAFE_CLOSE_LOCK_BUDGET_S}s "
                                f"(owner inside libpq, DB-25090)"
                            )
                        else:
                            logger.log(
                                5,   # TRACE
                                "drain: close() raised on %s: %r",
                                conn_id, exc,
                            )
                    else:
                        # close() hung past the budget. Hard-close the
                        # fd directly — releases the driver's grip
                        # even though libpq's PQfinish send() may
                        # still be blocked in the worker thread. The
                        # OS closes the socket immediately; libpq's
                        # blocked send() will error out when the peer
                        # (or the closed local half) becomes evident.
                        needs_hard_close = True
                        hard_close_reason = (
                            f"close() hung past "
                            f"{_KILL_AT_TIMEOUT_BUDGET_S}s budget"
                        )
                    if needs_hard_close:
                        hung += 1
                        if fd is not None:
                            try:
                                if hard_close_mode == "shutdown":
                                    # agent-K PR #21:672 — SHUT_RDWR
                                    # unblocks the owner's libpq read /
                                    # write without freeing the fd
                                    # number. Owner's eventual close()
                                    # (triggered by the broken-socket
                                    # error their next call sees) runs
                                    # the full Connection.close path
                                    # including the registry decrement.
                                    #
                                    # Agent-K PR #21:G4 — hold the
                                    # socket in a local, then call
                                    # ``detach()`` so the socket
                                    # object's ``__del__`` does NOT
                                    # close the fd when it is GC'd
                                    # right after this block. The whole
                                    # point of using ``shutdown``
                                    # instead of ``os.close`` is to
                                    # keep the fd number reserved so
                                    # the owner thread's stale libpq
                                    # read/write can't race a reused
                                    # fd belonging to another conn;
                                    # a GC'd socket silently defeated
                                    # that reservation.
                                    import socket as _socket
                                    try:
                                        sock = _socket.socket(fileno=fd)
                                        try:
                                            sock.shutdown(_socket.SHUT_RDWR)
                                        finally:
                                            # Releases ownership of the
                                            # fd back to the kernel's
                                            # ``close`` boundary (the
                                            # owner thread will do it);
                                            # the socket's ``__del__``
                                            # becomes a no-op.
                                            sock.detach()
                                    except OSError:
                                        # fd already gone / not a socket
                                        pass
                                else:
                                    os.close(fd)
                                logger.warning(
                                    "kill-at-timeout: %s %s "
                                    "(%s on %s)",
                                    conn_id,
                                    (
                                        "hard-fd-shutdown"
                                        if hard_close_mode == "shutdown"
                                        else "hard-fd-closed"
                                    ),
                                    hard_close_reason,
                                    outgoing.uuid,
                                )
                            except OSError:
                                # Either close() completed racily and
                                # already released the fd, or the fd
                                # was invalid to begin with. Nothing
                                # more to do.
                                pass
                        else:
                            logger.warning(
                                "kill-at-timeout: %s needs hard-close but "
                                "no fd available (%s on %s)",
                                conn_id, hard_close_reason, outgoing.uuid,
                            )
            finally:
                # wait=False: never block on hung workers. Any still-
                # running _safe_close call keeps executing after we
                # return — its future is abandoned and it eventually
                # unwinds when the OS releases the wedged send().
                ex.shutdown(wait=False)

            logger.warning(
                "kill-at-timeout complete on %s: %d force-closed "
                "(%d needed hard-fd escape hatch, primary_uuid=%s)",
                outgoing.uuid, len(survivors), hung, group.primary.uuid,
            )

        # Phase 1 complete — record the boundary and log the outcome so
        # the operator can tell at a glance which of the three cases
        # this was:
        #   (a) initial_tracked == 0 — no work at all.
        #   (b) drained naturally — all in-flight txns completed within
        #       the window; closed any remaining IDLE tracked conns at
        #       the flip boundary to prevent post-flip dual-write
        #       (DB-24801).
        #   (c) deadline hit — force-closed every tracked conn still
        #       alive at the deadline (in-flight + idle). The in-flight
        #       count at the deadline moment is reported separately.
        now = time.monotonic()
        phase1_elapsed = now - (
            group.last_failover_start_ts or now
        )
        if initial_tracked == 0:
            logger.info(
                "xCluster drain: Phase 1 exited early after %.2fs — "
                "no tracked conns on %s",
                phase1_elapsed, outgoing.uuid,
            )
        elif drained_naturally:
            # agent-K PR #19:466 — include total_idle_closed so the
            # operator can see how many persistent conns were closed
            # during the drain loop (the DB-25012 H7 path), separate
            # from the ones closed at the flip boundary.
            logger.info(
                "xCluster drain: Phase 1 completed in %.2fs — all %d "
                "in-flight txn(s) drained naturally; closed %d idle "
                "tracked conn(s) at flip boundary (DB-24801), %d "
                "additional idle conn(s) closed during drain loop "
                "(DB-25012)",
                phase1_elapsed, initial_in_flight, len(survivors),
                total_idle_closed,
            )
        else:
            # Count how many survivors were in-flight at the deadline
            # moment vs. just idle tracked conns we closed for the
            # flip-boundary guarantee.
            at_deadline_in_flight = len(in_txn)
            at_deadline_idle = len(survivors) - at_deadline_in_flight
            logger.info(
                "xCluster drain: Phase 1 hit the %ds drain-wait deadline "
                "→ kill-at-timeout fired: %d tracked conn(s) force-closed "
                "(%d in-flight + %d idle at deadline), %d completed "
                "naturally before deadline, %d idle conn(s) closed "
                "during drain loop (DB-25012)",
                drain_timeout_s, len(survivors),
                at_deadline_in_flight, at_deadline_idle,
                initial_in_flight - at_deadline_in_flight,
                total_idle_closed,
            )
        with group.lock:
            group.last_phase1_complete_ts = now
        closed_any = bool(survivors) or total_idle_closed > 0
        closed_at = time.monotonic()

        # 4b. Phase 2 — YBA replication-lag wait. Direction-aware:
        #   * Failover  (UNHEALTHY): poll the A→B config UUID iff set.
        #   * Failback  (HEALTHY):   poll the B→A config UUID iff set
        #                            (per §3.5.1 symmetric behaviour).
        # A missing config UUID in either direction fails open — same
        # semantics as YBA being unreachable.
        #
        # Wrap in `Phase2TimeoutError` catch. On
        # ``onPhase2Timeout="fail"`` (opt-in), a failback lag-wait raises
        # when it times out before converging. We log, record the
        # timeout timestamp, and SKIP the status change — routing
        # stays on the current (known-good) side. Dispatch still
        # resumes in the finally so the app isn't stuck. The probe
        # will try again on its next tick.
        phase2_aborted = False
        if group.yba_client is not None and not skip_phase2:
            try:
                if new_status == HealthResult.UNHEALTHY and group.xcluster_config_uuid:
                    _wait_for_replication_lag(group, which_cluster, is_failback=False)
                elif (
                    new_status == HealthResult.HEALTHY
                    and group.failback_xcluster_config_uuid
                ):
                    _wait_for_replication_lag(group, which_cluster, is_failback=True)
            except Phase2TimeoutError as exc:
                phase2_aborted = True
                abort_now = time.monotonic()
                with group.lock:
                    group.last_phase2_timeout_ts = abort_now
                    # agent-K PR #18:608. Stamp the per-side transition
                    # time so the probe's cooldown gate (health_probe
                    # ``_tick_check`` cooldown check) blocks the next
                    # tick from re-firing this drain. Without this,
                    # the probe sees status unchanged (because the
                    # flip was aborted) and immediately re-calls
                    # trigger_drain on its next tick (every 5 s by
                    # default) → pause_dispatch again → force-close
                    # every tracked conn on the active side again →
                    # Phase 2 wait again → abort again. Shubham's
                    # failback case turns a lag outage into a near-
                    # total application outage. Writing the stamp
                    # arms the cooldown so the next retry is bounded.
                    if which_cluster == "primary":
                        group.primary_last_transition_time = abort_now
                    else:
                        group.secondary_last_transition_time = abort_now
                logger.error(
                    "xCluster drain ABORTED at Phase 2: %s (primary_uuid=%s, "
                    "which=%s, new_status=%s) — routing stays on current "
                    "side, re-arming %s cooldown so probe backs off",
                    exc, group.primary.uuid, which_cluster, new_status.value,
                    which_cluster,
                )

        # Phase 2 complete — record the boundary (fires even when Phase 2
        # was skipped, so downstream code sees a consistent timeline).
        with group.lock:
            group.last_phase2_complete_ts = time.monotonic()

        # 5. Apply the status change while dispatch is still paused so
        # the state machine transitions atomically before waiters wake.
        # SKIPPED when Phase 2 aborted on timeout (data-corruption fix) —
        # we don't flip to a target we know is behind.
        if not phase2_aborted:
            # Close fence: a conn we just closed is only noticed by its
            # owner on its next call. An app that holds session state
            # (an advisory-lock leader heartbeating every ~1s) would
            # otherwise still believe it holds the lock while a new
            # session on the incoming side acquires it. Hold dispatch
            # until the fence has elapsed since the last close; Phase 2
            # time counts towards it. Once per flip, only when something
            # was closed.
            if closed_any:
                remaining = closed_at + _CLOSE_FENCE_S - time.monotonic()
                if remaining > 0:
                    time.sleep(remaining)
            _apply_status_change(
                group, which_cluster, new_status, after_drain=True,
            )
    finally:
        # 6. Resume dispatch — release waiters.
        group.resume_dispatch()
        # Record the direction-appropriate completion timestamp — but
        # only when the flip actually happened. agent-K PR #18:612:
        # debounced CBs read ``last_failover_complete_ts`` as "the
        # last time the state machine actually transitioned";
        # stamping it on an aborted drain lies to them and lets a
        # subsequent debounce window close over a flip that never
        # occurred.
        if not phase2_aborted:
            now_done = time.monotonic()
            with group.lock:
                if is_failback:
                    group.last_failback_complete_ts = now_done
                else:
                    group.last_failover_complete_ts = now_done
    logger.info(
        "xCluster drain complete: dispatch resumed (primary_uuid=%s, "
        "outgoing=%s, force_closed=%d)",
        group.primary.uuid, outgoing.uuid,
        len(survivors) if 'survivors' in locals() else 0,
    )


# --------------------------------------------------------------------- Phase 2

# Fixed 1-second poll interval per Amogh's 2026-08-11 Slack message.
_LAG_POLL_INTERVAL_S = 1.0

# After this many consecutive YBA request failures the driver gives up on
# lag confirmation and proceeds with the routing flip. Amogh's flow.
_LAG_UNREACHABLE_STRIKES = 3

# DB-24483 — Prometheus scrape interval YBA uses, matched to the YBA
# default (reference.conf:985). Used by the liveness gate to tell
# whether two "post-barrier" samples came from different scrapes (good)
# or the same one (YBA's replay pathology).
_LAG_SCRAPE_HINT_S = 10.0


def _wait_for_replication_lag(
    group: "FailoverGroup",
    which_cluster: str,
    is_failback: bool = False,
) -> None:
    """Poll YBA for the outgoing side's replication lag every second, exit
    on any of three conditions:

      1. ``lag_ms ≤ threshold_replication_lag_ms`` — converged. Safe to
         switch; the target has caught up to within the operator's RPO.
      2. ``elapsed ≥ lag_wait_timeout_s`` — timed out. Switch anyway. Some
         data may be un-replicated; that's a known consequence.
      3. ``_LAG_UNREACHABLE_STRIKES`` consecutive YBA errors — fail-open.
         Better to switch without lag confirmation than to wedge failover
         because the YBA control plane happens to be down.

    Never blocks failover on YBA availability. Never blocks longer than
    ``lag_wait_timeout_s``. Log at WARNING on entry and exit so operators
    can read the outcome from the log timeline.

    ``is_failback`` picks between the A→B config + failover knobs
    (default) and the B→A config + failback knobs (see design doc §3.5.1).
    """
    from .yba_client import YBAClientError

    if is_failback:
        threshold_ms = float(group.failback_threshold_replication_lag_ms)
        timeout_s = group.failback_lag_wait_timeout_s
        xcluster_uuid = group.failback_xcluster_config_uuid
        freshness_s = float(getattr(group, "failback_lag_freshness_s", 0))
        required_samples = int(getattr(group, "failback_lag_confirmation_samples", 1))
        direction_label = "failback"
    else:
        threshold_ms = float(group.threshold_replication_lag_ms)
        timeout_s = group.lag_wait_timeout_s
        xcluster_uuid = group.xcluster_config_uuid
        freshness_s = float(getattr(group, "lag_freshness_s", 0))
        required_samples = int(getattr(group, "lag_confirmation_samples", 1))
        direction_label = "failover"

    # DB-24483 — snapshot the barrier timestamp set by ``pause_dispatch``.
    # Zero means "no barrier armed" — a fail-open path (shouldn't happen
    # inside a drain, but defensive): if we have no barrier, the
    # freshness gate degrades to "any sample is fresh" and the liveness
    # gate still runs on sample identity.
    barrier_ts = float(getattr(group, "barrier_ts", 0.0))

    logger.warning(
        "lag wait start: direction=%s outgoing=%s xcluster_config=%s "
        "threshold=%.1fms timeout=%ds freshness=%.1fs confirmations=%d "
        "barrier_ts=%.3f",
        direction_label, which_cluster, xcluster_uuid, threshold_ms, timeout_s,
        freshness_s, required_samples, barrier_ts,
    )

    started = time.monotonic()
    deadline = started + timeout_s
    unreachable_streak = 0
    # DB-24483 — liveness tracking: consecutive distinct post-barrier
    # samples at ≤ threshold.
    confirmed_count = 0
    # Dedupe repeated YBA replays of the same sample. Any sample_ts
    # equal to the previous one does NOT advance the liveness count.
    last_counted_sample_ts: float | None = None

    # agent-K PR #8:430: the deadline is checked only BEFORE each poll,
    # but a YBA request can take up to ~`max(connect, read)`×2 (reconnect
    # on stale-socket). To stay strictly bounded by `lag_wait_timeout_s`
    # we bail early when the remaining budget is too small to fit one
    # safe poll. Only activate this guard when the user's configured
    # timeout is LARGE enough to make the trade-off worthwhile — if the
    # user picked a short timeout (e.g. 2s), respect their choice and
    # accept a possible small overshoot rather than bailing before the
    # first poll.
    _one_poll_cost_s = max(
        getattr(group.yba_client, "connect_timeout_s", 2.0),
        getattr(group.yba_client, "read_timeout_s", 3.0),
    ) * 2.0
    _poll_guard = timeout_s >= _one_poll_cost_s * 2.0
    # Reserve twice the slowest poll seen so far (floor 0.5s), capped at
    # the worst case. Reserving the worst case outright cut a 30s wait
    # to 24s even though real polls take milliseconds.
    slowest_poll_s = 0.0

    while True:
        now = time.monotonic()
        if slowest_poll_s > 0.0:
            _MIN_POLL_BUDGET_S = min(_one_poll_cost_s, max(0.5, slowest_poll_s * 2.0))
        else:
            _MIN_POLL_BUDGET_S = _one_poll_cost_s
        if not _poll_guard:
            _MIN_POLL_BUDGET_S = 0.0
        if now >= deadline:
            elapsed = now - started
            # Phase 2 timeout policy. Agent-K PR #18:710,575. The `fail`
            # policy aborts the flip on timeout — correct for FAILBACK
            # (we don't want to send traffic BACK to a lagged primary
            # while B→A hasn't caught up). But for FAILOVER (new_status
            # UNHEALTHY), "stay on current side" means stay on the
            # universe the CB just declared dead. If A is dead, A→B
            # lag can never converge; `fail` would then block the
            # failover forever. Force `proceed` in the failover
            # direction regardless of the knob — the knob only matters
            # for the failback case where staying on current side is
            # a safe degraded-mode choice.
            policy = getattr(group, "on_phase2_timeout", "proceed")
            if policy == "fail" and is_failback:
                logger.error(
                    "lag wait timeout after %.1fs (direction=%s, "
                    "threshold=%.1fms) — ABORTING flip; routing stays "
                    "on current side (set yb.failover.onPhase2Timeout="
                    "proceed to flip anyway)",
                    elapsed, direction_label, threshold_ms,
                )
                raise Phase2TimeoutError(
                    f"lag wait timeout after {elapsed:.1f}s "
                    f"(direction={direction_label}, "
                    f"threshold={threshold_ms:.1f}ms)"
                )
            if policy == "fail" and not is_failback:
                logger.warning(
                    "lag wait timeout after %.1fs — proceeding with flip "
                    "despite on_phase2_timeout='fail' (direction=failover; "
                    "the current side was just marked UNHEALTHY so "
                    "staying on it is worse than flipping to a lagged "
                    "target)", elapsed,
                )
            else:
                logger.warning(
                    "lag wait timeout after %.1fs — proceeding with flip "
                    "(policy=proceed)",
                    elapsed,
                )
            return
        if deadline - now < _MIN_POLL_BUDGET_S:
            # Not enough budget left to safely run one more poll
            # without overshooting the lag_wait_timeout. Agent-K PR
            # #18:G3 — this branch is a timeout equivalent, so it
            # must honour the on_phase2_timeout="fail" policy for the
            # FAILBACK direction (same reasoning as the deadline-hit
            # branch above). Previously this branch just returned,
            # bypassing the abort and silently flipping onto a
            # known-lagged target during tight lag_wait_timeout_s
            # settings.
            # Same "lag wait timeout" prefix as the deadline branch so log
            # consumers see one timeout outcome, not two.
            elapsed = now - started
            policy = getattr(group, "on_phase2_timeout", "proceed")
            if policy == "fail" and is_failback:
                logger.error(
                    "lag wait timeout after %.1fs (direction=failback, "
                    "threshold=%.1fms; %.1fs left is below the %.1fs poll "
                    "budget) — ABORTING flip; routing stays on current side",
                    elapsed, threshold_ms, deadline - now, _MIN_POLL_BUDGET_S,
                )
                raise Phase2TimeoutError(
                    f"lag wait timeout after {elapsed:.1f}s "
                    f"(direction=failback, threshold={threshold_ms:.1f}ms)"
                )
            logger.warning(
                "lag wait timeout after %.1fs (%.1fs left is below the "
                "%.1fs poll budget) — proceeding with flip (direction=%s)",
                elapsed, deadline - now, _MIN_POLL_BUDGET_S, direction_label,
            )
            return

        poll_started = time.monotonic()
        try:
            # DB-24483: prefer the sample-returning variant so we can
            # log how old the YBA data was. Custom lag sources that
            # only implement the legacy ``get_committed_lag_ms`` fall
            # back to that — otherwise an AttributeError would land in
            # the except-Exception path below and silently skip the
            # lag wait (agent-K PR #15:698). We still trust YBA as the
            # lag source of truth; surfacing the age gives operators a
            # concrete number to reason about after a flip.
            if hasattr(group.yba_client, "get_committed_lag_sample"):
                sample = group.yba_client.get_committed_lag_sample(xcluster_uuid)
                lag_ms = None if sample is None else sample[0]
                sample_ts = None if sample is None else sample[1]
            else:
                lag_ms = group.yba_client.get_committed_lag_ms(xcluster_uuid)
                sample_ts = None
            slowest_poll_s = max(slowest_poll_s, time.monotonic() - poll_started)
            unreachable_streak = 0
            # One line per poll — operator can trace exactly what the
            # driver observed on each 1-second tick. INFO because
            # WARNING would spam in production; the demo bumps this
            # logger to INFO to make it visible.
            elapsed = time.monotonic() - started
            if sample_ts is not None:
                sample_age_s = max(0.0, time.time() - sample_ts)
                age_str = f" sample_age={sample_age_s:.1f}s"
            else:
                age_str = ""
            if lag_ms is None:
                logger.info(
                    "lag wait poll: no data reported by YBA "
                    "(elapsed=%.1fs / %ds)",
                    elapsed, timeout_s,
                )
            else:
                logger.info(
                    "lag wait poll: lag=%.1fms threshold=%.1fms%s "
                    "(elapsed=%.1fs / %ds)",
                    lag_ms, threshold_ms, age_str, elapsed, timeout_s,
                )
        except YBAClientError as exc:
            slowest_poll_s = max(slowest_poll_s, time.monotonic() - poll_started)
            unreachable_streak += 1
            logger.warning(
                "lag wait: YBA unreachable (%d/%d): %s",
                unreachable_streak, _LAG_UNREACHABLE_STRIKES, exc,
            )
            if unreachable_streak >= _LAG_UNREACHABLE_STRIKES:
                logger.warning(
                    "lag wait: YBA unreachable %d× — proceeding with flip "
                    "(no lag confirmation)",
                    _LAG_UNREACHABLE_STRIKES,
                )
                return
            lag_ms = None
            sample_ts = None
        except Exception:
            # Belt-and-braces — never let an unexpected exception in the
            # YBA client wedge the drain barrier.
            logger.warning(
                "lag wait: unexpected YBA error — proceeding with flip",
                exc_info=True,
            )
            return

        if lag_ms is not None and lag_ms <= threshold_ms:
            elapsed = time.monotonic() - started
            # DB-24483 — freshness gate. A YBA sample whose ``sample_ts``
            # predates the barrier (or sits in the ``barrier_ts +
            # freshness_s`` grace window) cannot have observed writes
            # that landed before ``pause_dispatch``. Trusting it is the
            # exact bug that let the driver flip to a 17s-behind B.
            # Legacy YBA clients that don't return ``sample_ts`` fall
            # back to the pre-fix single-sample convergence — the
            # operator opted out of the gate by not giving us the
            # timestamp. ``freshness_s == 0`` likewise opts out.
            if freshness_s > 0 and sample_ts is not None and barrier_ts > 0:
                min_fresh_ts = barrier_ts + freshness_s
                if sample_ts < min_fresh_ts:
                    stale_by = min_fresh_ts - sample_ts
                    logger.info(
                        "lag wait: sample at %.3f is %.1fs older than the "
                        "freshness floor (barrier=%.3f + %.1fs) — not "
                        "counting (elapsed=%.1fs / %ds)",
                        sample_ts, stale_by, barrier_ts, freshness_s,
                        elapsed, timeout_s,
                    )
                    # Reset liveness: a stale sample is not a true pass.
                    confirmed_count = 0
                    last_counted_sample_ts = None
                    sleep_s = min(
                        _LAG_POLL_INTERVAL_S,
                        max(0.0, deadline - time.monotonic()),
                    )
                    if sleep_s > 0:
                        time.sleep(sleep_s)
                    continue
            # DB-24483 — liveness gate. YBA replays its last sample for
            # ~5min after the primary dies (DB-24852-adjacent). A single
            # ``lag=0`` is not enough to tell "standby really caught up"
            # from "same stale replay reused". Require N distinct
            # ``sample_ts`` values at ≤ threshold before flipping.
            if (
                sample_ts is not None
                and barrier_ts > 0
                and sample_ts <= barrier_ts
            ):
                # A point collected before the barrier cannot show that
                # writes acknowledged before the barrier have arrived
                # (stress phase2_claim: converged on such a sample while
                # B was still missing batches). Applies at any
                # confirmation count: with one, it is the only guard.
                logger.info(
                    "lag wait: lag=%.1fms sample at %.3f predates the "
                    "barrier %.3f — not counting (confirmations=%d/%d, "
                    "elapsed=%.1fs / %ds)",
                    lag_ms, sample_ts, barrier_ts, confirmed_count,
                    required_samples, elapsed, timeout_s,
                )
            elif sample_ts is not None and sample_ts == last_counted_sample_ts:
                logger.info(
                    "lag wait: lag=%.1fms ≤ threshold but sample_ts %.3f "
                    "unchanged since last pass — YBA replay suspected, "
                    "not counting (confirmations=%d/%d, elapsed=%.1fs / %ds)",
                    lag_ms, sample_ts, confirmed_count, required_samples,
                    elapsed, timeout_s,
                )
            else:
                confirmed_count += 1
                last_counted_sample_ts = sample_ts
                logger.info(
                    "lag wait: lag=%.1fms ≤ threshold=%.1fms counted "
                    "(confirmations=%d/%d, elapsed=%.1fs / %ds)",
                    lag_ms, threshold_ms, confirmed_count, required_samples,
                    elapsed, timeout_s,
                )
                if confirmed_count >= required_samples:
                    # Log sample age at convergence so a post-mortem
                    # can tell whether we trusted a fresh reading.
                    if sample_ts is not None:
                        age_str = (
                            f", sample_age="
                            f"{max(0.0, time.time() - sample_ts):.1f}s"
                        )
                    else:
                        age_str = ""
                    logger.warning(
                        "lag wait converged in %.1fs "
                        "(lag=%.1fms ≤ threshold=%.1fms, confirmations="
                        "%d/%d%s) — switching now",
                        elapsed, lag_ms, threshold_ms,
                        confirmed_count, required_samples, age_str,
                    )
                    return
        else:
            # Lag back above threshold (or we have no data this tick).
            # Reset the liveness counter so a next convergence has to
            # earn the full N consecutive passes.
            if confirmed_count > 0:
                logger.info(
                    "lag wait: liveness reset — lag no longer ≤ threshold "
                    "(had confirmed %d/%d)",
                    confirmed_count, required_samples,
                )
            confirmed_count = 0
            last_counted_sample_ts = None

        # Poll again in ~1s. If the remaining budget is smaller, sleep
        # only that much so we don't overshoot the deadline.
        sleep_s = min(_LAG_POLL_INTERVAL_S, max(0.0, deadline - time.monotonic()))
        if sleep_s > 0:
            time.sleep(sleep_s)
