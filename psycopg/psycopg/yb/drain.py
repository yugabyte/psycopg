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
     close hasn't returned within the budget gets its socket shut
     down (``SHUT_RDWR`` through a dup of its fd; the fd itself is
     never closed by the drain) — a stuck TCP ``send()`` in libpq's
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

from . import TRACE, fork_follow
from . import hold as yb_hold
from .health import HealthResult

if TYPE_CHECKING:
    from .registry import ClusterState, FailoverGroup

logger = logging.getLogger(__name__)


class Phase2TimeoutError(Exception):
    """Raised by a FAILBACK Phase 2 lag wait when it times out, or YBA
    stays unreachable for ``_LAG_UNREACHABLE_STRIKES`` polls, and the
    group's ``failback_on_phase2_timeout`` is ``"fail"`` (the default;
    ``"proceed"`` flips instead).

    Caught by ``_drain_and_apply``, which then SKIPS the status
    transition (does NOT flip routing) and resumes dispatch.

    The message ends up in the "xCluster drain ABORTED at Phase 2" line.
    It must not repeat the wording of the ERROR line logged just before
    the raise ("lag wait timeout", "unexpected YBA error"): log consumers
    classify a line by the first phrase it contains, and the ABORTED line
    would then read as a second timeout instead of the abort.
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
# stuck at the end gets its socket shut down (see _shutdown_socket_fd).
_KILL_AT_TIMEOUT_BUDGET_S = 2.0


def _shutdown_socket_fd(dup_fd: int) -> None:
    """``shutdown(SHUT_RDWR)`` the socket behind ``dup_fd``, our own dup
    of a conn's fd. Unblocks any thread inside libpq on that socket
    (its read/write errors out) without closing the conn's fd, so the
    fd number can't be reused under libpq. The caller closes
    ``dup_fd``."""
    import socket as _socket
    try:
        sock = _socket.socket(fileno=dup_fd)
    except OSError:
        return   # not a socket
    try:
        sock.shutdown(_socket.SHUT_RDWR)
    except OSError:
        pass     # already shut down / disconnected
    finally:
        sock.detach()


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

    Idempotent per conn (review #18): ``ClusterRegistry.release_conn``
    decrements at most once, so a later ``close()`` or ``xcluster_check``
    on the same conn doesn't decrement again. Conns without
    ``_yb_uuid`` / ``_yb_host`` (control conns, test fakes) are skipped
    there.
    """
    try:
        from .registry import ClusterRegistry
        ClusterRegistry.instance().release_conn(c)
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


# Poll interval while waiting for an async conn's ``asyncio.Lock`` to be
# released (see ``_wait_async_lock_free``).
_ASYNC_LOCK_POLL_S = 0.005


def _wait_async_lock_free(c, budget: float) -> bool:
    """Wait until ``c.lock`` (an ``asyncio.Lock``) is not held, up to
    ``budget`` seconds; ``budget <= 0`` checks once. Returns False if it
    is still held. Only reads ``locked()`` — never acquires the lock
    from this (non-event-loop) thread."""
    lock = getattr(c, "lock", None)
    locked = getattr(lock, "locked", None)
    if locked is None:
        return True
    deadline = time.monotonic() + max(budget, 0.0)
    while locked():
        if time.monotonic() >= deadline:
            return False
        time.sleep(_ASYNC_LOCK_POLL_S)
    return True


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

    Async conns get the same guarantee without acquiring their lock:
    ``.lock`` is an ``asyncio.Lock`` and acquiring it from the drain
    thread would be a cross-thread asyncio-lock call (undefined
    behavior). ``AsyncConnection`` holds it around every libpq call
    (COMMIT included), so the drain thread waits, within the same
    budget, until ``lock.locked()`` reads False — a plain attribute
    read, no event-loop interaction — and only then finishes the
    pgconn. Without the wait, a force-close landing while a COMMIT
    was on the wire cut it mid-reply: the app saw "connection socket
    closed", an unknown outcome on the outgoing universe (r12 drain_zero
    c3 dual_write). Pool-borrowed sync conns take ``Connection.lock``
    before ``pgconn.finish()`` for the same reason. A statement started
    between the wait and the finish still fails on the closed socket.
    """
    try:
        # Lazy import — ``..connection_async`` pulls ``.yb.state`` which
        # pulls back into this subpackage; top-level import would cycle.
        from ..connection_async import AsyncConnection
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
        if isinstance(c, AsyncConnection):
            if not _wait_async_lock_free(c, budget):
                return _CloseLockTimeoutError(
                    f"async conn.lock still held after {budget}s — "
                    f"a statement (possibly COMMIT) is on the wire; "
                    f"escalate to hard-fd close"
                )
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
        # Sync branch — acquire the per-conn lock first (DB-25090).
        # Missing lock on a mock / test shim: fall through to the
        # direct close path (what we did pre-fix). Real Connection
        # instances always have one.
        # Pool-borrowed conn: ``psycopg_pool.ConnectionPool`` patches
        # ``Connection.close`` with ``close_returns=True`` so that
        # ``c.close()`` reroutes to ``pool.putconn(c)`` — the conn
        # ends up back in the idle pool, NOT closed. The next
        # ``pool.getconn()`` hands the same stale-cluster socket to
        # another worker and the dual-write window re-opens. Bypass
        # by calling ``pgconn.finish()`` directly for any conn whose
        # ``_pool`` is set (agent-K PR #13:370), under the same lock.
        lock = getattr(c, "lock", None)
        is_pool = getattr(c, "_pool", None) is not None
        if lock is None:
            if is_pool:
                try:
                    c.pgconn.finish()
                finally:
                    _release_registry_count(c)
            else:
                c.close()
            return None
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
            if is_pool:
                try:
                    c.pgconn.finish()
                finally:
                    _release_registry_count(c)
            else:
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


def _tracked_snapshot(state: "ClusterState") -> list:
    """List of ``state.tracked_conns`` taken under ``state.lock``, the
    lock connect holds to add / discard (Review #10, Amogh): iterating
    the WeakSet while another thread adds to it raises "Set changed size
    during iteration"."""
    lock = getattr(state, "lock", None)
    if lock is None:
        return list(state.tracked_conns)
    with lock:
        return list(state.tracked_conns)


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
    hold: "yb_hold.Hold | None" = None, held: "list | None" = None,
) -> int:
    """Close every tracked direct conn that is currently IDLE (not in a
    transaction) and not yet closed. Returns the number of conns closed.

    With ``hold`` (a failback that may be refused, DB-25803), hold each
    such conn instead and append it to ``held``; nothing is closed.

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
        if hold is not None:
            if getattr(c, "_yb_hold", None) is not hold:
                yb_hold.hold_conn(c, hold)
                assert held is not None
                held.append(c)
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


def _close_held(hold: "yb_hold.Hold", held: list, outgoing_uuid: str) -> None:
    """The failback goes ahead: fail the round-trips blocked on ``hold``
    and close the conns Phase 1 held. None of them has a round-trip on
    the wire, so each close is quick; a blocked owner releases the conn
    lock as soon as its round-trip raises. A conn whose close fails keeps
    its hold, so every later round-trip on it raises too."""
    yb_hold.close(hold)
    failed = 0
    for c in held:
        if getattr(c, "closed", False):
            continue
        if _safe_close(c) is not None:
            failed += 1
    if held:
        logger.info(
            "xCluster drain: closed %d held conn(s) on %s before the flip"
            "%s",
            len(held) - failed, outgoing_uuid,
            f" ({failed} close(s) failed)" if failed else "",
        )


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
    operator: bool = False,
) -> None:
    """Apply a per-cluster CB status transition, running the barrier
    drain against the outgoing cluster iff the routing target actually
    changes.

    Called from the probe thread — see health_probe.HealthProbe._maybe_apply
    — and, with ``operator=True``, from
    ``ClusterRegistry.reset_failover_group``. An operator failback runs
    the same drain and Phase 2 as an automatic one; only the
    autoFailbackEnabled gate does not apply, since the operator is the
    manual failback that setting waits for.
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
        with group.lock:
            group.active_before_no_viable = before_active
        _apply_status_change(group, which_cluster, new_status)
        return

    if before_active is None:
        # Recovery from no-viable. Routing changes (None → after_active),
        # so run a real drain against the OTHER side: it closes anything
        # still open there (e.g. conns that outlived a probe-only
        # outage) before dispatch resumes onto after_active.
        #
        # Phase 2 runs when the other side was the one serving when
        # no-viable began: writes it committed just before going down
        # may still be replicating. The primary recovering after the
        # secondary served is a failback: it runs the failback lag wait,
        # but with no healthy side to stay on, neither
        # onPhase2Timeout=fail nor autoFailbackEnabled=false holds it
        # back: both only keep routing on a secondary that is still
        # serving, and a healthy primary that may not have caught up
        # beats no cluster at all (same reasoning as failover). The
        # secondary recovering after the primary served runs the A→B lag
        # wait, which like any failover proceeds on timeout. Otherwise
        # Phase 2 is skipped: the other side was not serving, so there
        # is nothing in flight to replicate.
        outgoing = group.secondary if after_active == "primary" else group.primary
        outgoing_label = "secondary" if after_active == "primary" else "primary"
        # Unknown (None): assume it served, so the lag wait still runs.
        outgoing_served = group.active_before_no_viable in (outgoing_label, None)
        is_failback = after_active == "primary" and outgoing_served
        if is_failback and not group.auto_failback_enabled and not operator:
            logger.warning(
                "xCluster failback despite autoFailbackEnabled=false: %s CB "
                "reported HEALTHY and the secondary is down, so there is no "
                "secondary to stay on (primary_uuid=%s).",
                which_cluster, group.primary.uuid,
            )
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
            skip_phase2=not outgoing_served,
            phase2_may_abort=False,
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
    if is_failback and not group.auto_failback_enabled and not operator:
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
    if fork_follow.is_leader(group):
        group._follow.publish_statuses(group.primary_status, group.secondary_status)
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
    phase2_may_abort: bool = True,
) -> None:
    """Full barrier-drain-flip sequence. See module docstring.

    ``skip_phase2`` is set for recovery from no-viable, where the
    outgoing side was not serving and there is no lag to wait for.
    ``phase2_may_abort=False`` (failback out of no-viable) makes the
    failback lag wait proceed where onPhase2Timeout=fail would abort:
    there is no healthy side to stay on."""
    # Snapshot the initial tracked-conn count for the log line — the
    # weakref set may shrink under GC during the drain.
    drain_started = time.monotonic()
    tracked_now = _tracked_snapshot(outgoing)
    initial_tracked = len(tracked_now)
    initial_in_flight = len(_in_flight_conns(tracked_now))
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

    # DB-25803: a failback that onPhase2Timeout=fail may refuse holds the
    # secondary's conns instead of closing them, so a refusal leaves the
    # side that keeps serving untouched. See yb/hold.py.
    hold = None
    held: list = []
    if (
        is_failback and phase2_may_abort and not skip_phase2
        and getattr(group, "failback_on_phase2_timeout", "fail") == "fail"
    ):
        hold = yb_hold.Hold()

    # 1. Pause dispatch (new opens block on the Condition).
    group.pause_dispatch()
    # DB-25060 — forked workers pause with us and report their in-flight
    # transactions back (fork_follow.py).
    follow = group._follow if fork_follow.is_leader(group) else None
    gen = group.barrier_generation
    if follow is not None:
        follow.publish_pause(gen, outgoing is group.primary)

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
                _tracked_snapshot(outgoing), outgoing.uuid, hold, held,
            )
            total_idle_closed += idle_closed
            in_txn = _in_flight_conns(_tracked_snapshot(outgoing))
            children_in_txn = (
                follow.children_in_flight(gen) if follow is not None else 0
            )
            if not in_txn and not children_in_txn:
                drained_naturally = True
                logger.debug(
                    "drain: all in-flight txns completed on %s "
                    "(tracked=%d, idle_closed_this_drain=%d)",
                    outgoing.uuid, len(_tracked_snapshot(outgoing)),
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
            c for c in _tracked_snapshot(outgoing)
            if c is not None and not getattr(c, "closed", False)
        ]
        if hold is not None:
            # Hold every survivor; close only those with a round-trip on
            # the wire, which a hold can't stop (an autocommit statement
            # would commit on the outgoing side during Phase 2).
            busy = []
            held_ids = {id(c) for c in held}
            for c in survivors:
                if yb_hold.hold_conn(c, hold):
                    busy.append(c)
                elif id(c) not in held_ids:
                    held.append(c)
            survivors = busy
            if held:
                logger.info(
                    "xCluster drain: holding %d tracked conn(s) on %s "
                    "through Phase 2 instead of closing them (failback "
                    "may be refused, DB-25803)",
                    len(held), outgoing.uuid,
                )
        children_live = 0
        if follow is not None:
            children_live = follow.live_children()
            follow.publish_kill(gen)

        # 4. Kill at timeout (step 3 in design doc §3). Force-close
        # any conns still in-flight at the deadline. Only reached with
        # a finite deadline; the -1 branch loops until in_txn is empty.
        #
        # Concurrency: fire close() on every survivor in parallel via
        # a bounded thread pool. A serial for-loop of c.close() blocks
        # on the FIRST hung TCP session for ~tcp_retries2 (up to ~15
        # min on Linux, worse with keepalive=0), which would wedge
        # drain past its own deadline. For any close() that doesn't
        # return within _KILL_AT_TIMEOUT_BUDGET_S (or never started),
        # escape by shutting its socket down — the blocked libpq call
        # errors out, and drain proceeds. Bounds total wall-clock at
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
            # Snapshot fds BEFORE close() nulls the pgconn state, and
            # dup each one: the escape hatch below shuts the socket down
            # through our own dup, so it can only ever reach this conn's
            # socket, even if libpq closes its fd and the number is
            # reused meanwhile. Review #6 (Amogh): the escape hatch
            # used to os.close() the borrowed fd number.
            snapshot = []
            for c in survivors:
                try:
                    fd = c.pgconn.socket
                    conn_id = f"conn=fd{fd}"
                except Exception:
                    fd, conn_id = None, f"conn={id(c):x}"
                dup_fd = None
                if isinstance(fd, int) and fd >= 0:
                    try:
                        dup_fd = os.dup(fd)
                    except OSError:
                        dup_fd = None
                snapshot.append((c, dup_fd, conn_id))

            hung = 0
            # NB: intentionally NOT using `with ThreadPoolExecutor()` —
            # its __exit__ calls shutdown(wait=True), which would block
            # on hung workers and re-introduce the very problem we're
            # solving. Manual instantiation + explicit shutdown(wait=
            # False) in finally lets a stuck close() worker remain
            # detached in the background; its socket has been shut
            # down (see below), so the stuck send() errors out and the
            # worker unwinds, and libpq closes its own fd.
            ex = concurrent.futures.ThreadPoolExecutor(
                max_workers=min(len(survivors), 32),
                thread_name_prefix="yb-kill",
            )
            try:
                fut_by_conn = {
                    ex.submit(_safe_close, c): (c, dup_fd, conn_id)
                    for c, dup_fd, conn_id in snapshot
                }
                concurrent.futures.wait(
                    list(fut_by_conn),
                    timeout=_KILL_AT_TIMEOUT_BUDGET_S,
                )
                for fut, (c, dup_fd, conn_id) in fut_by_conn.items():
                    # Escape hatch: shutdown(SHUT_RDWR), never close.
                    # The fd is not ours: its owner (a thread inside
                    # libpq, or the close() worker blocked in PQfinish)
                    # still uses it. Shutdown unblocks that thread's
                    # read/write; the fd number stays allocated until
                    # libpq closes it, so no other socket can be hit
                    # (agent-K PR #21:672, Review #6).
                    needs_hard_close = False
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
                            # dual-write.
                            needs_hard_close = True
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
                    elif fut.cancel():
                        # Never started: queued behind max_workers past
                        # the budget. Nothing closed this conn, and its
                        # owner may be inside libpq right now.
                        needs_hard_close = True
                        hard_close_reason = (
                            f"close() still queued after "
                            f"{_KILL_AT_TIMEOUT_BUDGET_S}s budget"
                        )
                    else:
                        # Started but not returned: blocked waiting for
                        # conn.lock, or close() hung in PQfinish's
                        # send(). Shutdown unblocks either.
                        needs_hard_close = True
                        hard_close_reason = (
                            f"close() still running past "
                            f"{_KILL_AT_TIMEOUT_BUDGET_S}s budget"
                        )
                    if needs_hard_close:
                        hung += 1
                        if dup_fd is not None:
                            _shutdown_socket_fd(dup_fd)
                            logger.warning(
                                "kill-at-timeout: %s hard-fd-shutdown "
                                "(%s on %s)",
                                conn_id, hard_close_reason, outgoing.uuid,
                            )
                        else:
                            logger.warning(
                                "kill-at-timeout: %s needs hard-close but "
                                "no fd available (%s on %s)",
                                conn_id, hard_close_reason, outgoing.uuid,
                            )
            finally:
                # wait=False: never block on hung workers. Any still-
                # running _safe_close call keeps executing after we
                # return — its future is abandoned and it unwinds once
                # its shut-down socket errors out.
                ex.shutdown(wait=False)
                # Our dups only: closing them never touches the conns'
                # own fds.
                for _c, dup_fd, _conn_id in snapshot:
                    if dup_fd is not None:
                        try:
                            os.close(dup_fd)
                        except OSError:
                            pass

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
        if follow is not None and children_live:
            pending = follow.wait_children_killed(gen, _KILL_AT_TIMEOUT_BUDGET_S)
            if pending:
                logger.warning(
                    "xCluster drain: %d forked worker(s) did not confirm "
                    "closing their conns on %s within %ss",
                    pending, outgoing.uuid, _KILL_AT_TIMEOUT_BUDGET_S,
                )

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
            # Re-stamp the freshness anchor now that the outgoing side
            # has stopped committing. Phase 1 only stops new opens; work
            # in flight kept committing until here, and Phase 2 has to
            # see those commits replicated, not just the ones before
            # pause_dispatch.
            group.barrier_ts = time.time()
        closed_any = bool(survivors) or total_idle_closed > 0 or children_live > 0
        closed_at = time.monotonic()

        # 4b. Phase 2 — YBA replication-lag wait. Direction-aware:
        #   * Failover  (UNHEALTHY): poll the A→B config UUID iff set.
        #   * Failback  (HEALTHY):   poll the B→A config UUID iff set
        #                            (per §3.5.1 symmetric behaviour).
        # A missing config UUID in either direction fails open — same
        # semantics as YBA being unreachable.
        #
        # Wrap in `Phase2TimeoutError` catch. On
        # ``yb.failback.onPhase2Timeout="fail"`` (the default), a failback lag-wait raises
        # when it times out before converging or YBA stays unreachable. We log, record the
        # timeout timestamp, and SKIP the status change — routing
        # stays on the current (known-good) side. Dispatch still
        # resumes in the finally so the app isn't stuck. The probe
        # will try again on its next tick.
        phase2_aborted = False
        # Direction of the drain, not of the CB report: recovery from
        # no-viable to the secondary is a HEALTHY report but drains A→B.
        lag_failback = is_failback
        phase2_budget = _budget_deadline(
            group.failback_global_time_budget_secs if lag_failback
            else group.global_time_budget_secs,
            drain_started,
        )
        if not skip_phase2:
            phase2_budget = _retry_pending_yba(group, lag_failback, phase2_budget)
        cert_error = getattr(group, "yba_cert_error", "")
        refuse_unverified = (
            lag_failback and phase2_may_abort and not skip_phase2
            and group.yba_client is None and cert_error
            and getattr(group, "failback_on_phase2_timeout", "fail") == "fail"
        )
        if refuse_unverified or (group.yba_client is not None and not skip_phase2):
            try:
                if refuse_unverified:
                    # Review #8: a YBA certificate that fails verification
                    # turns the lag phase off, and a failback that went
                    # ahead then would skip the lag check that
                    # onPhase2Timeout=fail exists to enforce.
                    raise Phase2TimeoutError(
                        f"failback lag can't be checked: {cert_error}"
                    )
                if not lag_failback and group.xcluster_config_uuid:
                    _wait_for_replication_lag(
                        group, which_cluster, is_failback=False,
                        budget_deadline=phase2_budget,
                        outgoing=outgoing,
                    )
                elif lag_failback and group.failback_xcluster_config_uuid:
                    # Passed only when set, so stand-ins without the
                    # parameter keep working.
                    kw = {} if phase2_may_abort else {"may_abort": False}
                    _wait_for_replication_lag(
                        group, which_cluster, is_failback=True,
                        budget_deadline=phase2_budget,
                        outgoing=outgoing,
                        **kw,
                    )
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
            if hold is not None:
                # The failback goes ahead: close what Phase 1 held.
                _close_held(hold, held, outgoing.uuid)
                if held:
                    closed_any = True
                    closed_at = time.monotonic()
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
        if hold is not None and hold.verdict is None:
            # Refused (or the drain failed): routing stays on the
            # secondary, so its held conns carry on. After the resume, so
            # a pool check waiting on a hold sees dispatch running again.
            yb_hold.release(hold, held)
            if held:
                logger.info(
                    "xCluster drain: released %d held conn(s) on %s; "
                    "routing stays there",
                    len(held), outgoing.uuid,
                )
        if follow is not None:
            follow.publish_resume(group.primary_status, group.secondary_status)
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
# default (reference.conf:985). A YBA sample's ``x`` is the range-query
# evaluation step, which follows the request time, not the scrape time.
# The liveness gate therefore counts a confirmation only when it is at
# least one scrape interval after the previous one, so N confirmations
# span N distinct scrapes.
_LAG_SCRAPE_HINT_S = 10.0

# Time kept back from ``globalTimeBudgetSecs`` for the work after the lag
# wait returns (status change, dispatch resume). Stress budget_split
# measured drain start → drain complete at 20.0s against a 20s budget:
# the 40/60 split leaves no room for Phase 1's force-close or the flip.
_BUDGET_FLIP_MARGIN_S = 0.25


def _budget_deadline(budget_secs: "int | None", drain_started: float) -> "float | None":
    """Monotonic time by which the lag wait must return so that the whole
    drain fits ``globalTimeBudgetSecs``; None without a budget."""
    if budget_secs is None:
        return None
    return drain_started + budget_secs - _BUDGET_FLIP_MARGIN_S


def _retry_pending_yba(
    group: "FailoverGroup", is_failback: bool, budget_deadline: "float | None",
) -> "float | None":
    """Retry a bootstrap YBA probe that failed transiently (see
    ``ClusterRegistry._wire_yba_lag_phase``) before this drain's lag wait.

    The probe and the lag wait share one lag-wait timeout: the probe is
    capped at ``lag_wait_timeout_s`` from now (and the global budget), and
    when it enables Phase 2 the returned deadline makes the lag wait end
    there too. Otherwise ``budget_deadline`` comes back unchanged."""
    if group.yba_client is not None or getattr(group, "yba_pending_params", None) is None:
        return budget_deadline
    timeout_s = group.failback_lag_wait_timeout_s if is_failback else group.lag_wait_timeout_s
    limit = time.monotonic() + timeout_s
    if budget_deadline is not None:
        limit = min(limit, budget_deadline)
    from .registry import ClusterRegistry      # late import; cycle-safe
    ClusterRegistry.instance().retry_yba_lag_phase(group, deadline=limit)
    return limit if group.yba_client is not None else budget_deadline


def _all_nodes_down(state: "ClusterState") -> bool:
    """True when the registry has every node of ``state`` quarantined."""
    with state.lock:
        nodes = list(state.nodes.values())
    return bool(nodes) and all(n.is_down for n in nodes)


def _wait_for_replication_lag(
    group: "FailoverGroup",
    which_cluster: str,
    is_failback: bool = False,
    budget_deadline: "float | None" = None,
    outgoing: "ClusterState | None" = None,
    may_abort: bool = True,
) -> None:
    """Poll YBA for the outgoing side's replication lag every second, exit
    on any of three conditions:

      1. ``lag_ms ≤ threshold_replication_lag_ms`` — converged. Safe to
         switch; the target has caught up to within the operator's RPO.
      2. ``elapsed ≥ lag_wait_timeout_s`` — timed out. Switch anyway. Some
         data may be un-replicated; that's a known consequence.
      3. ``_LAG_UNREACHABLE_STRIKES`` consecutive YBA errors — fail-open.
         Better to switch without lag confirmation than to wedge failover
         because the YBA control plane happens to be down. A failback
         with ``yb.failback.onPhase2Timeout="fail"`` aborts instead (stays on the
         healthy secondary), as on a timeout.

    Never blocks failover on YBA availability. Never blocks longer than
    ``lag_wait_timeout_s``, except that a failback which would abort with
    only pre-barrier ≤ threshold samples waits once for the next YBA
    sample (at most one scrape interval plus one poll). Log at WARNING on
    entry and exit so operators
    can read the outcome from the log timeline.

    ``is_failback`` picks between the A→B config + failover knobs
    (default) and the B→A config + failback knobs (see design doc §3.5.1).

    ``outgoing`` is the outgoing cluster's state; while the driver has
    every one of its nodes quarantined, no sample counts toward
    convergence.

    ``may_abort=False`` treats ``onPhase2Timeout=fail`` as ``proceed``
    (failback out of no-viable: there is no healthy side to stay on).
    """
    from .yba_client import YBAClientError

    policy = getattr(group, "failback_on_phase2_timeout", "proceed")
    if is_failback and policy == "fail" and not may_abort:
        logger.warning(
            "lag wait: yb.failback.onPhase2Timeout=fail not applied — "
            "failback out of no viable cluster has no healthy side to "
            "stay on; the flip proceeds on timeout"
        )
        policy = "proceed"

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

    # DB-24483 — snapshot the freshness anchor: stamped by
    # ``pause_dispatch`` and re-stamped when Phase 1 completes, so it
    # marks the last commit on the outgoing side. Zero means "no barrier
    # armed" — a fail-open path (shouldn't happen inside a drain, but
    # defensive): the freshness gate then treats any sample as fresh.
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
    if budget_deadline is not None and budget_deadline < deadline:
        # Phase 1 ran past its share of globalTimeBudgetSecs; the lag
        # wait gets what is left of the budget, not its full share.
        logger.info(
            "lag wait: %.1fs left of the global time budget (lag wait "
            "timeout %ds)", max(0.0, budget_deadline - started), timeout_s,
        )
        deadline = budget_deadline
    # A YBA call made under a budget is cut at the deadline as well.
    from .yba_client import YBAClient
    pass_deadline = (
        budget_deadline is not None and isinstance(group.yba_client, YBAClient)
    )
    unreachable_streak = 0
    # DB-24483 — liveness tracking: consecutive distinct post-barrier
    # samples at ≤ threshold.
    confirmed_count = 0
    # A sample_ts less than one scrape interval after the last counted
    # one does NOT advance the liveness count.
    last_counted_sample_ts: float | None = None
    warned_no_sample_ts = False
    # sample_ts of the latest ≤ threshold sample that was not counted only
    # because it predates the barrier. None once a later poll says
    # anything else (lagged, counted, no data).
    pre_barrier_pass_ts: float | None = None
    extended = False

    def _extend_to_next_sample() -> bool:
        """A failback that would abort while the only ≤ threshold samples
        predate the barrier waits once for YBA's next sample instead.
        Re-stamping the barrier at the end of Phase 1 leaves a short lag
        wait (6s on a 10s budget) less than one sample interval after the
        barrier, so a caught-up universe could never confirm (stress
        failback_refused_on_timeout/budget_10). The extension can run past
        globalTimeBudgetSecs by up to one sample interval plus one poll."""
        nonlocal deadline, extended
        if extended or pre_barrier_pass_ts is None:
            return False
        if not (policy == "fail" and is_failback):
            return False
        next_sample_in = pre_barrier_pass_ts + _LAG_SCRAPE_HINT_S - time.time()
        new_deadline = min(
            time.monotonic() + max(0.0, next_sample_in) + _LAG_POLL_INTERVAL_S,
            deadline + _LAG_SCRAPE_HINT_S + _LAG_POLL_INTERVAL_S,
        )
        if new_deadline <= deadline:
            return False
        extended = True
        logger.warning(
            "lag wait: lag=0 so far only on samples that predate the "
            "barrier; extending the failback lag wait by %.1fs for YBA's "
            "next sample (past the global time budget if one is set)",
            new_deadline - deadline,
        )
        deadline = new_deadline
        return True

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
        timed_out = now >= deadline or deadline - now < _MIN_POLL_BUDGET_S
        if timed_out and _extend_to_next_sample():
            continue
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
            if policy == "fail" and is_failback:
                logger.error(
                    "lag wait timeout after %.1fs (direction=%s, "
                    "threshold=%.1fms) — ABORTING flip; routing stays "
                    "on current side (set yb.failback.onPhase2Timeout="
                    "proceed to flip anyway)",
                    elapsed, direction_label, threshold_ms,
                )
                raise Phase2TimeoutError(
                    f"lag wait timed out after {elapsed:.1f}s "
                    f"(direction={direction_label}, "
                    f"threshold={threshold_ms:.1f}ms)"
                )
            if policy == "fail" and not is_failback:
                logger.warning(
                    "lag wait timeout after %.1fs — proceeding with flip "
                    "despite failback_on_phase2_timeout='fail' (direction=failover; "
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
            # must honour the failback_on_phase2_timeout="fail" policy for the
            # FAILBACK direction (same reasoning as the deadline-hit
            # branch above). Previously this branch just returned,
            # bypassing the abort and silently flipping onto a
            # known-lagged target during tight lag_wait_timeout_s
            # settings.
            # Same "lag wait timeout" prefix as the deadline branch so log
            # consumers see one timeout outcome, not two.
            elapsed = now - started
            if policy == "fail" and is_failback:
                logger.error(
                    "lag wait timeout after %.1fs (direction=failback, "
                    "threshold=%.1fms; %.1fs left is below the %.1fs poll "
                    "budget) — ABORTING flip; routing stays on current side",
                    elapsed, threshold_ms, deadline - now, _MIN_POLL_BUDGET_S,
                )
                raise Phase2TimeoutError(
                    f"lag wait timed out after {elapsed:.1f}s "
                    f"(direction=failback, threshold={threshold_ms:.1f}ms)"
                )
            logger.warning(
                "lag wait timeout after %.1fs (%.1fs left is below the "
                "%.1fs poll budget) — proceeding with flip (direction=%s)",
                elapsed, deadline - now, _MIN_POLL_BUDGET_S, direction_label,
            )
            return

        poll_started = time.monotonic()
        report = None
        try:
            # DB-24483: prefer the sample-returning variant so we can
            # log how old the YBA data was. Custom lag sources that
            # only implement the legacy ``get_committed_lag_ms`` fall
            # back to that — otherwise an AttributeError would land in
            # the except-Exception path below and silently skip the
            # lag wait (agent-K PR #15:698). We still trust YBA as the
            # lag source of truth; surfacing the age gives operators a
            # concrete number to reason about after a flip.
            if hasattr(group.yba_client, "get_committed_lag_report"):
                kw = {"deadline": deadline} if pass_deadline else {}
                report = group.yba_client.get_committed_lag_report(
                    xcluster_uuid, **kw,
                )
                lag_ms = None if report is None else report["lag_ms"]
                sample_ts = None if report is None else report["sample_ts"]
            elif hasattr(group.yba_client, "get_committed_lag_sample"):
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
            if report is not None:
                # Per producer node, so a converge on a universe whose
                # nodes stopped reporting can be read off the log.
                age_str += " nodes=" + ",".join(
                    f"{node}:{value:g}" for node, value in report["values"].items()
                )
                if report["stale"]:
                    age_str += f" stale={','.join(report['stale'])}"
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
            if pass_deadline and time.monotonic() >= deadline:
                # Cut at the budget deadline: a timeout, not a strike.
                logger.info("lag wait: YBA call cut at the budget deadline: %s", exc)
                continue
            unreachable_streak += 1
            logger.warning(
                "lag wait: YBA unreachable (%d/%d): %s",
                unreachable_streak, _LAG_UNREACHABLE_STRIKES, exc,
            )
            if unreachable_streak >= _LAG_UNREACHABLE_STRIKES:
                # yb.failback.onPhase2Timeout="fail" means "don't fail back onto a
                # primary we can't confirm has caught up"; an unreachable
                # YBA is exactly that case. Staying on the secondary is
                # safe (it's healthy), so a failback aborts like a
                # timeout. A failover still proceeds: the current side
                # was just declared unhealthy.
                if policy == "fail" and is_failback:
                    elapsed = time.monotonic() - started
                    logger.error(
                        "lag wait: YBA unreachable %d× after %.1fs "
                        "(direction=failback) — ABORTING flip; routing "
                        "stays on current side (set yb.failback."
                        "onPhase2Timeout=proceed to flip anyway)",
                        _LAG_UNREACHABLE_STRIKES, elapsed,
                    )
                    raise Phase2TimeoutError(
                        f"YBA unreachable {_LAG_UNREACHABLE_STRIKES}x after "
                        f"{elapsed:.1f}s (direction=failback)"
                    )
                logger.warning(
                    "lag wait: YBA unreachable %d× — proceeding with flip "
                    "(no lag confirmation)",
                    _LAG_UNREACHABLE_STRIKES,
                )
                return
            lag_ms = None
            sample_ts = None
        except Exception as exc:
            # Belt-and-braces — never let an unexpected exception in the
            # YBA client wedge the drain barrier. A failback under
            # yb.failback.onPhase2Timeout="fail" aborts instead, as in
            # the unreachable branch above: no lag confirmation means no
            # failback onto the primary.
            if policy == "fail" and is_failback:
                elapsed = time.monotonic() - started
                logger.error(
                    "lag wait: unexpected YBA error after %.1fs "
                    "(direction=failback) — ABORTING flip; routing stays "
                    "on current side (set yb.failback.onPhase2Timeout="
                    "proceed to flip anyway)",
                    elapsed, exc_info=True,
                )
                raise Phase2TimeoutError(
                    f"YBA lag read failed after {elapsed:.1f}s "
                    f"(direction=failback): {exc!r}"
                ) from exc
            logger.warning(
                "lag wait: unexpected YBA error — proceeding with flip",
                exc_info=True,
            )
            return

        pre_barrier_pass = False
        if lag_ms is not None and lag_ms <= threshold_ms:
            elapsed = time.monotonic() - started
            # DB-24483 — freshness gate. A YBA sample whose ``sample_ts``
            # predates the barrier (or sits in the ``barrier_ts +
            # freshness_s`` grace window) cannot have observed writes
            # that landed before ``pause_dispatch``. Trusting it is the
            # exact bug that let the driver flip to a 17s-behind B.
            # A sample with no ``sample_ts`` (YBA sent no ``x`` axis)
            # skips both this gate and the liveness spacing below: every
            # ≤ threshold poll counts. That is logged once per lag wait.
            # ``freshness_s == 0`` opts out of this gate.
            if sample_ts is None and not warned_no_sample_ts:
                warned_no_sample_ts = True
                logger.warning(
                    "lag wait: YBA returned a lag sample without a "
                    "timestamp — the freshness and liveness gates can't "
                    "run; counting every poll at or below the threshold",
                )
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
                    pre_barrier_pass_ts = None
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
            # from "same stale replay reused". Require N samples at
            # ≤ threshold, each at least one scrape interval after the
            # previous counted one, before flipping.
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
                pre_barrier_pass = True
                logger.info(
                    "lag wait: lag=%.1fms sample at %.3f predates the "
                    "barrier %.3f — not counting (confirmations=%d/%d, "
                    "elapsed=%.1fs / %ds)",
                    lag_ms, sample_ts, barrier_ts, confirmed_count,
                    required_samples, elapsed, timeout_s,
                )
            elif report is not None and (report["stale"] or not report["leader_reported"]):
                # A producer whose tablets lost their leaders reports 0
                # from the nodes still up, and nothing from the nodes
                # down (stress phase2_claim, a_majority_down: converged
                # on lag=0 two seconds after the barrier while a batch
                # could not leave A). Neither is a caught-up universe.
                why = (
                    f"node(s) {','.join(report['stale'])} stopped reporting"
                    if report["stale"] else "no tablet leader reporting (all 0)"
                )
                logger.info(
                    "lag wait: lag=%.1fms but %s — not counting "
                    "(confirmations=%d/%d, elapsed=%.1fs / %ds)",
                    lag_ms, why, confirmed_count, required_samples,
                    elapsed, timeout_s,
                )
                confirmed_count = 0
                last_counted_sample_ts = None
            elif outgoing is not None and _all_nodes_down(outgoing):
                # Every outgoing node refused the driver, so no producer
                # tserver can have produced this sample: YBA keeps
                # returning the last value scraped before they went down
                # (stress phase2_claim, region_outage_with_yba and
                # a_outage_into_limited_b: all three A nodes at 1us ten
                # seconds after their processes were stopped, batches
                # committed before the stop still missing on B).
                logger.info(
                    "lag wait: lag=%.1fms but every node of the outgoing "
                    "universe is down — not counting (confirmations=%d/%d, "
                    "elapsed=%.1fs / %ds)",
                    lag_ms, confirmed_count, required_samples,
                    elapsed, timeout_s,
                )
                confirmed_count = 0
                last_counted_sample_ts = None
            elif (
                sample_ts is not None
                and last_counted_sample_ts is not None
                and sample_ts < last_counted_sample_ts + _LAG_SCRAPE_HINT_S
            ):
                logger.info(
                    "lag wait: lag=%.1fms ≤ threshold but sample_ts %.3f "
                    "is within one scrape interval (%.0fs) of the last "
                    "counted sample — not counting (confirmations=%d/%d, "
                    "elapsed=%.1fs / %ds)",
                    lag_ms, sample_ts, _LAG_SCRAPE_HINT_S, confirmed_count,
                    required_samples, elapsed, timeout_s,
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
        pre_barrier_pass_ts = sample_ts if pre_barrier_pass else None

        # Poll again in ~1s. If the remaining budget is smaller, sleep
        # only that much so we don't overshoot the deadline.
        sleep_s = min(_LAG_POLL_INTERVAL_S, max(0.0, deadline - time.monotonic()))
        if sleep_s > 0:
            time.sleep(sleep_s)
