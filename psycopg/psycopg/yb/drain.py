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
  3. **Force-close survivors at the deadline.** If ``drain_timeout_s`` is
     positive and elapses with in-flight conns still open, ``conn.close()``
     each. The application sees ``psycopg.OperationalError`` and treats
     it like any transient disconnect. Sentinels:
       * ``-1`` — wait indefinitely; never force-close.
       * ``0`` — force-close immediately; no drain window.
       * ``N > 0`` — poll for N seconds, then force-close.
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

import logging
import time
import weakref
from typing import TYPE_CHECKING

from .health import HealthResult

if TYPE_CHECKING:
    from .registry import ClusterState, FailoverGroup

logger = logging.getLogger(__name__)

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

    if before_active == after_active or before_active is None:
        # No routing change — apply the status without disturbing
        # dispatch. This covers e.g. "secondary flips while primary is
        # HEALTHY" and "before was no-viable; new status doesn't change
        # active".
        _apply_status_change(group, which_cluster, new_status)
        return

    outgoing = group.primary if before_active == "primary" else group.secondary
    _drain_and_apply(
        group=group,
        which_cluster=which_cluster,
        new_status=new_status,
        outgoing=outgoing,
        drain_timeout_s=drain_timeout_s,
    )


def _apply_status_change(
    group: "FailoverGroup",
    which_cluster: str,
    new_status: HealthResult,
) -> None:
    """Non-drain status write. Under ``group.lock`` — atomic with
    reads by the dispatcher's ``active_cluster``."""
    now = time.monotonic()
    status_attr = f"{which_cluster}_status"
    transition_attr = f"{which_cluster}_last_transition_time"
    with group.lock:
        previous = getattr(group, status_attr)
        setattr(group, status_attr, new_status)
        setattr(group, transition_attr, now)
    logger.info(
        "xCluster %s status transition: %s → %s (primary_uuid=%s, "
        "no drain needed — routing unchanged)",
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
) -> None:
    """Full barrier-drain-flip sequence. See module docstring."""
    # Snapshot the initial tracked-conn count for the log line — the
    # weakref set may shrink under GC during the drain.
    initial_tracked = len(outgoing.tracked_conns)
    logger.info(
        "xCluster drain start: outgoing=%s → new=%s (primary_uuid=%s, "
        "drain_timeout_s=%s, initial_tracked=%d)",
        outgoing.uuid, new_status.value, group.primary.uuid,
        drain_timeout_s, initial_tracked,
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

        # 3. Poll until either all conns drain OR the deadline hits.
        survivors: list = []
        while True:
            in_txn = _in_flight_conns(outgoing.tracked_conns)
            if not in_txn:
                logger.debug(
                    "drain: all conns idle on %s (tracked=%d)",
                    outgoing.uuid, len(outgoing.tracked_conns),
                )
                break
            now = time.monotonic()
            if now >= deadline:
                survivors = in_txn
                break
            # Sleep in short slices so `drain_timeout_s == -1` still
            # returns promptly if all conns drain. 100 ms matches the
            # design doc §3.2 pseudocode.
            time.sleep(0.1)

        # 4. Force-close survivors (only reached with a finite deadline;
        # the -1 branch loops until in_txn is empty).
        for c in survivors:
            try:
                c.close()
            except Exception:
                logger.log(
                    5,   # TRACE
                    "drain: close() raised on %r",
                    c, exc_info=True,
                )
        if survivors:
            logger.warning(
                "drain deadline reached on %s; force-closed %d in-flight "
                "conn(s) (primary_uuid=%s)",
                outgoing.uuid, len(survivors), group.primary.uuid,
            )

        # 5. Apply the status change while dispatch is still paused so
        # the state machine transitions atomically before waiters wake.
        _apply_status_change(group, which_cluster, new_status)
    finally:
        # 6. Resume dispatch — release waiters.
        group.resume_dispatch()
    logger.info(
        "xCluster drain complete: dispatch resumed (primary_uuid=%s, "
        "outgoing=%s, force_closed=%d)",
        group.primary.uuid, outgoing.uuid,
        len(survivors) if 'survivors' in locals() else 0,
    )
