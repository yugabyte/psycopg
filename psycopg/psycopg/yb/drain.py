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
) -> None:
    """Full barrier-drain-flip sequence. See module docstring."""
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

        # 4. Kill at timeout (step 3 in design doc §3). Force-close
        # any conns still in-flight at the deadline. Only reached with
        # a finite deadline; the -1 branch loops until in_txn is empty.
        if survivors:
            logger.warning(
                "xCluster drain-wait deadline (%ss elapsed) — "
                "force-closing %d survivor(s) on %s",
                drain_timeout_s, len(survivors), outgoing.uuid,
            )
        # One INFO per conn so operators can trace exactly which
        # sessions got forced.
        for c in survivors:
            try:
                # Snapshot addressable identity before close() nulls state.
                try:
                    conn_id = f"conn=fd{c.pgconn.socket}"
                except Exception:
                    conn_id = f"conn={id(c):x}"
                c.close()
                logger.info(
                    "kill-at-timeout: %s force-closed (was in-flight on %s)",
                    conn_id, outgoing.uuid,
                )
            except Exception:
                logger.log(
                    5,   # TRACE
                    "drain: close() raised on %r",
                    c, exc_info=True,
                )
        if survivors:
            logger.warning(
                "kill-at-timeout complete on %s: %d force-closed "
                "(primary_uuid=%s)",
                outgoing.uuid, len(survivors), group.primary.uuid,
            )

        # Phase 1 complete — record the boundary and log the outcome so
        # the operator can tell at a glance which of the three cases
        # this was: (a) all conns naturally drained, (b) some conns
        # force-closed at deadline, (c) no work to do because they were
        # already idle/closed at start.
        now = time.monotonic()
        phase1_elapsed = now - (
            group.last_failover_start_ts or now
        )
        if initial_in_flight == 0:
            logger.info(
                "xCluster drain: Phase 1 exited early after %.2fs — "
                "no in-flight txns to wait on (initial_tracked=%d, "
                "in_flight_at_start=0)",
                phase1_elapsed, initial_tracked,
            )
        elif not survivors:
            logger.info(
                "xCluster drain: Phase 1 completed in %.2fs — all %d "
                "in-flight conns drained naturally before deadline "
                "(force_closed=0)",
                phase1_elapsed, initial_in_flight,
            )
        else:
            logger.info(
                "xCluster drain: Phase 1 hit the %ds drain-wait deadline "
                "→ kill-at-timeout fired: %d/%d conns force-closed, "
                "%d completed naturally",
                drain_timeout_s, len(survivors), initial_in_flight,
                initial_in_flight - len(survivors),
            )
        with group.lock:
            group.last_phase1_complete_ts = now

        # 4b. Phase 2 — YBA replication-lag wait. Direction-aware:
        #   * Failover  (UNHEALTHY): poll the A→B config UUID iff set.
        #   * Failback  (HEALTHY):   poll the B→A config UUID iff set
        #                            (per §3.5.1 symmetric behaviour).
        # A missing config UUID in either direction fails open — same
        # semantics as YBA being unreachable.
        if group.yba_client is not None:
            if new_status == HealthResult.UNHEALTHY and group.xcluster_config_uuid:
                _wait_for_replication_lag(group, which_cluster, is_failback=False)
            elif (
                new_status == HealthResult.HEALTHY
                and group.failback_xcluster_config_uuid
            ):
                _wait_for_replication_lag(group, which_cluster, is_failback=True)

        # Phase 2 complete — record the boundary (fires even when Phase 2
        # was skipped, so downstream code sees a consistent timeline).
        with group.lock:
            group.last_phase2_complete_ts = time.monotonic()

        # 5. Apply the status change while dispatch is still paused so
        # the state machine transitions atomically before waiters wake.
        # after_drain=True so the log line reflects "routing flipped"
        # rather than the "routing unchanged" wording used by the
        # non-drain no-op transition path.
        _apply_status_change(
            group, which_cluster, new_status, after_drain=True,
        )
    finally:
        # 6. Resume dispatch — release waiters.
        group.resume_dispatch()
        # Record the direction-appropriate completion timestamp.
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
        direction_label = "failback"
    else:
        threshold_ms = float(group.threshold_replication_lag_ms)
        timeout_s = group.lag_wait_timeout_s
        xcluster_uuid = group.xcluster_config_uuid
        direction_label = "failover"

    logger.warning(
        "lag wait start: direction=%s outgoing=%s xcluster_config=%s "
        "threshold=%.1fms timeout=%ds",
        direction_label, which_cluster, xcluster_uuid, threshold_ms, timeout_s,
    )

    started = time.monotonic()
    deadline = started + timeout_s
    unreachable_streak = 0

    while True:
        now = time.monotonic()
        if now >= deadline:
            elapsed = now - started
            logger.warning(
                "lag wait timeout after %.1fs — proceeding with flip",
                elapsed,
            )
            return

        try:
            lag_ms = group.yba_client.get_committed_lag_ms(xcluster_uuid)
            unreachable_streak = 0
            # One line per poll — operator can trace exactly what the
            # driver observed on each 1-second tick. INFO because
            # WARNING would spam in production; the demo bumps this
            # logger to INFO to make it visible.
            elapsed = time.monotonic() - started
            if lag_ms is None:
                logger.info(
                    "lag wait poll: no data reported by YBA "
                    "(elapsed=%.1fs / %ds)",
                    elapsed, timeout_s,
                )
            else:
                logger.info(
                    "lag wait poll: lag=%.1fms threshold=%.1fms "
                    "(elapsed=%.1fs / %ds)",
                    lag_ms, threshold_ms, elapsed, timeout_s,
                )
        except YBAClientError as exc:
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
            logger.warning(
                "lag wait converged in %.1fs (lag=%.1fms ≤ threshold=%.1fms) "
                "— switching now",
                elapsed, lag_ms, threshold_ms,
            )
            return

        # Poll again in ~1s. If the remaining budget is smaller, sleep
        # only that much so we don't overshoot the deadline.
        sleep_s = min(_LAG_POLL_INTERVAL_S, max(0.0, deadline - time.monotonic()))
        if sleep_s > 0:
            time.sleep(sleep_s)
