"""
Background health-probe thread for xCluster failover.

One daemon ``threading.Thread`` per ``FailoverGroup`` (so per primary
cluster) per Python process. On each tick the thread calls
``cluster_status_check`` (currently a stub that always returns HEALTHY)
and, if the result differs from the current ``group.status`` AND the
cool-down has elapsed, atomically writes the new status under
``group.lock``.

The probe is always sync regardless of whether the application uses
``Connection`` or ``AsyncConnection`` — it uses the primary's
``control_sync`` connection and avoids the "whose event loop is this"
problem entirely (see design doc §10).

Concurrency invariants:
  * The probe NEVER blocks the application path. The dispatcher reads
    ``group.status`` under ``group.lock`` (microseconds); the probe writes
    it under the same lock.
  * Probe exceptions are caught and logged at WARNING. The probe loop
    NEVER exits on a strategy error — that would make failover silently
    stop working, which is far worse than a noisy log.

Spec / design references:
  * /tmp/xcluster_failover_design.html §4 (architecture overview)
  * /tmp/xcluster_failover_design.html §6 (failback hysteresis + cool-down)
  * Functional spec — "Circuit Breaker (CB) — How it'll work"
"""

# Copyright (C) 2026 Yugabyte

from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING

from .health import HealthResult, cluster_status_check

if TYPE_CHECKING:
    from .registry import FailoverGroup

logger = logging.getLogger(__name__)


# One-time INFO emitted by the FIRST probe instance to start in this process.
# Resolves design doc §15 "Stub vs. real surprise" — operators should know the
# detection logic is a stub so they don't expect automatic failover to fire on
# real cluster failures yet.
_STUB_NOTICE_EMITTED = False
_STUB_NOTICE_LOCK = threading.Lock()


def _emit_stub_notice_once() -> None:
    global _STUB_NOTICE_EMITTED
    with _STUB_NOTICE_LOCK:
        if _STUB_NOTICE_EMITTED:
            return
        _STUB_NOTICE_EMITTED = True
    logger.info(
        "xCluster failover plumbing active; health detection is currently a "
        "stub (always HEALTHY). Real tracker-table-based detection lands in "
        "a follow-on patch; until then, automatic failover only triggers via "
        "FailoverGroup.force_status or ClusterRegistry.reset_failover_group."
    )


class HealthProbe:
    """Daemon thread that polls cluster health and updates ``group.status``.

    Lifecycle: created during failover-group bootstrap, ``start()`` called
    immediately. ``stop()`` called from ``ClusterRegistry.clear()`` /
    ``aclear()``. Idempotent — repeated ``start()`` / ``stop()`` calls are
    safe.

    The thread is a daemon so a hung join during process teardown can't
    block interpreter shutdown. ``stop()`` still joins with a timeout for
    clean test teardown.
    """

    def __init__(self, group: "FailoverGroup", interval_s: int) -> None:
        self._group = group
        # Clamp to a minimum so tests can use very small intervals (e.g.
        # 0.05s) without us spinning. Anything sub-millisecond is silly.
        self._interval_s = max(float(interval_s), 0.001)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._started = False

    def start(self) -> None:
        """Spin up the daemon thread. Idempotent."""
        if self._started:
            return
        self._started = True
        _emit_stub_notice_once()
        self._thread = threading.Thread(
            target=self._run,
            name=f"yb-health-probe-{id(self._group):x}",
            daemon=True,
        )
        self._thread.start()
        logger.debug(
            "HealthProbe started: primary_uuid=%s, interval_s=%s",
            self._group.primary.uuid, self._interval_s,
        )

    def stop(self, join_timeout: float = 2.0) -> None:
        """Signal the thread to exit and wait up to ``join_timeout`` seconds
        for it to drain. Idempotent."""
        if not self._started:
            return
        self._stop_event.set()
        t = self._thread
        if t is not None and t.is_alive():
            t.join(timeout=join_timeout)
            if t.is_alive():
                logger.warning(
                    "HealthProbe thread did not stop within %.1fs (primary_uuid=%s)",
                    join_timeout, self._group.primary.uuid,
                )
        logger.debug(
            "HealthProbe stopped: primary_uuid=%s", self._group.primary.uuid
        )

    def _run(self) -> None:
        """Probe loop. Sleeps `interval_s` between ticks via the stop event
        so `stop()` interrupts the sleep promptly."""
        group = self._group
        while not self._stop_event.wait(timeout=self._interval_s):
            try:
                result = cluster_status_check(group)
            except Exception:
                logger.warning(
                    "cluster_status_check raised; treating as no-change "
                    "(primary_uuid=%s)", group.primary.uuid,
                    exc_info=True,
                )
                continue
            self._maybe_apply(result)

    def _maybe_apply(self, result: HealthResult) -> None:
        """Apply ``result`` to ``group.status`` if it represents a transition
        AND the cool-down has elapsed. Cool-down deferral is logged at DEBUG
        so an operator can see the probe noticed but chose to wait."""
        group = self._group
        now = time.monotonic()
        with group.lock:
            if group.status == result:
                logger.log(
                    5,  # TRACE — defined in psycopg.yb.__init__
                    "probe tick: no-op (status=%s, primary_uuid=%s)",
                    result.value, group.primary.uuid,
                )
                return
            if not group.can_transition(now):
                logger.debug(
                    "probe tick: transition %s → %s deferred by cool-down "
                    "(primary_uuid=%s, remaining=%.1fs)",
                    group.status.value, result.value, group.primary.uuid,
                    group.cooldown_s - (now - group.last_transition_time),
                )
                return
            previous = group.status
            group.status = result
            group.last_transition_time = now
        # Logged OUTSIDE the group lock — emit at INFO so operators see
        # transitions clearly.
        logger.info(
            "xCluster status transition: %s → %s (primary_uuid=%s)",
            previous.value, result.value, group.primary.uuid,
        )
