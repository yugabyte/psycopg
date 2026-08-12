"""
Per-cluster background health-probe threads for xCluster failover.

Each ``FailoverGroup`` owns **two** daemon threads — one per cluster —
constructed by ``ClusterRegistry.get_or_bootstrap_failover_group`` at
bootstrap. Each thread on every tick:

  1. Calls its cluster's circuit breaker (``check_primary_cluster`` or
     ``check_secondary_cluster``) and applies the result to that cluster's
     status field, subject to the per-cluster cool-down.
  2. Runs ``yb_servers()`` on the same cluster's control connection and
     merges the result into ``ClusterState.nodes`` via the registry's
     ``_do_refresh_sync`` helper.

Rolling both operations onto the same thread gives two properties from
design doc §4.1:

  * No contention on ``state.control_sync`` — the probe is the sole owner
    of that connection.
  * Topology stays fresh even when the app is idle. The old lazy-refresh
    path (in the dispatcher) is skipped for xCluster deployments; the
    probe cadence is authoritative.

Probes are always sync regardless of whether the application uses
``Connection`` or ``AsyncConnection`` — see design doc §10.

Concurrency invariants:
  * A probe NEVER blocks the application path. The dispatcher reads
    the per-cluster status flags under ``group.lock`` (microseconds);
    the probe writes them under the same lock.
  * CB exceptions are caught and logged at WARNING. The probe loop
    NEVER exits on a strategy error — that would make failover silently
    stop working, which is far worse than a noisy log.
"""

# Copyright (C) 2026 Yugabyte

from __future__ import annotations

import concurrent.futures
import logging
import threading
import time
from typing import TYPE_CHECKING, Callable

from . import health as _health
from .health import HealthResult

if TYPE_CHECKING:
    from .registry import FailoverGroup

logger = logging.getLogger(__name__)


# One-time INFO emitted by the FIRST probe instance to start in this process.
# Lets operators see "xCluster failover is active in this process" in their
# logs even when no transitions fire.
_STARTUP_NOTICE_EMITTED = False
_STARTUP_NOTICE_LOCK = threading.Lock()


def _emit_startup_notice_once() -> None:
    global _STARTUP_NOTICE_EMITTED
    with _STARTUP_NOTICE_LOCK:
        if _STARTUP_NOTICE_EMITTED:
            return
        _STARTUP_NOTICE_EMITTED = True
    logger.info(
        "xCluster failover plumbing active; per-cluster probes started. "
        "Health detection via the CircuitBreaker attached to "
        "group.primary_circuit_breaker / group.secondary_circuit_breaker "
        "(see demo/samples/ for reference implementations). Status "
        "transitions will be logged at INFO."
    )


# Tests monkey-patch these NAMES at module scope to steer the probe's decision.
# The probe reads them via ``_health`` on each tick, so patching
# ``psycopg.yb.health_probe.check_primary_cluster`` OR
# ``psycopg.yb.health.check_primary_cluster`` both work.
check_primary_cluster = _health.check_primary_cluster
check_secondary_cluster = _health.check_secondary_cluster


class HealthProbe:
    """Daemon thread that polls ONE cluster's health and refreshes its
    topology on every tick.

    Two instances per ``FailoverGroup`` — one with ``which_cluster="primary"``
    and one with ``which_cluster="secondary"``.

    Lifecycle: created during failover-group bootstrap, ``start()`` called
    immediately. ``stop()`` called from ``ClusterRegistry.clear()`` /
    ``aclear()``. Idempotent — repeated ``start()`` / ``stop()`` calls are
    safe.

    The thread is a daemon so a hung join during process teardown can't
    block interpreter shutdown. ``stop()`` still joins with a timeout for
    clean test teardown.
    """

    def __init__(
        self,
        group: "FailoverGroup",
        interval_s: float,
        which_cluster: str = "primary",
        check_timeout_s: float = 1.0,
        drain_timeout_s: int = 10,
    ) -> None:
        if which_cluster not in ("primary", "secondary"):
            raise ValueError(
                f"which_cluster must be 'primary' or 'secondary', "
                f"got {which_cluster!r}"
            )
        self._group = group
        self._which = which_cluster
        # Clamp to a minimum so tests can use very small intervals (e.g.
        # 0.05s) without us spinning. Anything sub-millisecond is silly.
        self._interval_s = max(float(interval_s), 0.001)
        # Wall-clock cap on each check() call. Floored at 1ms for tests.
        self._check_timeout_s = max(float(check_timeout_s), 0.001)
        # Passed through to trigger_drain when the CB transitions.
        # Sentinel semantics (see psycopg.yb.drain): -1 wait forever,
        # 0 kill immediately, N wait N then kill.
        self._drain_timeout_s = drain_timeout_s
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._started = False
        # Persistent single-worker executor so a stuck check() doesn't
        # multiply into N threads across ticks. If a check hangs past
        # the cap, we abandon its future — the executor thread stays
        # occupied until the underlying call returns (or the process
        # exits), and _tick_check skips subsequent ticks until it clears.
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix=f"yb-check-{which_cluster}-{id(group):x}",
        )
        self._pending_future: concurrent.futures.Future | None = None

    def start(self) -> None:
        """Spin up the daemon thread. Idempotent."""
        if self._started:
            return
        self._started = True
        _emit_startup_notice_once()
        self._thread = threading.Thread(
            target=self._run,
            name=f"yb-probe-{self._which}-{id(self._group):x}",
            daemon=True,
        )
        self._thread.start()
        logger.debug(
            "HealthProbe (%s) started: primary_uuid=%s, interval_s=%s",
            self._which, self._group.primary.uuid, self._interval_s,
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
                    "HealthProbe (%s) thread did not stop within %.1fs "
                    "(primary_uuid=%s)",
                    self._which, join_timeout, self._group.primary.uuid,
                )
        # Fire-and-forget executor shutdown — a stuck check() may still
        # be holding the worker. `wait=False` lets us tear down cleanly
        # without blocking on it (the daemon thread dies with the process).
        self._executor.shutdown(wait=False)
        logger.debug(
            "HealthProbe (%s) stopped: primary_uuid=%s",
            self._which, self._group.primary.uuid,
        )

    def _run(self) -> None:
        """Probe loop. On each tick:
          1. Call this cluster's CB and apply the result to its status.
          2. Refresh ``yb_servers()`` on the same cluster's control conn.

        Sleeps `interval_s` between ticks via the stop event so `stop()`
        interrupts the sleep promptly."""
        group = self._group
        while not self._stop_event.wait(timeout=self._interval_s):
            self._tick_check()
            self._tick_refresh_topology()

    # ---------- CB check
    def _tick_check(self) -> None:
        """Call the CB with a wall-clock cap of ``check_timeout_s`` and
        apply the result. Isolated so any exception is contained — the
        refresh step still runs.

        Enforcement: submit to the persistent single-worker executor and
        wait via ``future.result(timeout=)``. If the check exceeds the
        cap, we abandon the future and log a WARNING; the previous status
        is preserved (the check is NOT treated as UNHEALTHY — a stuck
        check is a CB bug, not a cluster signal). If the executor is
        still busy with a previous stuck check, skip this tick entirely
        so a wedged CB doesn't accumulate a backlog."""
        # Previous check still running past its cap? Skip this tick.
        if (
            self._pending_future is not None
            and not self._pending_future.done()
        ):
            logger.warning(
                "%s CB check still running past checkTimeoutSecs=%.1fs; "
                "skipping tick (primary_uuid=%s)",
                self._which, self._check_timeout_s,
                self._group.primary.uuid,
            )
            return
        # Read from module globals so tests can monkeypatch either
        # ``psycopg.yb.health_probe.check_primary_cluster`` or
        # ``psycopg.yb.health.check_primary_cluster``.
        module_scope = _get_module_scope()
        check_fn: Callable[["FailoverGroup"], HealthResult] = (
            module_scope[f"check_{self._which}_cluster"]
        )
        future = self._executor.submit(check_fn, self._group)
        self._pending_future = future
        try:
            result = future.result(timeout=self._check_timeout_s)
        except concurrent.futures.TimeoutError:
            logger.warning(
                "%s CB check exceeded checkTimeoutSecs=%.1fs; preserving "
                "previous status (primary_uuid=%s)",
                self._which, self._check_timeout_s,
                self._group.primary.uuid,
            )
            # Leave `_pending_future` in place so the next tick sees the
            # still-running check and skips itself.
            return
        except Exception:
            logger.warning(
                "%s CB check raised; treating as no-change (primary_uuid=%s)",
                self._which, self._group.primary.uuid,
                exc_info=True,
            )
            self._pending_future = None
            return
        self._pending_future = None
        self._maybe_apply(result)

    def _maybe_apply(self, result: HealthResult) -> None:
        """If ``result`` differs from the current per-cluster status AND
        the cool-down has elapsed, delegate to ``trigger_drain`` for the
        barrier-drain-and-flip sequence (Phase E).

        Cool-down and no-op detection stay on the probe (fast path); the
        drain itself is in the ``drain`` module because it composes the
        pause / poll / force-close / resume phases against the outgoing
        cluster's ``tracked_conns``.
        """
        group = self._group
        now = time.monotonic()
        status_attr = f"{self._which}_status"
        transition_attr = f"{self._which}_last_transition_time"
        can_transition = getattr(group, f"can_transition_{self._which}")
        with group.lock:
            current = getattr(group, status_attr)
            if current == result:
                logger.log(
                    5,  # TRACE — defined in psycopg.yb.__init__
                    "probe tick: %s no-op (status=%s, primary_uuid=%s)",
                    self._which, result.value, group.primary.uuid,
                )
                return
            if not can_transition(now):
                logger.debug(
                    "probe tick: %s transition %s → %s deferred by cool-down "
                    "(primary_uuid=%s, remaining=%.1fs)",
                    self._which, current.value, result.value,
                    group.primary.uuid,
                    group.cooldown_s - (now - getattr(group, transition_attr)),
                )
                return
        # Fall through the lock — trigger_drain re-acquires it as it
        # mutates the group state. See drain.trigger_drain for the full
        # sequence (pause / drain outgoing / force-close / apply / resume).
        from .drain import trigger_drain            # late import; cycle-safe
        trigger_drain(
            group=group,
            which_cluster=self._which,
            new_status=result,
            drain_timeout_s=self._drain_timeout_s,
        )

    # ---------- topology refresh
    def _tick_refresh_topology(self) -> None:
        """Run ``yb_servers()`` on this cluster's control connection and
        merge the result into ``ClusterState.nodes``. Wraps the registry's
        existing ``_do_refresh_sync`` so both the CB check and the refresh
        share the same conn without contending."""
        group = self._group
        state = getattr(group, self._which)
        from .registry import ClusterRegistry  # late import; cycle-safe
        try:
            ClusterRegistry.instance()._do_refresh_sync(state)
        except Exception:
            logger.warning(
                "%s topology refresh raised; skipping this tick "
                "(primary_uuid=%s)",
                self._which, group.primary.uuid,
                exc_info=True,
            )


def _get_module_scope() -> dict:
    """Return this module's globals so ``_tick_check`` can pick up
    monkey-patched values of ``check_primary_cluster`` /
    ``check_secondary_cluster`` at call time (not import time)."""
    return globals()
