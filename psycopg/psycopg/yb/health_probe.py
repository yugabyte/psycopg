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
        #
        # agent-K PR #9:179: ThreadPoolExecutor workers are NOT daemon
        # threads on 3.10+. The interpreter's exit hook joins them, so
        # a stuck CB check would block process exit indefinitely. Fix:
        # register an atexit hook that force-shuts the executor via
        # cancel_futures=True — matches the "daemon dies with process"
        # contract the docstring originally promised.
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix=f"yb-check-{which_cluster}-{id(group):x}",
        )
        import atexit
        atexit.register(self._executor.shutdown, wait=False, cancel_futures=True)
        self._pending_future: concurrent.futures.Future | None = None
        # Set after the first tick completes (success, timeout, or raised).
        # ``FailoverGroup.wait_for_first_check`` waits on this so callers
        # can synchronise routing decisions with actual CB observations
        # (design doc §4.6). See ``run_first_check_synchronously``.
        self._first_check_complete = threading.Event()

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
    def _tick_check(self, apply_verdict: bool = True) -> None:
        """Call the CB with a wall-clock cap of ``check_timeout_s`` and
        (optionally) apply the result. Isolated so any exception is
        contained — the refresh step still runs.

        ``apply_verdict=False`` runs the CB and marks the first check
        complete WITHOUT calling ``_maybe_apply`` (so no ``trigger_drain``
        fires). This is the bootstrap-time path exercised by
        ``run_first_check_synchronously`` — the goal there is to seed the
        gate for the first connect, not to fire a full drain on the
        caller's thread while they're still inside
        ``bootstrap_failover_group`` (agent-K PR #10:284). The next
        natural daemon tick will apply the verdict normally.

        Enforcement: submit to the persistent single-worker executor and
        wait via ``future.result(timeout=)``. If the check exceeds the
        cap, we abandon the future and log a WARNING; the previous status
        is preserved (the check is NOT treated as UNHEALTHY — a stuck
        check is a CB bug, not a cluster signal). If the executor is
        still busy with a previous stuck check, skip this tick entirely
        so a wedged CB doesn't accumulate a backlog."""
        try:
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
            try:
                if apply_verdict:
                    self._maybe_apply(result)
                else:
                    # Bootstrap-time path (agent-K PR #10:284): don't
                    # fire a drain inside wait_for_first_check — the
                    # caller is still inside bootstrap_failover_group
                    # and trigger_drain's Phase 2 lag wait would
                    # dwarf wait_for_first_check's own timeout. Mirror
                    # ``_maybe_apply``'s "same status, no-op" fast path
                    # here: when the verdict matches the current
                    # status (the common bootstrap case) nothing was
                    # going to fire anyway, so stashing the result for
                    # the next daemon tick is correct. When they
                    # differ, the daemon's natural first tick (interval
                    # later) will apply the verdict normally.
                    logger.debug(
                        "%s probe: first-check verdict %s (apply "
                        "deferred to next daemon tick, primary_uuid=%s)",
                        self._which, result.value,
                        self._group.primary.uuid,
                    )
                # Only count a successfully applied verdict as "first
                # check complete". agent-K PR #9:260: previously the
                # finally fired the event on skip/timeout paths too,
                # so wait_for_first_check could release before any CB
                # verdict had reached the status machine — the first
                # connect would then route on constructor defaults.
                self._first_check_complete.set()
            except Exception:
                # agent-K PR #9:254: trigger_drain can raise for a
                # variety of reasons. Previously that killed the probe
                # thread and failover silently stopped for this cluster.
                # ERROR-log and swallow so the probe keeps ticking.
                logger.error(
                    "%s probe: _maybe_apply raised; probe will keep "
                    "ticking (primary_uuid=%s)",
                    self._which, self._group.primary.uuid,
                    exc_info=True,
                )
        except Exception:
            # Belt-and-braces for anything else in this tick (module
            # scope lookup, executor submission failures). Probe
            # survives so failover doesn't go silent (agent-K PR #9:254).
            logger.error(
                "%s probe: unexpected error in _tick_check; probe "
                "will keep ticking (primary_uuid=%s)",
                self._which, self._group.primary.uuid,
                exc_info=True,
            )

    @property
    def first_check_complete(self) -> bool:
        return self._first_check_complete.is_set()

    def run_first_check_synchronously(self) -> None:
        """Force one synchronous ``_tick_check`` and block until it
        completes. Used by ``FailoverGroup.wait_for_first_check``.
        Idempotent — a no-op if the first tick has already fired.

        Bounded by ``check_timeout_s`` inside ``_tick_check`` — if the
        CB hangs, this returns anyway. The verdict is NOT applied on
        this path (agent-K PR #10:284): we only want to seed the gate
        before the first connect, not fire a Phase 2 lag wait on the
        caller's thread while they're still inside
        ``bootstrap_failover_group``. The daemon's next natural tick
        will apply the verdict with the normal cooldown + drain
        semantics."""
        if self._first_check_complete.is_set():
            return
        self._tick_check(apply_verdict=False)

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
        # agent-K PR #9:313: don't start a new drain while another
        # cluster's probe is already running one. The two probes are
        # independent threads; without this gate, the secondary probe
        # could enter trigger_drain while the primary probe was already
        # mid-drain, interleaving pause/resume and status transitions.
        # Fall through the lock — trigger_drain re-acquires it as it
        # mutates the group state.
        with group.lock:
            if group.dispatch_paused:
                logger.info(
                    "%s probe: another drain is in flight "
                    "(dispatch_paused=True); deferring transition "
                    "%s → %s to next tick (primary_uuid=%s)",
                    self._which, current.value, result.value,
                    group.primary.uuid,
                )
                return
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
        existing ``_do_refresh_sync``.

        agent-K PR #9:326: skip when a previous CB check is still
        running — otherwise refresh would race the CB on the same
        control connection and could close it mid-check.

        agent-K PR #9:331: check if this probe's group is still in the
        registry. If not, ``ClusterRegistry.clear()`` has disowned us
        — stop ticking rather than opening a fresh control conn on
        orphaned state.
        """
        group = self._group
        if (
            self._pending_future is not None
            and not self._pending_future.done()
        ):
            # CB check still running on the control conn; skip refresh
            # this tick to avoid the race with it.
            return
        from .registry import ClusterRegistry  # late import; cycle-safe
        reg = ClusterRegistry.instance()
        is_registered_now = (
            reg.get_failover_group(group.primary.uuid) is group
        )
        if is_registered_now:
            # Record that we've seen this group registered — so a later
            # observation of "not registered" means we were disowned
            # (clear() ran), not merely that we were never installed.
            self._was_ever_registered = True
        elif getattr(self, "_was_ever_registered", False):
            # Registry used to track our group but doesn't anymore —
            # clear() must have run. Stop ourselves rather than opening
            # a fresh conn on orphaned state. agent-K PR #9:331.
            logger.info(
                "%s probe: group disowned by registry; stopping self "
                "(primary_uuid=%s)",
                self._which, group.primary.uuid,
            )
            self._stop_event.set()
            return
        # else: never registered — test fixture or pre-install state;
        # just continue, nothing to clean up.
        state = getattr(group, self._which)
        try:
            reg._do_refresh_sync(state)
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
