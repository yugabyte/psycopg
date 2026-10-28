"""
Sample CircuitBreaker implementation: health probe combined with a
YBA replication-lag check.

This is a **reference implementation**, not driver-shipped code. It
belongs alongside :class:`TrackerTableCircuitBreaker` and
:class:`ExternalSignalCircuitBreaker` in ``demo/samples/`` — applications
import it explicitly.

Motivation
----------

The driver's Phase 2 (lag wait) polls YBA between "primary CB says
unhealthy" and "flip routing", to avoid switching to a secondary that
hasn't caught up yet. Some deployments would rather fold that check
into the CB itself — so by the time the CB reports UNHEALTHY, the
secondary is already known caught-up, and the driver can skip Phase 2
entirely (and can safely set ``drainTimeoutSecs=0`` if the workload
tolerates it).

That is what this CB does. It composes:

  1. A **health probe** — default :class:`TrackerTableCircuitBreaker`,
     but any object implementing ``check(group) -> HealthResult``
     works (``ExternalSignalCircuitBreaker`` or a custom class).
  2. A **lag check** via YBA — only consulted when the health probe
     says the target cluster is bad. Uses the same
     :class:`psycopg.yb.yba_client.YBAClient` and same
     ``xcluster_config_uuid`` the driver's Phase 2 uses; either the
     driver has wired them at bootstrap (via
     ``yb.failover.ybaEndpoint`` etc), or the operator passes them
     to this CB's constructor.

Decision table (per tick):

  ┌────────────────────┬──────────────────┬───────────────────────┐
  │  Health probe      │  Lag check       │  This CB says         │
  ├────────────────────┼──────────────────┼───────────────────────┤
  │  HEALTHY           │  (not consulted) │  HEALTHY              │
  │  UNHEALTHY         │  lag ≤ threshold │  UNHEALTHY            │
  │  UNHEALTHY         │  lag >  threshold│  previous verdict     │
  │                    │                  │  (hold — see DB-25154)│
  │  UNHEALTHY         │  YBA unreachable │  UNHEALTHY            │
  │  UNHEALTHY         │  no config UUID  │  UNHEALTHY            │
  └────────────────────┴──────────────────┴───────────────────────┘

The hold row preserves ``_last_reported`` rather than returning
``HEALTHY``: a HEALTHY verdict after an UNHEALTHY one is a recovery
to the driver, which then starts the failback cool-down and drain —
onto a primary the health probe is still saying is bad (DB-25154).

The "YBA unreachable" and "no config UUID" rows fall through to
UNHEALTHY (fail-open) so a broken control plane never wedges failover.
Matches the driver's own Phase 2 behaviour.

Symmetric per slot:

  * Attached to ``group.primary_circuit_breaker``  → consults the A→B
    xCluster config (``group.xcluster_config_uuid``). This is the
    failover direction: primary CB says primary is bad → wait for
    the target (secondary) to catch up before agreeing to trip.
  * Attached to ``group.secondary_circuit_breaker`` → consults the
    B→A xCluster config (``group.failback_xcluster_config_uuid``) —
    only meaningful during failback. If B→A isn't wired, the lag
    check is skipped and the CB behaves like its wrapped health
    probe.

Usage
-----

    from psycopg.yb import bootstrap_failover_group
    from demo.samples.health_and_lag_cb import HealthAndLagCircuitBreaker

    group = bootstrap_failover_group(dsn)   # DSN wires YBA + replicationName

    group.primary_circuit_breaker = HealthAndLagCircuitBreaker(
        which_cluster="primary",
        threshold_lag_ms=0,                 # strict: wait for full convergence
    )
    group.secondary_circuit_breaker = HealthAndLagCircuitBreaker(
        which_cluster="secondary",
        threshold_lag_ms=0,
    )
    group.wait_for_first_check(timeout_s=10)

    # Because this CB gates its own trip on lag, Phase 2 in the driver
    # becomes redundant and drainTimeoutSecs can be small:
    #   yb.failover.drainTimeoutSecs=0
    # (only safe when the workload tolerates in-flight force-close.)

The wrapped health probe defaults to ``TrackerTableCircuitBreaker`` but
can be any CB — swap it via the ``health_probe`` kwarg to compose with
``ExternalSignalCircuitBreaker`` or a custom implementation.
"""

# Copyright (C) 2026 Yugabyte

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from psycopg.yb.health import HealthResult

from .tracker_table_cb import TrackerTableCircuitBreaker

if TYPE_CHECKING:
    from psycopg.yb.registry import FailoverGroup

logger = logging.getLogger(__name__)


class HealthAndLagCircuitBreaker:
    """A CB that gates its UNHEALTHY verdict on both a health probe and
    a replication-lag check. See module docstring for the full decision
    table.

    Instance state — the probe thread is single-threaded per
    (group, which_cluster), so no lock needed.
    """

    def __init__(
        self,
        which_cluster: str = "primary",
        threshold_lag_ms: int = 0,
        health_probe: Any = None,
        # Optional YBA overrides — else pulled from the FailoverGroup at
        # check() time. Explicit overrides let this CB work even when
        # the driver's own Phase 2 is disabled (no ybaEndpoint in DSN).
        yba_client: Any = None,
        xcluster_config_uuid: str | None = None,
        # Constructor kwargs passed through to the default health probe
        # when ``health_probe`` is None. Ignored when a custom probe is
        # supplied.
        tracker_table_tablets: int = 9,
        max_health_failures_allowed: int = 0,
        # DB-24852 — bounded-hold backstop. After this many consecutive
        # "health UNHEALTHY but lag > threshold" ticks we trip to
        # UNHEALTHY anyway, on the theory that if the primary is really
        # dead the lag number can never move: nobody is left to report
        # it, and YBA keeps replaying the last sample for its
        # ~5-minute Prometheus window. The ``max_hold_ticks`` default
        # (12) bounds the hold at 12 × ``yb_servers_refresh_interval``.
        # The driver's default refresh interval is 300 s (5 min), so
        # the default bound is ~60 min — too long for a dead-primary
        # outage. Operators who care about dead-primary latency should
        # EITHER lower this to a small number of ticks AND set
        # ``yb_servers_refresh_interval`` to a short value in the DSN
        # (e.g. refresh=5, max_hold_ticks=12 → ~60 s bound), OR size
        # ``max_hold_ticks`` explicitly in wall-clock terms:
        # ``int(desired_bound_s / yb_servers_refresh_interval)``. Set
        # to 0 to disable (preserves the pre-fix infinite-hold
        # behaviour — not recommended). Agent-K PR #15:137.
        max_hold_ticks: int = 12,
    ) -> None:
        if which_cluster not in ("primary", "secondary"):
            raise ValueError(
                f"which_cluster must be 'primary' or 'secondary', "
                f"got {which_cluster!r}"
            )
        if threshold_lag_ms < 0:
            raise ValueError(
                f"threshold_lag_ms must be >= 0, got {threshold_lag_ms}"
            )
        if max_hold_ticks < 0:
            raise ValueError(
                f"max_hold_ticks must be >= 0, got {max_hold_ticks}"
            )
        self.which_cluster = which_cluster
        self.threshold_lag_ms = threshold_lag_ms
        self.max_hold_ticks = max_hold_ticks
        self._yba_client_override = yba_client
        self._xcluster_config_uuid_override = xcluster_config_uuid
        # DB-24852: consecutive "hold HEALTHY" decisions. Reset on any
        # health-HEALTHY or lag-converged tick.
        self._hold_ticks = 0
        # Consecutive "hold UNHEALTHY (failback gated)" decisions.
        self._failback_hold_ticks = 0

        # Compose in a health probe. Default: tracker-table with the same
        # tuning knobs the pure TrackerTable CB accepts. Callers pass their
        # own for other strategies.
        if health_probe is None:
            health_probe = TrackerTableCircuitBreaker(
                which_cluster=which_cluster,
                tracker_table_tablets=tracker_table_tablets,
                max_update_failures_allowed=max_health_failures_allowed,
            )
        self.health_probe = health_probe

        # Diagnostics — same shape as TrackerTable's fields so an
        # operator watching group.*_status can compute detection latency
        # the same way regardless of which sample CB is attached.
        self.last_trip_time: float = 0.0
        self.last_recovery_time: float = 0.0
        self._last_reported: HealthResult = HealthResult.HEALTHY
        # agent-K PR #15:G9 — DB-24852/DB-24483 interaction. If the
        # operator attaches this CB AND the driver's Phase 2 lag wait
        # is also enabled on the group (``yba_client`` set), both gates
        # run back-to-back on every planned failover, doubling the
        # wall-clock. Log once per CB instance when ``check()`` first
        # observes both are wired, so operators see the misconfig in
        # their logs without hunting through two layers of config.
        self._double_wait_warned = False

    # ---------------------------------------------------------- check()

    def check(self, group: "FailoverGroup") -> HealthResult:
        # agent-K PR #15:G9 — warn once if driver Phase 2 is ALSO
        # enabled on the group. The CB already gates on lag inside its
        # UNHEALTHY path; running the driver's Phase 2 after that is
        # redundant and doubles planned-failover wall-clock. Operators
        # should set ``drainTimeoutSecs=0`` (and treat Phase 2 as a
        # no-op) when attaching this CB. Checked here rather than at
        # construction time because the CB is attached BEFORE
        # ``_bootstrap_yba_lag_phase`` wires ``yba_client`` on the group.
        if not self._double_wait_warned and getattr(group, "yba_client", None):
            logger.warning(
                "%s: HealthAndLagCircuitBreaker is attached AND driver "
                "Phase 2 (yb.failover.ybaEndpoint) is wired on the "
                "group — BOTH gate on lag back-to-back, doubling the "
                "planned-failover wall-clock. Either remove the YBA "
                "DSN knobs or set yb.failover.drainTimeoutSecs=0 so "
                "Phase 2 skips the lag wait (the CB will do it).",
                self.which_cluster,
            )
            self._double_wait_warned = True

        # Step 1: run the health probe. Any exception the wrapped CB
        # doesn't handle bubbles up — the driver's probe wraps this
        # call in a try/except anyway and preserves the previous status.
        health = self.health_probe.check(group)

        if health == HealthResult.HEALTHY:
            # Failback gate: when the PRIMARY slot is recovering from
            # UNHEALTHY → HEALTHY, don't let routing return until the
            # B→A lag has drained — otherwise writes committed on the
            # secondary during the outage haven't replicated back and
            # reading from the primary post-failback shows a stale
            # view. This is the symmetric counterpart to the failover
            # lag gate; only the primary slot owns it because that's
            # the slot whose HEALTHY verdict triggers failback.
            # agent-K PR #12:58.
            if (
                self.which_cluster == "primary"
                and self._last_reported == HealthResult.UNHEALTHY
                and not self._failback_lag_drained(group)
            ):
                logger.info(
                    "primary: health recovered but B→A lag still above "
                    "threshold — holding UNHEALTHY (failback gated)"
                )
                return HealthResult.UNHEALTHY
            self._record_healthy()
            return HealthResult.HEALTHY

        # Step 2: health says bad. Consult lag before agreeing.
        client, config_uuid = self._resolve_yba(group)

        # No YBA path available — fall back to health-only. Fail-open:
        # better to trip than to wedge waiting for a control plane we
        # don't have.
        # agent-K PR #15:294 — each fail-open UNHEALTHY return breaks
        # the "consecutive holds" chain the DB-24852 backstop measures.
        # Reset the counter so a later re-hold starts fresh; otherwise
        # a YBA blip between stuck ticks would silently preserve the
        # count and let the backstop fire before N uninterrupted
        # stuck-lag ticks.
        if client is None or not config_uuid:
            logger.warning(
                "%s health probe says UNHEALTHY and no YBA lag source is "
                "wired (client=%s, config_uuid=%r); reporting UNHEALTHY "
                "without lag confirmation (fail-open)",
                self.which_cluster,
                "set" if client is not None else "unset",
                config_uuid or "",
            )
            self._hold_ticks = 0
            self._record_unhealthy(reason="health-fail, no YBA")
            return HealthResult.UNHEALTHY

        # Consult YBA. Any error bubbles up as "unreachable" — fail-open.
        from psycopg.yb.yba_client import YBAClientError

        try:
            lag_ms = client.get_committed_lag_ms(config_uuid)
        except YBAClientError as exc:
            logger.warning(
                "%s health probe says UNHEALTHY and YBA lag lookup failed "
                "(%s); reporting UNHEALTHY without lag confirmation "
                "(fail-open)",
                self.which_cluster, exc,
            )
            self._hold_ticks = 0
            self._record_unhealthy(reason="health-fail, YBA error")
            return HealthResult.UNHEALTHY
        except Exception:
            # Belt-and-braces — never let an unexpected error in the
            # YBA client wedge the CB.
            logger.warning(
                "%s health probe says UNHEALTHY and YBA lag lookup raised "
                "unexpectedly; reporting UNHEALTHY (fail-open)",
                self.which_cluster, exc_info=True,
            )
            self._hold_ticks = 0
            self._record_unhealthy(reason="health-fail, YBA unexpected")
            return HealthResult.UNHEALTHY

        # YBA said "no data" — treat as unreachable / fail-open. Matches
        # the driver's own Phase 2.
        if lag_ms is None:
            logger.warning(
                "%s health probe says UNHEALTHY and YBA returned no lag "
                "data; reporting UNHEALTHY (fail-open)",
                self.which_cluster,
            )
            self._hold_ticks = 0
            self._record_unhealthy(reason="health-fail, YBA no-data")
            return HealthResult.UNHEALTHY

        if lag_ms <= self.threshold_lag_ms:
            logger.warning(
                "🔴 %s: health probe UNHEALTHY AND lag %.1fms ≤ "
                "threshold %dms — CB tripping to UNHEALTHY",
                self.which_cluster, lag_ms, self.threshold_lag_ms,
            )
            # This is the "real" trip path (standby caught up). Reset
            # the DB-24852 hold counter so a later re-hold starts
            # fresh; `_record_unhealthy` deliberately does not reset
            # to protect the sustained-stuck trip path from
            # ping-ponging.
            self._hold_ticks = 0
            self._record_unhealthy(
                reason=f"health-fail, lag {lag_ms:.1f}ms ≤ threshold"
            )
            return HealthResult.UNHEALTHY

        # Health is bad but the target still hasn't caught up.
        # DB-24852: track consecutive holds; if we've been holding past
        # ``max_hold_ticks`` the primary is likely dead (nothing is
        # left to write to it, so the lag number YBA replays will
        # never move). Trip anyway — the standby being behind is
        # better than the application staying pointed at a dead
        # universe.
        self._hold_ticks += 1
        if self.max_hold_ticks > 0 and self._hold_ticks >= self.max_hold_ticks:
            # agent-K PR #15:296 — fire the WARNING only on the
            # transition tick, not every subsequent tick while the
            # backstop stays tripped. ``_record_unhealthy`` leaves
            # ``_last_reported`` at UNHEALTHY, so an ongoing stuck
            # state after the first trip comes back here with
            # ``_last_reported == UNHEALTHY`` and skips the WARNING.
            if self._last_reported == HealthResult.HEALTHY:
                logger.warning(
                    "🔴 %s: health probe UNHEALTHY for %d consecutive ticks "
                    "with lag stuck at %.1fms (> threshold %dms) — tripping "
                    "to UNHEALTHY anyway (DB-24852 backstop: primary may be "
                    "dead and the YBA lag sample can only move once the "
                    "primary comes back)",
                    self.which_cluster, self._hold_ticks, lag_ms,
                    self.threshold_lag_ms,
                )
            self._record_unhealthy(
                reason=(
                    f"sustained UNHEALTHY {self._hold_ticks} ticks, "
                    f"lag stuck at {lag_ms:.1f}ms"
                )
            )
            return HealthResult.UNHEALTHY

        # DB-25154: hold means "do not change the verdict." Returning
        # HEALTHY here after an earlier UNHEALTHY looks like a recovery
        # to the driver, which then runs the cool-down + failback drain
        # back onto a primary the health probe is still flagging. Report
        # whatever we last told the driver — it matches _last_reported,
        # which _record_unhealthy / _record_healthy maintain.
        held_as = self._last_reported
        logger.info(
            "%s: health probe UNHEALTHY but lag %.1fms > threshold %dms "
            "— holding %s (%d/%d, target not ready)",
            self.which_cluster, lag_ms, self.threshold_lag_ms,
            held_as.name,
            self._hold_ticks,
            self.max_hold_ticks if self.max_hold_ticks > 0 else -1,
        )
        return held_as

    # ---------------------------------------------------------- helpers

    def _resolve_yba(self, group: "FailoverGroup") -> tuple[Any, str]:
        """Return ``(yba_client, xcluster_config_uuid)`` for this CB
        slot. Constructor overrides win; otherwise pull from the group.

        For the primary slot the A→B config is used (failover direction).
        For the secondary slot the B→A config is used (failback
        direction) — only meaningful if the operator wired
        ``yb.failback.replicationName``. If the B→A config isn't
        available on the secondary slot, ``config_uuid`` comes back as
        ``""`` and the caller falls through to health-only.
        """
        client = self._yba_client_override or getattr(group, "yba_client", None)
        if self._xcluster_config_uuid_override is not None:
            return client, self._xcluster_config_uuid_override
        if self.which_cluster == "primary":
            config_uuid = getattr(group, "xcluster_config_uuid", "") or ""
        else:
            config_uuid = (
                getattr(group, "failback_xcluster_config_uuid", "") or ""
            )
        return client, config_uuid

    def _failback_lag_drained(self, group: "FailoverGroup") -> bool:
        """True iff the B→A lag source reports lag ≤ threshold, OR
        there's no B→A source wired (no failback replication → operator
        hasn't opted in to the gate, fail-open).

        Called only from the PRIMARY slot's recovery path. Reads the
        failback xcluster config (B→A direction) via YBA, same client
        the secondary slot would use when it goes UNHEALTHY. On any
        error, returns True (fail-open) — the recovery check is a
        safety belt, not a hard interlock.
        """
        failback_config = (
            getattr(group, "failback_xcluster_config_uuid", "") or ""
        )
        if (
            self._yba_client_override is None
            and getattr(group, "yba_client", None) is not None
            and failback_config
        ):
            # The driver's own failback Phase 2 is wired: it waits on
            # B→A lag after the dispatch barrier (so the source is
            # quiesced and lag can actually drain) and is bounded by
            # failbackLagWaitTimeoutSecs. Gating here as well measures
            # lag while the app is still writing to B, which never
            # reaches a 0 ms threshold — failback would hold forever.
            return True
        client = (
            self._yba_client_override
            or getattr(group, "yba_client", None)
        )
        if client is None or not failback_config:
            # No failback source wired — operator did not opt into the
            # gate. Treat as drained.
            return True
        from psycopg.yb.yba_client import YBAClientError
        try:
            lag_ms = client.get_committed_lag_ms(failback_config)
        except (YBAClientError, Exception) as exc:  # noqa: BLE001
            logger.warning(
                "primary: failback lag lookup failed (%s); treating as "
                "drained (fail-open on the recovery gate)",
                exc,
            )
            return True
        threshold = getattr(
            group, "failback_threshold_replication_lag_ms", self.threshold_lag_ms,
        )
        drained = lag_ms <= threshold
        if drained:
            self._failback_hold_ticks = 0
            return True
        # Same bounded-hold backstop as the failover direction: the app
        # keeps writing to B while we wait, so lag may never reach the
        # threshold. Release after max_hold_ticks.
        self._failback_hold_ticks += 1
        if self.max_hold_ticks > 0 and self._failback_hold_ticks >= self.max_hold_ticks:
            logger.warning(
                "primary: failback lag %.1fms > threshold %dms for %d ticks "
                "— releasing failback gate (max_hold_ticks=%d)",
                lag_ms, threshold, self._failback_hold_ticks, self.max_hold_ticks,
            )
            self._failback_hold_ticks = 0
            return True
        logger.info(
            "primary: failback lag %.1fms > threshold %dms",
            lag_ms, threshold,
        )
        return False

    def _record_healthy(self) -> None:
        if self._last_reported == HealthResult.UNHEALTHY:
            self.last_recovery_time = time.monotonic()
            logger.warning(
                "🟢 %s: health + lag both agree — CB recovering to HEALTHY",
                self.which_cluster,
            )
        self._last_reported = HealthResult.HEALTHY
        # Reset the DB-24852 hold counter: either health recovered, or
        # lag finally converged on this tick — either way, the next
        # hold starts fresh.
        self._hold_ticks = 0

    def _record_unhealthy(self, reason: str) -> None:
        if self._last_reported == HealthResult.HEALTHY:
            self.last_trip_time = time.monotonic()
            logger.debug(
                "%s CB trip cause: %s", self.which_cluster, reason,
            )
        self._last_reported = HealthResult.UNHEALTHY
        # Deliberately NOT clearing ``_hold_ticks`` here. If a sustained-
        # stuck trip (DB-24852 backstop) called us, the next tick is
        # still a hold candidate — resetting would let the CB flip back
        # to HEALTHY on the very next probe and ping-pong the status.
        # The lag-converge trip path explicitly resets before calling
        # us (that's a "real" trip, not a stuck one). ``_record_healthy``
        # also resets on any genuine recovery.
