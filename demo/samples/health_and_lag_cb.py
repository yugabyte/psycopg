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

  ┌────────────────────┬──────────────────┬────────────────┐
  │  Health probe      │  Lag check       │  This CB says  │
  ├────────────────────┼──────────────────┼────────────────┤
  │  HEALTHY           │  (not consulted) │  HEALTHY       │
  │  UNHEALTHY         │  lag ≤ threshold │  UNHEALTHY     │
  │  UNHEALTHY         │  lag >  threshold│  HEALTHY (hold)│
  │  UNHEALTHY         │  YBA unreachable │  UNHEALTHY     │
  │  UNHEALTHY         │  no config UUID  │  UNHEALTHY     │
  └────────────────────┴──────────────────┴────────────────┘

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
        self.which_cluster = which_cluster
        self.threshold_lag_ms = threshold_lag_ms
        self._yba_client_override = yba_client
        self._xcluster_config_uuid_override = xcluster_config_uuid

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

    # ---------------------------------------------------------- check()

    def check(self, group: "FailoverGroup") -> HealthResult:
        # Step 1: run the health probe. Any exception the wrapped CB
        # doesn't handle bubbles up — the driver's probe wraps this
        # call in a try/except anyway and preserves the previous status.
        health = self.health_probe.check(group)

        if health == HealthResult.HEALTHY:
            self._record_healthy()
            return HealthResult.HEALTHY

        # Step 2: health says bad. Consult lag before agreeing.
        client, config_uuid = self._resolve_yba(group)

        # No YBA path available — fall back to health-only. Fail-open:
        # better to trip than to wedge waiting for a control plane we
        # don't have.
        if client is None or not config_uuid:
            logger.warning(
                "%s health probe says UNHEALTHY and no YBA lag source is "
                "wired (client=%s, config_uuid=%r); reporting UNHEALTHY "
                "without lag confirmation (fail-open)",
                self.which_cluster,
                "set" if client is not None else "unset",
                config_uuid or "",
            )
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
            self._record_unhealthy(reason="health-fail, YBA no-data")
            return HealthResult.UNHEALTHY

        if lag_ms <= self.threshold_lag_ms:
            logger.warning(
                "🔴 %s: health probe UNHEALTHY AND lag %.1fms ≤ "
                "threshold %dms — CB tripping to UNHEALTHY",
                self.which_cluster, lag_ms, self.threshold_lag_ms,
            )
            self._record_unhealthy(
                reason=f"health-fail, lag {lag_ms:.1f}ms ≤ threshold"
            )
            return HealthResult.UNHEALTHY

        # Health is bad but the target still hasn't caught up. Hold
        # HEALTHY so the driver doesn't trip; next tick will consult
        # both signals again.
        logger.info(
            "%s: health probe UNHEALTHY but lag %.1fms > threshold %dms "
            "— holding HEALTHY (target not ready)",
            self.which_cluster, lag_ms, self.threshold_lag_ms,
        )
        # Deliberately NOT calling _record_unhealthy — this is a hold,
        # not a trip. Keep last_reported as-is.
        return HealthResult.HEALTHY

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

    def _record_healthy(self) -> None:
        if self._last_reported == HealthResult.UNHEALTHY:
            self.last_recovery_time = time.monotonic()
            logger.warning(
                "🟢 %s: health + lag both agree — CB recovering to HEALTHY",
                self.which_cluster,
            )
        self._last_reported = HealthResult.HEALTHY

    def _record_unhealthy(self, reason: str) -> None:
        if self._last_reported == HealthResult.HEALTHY:
            self.last_trip_time = time.monotonic()
            logger.debug(
                "%s CB trip cause: %s", self.which_cluster, reason,
            )
        self._last_reported = HealthResult.UNHEALTHY
