"""
Circuit-breaker Protocol for xCluster health detection.

The probe thread (``HealthProbe``) calls ``check_primary_cluster`` and
``check_secondary_cluster`` on each tick — one per cluster. Each function
delegates to the matching CB slot on the ``FailoverGroup``
(``group.primary_circuit_breaker`` / ``group.secondary_circuit_breaker``).

**The driver does not ship a default CB implementation.** After calling
``bootstrap_failover_group(dsn)``, the application MUST assign a CB
implementation to both slots before opening any connection. The
dispatcher raises ``MissingCircuitBreakerError`` if either slot is still
``None`` at connect time — this is a fail-fast guarantee that xCluster
routing never runs against an unpolled cluster.

Reference implementations live under ``demo/samples/``:

  * ``demo/samples/tracker_table_cb.py`` — ``TrackerTableCircuitBreaker``:
    periodically runs an UPDATE against a sharded tracker table; UNHEALTHY
    after N consecutive UPDATE failures (symmetric hysteresis).
  * ``demo/samples/external_signal_cb.py`` — ``ExternalSignalCircuitBreaker``:
    operator-controlled failover via a signal table on the cluster's
    control connection.

Copy the sample that fits, tune the constructor args, and attach it:

    from psycopg.yb import bootstrap_failover_group
    from demo.samples.tracker_table_cb import TrackerTableCircuitBreaker

    group = bootstrap_failover_group(dsn)
    group.primary_circuit_breaker = TrackerTableCircuitBreaker(
        which_cluster="primary")
    group.secondary_circuit_breaker = TrackerTableCircuitBreaker(
        which_cluster="secondary")
    conn = psycopg.connect(dsn)

Any class matching the ``CircuitBreaker`` Protocol works — swap in a
different strategy without touching the dispatcher, the registry, or
the probe thread.
"""

# Copyright (C) 2026 Yugabyte

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from .health import HealthResult

if TYPE_CHECKING:
    from .registry import FailoverGroup


# ----------------------------------------------------------------- protocol

class CircuitBreaker(Protocol):
    """The contract every circuit-breaker implementation must satisfy.

    A single instance is owned by one ``FailoverGroup``. The probe thread
    calls ``check(group)`` on each tick; the return value flows back into
    ``HealthProbe._maybe_apply`` which writes the new status (subject to
    cool-down) under ``group.lock``.

    Implementations MUST be thread-safe with respect to the probe thread
    only — the probe is the sole caller. But the implementation may
    maintain mutable state across calls (e.g. failure counters), so don't
    create one instance and share it across multiple FailoverGroups.
    """

    def check(self, group: "FailoverGroup") -> HealthResult: ...


# ----------------------------------------------------------------- stub

class AlwaysHealthyCircuitBreaker:
    """Inert CB — always reports HEALTHY. Useful for tests that want the
    delegation path to run without any real health-check logic, or as
    a documented "what does a minimal CB look like" example.

    Not suitable for production: with this attached to both clusters,
    the driver will never detect a failure and will never fail over.
    """

    def check(self, group: "FailoverGroup") -> HealthResult:
        return HealthResult.HEALTHY
