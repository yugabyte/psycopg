"""
Unit tests for the ``HealthAndLagCircuitBreaker`` sample CB.

Verifies the decision table from the module docstring:

  * health HEALTHY                          → HEALTHY
  * health UNHEALTHY + lag ≤ threshold      → UNHEALTHY (trip)
  * health UNHEALTHY + lag >  threshold     → HEALTHY   (hold)
  * health UNHEALTHY + YBAClientError       → UNHEALTHY (fail-open)
  * health UNHEALTHY + client=None          → UNHEALTHY (fail-open)
  * health UNHEALTHY + no config UUID       → UNHEALTHY (fail-open)
  * health UNHEALTHY + YBA returns None     → UNHEALTHY (fail-open)

Also verifies constructor override paths, the primary-vs-secondary
config-UUID selection, and that health_probe is a swappable dependency
(any object with ``check(group) -> HealthResult``).
"""

# Copyright (C) 2026 Yugabyte

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from psycopg.yb.health import HealthResult
from psycopg.yb.yba_client import YBAClientError

from demo.samples.health_and_lag_cb import HealthAndLagCircuitBreaker


pytestmark = pytest.mark.yb_unit


# ------------------------------------------------------- test doubles


class _StubHealthProbe:
    """CB that returns whatever ``verdict`` says, no I/O."""

    def __init__(self, verdict: HealthResult) -> None:
        self.verdict = verdict
        self.calls = 0

    def check(self, group) -> HealthResult:
        self.calls += 1
        return self.verdict


class _StubYBAClient:
    """YBA test double. Each call to ``get_committed_lag_ms`` returns
    (or raises) the next item from ``responses``. Records the config
    UUID(s) it was asked about."""

    def __init__(self, responses: list) -> None:
        self.responses = list(responses)
        self.calls: list[str] = []

    def get_committed_lag_ms(self, xcluster_uuid):
        self.calls.append(xcluster_uuid)
        if not self.responses:
            return None
        r = self.responses[0]
        if len(self.responses) > 1:
            self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def _fake_group(
    *,
    yba_client=None,
    xcluster_config_uuid: str = "",
    failback_xcluster_config_uuid: str = "",
):
    """Bare-bones stand-in for ``FailoverGroup`` — the CB only reads
    ``yba_client``, ``xcluster_config_uuid``, and
    ``failback_xcluster_config_uuid`` off it. Full FailoverGroup
    construction is overkill for these tests."""
    return SimpleNamespace(
        yba_client=yba_client,
        xcluster_config_uuid=xcluster_config_uuid,
        failback_xcluster_config_uuid=failback_xcluster_config_uuid,
        lock=threading.Lock(),
    )


# ------------------------------------------------------- happy paths


def test_health_healthy_short_circuits_no_yba_call():
    """When the health probe says HEALTHY, YBA is never consulted."""
    yba = _StubYBAClient(responses=[500.0])
    cb = HealthAndLagCircuitBreaker(
        which_cluster="primary",
        threshold_lag_ms=0,
        health_probe=_StubHealthProbe(HealthResult.HEALTHY),
        yba_client=yba,
        xcluster_config_uuid="xcc-a2b",
    )
    group = _fake_group(yba_client=yba, xcluster_config_uuid="xcc-a2b")
    assert cb.check(group) is HealthResult.HEALTHY
    assert yba.calls == []


def test_health_unhealthy_and_lag_below_threshold_trips():
    """The core positive case: health bad + lag caught up → trip."""
    yba = _StubYBAClient(responses=[0.5])
    cb = HealthAndLagCircuitBreaker(
        which_cluster="primary",
        threshold_lag_ms=1,
        health_probe=_StubHealthProbe(HealthResult.UNHEALTHY),
        yba_client=yba,
        xcluster_config_uuid="xcc-a2b",
    )
    group = _fake_group(yba_client=yba, xcluster_config_uuid="xcc-a2b")
    assert cb.check(group) is HealthResult.UNHEALTHY
    assert yba.calls == ["xcc-a2b"]
    assert cb.last_trip_time > 0.0


def test_health_unhealthy_and_lag_at_threshold_trips():
    """Boundary: lag == threshold satisfies ≤ and trips."""
    yba = _StubYBAClient(responses=[100.0])
    cb = HealthAndLagCircuitBreaker(
        which_cluster="primary",
        threshold_lag_ms=100,
        health_probe=_StubHealthProbe(HealthResult.UNHEALTHY),
        yba_client=yba,
        xcluster_config_uuid="xcc-a2b",
    )
    group = _fake_group(yba_client=yba, xcluster_config_uuid="xcc-a2b")
    assert cb.check(group) is HealthResult.UNHEALTHY


def test_health_unhealthy_but_lag_above_threshold_holds():
    """Health bad + target still behind → hold HEALTHY (no trip)."""
    yba = _StubYBAClient(responses=[500.0])
    cb = HealthAndLagCircuitBreaker(
        which_cluster="primary",
        threshold_lag_ms=100,
        health_probe=_StubHealthProbe(HealthResult.UNHEALTHY),
        yba_client=yba,
        xcluster_config_uuid="xcc-a2b",
    )
    group = _fake_group(yba_client=yba, xcluster_config_uuid="xcc-a2b")
    assert cb.check(group) is HealthResult.HEALTHY
    assert cb.last_trip_time == 0.0


# ------------------------------------------------------- fail-open paths


def test_health_unhealthy_and_yba_error_fails_open_to_trip():
    """YBAClientError is fail-open: trip UNHEALTHY without lag data."""
    yba = _StubYBAClient(responses=[YBAClientError("boom")])
    cb = HealthAndLagCircuitBreaker(
        which_cluster="primary",
        threshold_lag_ms=0,
        health_probe=_StubHealthProbe(HealthResult.UNHEALTHY),
        yba_client=yba,
        xcluster_config_uuid="xcc-a2b",
    )
    group = _fake_group(yba_client=yba, xcluster_config_uuid="xcc-a2b")
    assert cb.check(group) is HealthResult.UNHEALTHY


def test_health_unhealthy_and_yba_none_fails_open_to_trip():
    """YBA returning None (no data) is fail-open: trip UNHEALTHY."""
    yba = _StubYBAClient(responses=[None])
    cb = HealthAndLagCircuitBreaker(
        which_cluster="primary",
        threshold_lag_ms=0,
        health_probe=_StubHealthProbe(HealthResult.UNHEALTHY),
        yba_client=yba,
        xcluster_config_uuid="xcc-a2b",
    )
    group = _fake_group(yba_client=yba, xcluster_config_uuid="xcc-a2b")
    assert cb.check(group) is HealthResult.UNHEALTHY


def test_health_unhealthy_and_no_yba_client_fails_open():
    """No YBA client anywhere → fail-open trip."""
    cb = HealthAndLagCircuitBreaker(
        which_cluster="primary",
        threshold_lag_ms=0,
        health_probe=_StubHealthProbe(HealthResult.UNHEALTHY),
        yba_client=None,
        xcluster_config_uuid=None,
    )
    group = _fake_group()   # no yba_client, no config UUID
    assert cb.check(group) is HealthResult.UNHEALTHY


def test_health_unhealthy_and_no_config_uuid_fails_open():
    """Client set but no config UUID (constructor override or group) →
    fail-open. Covers the secondary-slot-without-failback-config case."""
    yba = _StubYBAClient(responses=[10.0])   # would say caught-up if consulted
    cb = HealthAndLagCircuitBreaker(
        which_cluster="secondary",
        threshold_lag_ms=0,
        health_probe=_StubHealthProbe(HealthResult.UNHEALTHY),
        yba_client=yba,
    )   # no override, and the group below has no failback_xcluster_config_uuid
    group = _fake_group(yba_client=yba, failback_xcluster_config_uuid="")
    assert cb.check(group) is HealthResult.UNHEALTHY
    # YBA should NOT have been consulted — no UUID to consult it with.
    assert yba.calls == []


# ------------------------------------------------------- resolution paths


def test_primary_slot_uses_a2b_config_uuid_from_group():
    """When no override is provided, the CB reads
    ``group.xcluster_config_uuid`` for the primary slot."""
    yba = _StubYBAClient(responses=[0.0])
    cb = HealthAndLagCircuitBreaker(
        which_cluster="primary",
        threshold_lag_ms=10,
        health_probe=_StubHealthProbe(HealthResult.UNHEALTHY),
    )   # no yba_client override, no config_uuid override
    group = _fake_group(yba_client=yba, xcluster_config_uuid="xcc-a2b-from-group")
    assert cb.check(group) is HealthResult.UNHEALTHY
    assert yba.calls == ["xcc-a2b-from-group"]


def test_secondary_slot_uses_b2a_config_uuid_from_group():
    """Symmetric — the secondary slot reads the B→A UUID for failback."""
    yba = _StubYBAClient(responses=[0.0])
    cb = HealthAndLagCircuitBreaker(
        which_cluster="secondary",
        threshold_lag_ms=10,
        health_probe=_StubHealthProbe(HealthResult.UNHEALTHY),
    )
    group = _fake_group(
        yba_client=yba,
        xcluster_config_uuid="xcc-a2b",              # would be wrong for failback
        failback_xcluster_config_uuid="xcc-b2a",
    )
    assert cb.check(group) is HealthResult.UNHEALTHY
    assert yba.calls == ["xcc-b2a"]


def test_constructor_override_wins_over_group():
    """Explicit constructor args override whatever the group carries."""
    yba_group = _StubYBAClient(responses=[500.0])   # would hold
    yba_override = _StubYBAClient(responses=[0.0])  # would trip
    cb = HealthAndLagCircuitBreaker(
        which_cluster="primary",
        threshold_lag_ms=10,
        health_probe=_StubHealthProbe(HealthResult.UNHEALTHY),
        yba_client=yba_override,
        xcluster_config_uuid="xcc-explicit",
    )
    group = _fake_group(
        yba_client=yba_group, xcluster_config_uuid="xcc-from-group",
    )
    assert cb.check(group) is HealthResult.UNHEALTHY
    assert yba_group.calls == []
    assert yba_override.calls == ["xcc-explicit"]


# ------------------------------------------------------- construction


def test_defaults_wrap_tracker_table_cb():
    """When no explicit ``health_probe`` is supplied, the CB constructs
    a ``TrackerTableCircuitBreaker`` and delegates to it."""
    from demo.samples.tracker_table_cb import TrackerTableCircuitBreaker
    cb = HealthAndLagCircuitBreaker(which_cluster="primary")
    assert isinstance(cb.health_probe, TrackerTableCircuitBreaker)
    assert cb.health_probe.which_cluster == "primary"


def test_invalid_which_cluster_raises():
    with pytest.raises(ValueError, match="which_cluster"):
        HealthAndLagCircuitBreaker(which_cluster="tertiary")


def test_negative_threshold_raises():
    with pytest.raises(ValueError, match="threshold_lag_ms"):
        HealthAndLagCircuitBreaker(
            which_cluster="primary", threshold_lag_ms=-1,
        )


# ------------------------------------------------------- state tracking


def test_trip_and_recovery_timestamps():
    """last_trip_time / last_recovery_time move on real transitions,
    not on holds."""
    yba = _StubYBAClient(responses=[500.0, 500.0, 0.0, 0.0])
    probe = _StubHealthProbe(HealthResult.UNHEALTHY)
    cb = HealthAndLagCircuitBreaker(
        which_cluster="primary",
        threshold_lag_ms=100,
        health_probe=probe,
        yba_client=yba,
        xcluster_config_uuid="xcc-a2b",
    )
    group = _fake_group(yba_client=yba, xcluster_config_uuid="xcc-a2b")

    # Tick 1: health bad, lag=500 → hold. No trip time set.
    assert cb.check(group) is HealthResult.HEALTHY
    assert cb.last_trip_time == 0.0

    # Tick 2: same. Still no trip.
    assert cb.check(group) is HealthResult.HEALTHY
    assert cb.last_trip_time == 0.0

    # Tick 3: lag=0 → trip. Trip time populated.
    assert cb.check(group) is HealthResult.UNHEALTHY
    trip_at = cb.last_trip_time
    assert trip_at > 0.0

    # Flip probe to healthy for recovery.
    probe.verdict = HealthResult.HEALTHY
    assert cb.check(group) is HealthResult.HEALTHY
    assert cb.last_recovery_time >= trip_at
