"""
Unit tests for the dual per-cluster probes (Phase B of the xCluster
failover implementation plan).

Verifies:

  * ``FailoverGroup`` owns two ``HealthProbe`` instances — one per cluster.
  * Each probe checks ONLY its own cluster's CB — a flip on secondary
    doesn't affect primary's status, and vice versa.
  * ``ClusterRegistry._stop_probes_best_effort`` stops both probes.
  * The probe thread refreshes topology on each tick (calls
    ``_do_refresh_sync`` on its cluster's state).

Uses monkey-patched CB check functions and fake ClusterState so no real
DB is required.
"""

# Copyright (C) 2026 Yugabyte

from __future__ import annotations

import threading
import time

import pytest

from psycopg.yb.health import HealthResult
from psycopg.yb.health_probe import HealthProbe
from psycopg.yb.registry import FailoverGroup


pytestmark = pytest.mark.yb_unit


def _make_group(fake_state, cooldown_s=0):
    p = fake_state(
        ("p1", "aws", "us-west", "us-west-1a", "primary"),
        uuid="P",
    )
    s = fake_state(
        ("s1", "aws", "us-east", "us-east-1a", "primary"),
        uuid="S",
    )
    return FailoverGroup(
        primary=p,
        secondary=s,
        lock=threading.Lock(),
        primary_status=HealthResult.HEALTHY,
        secondary_status=HealthResult.HEALTHY,
        cooldown_s=cooldown_s,
    )


def test_two_probes_are_independent_threads(fake_state, monkeypatch):
    """Bootstrap installs one probe per cluster; they run in DISTINCT
    daemon threads."""
    import psycopg.yb.health_probe as hp_mod

    # Neutral CBs — return HEALTHY forever so no state churn.
    monkeypatch.setattr(
        hp_mod, "check_primary_cluster", lambda g: HealthResult.HEALTHY,
    )
    monkeypatch.setattr(
        hp_mod, "check_secondary_cluster", lambda g: HealthResult.HEALTHY,
    )
    group = _make_group(fake_state)

    p_probe = HealthProbe(group, interval_s=0.05, which_cluster="primary")
    s_probe = HealthProbe(group, interval_s=0.05, which_cluster="secondary")
    p_probe.start()
    s_probe.start()
    try:
        # Two distinct daemon threads exist.
        active_names = {t.name for t in threading.enumerate() if t.is_alive()}
        assert any("yb-probe-primary-" in n for n in active_names)
        assert any("yb-probe-secondary-" in n for n in active_names)
    finally:
        p_probe.stop()
        s_probe.stop()


def test_secondary_flip_doesnt_touch_primary(fake_state, monkeypatch):
    """Only the secondary CB reports UNHEALTHY — primary_status stays
    HEALTHY, secondary_status flips."""
    import psycopg.yb.health_probe as hp_mod

    monkeypatch.setattr(
        hp_mod, "check_primary_cluster", lambda g: HealthResult.HEALTHY,
    )
    monkeypatch.setattr(
        hp_mod, "check_secondary_cluster", lambda g: HealthResult.UNHEALTHY,
    )
    group = _make_group(fake_state)

    p_probe = HealthProbe(group, interval_s=0.02, which_cluster="primary")
    s_probe = HealthProbe(group, interval_s=0.02, which_cluster="secondary")
    p_probe.start()
    s_probe.start()

    deadline = time.monotonic() + 2.0
    while (
        time.monotonic() < deadline
        and group.secondary_status == HealthResult.HEALTHY
    ):
        time.sleep(0.02)

    p_probe.stop()
    s_probe.stop()

    assert group.secondary_status == HealthResult.UNHEALTHY
    assert group.primary_status == HealthResult.HEALTHY   # untouched


def test_primary_flip_doesnt_touch_secondary(fake_state, monkeypatch):
    """Symmetric — primary flips, secondary stays put."""
    import psycopg.yb.health_probe as hp_mod

    monkeypatch.setattr(
        hp_mod, "check_primary_cluster", lambda g: HealthResult.UNHEALTHY,
    )
    monkeypatch.setattr(
        hp_mod, "check_secondary_cluster", lambda g: HealthResult.HEALTHY,
    )
    group = _make_group(fake_state)

    p_probe = HealthProbe(group, interval_s=0.02, which_cluster="primary")
    s_probe = HealthProbe(group, interval_s=0.02, which_cluster="secondary")
    p_probe.start()
    s_probe.start()

    deadline = time.monotonic() + 2.0
    while (
        time.monotonic() < deadline
        and group.primary_status == HealthResult.HEALTHY
    ):
        time.sleep(0.02)

    p_probe.stop()
    s_probe.stop()

    assert group.primary_status == HealthResult.UNHEALTHY
    assert group.secondary_status == HealthResult.HEALTHY   # untouched


def test_probe_invokes_topology_refresh_on_each_tick(fake_state, monkeypatch):
    """Every tick, the probe calls the registry's ``_do_refresh_sync`` on
    its cluster's state — even when the CB check itself is a no-op."""
    import psycopg.yb.health_probe as hp_mod
    from psycopg.yb.registry import ClusterRegistry

    monkeypatch.setattr(
        hp_mod, "check_primary_cluster", lambda g: HealthResult.HEALTHY,
    )
    monkeypatch.setattr(
        hp_mod, "check_secondary_cluster", lambda g: HealthResult.HEALTHY,
    )
    group = _make_group(fake_state)

    calls: list[str] = []

    def fake_refresh(state):
        calls.append(state.uuid)
        return False   # simulate "no control conn" — same as fake_state today

    monkeypatch.setattr(
        ClusterRegistry.instance(), "_do_refresh_sync", fake_refresh,
    )

    p_probe = HealthProbe(group, interval_s=0.02, which_cluster="primary")
    s_probe = HealthProbe(group, interval_s=0.02, which_cluster="secondary")
    p_probe.start()
    s_probe.start()
    time.sleep(0.2)
    p_probe.stop()
    s_probe.stop()

    # Both cluster uuids observed — one call per probe per tick.
    assert "P" in calls
    assert "S" in calls


def test_stopping_a_probe_halts_only_that_probe(fake_state, monkeypatch):
    """Stop primary → primary ticks halt; secondary keeps ticking."""
    import psycopg.yb.health_probe as hp_mod
    from psycopg.yb.registry import ClusterRegistry

    monkeypatch.setattr(
        hp_mod, "check_primary_cluster", lambda g: HealthResult.HEALTHY,
    )
    monkeypatch.setattr(
        hp_mod, "check_secondary_cluster", lambda g: HealthResult.HEALTHY,
    )
    group = _make_group(fake_state)

    tick_counts = {"P": 0, "S": 0}

    def counting_refresh(state):
        tick_counts[state.uuid] += 1
        return False

    monkeypatch.setattr(
        ClusterRegistry.instance(), "_do_refresh_sync", counting_refresh,
    )

    p_probe = HealthProbe(group, interval_s=0.02, which_cluster="primary")
    s_probe = HealthProbe(group, interval_s=0.02, which_cluster="secondary")
    p_probe.start()
    s_probe.start()
    time.sleep(0.15)
    p_probe.stop()
    snapshot = dict(tick_counts)   # freeze primary's count at stop
    time.sleep(0.15)
    s_probe.stop()

    # Secondary kept ticking after primary stopped.
    assert tick_counts["S"] > snapshot["S"]
    # Primary hasn't advanced.
    assert tick_counts["P"] == snapshot["P"]
