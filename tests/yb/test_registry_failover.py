"""
Unit tests for FailoverGroup, ClusterRegistry's xCluster methods, and the
HealthProbe daemon thread. Covers Phases 2, 3, and 4 of the xCluster
failover implementation plan.

Auto-tagged ``yb_unit`` via the filename prefix (see conftest.py).

These tests do not open real connections — they monkeypatch the registry's
``get_or_bootstrap`` to return ``fake_state`` instances. Real-cluster
integration coverage lives in ``test_xcluster_failover.py`` (Phase 7).
"""

# Copyright (C) 2026 Yugabyte

from __future__ import annotations

import threading
import time

import pytest

from psycopg.yb.health import (
    HealthResult,
    check_primary_cluster,
    check_secondary_cluster,
)
from psycopg.yb.health_probe import HealthProbe
from psycopg.yb.params import YBParams
from psycopg.yb.registry import ClusterRegistry, ClusterState, FailoverGroup


# ----------------------------------------------------------------- helpers

def _make_yb_params(secondary_hosts, cooldown_s=5, refresh_s=300):
    """Build a YBParams shaped for xCluster — both opt-in conditions on."""
    return YBParams(
        smart_driver_enabled=True,
        secondary_cluster_hosts=list(secondary_hosts),
        cooldown_s=cooldown_s,
        refresh_interval_s=refresh_s,
    )


def _install_group(
    reg: ClusterRegistry,
    primary: ClusterState,
    secondary: ClusterState,
    cooldown_s: int = 5,
) -> FailoverGroup:
    """Bypass bootstrap — install a synthetic group + cluster states directly.
    Used by tests that aren't exercising the bootstrap path."""
    group = FailoverGroup(
        primary=primary,
        secondary=secondary,
        lock=threading.Lock(),
        primary_status=HealthResult.HEALTHY, secondary_status=HealthResult.HEALTHY,
        cooldown_s=cooldown_s,
    )
    reg._clusters[primary.uuid] = primary
    reg._clusters[secondary.uuid] = secondary
    reg._failover_groups[primary.uuid] = group
    return group


# ----------------------------------------------------------------- FailoverGroup dataclass

def test_failover_group_initial_state(fake_state):
    primary = fake_state(("p1", "aws", "us-west", "us-west-1a", "primary"))
    secondary = fake_state(
        ("s1", "aws", "us-east", "us-east-1a", "primary"),
        uuid="secondary-uuid",
    )
    group = FailoverGroup(
        primary=primary, secondary=secondary,
        lock=threading.Lock(), primary_status=HealthResult.HEALTHY, secondary_status=HealthResult.HEALTHY,
    )
    assert group.primary_status == HealthResult.HEALTHY
    assert group.primary_last_transition_time == 0.0   # initial-state, can transition immediately
    assert group.primary_probe is None
    assert group.secondary_probe is None


def test_force_status_flips_and_updates_timestamp(fake_state):
    p = fake_state(("p1", "aws", "us-west", "us-west-1a", "primary"))
    s = fake_state(("s1", "aws", "us-east", "us-east-1a", "primary"), uuid="sec")
    group = FailoverGroup(
        primary=p, secondary=s, lock=threading.Lock(),
        primary_status=HealthResult.HEALTHY, secondary_status=HealthResult.HEALTHY,
    )
    before = time.monotonic()
    group.force_primary_status(HealthResult.UNHEALTHY)
    after = time.monotonic()
    assert group.primary_status == HealthResult.UNHEALTHY
    assert before <= group.primary_last_transition_time <= after


def test_can_transition_initial_true(fake_state):
    """Fresh group must always be allowed to transition — no cool-down floor."""
    p = fake_state(("p1", "aws", "us-west", "us-west-1a", "primary"))
    s = fake_state(("s1", "aws", "us-east", "us-east-1a", "primary"), uuid="sec")
    group = FailoverGroup(
        primary=p, secondary=s, lock=threading.Lock(),
        primary_status=HealthResult.HEALTHY, secondary_status=HealthResult.HEALTHY, cooldown_s=999,
    )
    assert group.can_transition_primary(time.monotonic()) is True


def test_can_transition_honours_cooldown(fake_state):
    p = fake_state(("p1", "aws", "us-west", "us-west-1a", "primary"))
    s = fake_state(("s1", "aws", "us-east", "us-east-1a", "primary"), uuid="sec")
    group = FailoverGroup(
        primary=p, secondary=s, lock=threading.Lock(),
        primary_status=HealthResult.HEALTHY, secondary_status=HealthResult.HEALTHY, cooldown_s=999,
    )
    group.force_primary_status(HealthResult.UNHEALTHY)
    # Cool-down just started; no second transition allowed.
    assert group.can_transition_primary(time.monotonic()) is False
    # Time-warp past cool-down → True.
    assert group.can_transition_primary(group.primary_last_transition_time + 1000) is True


# ----------------------------------------------------------------- registry lookups

def test_get_failover_group_by_primary_uuid(fresh_registry, fake_state):
    p = fake_state(("p1", "aws", "us-west", "us-west-1a", "primary"), uuid="P")
    s = fake_state(("s1", "aws", "us-east", "us-east-1a", "primary"), uuid="S")
    group = _install_group(fresh_registry, p, s)
    assert fresh_registry.get_failover_group("P") is group
    assert fresh_registry.get_failover_group("S") is None   # secondary uuid is NOT primary key
    assert fresh_registry.get_failover_group("nonexistent") is None


def test_get_failover_group_by_uuid_searches_both_sides(fresh_registry, fake_state):
    p = fake_state(("p1", "aws", "us-west", "us-west-1a", "primary"), uuid="P")
    s = fake_state(("s1", "aws", "us-east", "us-east-1a", "primary"), uuid="S")
    group = _install_group(fresh_registry, p, s)
    assert fresh_registry.get_failover_group_by_uuid("P") is group
    assert fresh_registry.get_failover_group_by_uuid("S") is group
    assert fresh_registry.get_failover_group_by_uuid("nonexistent") is None


# ----------------------------------------------------------------- operator API

def test_reset_failover_group_returns_false_for_unknown(fresh_registry):
    assert fresh_registry.reset_failover_group("nonexistent") is False


def test_reset_failover_group_flips_to_healthy(fresh_registry, fake_state):
    p = fake_state(("p1", "aws", "us-west", "us-west-1a", "primary"), uuid="P")
    s = fake_state(("s1", "aws", "us-east", "us-east-1a", "primary"), uuid="S")
    group = _install_group(fresh_registry, p, s)
    group.force_primary_status(HealthResult.UNHEALTHY)
    assert group.primary_status == HealthResult.UNHEALTHY

    assert fresh_registry.reset_failover_group("P") is True
    assert group.primary_status == HealthResult.HEALTHY


def test_get_failover_status_returns_state_and_timestamp(fresh_registry, fake_state):
    p = fake_state(("p1", "aws", "us-west", "us-west-1a", "primary"), uuid="P")
    s = fake_state(("s1", "aws", "us-east", "us-east-1a", "primary"), uuid="S")
    group = _install_group(fresh_registry, p, s)
    p_status, s_status, p_ts, s_ts = fresh_registry.get_failover_status("P")
    assert p_status == HealthResult.HEALTHY
    assert s_status == HealthResult.HEALTHY
    assert p_ts == 0.0   # initial
    assert s_ts == 0.0   # initial

    group.force_primary_status(HealthResult.UNHEALTHY)
    p_status, s_status, p_ts, s_ts = fresh_registry.get_failover_status("P")
    assert p_status == HealthResult.UNHEALTHY
    assert s_status == HealthResult.HEALTHY   # untouched
    assert p_ts > 0.0
    assert s_ts == 0.0

    group.force_secondary_status(HealthResult.UNHEALTHY)
    p_status, s_status, _, s_ts2 = fresh_registry.get_failover_status("P")
    assert s_status == HealthResult.UNHEALTHY
    assert s_ts2 > 0.0

    assert fresh_registry.get_failover_status("nonexistent") is None


# ----------------------------------------------------------------- bootstrap

def test_bootstrap_creates_group_with_correct_uuids(fresh_registry, fake_state, monkeypatch):
    """Monkeypatch `get_or_bootstrap` so we never open a real connection.
    Verify that a FailoverGroup with the right primary + secondary states
    lands in the registry."""
    p = fake_state(("p1", "aws", "us-west", "us-west-1a", "primary"), uuid="P")
    s = fake_state(("s1", "aws", "us-east", "us-east-1a", "primary"), uuid="S")

    def fake_bootstrap(key, conninfo, kwargs):
        # Decide which side based on the host kwarg.
        return s if "s1" in str(kwargs.get("host", "")) else p

    monkeypatch.setattr(fresh_registry, "get_or_bootstrap", fake_bootstrap)
    # Monkeypatch the conninfo parser to avoid pulling in libpq.
    monkeypatch.setattr(
        fresh_registry, "_param_dict_for_key",
        lambda conninfo, kwargs: {"host": str(kwargs.get("host", ""))},
    )
    # Don't start the probe thread in this test (we test that separately).
    monkeypatch.setattr(fresh_registry, "_start_probe", lambda *a, **k: None)

    yb = _make_yb_params(["s1"])
    group = fresh_registry.get_or_bootstrap_failover_group(
        yb, "host=p1", {"host": "p1"},
    )
    assert group.primary.uuid == "P"
    assert group.secondary.uuid == "S"
    assert group.primary_status == HealthResult.HEALTHY
    assert group.cooldown_s == 5
    assert "P" in fresh_registry._failover_groups
    # (_clusters bookkeeping is tested separately — this test monkeypatches
    # `get_or_bootstrap` so the internal _clusters dict won't be populated.)


def test_bootstrap_is_idempotent(fresh_registry, fake_state, monkeypatch):
    p = fake_state(("p1", "aws", "us-west", "us-west-1a", "primary"), uuid="P")
    s = fake_state(("s1", "aws", "us-east", "us-east-1a", "primary"), uuid="S")

    call_count = {"n": 0}

    def fake_bootstrap(key, conninfo, kwargs):
        call_count["n"] += 1
        return s if "s1" in str(kwargs.get("host", "")) else p

    monkeypatch.setattr(fresh_registry, "get_or_bootstrap", fake_bootstrap)
    monkeypatch.setattr(
        fresh_registry, "_param_dict_for_key",
        lambda conninfo, kwargs: {"host": str(kwargs.get("host", ""))},
    )
    monkeypatch.setattr(fresh_registry, "_start_probe", lambda *a, **k: None)

    yb = _make_yb_params(["s1"])
    g1 = fresh_registry.get_or_bootstrap_failover_group(
        yb, "host=p1", {"host": "p1"})
    g2 = fresh_registry.get_or_bootstrap_failover_group(
        yb, "host=p1", {"host": "p1"})
    assert g1 is g2
    # First call: 2 bootstraps (primary + secondary). Second call: 1
    # (`get_or_bootstrap` for primary always runs because the failover-group
    # cache check happens AFTER primary bootstrap; primary itself is
    # internally cached by ClusterKey).
    assert call_count["n"] == 3


def test_bootstrap_secondary_kwargs_replace_host(fresh_registry, fake_state, monkeypatch):
    """The secondary cluster's bootstrap must use `secondary_cluster_hosts`,
    not the primary's host list — everything else carries over."""
    p = fake_state(("p1", "aws", "us-west", "us-west-1a", "primary"), uuid="P")
    s = fake_state(("s1", "aws", "us-east", "us-east-1a", "primary"), uuid="S")

    seen_kwargs: list[dict] = []

    def fake_bootstrap(key, conninfo, kwargs):
        seen_kwargs.append(dict(kwargs))
        return s if "s1" in str(kwargs.get("host", "")) else p

    monkeypatch.setattr(fresh_registry, "get_or_bootstrap", fake_bootstrap)
    monkeypatch.setattr(
        fresh_registry, "_param_dict_for_key",
        lambda conninfo, kwargs: {"host": str(kwargs.get("host", ""))},
    )
    monkeypatch.setattr(fresh_registry, "_start_probe", lambda *a, **k: None)

    yb = _make_yb_params(["sec-h1", "sec-h2"])
    primary_kwargs = {
        "host": "primary-h1,primary-h2",
        "port": 5433,
        "user": "yugabyte",
        "dbname": "yugabyte",
    }
    fresh_registry.get_or_bootstrap_failover_group(
        yb, "host=primary-h1,primary-h2", primary_kwargs,
    )
    assert len(seen_kwargs) == 2
    primary_call, secondary_call = seen_kwargs
    assert primary_call["host"] == "primary-h1,primary-h2"
    # Secondary kwargs carry over all non-host params.
    assert secondary_call["host"] == "sec-h1,sec-h2"
    assert secondary_call["port"] == 5433
    assert secondary_call["user"] == "yugabyte"
    assert secondary_call["dbname"] == "yugabyte"


# ----------------------------------------------------------------- clear()

def test_clear_drops_groups_and_stops_probes(fresh_registry, fake_state):
    p = fake_state(("p1", "aws", "us-west", "us-west-1a", "primary"), uuid="P")
    s = fake_state(("s1", "aws", "us-east", "us-east-1a", "primary"), uuid="S")
    group = _install_group(fresh_registry, p, s)

    # Attach fake probes to verify stop() is called on BOTH.
    stop_counts = {"primary": 0, "secondary": 0}
    class FakeProbe:
        def __init__(self, name: str) -> None:
            self._name = name
        def stop(self):
            stop_counts[self._name] += 1
    group.primary_probe = FakeProbe("primary")
    group.secondary_probe = FakeProbe("secondary")

    fresh_registry.clear()
    assert fresh_registry._failover_groups == {}
    assert stop_counts == {"primary": 1, "secondary": 1}


# ----------------------------------------------------------------- CB-None fallback

def test_check_functions_fall_back_to_healthy_when_cb_is_none(fake_state):
    """When a per-cluster CB slot is ``None`` (e.g. synthetic test group),
    ``check_primary_cluster`` and ``check_secondary_cluster`` return HEALTHY
    — matches the pre-CB stub semantics."""
    p = fake_state(("p1", "aws", "us-west", "us-west-1a", "primary"), uuid="P")
    s = fake_state(("s1", "aws", "us-east", "us-east-1a", "primary"), uuid="S")
    group = FailoverGroup(
        primary=p, secondary=s, lock=threading.Lock(),
        primary_status=HealthResult.UNHEALTHY, secondary_status=HealthResult.UNHEALTHY,
    )
    # No CBs installed → both fall back to HEALTHY regardless of stored status.
    assert group.primary_circuit_breaker is None
    assert group.secondary_circuit_breaker is None
    assert check_primary_cluster(group) == HealthResult.HEALTHY
    assert check_secondary_cluster(group) == HealthResult.HEALTHY


# ----------------------------------------------------------------- HealthProbe

def _make_probe_target_group(fake_state, cooldown_s=0):
    p = fake_state(("p1", "aws", "us-west", "us-west-1a", "primary"), uuid="P")
    s = fake_state(("s1", "aws", "us-east", "us-east-1a", "primary"), uuid="S")
    return FailoverGroup(
        primary=p, secondary=s, lock=threading.Lock(),
        primary_status=HealthResult.HEALTHY, secondary_status=HealthResult.HEALTHY, cooldown_s=cooldown_s,
    )


def test_probe_start_stop_idempotent(fake_state):
    group = _make_probe_target_group(fake_state)
    probe = HealthProbe(group, interval_s=10)
    probe.start()
    probe.start()  # idempotent
    assert probe._thread is not None
    probe.stop()
    probe.stop()  # idempotent
    assert not probe._thread.is_alive()


def test_probe_flips_status_on_first_tick(fake_state, monkeypatch):
    """With the stub returning UNHEALTHY (monkeypatched) and the initial
    `last_transition_time=0.0`, the very first tick must flip status."""
    import psycopg.yb.health_probe as hp_mod
    monkeypatch.setattr(
        hp_mod, "check_primary_cluster",
        lambda g: HealthResult.UNHEALTHY,
    )
    # Keep secondary HEALTHY so we only exercise the primary transition path.
    monkeypatch.setattr(
        hp_mod, "check_secondary_cluster",
        lambda g: HealthResult.HEALTHY,
    )
    group = _make_probe_target_group(fake_state, cooldown_s=0)
    probe = HealthProbe(group, interval_s=0.02)
    probe.start()
    # Allow a few ticks.
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline and group.primary_status == HealthResult.HEALTHY:
        time.sleep(0.02)
    probe.stop()
    assert group.primary_status == HealthResult.UNHEALTHY


def test_probe_honours_cooldown(fake_state, monkeypatch):
    """Pre-load `last_transition_time` so the cool-down is active; probe must
    NOT flip status even with the strategy returning UNHEALTHY."""
    import psycopg.yb.health_probe as hp_mod
    monkeypatch.setattr(
        hp_mod, "check_primary_cluster",
        lambda g: HealthResult.UNHEALTHY,
    )
    # Keep secondary HEALTHY so we only exercise the primary transition path.
    monkeypatch.setattr(
        hp_mod, "check_secondary_cluster",
        lambda g: HealthResult.HEALTHY,
    )
    group = _make_probe_target_group(fake_state, cooldown_s=999)
    group.primary_last_transition_time = time.monotonic()  # cool-down just started
    probe = HealthProbe(group, interval_s=0.02)
    probe.start()
    time.sleep(0.2)  # several probe ticks
    probe.stop()
    assert group.primary_status == HealthResult.HEALTHY  # cool-down kept us pinned


def test_probe_swallows_exceptions(fake_state, monkeypatch):
    """If a CB check raises, the probe loop must NOT exit."""
    import psycopg.yb.health_probe as hp_mod

    call_count = {"n": 0}
    def boom(g):
        call_count["n"] += 1
        raise RuntimeError("synthetic")

    monkeypatch.setattr(hp_mod, "check_primary_cluster", boom)
    monkeypatch.setattr(hp_mod, "check_secondary_cluster", boom)
    group = _make_probe_target_group(fake_state)
    probe = HealthProbe(group, interval_s=0.02)
    probe.start()
    time.sleep(0.2)
    probe.stop()
    # Loop ran multiple times despite the exception each time.
    assert call_count["n"] >= 2
    # Status unchanged.
    assert group.primary_status == HealthResult.HEALTHY
