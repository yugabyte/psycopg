"""
Unit tests for the bootstrap first-poll gate (P1.6, design doc §4.6).

Verifies:

  * ``FailoverGroup.wait_for_first_check`` blocks until each probe's
    first ``check()`` has run — either the daemon's natural first tick
    or the synchronous fallback fired from the main thread.
  * A stuck / slow CB does not block bootstrap longer than
    ``checkTimeoutSecs`` per probe.
  * A raising or timed-out CB releases the gate (it doesn't block),
    but ``wait_for_first_check`` reports ``False`` for it.
  * Idempotent — calling twice with the first check already complete
    returns immediately.
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
        ("p1", "aws", "us-west", "us-west-1a", "primary"), uuid="P",
    )
    s = fake_state(
        ("s1", "aws", "us-east", "us-east-1a", "primary"), uuid="S",
    )
    return FailoverGroup(
        primary=p, secondary=s, lock=threading.Lock(),
        primary_status=HealthResult.HEALTHY,
        secondary_status=HealthResult.HEALTHY,
        cooldown_s=cooldown_s,
    )


# --------------------------------------------------- happy path

def test_wait_for_first_check_returns_true_after_both_ticks(fake_state, monkeypatch):
    """With fast CBs on both sides, wait_for_first_check returns True
    and both probes have fired at least one check."""
    import psycopg.yb.health_probe as hp_mod
    monkeypatch.setattr(
        hp_mod, "check_primary_cluster", lambda g: HealthResult.HEALTHY,
    )
    monkeypatch.setattr(
        hp_mod, "check_secondary_cluster", lambda g: HealthResult.HEALTHY,
    )
    group = _make_group(fake_state)
    # Long interval — daemon won't run first tick naturally within the
    # test window. wait_for_first_check should force it from main thread.
    group.primary_probe = HealthProbe(
        group, interval_s=999, which_cluster="primary", check_timeout_s=1.0,
    )
    group.secondary_probe = HealthProbe(
        group, interval_s=999, which_cluster="secondary", check_timeout_s=1.0,
    )
    group.primary_probe.start()
    group.secondary_probe.start()
    try:
        result = group.wait_for_first_check(timeout_s=3.0)
        assert result is True
        assert group.primary_probe.first_check_complete is True
        assert group.secondary_probe.first_check_complete is True
    finally:
        group.primary_probe.stop()
        group.secondary_probe.stop()


def test_wait_for_first_check_idempotent(fake_state, monkeypatch):
    """Calling twice after first checks have completed returns
    immediately the second time."""
    import psycopg.yb.health_probe as hp_mod
    monkeypatch.setattr(
        hp_mod, "check_primary_cluster", lambda g: HealthResult.HEALTHY,
    )
    monkeypatch.setattr(
        hp_mod, "check_secondary_cluster", lambda g: HealthResult.HEALTHY,
    )
    group = _make_group(fake_state)
    group.primary_probe = HealthProbe(
        group, interval_s=999, which_cluster="primary", check_timeout_s=1.0,
    )
    group.secondary_probe = HealthProbe(
        group, interval_s=999, which_cluster="secondary", check_timeout_s=1.0,
    )
    group.primary_probe.start()
    group.secondary_probe.start()
    try:
        group.wait_for_first_check(timeout_s=3.0)
        t0 = time.monotonic()
        group.wait_for_first_check(timeout_s=3.0)   # second call
        elapsed = time.monotonic() - t0
        # Second call should be effectively instant.
        assert elapsed < 0.1
    finally:
        group.primary_probe.stop()
        group.secondary_probe.stop()


# --------------------------------------------------- CB raising

def test_first_check_complete_not_set_on_cb_raise(fake_state, monkeypatch):
    """agent-K PR #9:260 fix: a CB that always raises must NOT set the
    `_first_check_complete` event. Pre-fix: event fired in a finally
    regardless of outcome, so the gate released with no verdict applied
    and the first connect routed on constructor defaults. Post-fix: the
    event is set only inside the verdict-applied branch of _tick_check.

    The registry-level `wait_for_first_check` result for this case is
    covered by `test_wait_for_first_check_false_when_cb_raises`."""
    import psycopg.yb.health_probe as hp_mod

    def raising(_g):
        raise RuntimeError("boom")

    monkeypatch.setattr(hp_mod, "check_primary_cluster", raising)
    monkeypatch.setattr(hp_mod, "check_secondary_cluster", raising)
    group = _make_group(fake_state)
    group.primary_probe = HealthProbe(
        group, interval_s=0.05, which_cluster="primary", check_timeout_s=0.5,
    )
    group.secondary_probe = HealthProbe(
        group, interval_s=0.05, which_cluster="secondary", check_timeout_s=0.5,
    )
    group.primary_probe.start()
    group.secondary_probe.start()
    try:
        # Let several ticks fire — each one should hit the CB-raised
        # branch and NOT set the event.
        time.sleep(0.3)
        assert group.primary_probe.first_check_complete is False, (
            "primary probe first_check_complete should stay False while "
            "CB keeps raising — agent-K PR #9:260"
        )
        assert group.secondary_probe.first_check_complete is False
    finally:
        group.primary_probe.stop()
        group.secondary_probe.stop()


def test_wait_for_first_check_false_when_cb_raises(fake_state, monkeypatch):
    """Review #10: with no deadline, a CB that raises used to make
    wait_for_first_check return True although no check completed. It
    must report False, without blocking."""
    import psycopg.yb.health_probe as hp_mod

    def raising(_g):
        raise RuntimeError("boom")

    monkeypatch.setattr(hp_mod, "check_primary_cluster", raising)
    monkeypatch.setattr(
        hp_mod, "check_secondary_cluster", lambda g: HealthResult.HEALTHY,
    )
    group = _make_group(fake_state)
    group.primary_probe = HealthProbe(
        group, interval_s=999, which_cluster="primary", check_timeout_s=0.5,
    )
    group.secondary_probe = HealthProbe(
        group, interval_s=999, which_cluster="secondary", check_timeout_s=0.5,
    )
    group.primary_probe.start()
    group.secondary_probe.start()
    try:
        assert group.wait_for_first_check() is False
        assert group.wait_for_first_check(timeout_s=3.0) is False
        assert group.primary_probe.first_check_complete is False
        assert group.secondary_probe.first_check_complete is True
    finally:
        group.primary_probe.stop()
        group.secondary_probe.stop()


def test_wait_for_first_check_false_when_cb_times_out(fake_state, monkeypatch):
    """A CB stalling past check_timeout_s returns within the cap, and
    the result is False rather than True."""
    import psycopg.yb.health_probe as hp_mod

    stall = threading.Event()

    def slow(_g):
        stall.wait(timeout=10)
        return HealthResult.HEALTHY

    monkeypatch.setattr(hp_mod, "check_primary_cluster", slow)
    monkeypatch.setattr(
        hp_mod, "check_secondary_cluster", lambda g: HealthResult.HEALTHY,
    )
    group = _make_group(fake_state)
    group.primary_probe = HealthProbe(
        group, interval_s=999, which_cluster="primary", check_timeout_s=0.2,
    )
    group.secondary_probe = HealthProbe(
        group, interval_s=999, which_cluster="secondary", check_timeout_s=0.2,
    )
    group.primary_probe.start()
    group.secondary_probe.start()
    try:
        t0 = time.monotonic()
        assert group.wait_for_first_check() is False
        assert time.monotonic() - t0 < 2.0
    finally:
        stall.set()
        group.primary_probe.stop()
        group.secondary_probe.stop()


# --------------------------------------------------- slow CB

def test_first_check_complete_not_set_on_slow_cb(fake_state, monkeypatch):
    """agent-K PR #9:260: a CB that stalls past check_timeout_s causes
    the tick to time out WITHOUT applying a verdict. The probe's
    `_first_check_complete` event must NOT fire on that path. The
    secondary CB returns normally and does set its event — asymmetric
    verifies only the slow side's event stays unset."""
    import psycopg.yb.health_probe as hp_mod

    stall = threading.Event()

    def slow(_g):
        stall.wait(timeout=10)  # sleep past check_timeout_s
        return HealthResult.HEALTHY

    monkeypatch.setattr(hp_mod, "check_primary_cluster", slow)
    monkeypatch.setattr(
        hp_mod, "check_secondary_cluster",
        lambda g: HealthResult.HEALTHY,
    )
    group = _make_group(fake_state)
    group.primary_probe = HealthProbe(
        group, interval_s=0.05, which_cluster="primary", check_timeout_s=0.3,
    )
    group.secondary_probe = HealthProbe(
        group, interval_s=0.05, which_cluster="secondary", check_timeout_s=0.3,
    )
    group.primary_probe.start()
    group.secondary_probe.start()
    try:
        # Wait long enough for several ticks + primary timeout.
        time.sleep(0.8)
        # Secondary returns a verdict → event fires.
        assert group.secondary_probe.first_check_complete is True, (
            "secondary CB applied a HEALTHY verdict → event should set"
        )
        # Primary CB keeps timing out → no verdict → event stays unset.
        assert group.primary_probe.first_check_complete is False, (
            "slow CB has no verdict applied → event must stay unset "
            "(agent-K PR #9:260)"
        )
    finally:
        stall.set()   # release the stall so cleanup finishes
        group.primary_probe.stop()
        group.secondary_probe.stop()


# --------------------------------------------------- no probes

def test_wait_for_first_check_no_probes_returns_true(fake_state):
    """When no probes have been wired (rare — bootstrap always wires
    them, but defensive), the gate is a no-op returning True."""
    group = _make_group(fake_state)
    # primary_probe / secondary_probe both None by default.
    assert group.wait_for_first_check() is True
