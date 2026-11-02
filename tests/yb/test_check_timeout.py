"""
Unit tests for the ``check_timeout_s`` wall-clock cap on ``CircuitBreaker.check()``
(Phase C of the xCluster failover implementation plan).

Verifies:

  * ``YBParams.check_timeout_s`` — default is derived from
    ``refresh_interval_s // 2`` at parse time; explicit DSN values win;
    accepts dotted / underscore / camelCase / dashed forms.
  * ``HealthProbe`` — a check() that blocks past the cap is abandoned
    with a WARNING, previous status is preserved, and the probe loop
    keeps ticking on subsequent iterations.
"""

# Copyright (C) 2026 Yugabyte

from __future__ import annotations

import logging
import threading
import time

import pytest

from psycopg.yb.health import HealthResult
from psycopg.yb.health_probe import HealthProbe
from psycopg.yb.params import extract_yb_params
from psycopg.yb.registry import FailoverGroup


pytestmark = pytest.mark.yb_unit


# ------------------------------------------------------------ params parsing

def test_check_timeout_default_is_half_of_refresh_interval():
    """No explicit ``checkTimeoutSecs`` → default = max(1, refresh // 2)."""
    params, _, _ = extract_yb_params(
        "load_balance_hosts=true yb_servers_refresh_interval=20", {}
    )
    assert params.refresh_interval_s == 20
    assert params.check_timeout_s == 10


def test_check_timeout_default_floors_at_1_second():
    """A very small refresh (or 0) still yields a 1s floor."""
    params, _, _ = extract_yb_params(
        "load_balance_hosts=true yb_servers_refresh_interval=1", {}
    )
    assert params.check_timeout_s == 1


@pytest.mark.parametrize("dsn_key", [
    "yb.failover.checkTimeoutSecs",
    "yb_failover_check_timeout_secs",
    "yb-failover-check-timeout-secs",
])
def test_check_timeout_dsn_forms(dsn_key):
    """All separator variants parse to the canonical field."""
    params, _, _ = extract_yb_params(
        f"load_balance_hosts=true {dsn_key}=3", {}
    )
    assert params.check_timeout_s == 3


def test_check_timeout_via_kwargs():
    """Kwargs form ({'yb.failover.checkTimeoutSecs': N}) is accepted."""
    params, _, cleaned = extract_yb_params(
        "load_balance_hosts=true",
        {"yb.failover.checkTimeoutSecs": 7},
    )
    assert params.check_timeout_s == 7
    # Key was stripped from the cleaned kwargs (so libpq doesn't see it).
    assert "yb.failover.checkTimeoutSecs" not in cleaned


def test_check_timeout_clamps_floor_at_one():
    """Values below 1 second get clamped up."""
    params, _, _ = extract_yb_params(
        "load_balance_hosts=true yb.failover.checkTimeoutSecs=0", {}
    )
    assert params.check_timeout_s == 1


# ------------------------------------------------------------ probe enforcement

def _make_group(fake_state):
    p = fake_state(("p1", "aws", "us-west", "us-west-1a", "primary"), uuid="P")
    s = fake_state(("s1", "aws", "us-east", "us-east-1a", "primary"), uuid="S")
    return FailoverGroup(
        primary=p, secondary=s, lock=threading.Lock(),
        primary_status=HealthResult.HEALTHY,
        secondary_status=HealthResult.HEALTHY,
        cooldown_s=0,
    )


def test_probe_abandons_slow_check_and_logs_warning(
    fake_state, monkeypatch, caplog,
):
    """A ``check()`` that sleeps > check_timeout_s is abandoned. The
    probe logs a WARNING, preserves the previous status, and keeps
    ticking."""
    import psycopg.yb.health_probe as hp_mod

    started = threading.Event()

    def slow_check(group):
        started.set()
        # Sleep well past the cap so future.result(timeout=) fires.
        time.sleep(0.5)
        return HealthResult.UNHEALTHY   # would flip if we ever saw it

    monkeypatch.setattr(hp_mod, "check_primary_cluster", slow_check)
    monkeypatch.setattr(
        hp_mod, "check_secondary_cluster",
        lambda g: HealthResult.HEALTHY,
    )

    group = _make_group(fake_state)
    probe = HealthProbe(
        group,
        interval_s=0.05,
        which_cluster="primary",
        check_timeout_s=0.05,   # aggressive cap for the test
    )

    caplog.set_level(logging.WARNING, logger="psycopg.yb.health_probe")
    probe.start()
    # Wait long enough for at least one tick to fire and time out.
    assert started.wait(timeout=1.0)
    time.sleep(0.15)
    probe.stop()

    # Previous status preserved (slow_check returned UNHEALTHY but we
    # abandoned the future before receiving it).
    assert group.primary_status == HealthResult.HEALTHY
    # WARNING was emitted.
    warning_messages = [
        r.getMessage() for r in caplog.records if r.levelname == "WARNING"
    ]
    assert any("checkTimeoutSecs" in m for m in warning_messages), (
        f"expected a checkTimeoutSecs WARNING, saw: {warning_messages}"
    )


def test_probe_continues_ticking_after_a_timed_out_check(
    fake_state, monkeypatch,
):
    """A single slow tick doesn't stall the loop — subsequent ticks
    that return normally get applied."""
    import psycopg.yb.health_probe as hp_mod

    call_count = {"n": 0}
    slow_done = threading.Event()

    def sometimes_slow(group):
        call_count["n"] += 1
        if call_count["n"] == 1:
            # First call: block long enough to exceed the cap AND fill
            # the executor's single worker for the second tick — we
            # want to prove the loop still recovers.
            time.sleep(0.3)
            slow_done.set()
            return HealthResult.HEALTHY
        return HealthResult.UNHEALTHY

    monkeypatch.setattr(hp_mod, "check_primary_cluster", sometimes_slow)
    monkeypatch.setattr(
        hp_mod, "check_secondary_cluster",
        lambda g: HealthResult.HEALTHY,
    )

    group = _make_group(fake_state)
    probe = HealthProbe(
        group,
        interval_s=0.05,
        which_cluster="primary",
        check_timeout_s=0.05,
    )
    probe.start()

    # Wait for the slow check to finish AND at least one more tick to
    # register the flip.
    deadline = time.monotonic() + 3.0
    while (
        time.monotonic() < deadline
        and group.primary_status == HealthResult.HEALTHY
    ):
        time.sleep(0.05)
    probe.stop()

    assert slow_done.is_set(), "slow check never completed inside the executor"
    assert group.primary_status == HealthResult.UNHEALTHY, (
        f"expected the probe to keep ticking and eventually apply "
        f"UNHEALTHY, saw {group.primary_status}"
    )


# ------------------------------------------------------------ hung check vs exit

_HUNG_CHECK_SCRIPT = r"""
import sys, threading
import psycopg.yb.health_probe as hp
from psycopg.yb.health import HealthResult
from psycopg.yb.registry import ClusterState, FailoverGroup

def mk(uuid):
    return ClusterState(uuid=uuid, nodes={}, lock=threading.Lock(), last_refresh=0.0)

group = FailoverGroup(
    primary=mk("P"), secondary=mk("S"), lock=threading.Lock(),
    primary_status=HealthResult.HEALTHY, secondary_status=HealthResult.HEALTHY,
    cooldown_s=0,
)
hp.check_primary_cluster = lambda g: threading.Event().wait()   # never returns
probe = hp.HealthProbe(group, interval_s=60, which_cluster="primary",
                       check_timeout_s=0.05)
probe._tick_check()
print("tick returned", flush=True)
sys.exit(0)
"""


def test_hung_check_does_not_block_process_exit():
    """Review #9 (Amogh): executor workers are joined at interpreter exit
    (before atexit handlers run), so a CB check that never returns hung
    the process on sys.exit(). A check thread is a daemon now."""
    import os
    import subprocess
    import sys

    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)
    proc = subprocess.run(
        [sys.executable, "-c", _HUNG_CHECK_SCRIPT],
        env=env, capture_output=True, text=True, timeout=30,
    )
    assert "tick returned" in proc.stdout, proc.stderr
    assert proc.returncode == 0, proc.stderr


def test_probe_registers_no_atexit_hook(fake_state, monkeypatch):
    """One atexit entry per probe leaked for the life of the process."""
    import atexit

    calls = []
    monkeypatch.setattr(atexit, "register", lambda *a, **kw: calls.append(a))
    group = _make_group(fake_state)
    for _ in range(3):
        HealthProbe(group, interval_s=60, which_cluster="primary")
    assert calls == []


def test_hung_check_skips_next_tick_without_a_second_thread(
    fake_state, monkeypatch, caplog,
):
    """While a timed-out check is still running, later ticks skip (no
    second check thread) and the previous status is preserved; once it
    finishes, the next tick runs a fresh check."""
    import psycopg.yb.health_probe as hp_mod

    release = threading.Event()
    calls = []

    def hung_check(group):
        calls.append(threading.current_thread())
        release.wait(timeout=10)
        return HealthResult.UNHEALTHY

    monkeypatch.setattr(hp_mod, "check_primary_cluster", hung_check)
    group = _make_group(fake_state)
    probe = HealthProbe(
        group, interval_s=60, which_cluster="primary", check_timeout_s=0.05,
    )
    caplog.set_level(logging.WARNING, logger="psycopg.yb.health_probe")

    probe._tick_check()
    probe._tick_check()

    assert len(calls) == 1
    assert calls[0].daemon is True
    assert group.primary_status == HealthResult.HEALTHY
    msgs = [r.getMessage() for r in caplog.records]
    assert any("exceeded checkTimeoutSecs" in m for m in msgs), msgs
    assert any("skipping tick" in m for m in msgs), msgs

    release.set()
    calls[0].join(timeout=2)
    monkeypatch.setattr(hp_mod, "check_primary_cluster", lambda g: HealthResult.HEALTHY)
    probe._tick_check()
    assert probe.first_check_complete


def test_raising_check_is_no_change(fake_state, monkeypatch, caplog):
    import psycopg.yb.health_probe as hp_mod

    def boom(group):
        raise RuntimeError("cb bug")

    monkeypatch.setattr(hp_mod, "check_primary_cluster", boom)
    group = _make_group(fake_state)
    probe = HealthProbe(group, interval_s=60, which_cluster="primary")
    caplog.set_level(logging.WARNING, logger="psycopg.yb.health_probe")

    probe._tick_check()

    assert group.primary_status == HealthResult.HEALTHY
    assert not probe.first_check_complete
    assert any("CB check raised" in r.getMessage() for r in caplog.records)
