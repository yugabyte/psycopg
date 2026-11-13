"""
Unit tests for the Phase 2 lag-wait loop in
``psycopg.yb.drain._wait_for_replication_lag``.

Verifies the three exit conditions from Amogh's 2026-08-11 flow:

  * lag ≤ threshold → converged (returns quickly, logs converged)
  * elapsed ≥ timeout → timeout (returns at deadline, logs timeout)
  * 3 consecutive YBA errors → unreachable (returns after strike 3)

Also verifies:
  * When ``group.yba_client is None`` the phase is skipped entirely
  * When new_status is HEALTHY (recovery) the phase is skipped
  * The lag phase reads the outgoing side's ``node_prefix`` correctly
"""

# Copyright (C) 2026 Yugabyte

from __future__ import annotations

import threading
from typing import Optional

import pytest

from psycopg.pq import TransactionStatus
from psycopg.yb.drain import trigger_drain
from psycopg.yb.health import HealthResult
from psycopg.yb.registry import FailoverGroup
from psycopg.yb.yba_client import YBAClientError


pytestmark = pytest.mark.yb_unit


# ------------------------------------------------------- fake YBA client

class _FakeYBAClient:
    """Test double for ``YBAClient``. Each call to
    ``get_committed_lag_ms`` pops the next value from ``responses`` — a
    ``float`` value is returned as-is, an ``Exception`` instance is
    raised. Any additional calls after the list is exhausted repeat
    the last value (or re-raise the last exception).

    Records every xcluster_uuid it received so tests can assert on the
    caller's use of the cached config UUID.
    """

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

    def get_committed_lag_sample(self, xcluster_uuid):
        """PR-H3 (DB-24483): drain.py now calls this variant. Return the
        same lag value as `get_committed_lag_ms` and a fake-fresh
        timestamp (`time.time()` so staleness checks see a 0-s age)."""
        import time as _time
        lag = self.get_committed_lag_ms(xcluster_uuid)
        return (lag, _time.time())


# ------------------------------------------------------- fake conn (mirrors test_drain.py)

class _FakeInfo:
    def __init__(self, status: TransactionStatus) -> None:
        self.transaction_status = status


class _FakeConn:
    def __init__(self, status: TransactionStatus = TransactionStatus.IDLE) -> None:
        self.info = _FakeInfo(status)
        self.closed = False

    def close(self) -> None:
        self.closed = True
        self.info.transaction_status = TransactionStatus.UNKNOWN


# ------------------------------------------------------- group with YBA wired

def _make_group_with_yba(
    fake_state,
    yba_responses: list,
    threshold_ms: int = 0,
    lag_wait_timeout_s: int = 5,
    with_yba_client: bool = True,
    # DB-24483: these tests predate the freshness + liveness gates and
    # exercise the pre-fix single-sample convergence. Opt out by
    # default; test_drain_db24483_freshness_gate.py covers the gate.
    lag_freshness_s: int = 0,
    lag_confirmation_samples: int = 1,
) -> tuple[FailoverGroup, _FakeYBAClient]:
    p = fake_state(("p1", "aws", "us-west", "us-west-1a", "primary"), uuid="P")
    s = fake_state(("s1", "aws", "us-east", "us-east-1a", "primary"), uuid="S")
    client = _FakeYBAClient(yba_responses) if with_yba_client else None
    group = FailoverGroup(
        primary=p, secondary=s, lock=threading.Lock(),
        primary_status=HealthResult.HEALTHY,
        secondary_status=HealthResult.HEALTHY,
        cooldown_s=0,
        yba_client=client,
        xcluster_config_uuid="xcc-abcd",
        threshold_replication_lag_ms=threshold_ms,
        lag_wait_timeout_s=lag_wait_timeout_s,
        lag_freshness_s=lag_freshness_s,
        lag_confirmation_samples=lag_confirmation_samples,
        failback_lag_freshness_s=lag_freshness_s,
        failback_lag_confirmation_samples=lag_confirmation_samples,
    )
    return group, client


# ------------------------------------------------------- exit condition: converged

def test_lag_wait_converges_when_lag_below_threshold(fake_state):
    """First poll returns lag ≤ threshold — exits immediately, no waiting."""
    group, client = _make_group_with_yba(
        fake_state, yba_responses=[0.5], threshold_ms=1, lag_wait_timeout_s=10,
    )
    import time
    t0 = time.monotonic()
    trigger_drain(group, "primary", HealthResult.UNHEALTHY, drain_timeout_s=0)
    elapsed = time.monotonic() - t0

    # Should return promptly (<1s).
    assert elapsed < 1.0
    assert group.primary_status == HealthResult.UNHEALTHY
    # YBA was called exactly once
    assert len(client.calls) == 1
    # Called for the outgoing (primary) side's prefix
    assert client.calls[0] == "xcc-abcd"


def test_lag_wait_converges_after_multiple_polls(fake_state):
    """First few polls return high lag, later polls converge — should exit
    at convergence, not at timeout."""
    group, client = _make_group_with_yba(
        fake_state,
        yba_responses=[500.0, 250.0, 100.0, 10.0, 0.5],
        threshold_ms=1,
        lag_wait_timeout_s=30,
    )
    trigger_drain(group, "primary", HealthResult.UNHEALTHY, drain_timeout_s=0)

    # Should have polled 5 times (until the 0.5 <= 1 exit)
    assert len(client.calls) == 5


# ------------------------------------------------------- exit condition: timeout

def test_lag_wait_timeout_failover_proceeds_even_with_fail_policy(fake_state):
    """`yb.failback.onPhase2Timeout=fail` only gates FAILBACK. A failover lag wait
    that times out still flips — staying on the side the CB just
    declared dead is worse than a lagged target."""
    group, client = _make_group_with_yba(
        fake_state,
        yba_responses=[500.0],   # sticky high value
        threshold_ms=1,
        lag_wait_timeout_s=2,
    )
    group.failback_on_phase2_timeout = "fail"
    import time
    t0 = time.monotonic()
    trigger_drain(group, "primary", HealthResult.UNHEALTHY, drain_timeout_s=0)
    elapsed = time.monotonic() - t0

    assert 1.5 <= elapsed <= 3.5
    assert len(client.calls) >= 2
    assert group.primary_status == HealthResult.UNHEALTHY
    assert group.dispatch_paused is False


def test_lag_wait_timeout_when_never_converges_proceed_policy(fake_state):
    """`yb.failback.onPhase2Timeout=proceed` (opt-in) — flip anyway once the lag
    wait times out."""
    group, client = _make_group_with_yba(
        fake_state,
        yba_responses=[500.0],
        threshold_ms=1,
        lag_wait_timeout_s=2,
    )
    group.failback_on_phase2_timeout = "proceed"
    import time
    t0 = time.monotonic()
    trigger_drain(group, "primary", HealthResult.UNHEALTHY, drain_timeout_s=0)
    elapsed = time.monotonic() - t0

    assert 1.5 <= elapsed <= 3.5
    assert group.primary_status == HealthResult.UNHEALTHY   # pre-fix behavior


# ------------------------------------------------------- exit condition: unreachable

def test_lag_wait_gives_up_after_3_consecutive_errors(fake_state):
    """3 consecutive YBAClientError → exits, proceeds with flip."""
    group, client = _make_group_with_yba(
        fake_state,
        yba_responses=[
            YBAClientError("net down"),
            YBAClientError("net down"),
            YBAClientError("net down"),
        ],
        threshold_ms=0,
        lag_wait_timeout_s=30,
    )
    import time
    t0 = time.monotonic()
    trigger_drain(group, "primary", HealthResult.UNHEALTHY, drain_timeout_s=0)
    elapsed = time.monotonic() - t0

    # Three polls, 1s between them → ~2-3s total.
    assert elapsed < 4.0
    assert group.primary_status == HealthResult.UNHEALTHY
    assert len(client.calls) == 3


def test_lag_wait_resets_error_streak_on_success(fake_state):
    """2 errors, then a success, then 2 more errors → does NOT give up
    (streak reset by the success in the middle)."""
    group, client = _make_group_with_yba(
        fake_state,
        yba_responses=[
            YBAClientError("blip"),
            YBAClientError("blip"),
            500.0,                       # success — resets streak
            YBAClientError("blip"),
            YBAClientError("blip"),
            0.0,                         # eventually converges
        ],
        threshold_ms=0,
        lag_wait_timeout_s=30,
    )
    trigger_drain(group, "primary", HealthResult.UNHEALTHY, drain_timeout_s=0)

    # Did NOT trip the 3-strike rule (never had 3 consecutive errors)
    # Should have polled all 6 responses
    assert len(client.calls) == 6
    assert group.primary_status == HealthResult.UNHEALTHY


# ------------------------------------------------------- lag phase skips

def test_lag_phase_skipped_when_no_yba_client(fake_state):
    """group.yba_client is None → no polling at all."""
    group, client = _make_group_with_yba(
        fake_state,
        yba_responses=[],
        with_yba_client=False,
    )
    assert group.yba_client is None
    trigger_drain(group, "primary", HealthResult.UNHEALTHY, drain_timeout_s=0)
    # (client is None; there's no calls list to check — the fact that
    # trigger_drain returned without exploding is the check.)
    assert group.primary_status == HealthResult.UNHEALTHY


def test_lag_phase_skipped_on_recovery(fake_state):
    """Recovery (new_status == HEALTHY) doesn't run the lag phase — lag
    only matters on failover (UNHEALTHY), not failback."""
    group, client = _make_group_with_yba(
        fake_state,
        yba_responses=[500.0],
    )
    # Flip primary to UNHEALTHY first, drain runs, lag phase fires.
    trigger_drain(group, "primary", HealthResult.UNHEALTHY, drain_timeout_s=0)
    lag_calls_after_failover = len(client.calls)

    # Recovery back to HEALTHY — no additional lag calls.
    trigger_drain(group, "primary", HealthResult.HEALTHY, drain_timeout_s=0)
    assert len(client.calls) == lag_calls_after_failover
    assert group.primary_status == HealthResult.HEALTHY


# ------------------------------------------------------- secondary-side flip

def test_no_lag_wait_when_last_viable_cluster_goes_down(fake_state):
    """Primary already UNHEALTHY, secondary now flips too: routing goes
    secondary → None. There is no target to switch to, so no drain runs
    and YBA is not polled."""
    group, client = _make_group_with_yba(
        fake_state,
        yba_responses=[0.0],
    )
    with group.lock:
        group.primary_status = HealthResult.UNHEALTHY
    client.calls.clear()

    trigger_drain(group, "secondary", HealthResult.UNHEALTHY, drain_timeout_s=0)

    assert client.calls == []
    assert group.secondary_status == HealthResult.UNHEALTHY


# ------------------------------------------------------- unexpected exceptions

def test_lag_wait_fails_open_on_unexpected_exception(fake_state):
    """A non-YBAClientError exception is treated as fail-open — proceeds
    with the flip rather than crashing the drain thread."""

    class _BrokenClient:
        def __init__(self):
            self.calls = []

        def get_committed_lag_ms(self, xu):
            self.calls.append(xu)
            raise RuntimeError("bug in yba_client")

        def get_committed_lag_sample(self, xu):
            # PR-H3 drain path uses this; mirror the ms-method
            # exception shape.
            return self.get_committed_lag_ms(xu)

    p = fake_state(("p1", "aws", "us-west", "us-west-1a", "primary"), uuid="P")
    s = fake_state(("s1", "aws", "us-east", "us-east-1a", "primary"), uuid="S")
    client = _BrokenClient()
    group = FailoverGroup(
        primary=p, secondary=s, lock=threading.Lock(),
        primary_status=HealthResult.HEALTHY,
        secondary_status=HealthResult.HEALTHY,
        cooldown_s=0,
        yba_client=client,
        xcluster_config_uuid="xcc-abcd",
        threshold_replication_lag_ms=0,
        lag_wait_timeout_s=30,
    )
    trigger_drain(group, "primary", HealthResult.UNHEALTHY, drain_timeout_s=0)
    assert group.primary_status == HealthResult.UNHEALTHY
    # First call raised → drain moves on
    assert len(client.calls) == 1


# ============================================================
# Failback (B→A) Phase 2 (P1.4)
# ============================================================

def _make_group_with_failback_yba(
    fake_state,
    yba_responses: list,
    failover_threshold_ms: int = 0,
    failover_timeout_s: int = 5,
    failback_threshold_ms: int = 0,
    failback_timeout_s: int = 5,
    failback_config_uuid: str = "xcc-b2a",
    # DB-24483: default both sides to pre-fix single-sample semantics so
    # existing tests keep testing the direction/UUID selection they were
    # written for; the gate itself has dedicated tests.
    lag_freshness_s: int = 0,
    lag_confirmation_samples: int = 1,
) -> tuple[FailoverGroup, _FakeYBAClient]:
    p = fake_state(("p1", "aws", "us-west", "us-west-1a", "primary"), uuid="P")
    s = fake_state(("s1", "aws", "us-east", "us-east-1a", "primary"), uuid="S")
    client = _FakeYBAClient(yba_responses)
    group = FailoverGroup(
        primary=p, secondary=s, lock=threading.Lock(),
        primary_status=HealthResult.UNHEALTHY,   # already failed over
        secondary_status=HealthResult.HEALTHY,
        cooldown_s=0,
        yba_client=client,
        xcluster_config_uuid="xcc-a2b",
        threshold_replication_lag_ms=failover_threshold_ms,
        lag_wait_timeout_s=failover_timeout_s,
        failback_xcluster_config_uuid=failback_config_uuid,
        failback_threshold_replication_lag_ms=failback_threshold_ms,
        failback_lag_wait_timeout_s=failback_timeout_s,
        lag_freshness_s=lag_freshness_s,
        lag_confirmation_samples=lag_confirmation_samples,
        failback_lag_freshness_s=lag_freshness_s,
        failback_lag_confirmation_samples=lag_confirmation_samples,
    )
    return group, client


def test_failback_lag_wait_fires_when_failback_config_set(fake_state):
    """Failback (HEALTHY on primary) with failback_xcluster_config_uuid
    set → Phase 2 runs against the B→A config."""
    group, client = _make_group_with_failback_yba(
        fake_state, yba_responses=[0.0],  # instant converge
    )
    trigger_drain(group, "primary", HealthResult.HEALTHY, drain_timeout_s=0)

    # One YBA poll happened; it used the B→A config UUID (not A→B).
    assert len(client.calls) == 1
    assert client.calls[0] == "xcc-b2a"
    # Failback completed.
    assert group.primary_status == HealthResult.HEALTHY


def test_failback_lag_wait_skipped_when_no_failback_config(fake_state):
    """Failback with failback_xcluster_config_uuid unset → Phase 2
    skipped (fail-open); YBA is never polled on the reverse direction."""
    group, client = _make_group_with_failback_yba(
        fake_state, yba_responses=[500.0],   # would not converge
        failback_config_uuid="",             # unset
    )
    trigger_drain(group, "primary", HealthResult.HEALTHY, drain_timeout_s=0)

    # No poll happened on the failback direction — the B→A UUID is empty
    # so Phase 2 is gated off.
    assert client.calls == []
    # Failback still completed (just without lag confirmation).
    assert group.primary_status == HealthResult.HEALTHY


def test_failback_uses_failback_threshold_not_failover(fake_state):
    """When failback threshold differs from failover threshold (Case 2
    asymmetric pattern: loose on failover, strict on failback), Phase 2
    on failback must honour the failback value."""
    group, client = _make_group_with_failback_yba(
        fake_state,
        # First response is 5 ms — would converge against failover
        # threshold=1000, but NOT against failback threshold=0.
        # Second response is 0 ms — converges everywhere.
        yba_responses=[5.0, 0.0],
        failover_threshold_ms=1000,   # loose
        failback_threshold_ms=0,      # strict
    )
    trigger_drain(group, "primary", HealthResult.HEALTHY, drain_timeout_s=0)

    # Phase 2 kept polling because 5 ms > failback threshold 0 ms.
    assert len(client.calls) == 2
    # All polls used the B→A UUID.
    assert all(c == "xcc-b2a" for c in client.calls)


def test_failback_uses_failback_timeout_not_failover(fake_state):
    """Failback Phase 2 honours failback_lag_wait_timeout_s, not the
    failover value."""
    import time as _time
    group, client = _make_group_with_failback_yba(
        fake_state,
        yba_responses=[500.0],   # sticky high — never converges
        failover_timeout_s=30,   # would be very slow if used
        failback_timeout_s=2,    # actual cap
        failback_threshold_ms=1, # keep it strict so 500ms never fits
    )
    t0 = _time.monotonic()
    trigger_drain(group, "primary", HealthResult.HEALTHY, drain_timeout_s=0)
    elapsed = _time.monotonic() - t0

    # Should have used the 2s failback timeout, not the 30s failover one.
    assert 1.5 <= elapsed <= 3.5


def test_failover_direction_still_uses_failover_config(fake_state):
    """Regression check: with failback config set, failover still uses
    the A→B config UUID and failover thresholds."""
    group, client = _make_group_with_failback_yba(
        fake_state, yba_responses=[0.0],
    )
    # Reset to healthy so we can fail over.
    with group.lock:
        group.primary_status = HealthResult.HEALTHY

    trigger_drain(group, "primary", HealthResult.UNHEALTHY, drain_timeout_s=0)

    # The poll used the A→B UUID (not the B→A).
    assert len(client.calls) == 1
    assert client.calls[0] == "xcc-a2b"



def test_lag_wait_timeout_uses_nearly_full_budget_with_fast_polls(fake_state, caplog):
    """Fast YBA polls must not cost 2×(connect+read) of the budget: a
    12s wait with millisecond polls runs to ≥11s, and the exit is logged
    as a plain "lag wait timeout"."""
    import logging
    import time
    group, client = _make_group_with_yba(
        fake_state,
        yba_responses=[500.0],
        threshold_ms=1,
        lag_wait_timeout_s=12,
    )
    caplog.set_level(logging.WARNING, logger="psycopg.yb.drain")
    t0 = time.monotonic()
    trigger_drain(group, "primary", HealthResult.UNHEALTHY, drain_timeout_s=0)
    elapsed = time.monotonic() - t0

    assert 11.0 <= elapsed <= 12.5, elapsed
    assert group.primary_status == HealthResult.UNHEALTHY
    msgs = [r.getMessage() for r in caplog.records]
    assert any(m.startswith("lag wait timeout after") for m in msgs), msgs
    assert not any("near-timeout" in m for m in msgs)


# ------------------------------------------------------- globalTimeBudgetSecs

class _InFlightConn(_FakeConn):
    """In a transaction that never commits; the drain deadline closes it."""

    def __init__(self) -> None:
        super().__init__(TransactionStatus.INTRANS)
        import types
        self.pgconn = types.SimpleNamespace(socket=-1)


def test_budget_caps_lag_wait_from_drain_start(fake_state):
    """Stress budget_split: drain start → drain complete must fit
    globalTimeBudgetSecs. Phase 1 used its whole share, so the lag wait
    only gets what is left of the budget, not its own full timeout."""
    import time

    group, client = _make_group_with_yba(
        fake_state, yba_responses=[500.0], threshold_ms=1, lag_wait_timeout_s=3,
    )
    group.global_time_budget_secs = 3
    conn = _InFlightConn()
    group.primary.tracked_conns.add(conn)

    t0 = time.monotonic()
    trigger_drain(group, "primary", HealthResult.UNHEALTHY, drain_timeout_s=1)
    elapsed = time.monotonic() - t0

    assert conn.closed
    assert 2.4 <= elapsed <= 3.0, elapsed
    assert group.primary_status == HealthResult.UNHEALTHY


def test_failback_uses_failback_budget(fake_state):
    import time

    group, client = _make_group_with_failback_yba(
        fake_state, yba_responses=[500.0], failback_threshold_ms=1,
        failback_timeout_s=3,
    )
    group.global_time_budget_secs = 60
    group.failback_global_time_budget_secs = 1
    t0 = time.monotonic()
    trigger_drain(group, "primary", HealthResult.HEALTHY, drain_timeout_s=0)
    assert time.monotonic() - t0 <= 1.2


# ------------------------------------------------------- per-node lag report

class _ReportYBAClient:
    """Returns per-node lag reports like YBAClient.get_committed_lag_report."""

    def __init__(self, reports: list) -> None:
        self.reports = list(reports)
        self.calls = 0

    def get_committed_lag_report(self, xcluster_uuid, deadline=None):
        import time as _time
        self.calls += 1
        r = self.reports[0] if len(self.reports) == 1 else self.reports.pop(0)
        return dict(r, sample_ts=_time.time() + 1)


def _report(values, stale=()):
    return {
        "lag_ms": max([v for v in values.values() if v > 1.0], default=0.0),
        "sample_ts": None,
        "values": values,
        "stale": list(stale),
        "leader_reported": any(v > 0 for n, v in values.items() if n not in stale),
    }


def _group_with_reports(fake_state, reports, timeout_s=2):
    group, _ = _make_group_with_yba(
        fake_state, yba_responses=[], threshold_ms=0, lag_wait_timeout_s=timeout_s,
    )
    client = _ReportYBAClient(reports)
    group.yba_client = client
    return group, client


def test_all_zero_lag_from_producer_does_not_converge(fake_state, caplog):
    """Stress phase2_claim (a_majority_down): the A nodes still up
    reported 0 because no tablet had a leader. Not a caught-up A."""
    import logging, time
    group, client = _group_with_reports(fake_state, [_report({"n1": 0.0, "n3": 0.0})])
    caplog.set_level(logging.INFO, logger="psycopg.yb.drain")
    t0 = time.monotonic()
    trigger_drain(group, "primary", HealthResult.UNHEALTHY, drain_timeout_s=0)
    assert time.monotonic() - t0 >= 1.5
    assert "no tablet leader reporting" in caplog.text
    assert "lag wait converged" not in caplog.text
    assert "lag wait timeout" in caplog.text


def test_node_that_stopped_reporting_blocks_converge(fake_state, caplog):
    import logging
    group, client = _group_with_reports(
        fake_state, [_report({"n1": 0.001, "n2": 0.001}, stale=["n2"])],
    )
    caplog.set_level(logging.INFO, logger="psycopg.yb.drain")
    trigger_drain(group, "primary", HealthResult.UNHEALTHY, drain_timeout_s=0)
    assert "n2 stopped reporting" in caplog.text
    assert "lag wait converged" not in caplog.text


def test_caught_up_leader_report_converges(fake_state, caplog):
    import logging
    group, client = _group_with_reports(
        fake_state, [_report({"n1": 0.001, "n2": 0.0, "n3": 0.001})], timeout_s=10,
    )
    caplog.set_level(logging.INFO, logger="psycopg.yb.drain")
    trigger_drain(group, "primary", HealthResult.UNHEALTHY, drain_timeout_s=0)
    assert client.calls == 1
    assert "lag wait converged" in caplog.text
    assert "nodes=n1:0.001,n2:0,n3:0.001" in caplog.text


def test_outgoing_all_nodes_down_does_not_converge(fake_state, caplog):
    """Stress phase2_claim (region_outage_with_yba, a_outage_into_limited_b):
    with every A process stopped, YBA kept returning 1us for all three A
    nodes after the barrier. The driver had all of them quarantined."""
    import logging
    group, client = _group_with_reports(
        fake_state, [_report({"n1": 0.001, "n2": 0.001, "n3": 0.001})],
    )
    for n in group.primary.nodes.values():
        n.is_down = True
    caplog.set_level(logging.INFO, logger="psycopg.yb.drain")
    trigger_drain(group, "primary", HealthResult.UNHEALTHY, drain_timeout_s=0)
    assert "every node of the outgoing universe is down" in caplog.text
    assert "lag wait converged" not in caplog.text
    assert "lag wait timeout" in caplog.text
    assert group.primary_status == HealthResult.UNHEALTHY


def test_outgoing_node_up_again_converges(fake_state, caplog):
    import logging
    group, client = _group_with_reports(
        fake_state, [_report({"n1": 0.001, "n2": 0.001, "n3": 0.001})], timeout_s=10,
    )
    node = next(iter(group.primary.nodes.values()))
    node.is_down = True
    group.primary.nodes["p2"] = type(node)(
        host="p2", public_ip=None, port=5433, placement=node.placement,
        node_type=node.node_type,
    )
    caplog.set_level(logging.INFO, logger="psycopg.yb.drain")
    trigger_drain(group, "primary", HealthResult.UNHEALTHY, drain_timeout_s=0)
    assert client.calls == 1
    assert "lag wait converged" in caplog.text


def test_failback_outgoing_all_nodes_down_does_not_converge(fake_state, caplog):
    import logging
    group, _ = _make_group_with_failback_yba(
        fake_state, yba_responses=[], failback_timeout_s=2,
    )
    group.yba_client = _ReportYBAClient([_report({"n1": 0.001})])
    for n in group.secondary.nodes.values():
        n.is_down = True
    caplog.set_level(logging.INFO, logger="psycopg.yb.drain")
    trigger_drain(group, "primary", HealthResult.HEALTHY, drain_timeout_s=0)
    assert "every node of the outgoing universe is down" in caplog.text
    assert "lag wait converged" not in caplog.text


# ------------------------------------------------------- unreachable + yb.failback.onPhase2Timeout=fail

def _failback_group(fake_state, responses, policy):
    group, client = _make_group_with_yba(
        fake_state, yba_responses=responses, threshold_ms=0, lag_wait_timeout_s=30,
    )
    group.primary_status = HealthResult.UNHEALTHY
    group.failback_xcluster_config_uuid = "xcc-ba"
    group.failback_lag_wait_timeout_s = 30
    group.failback_on_phase2_timeout = policy
    return group, client


def test_failback_yba_unreachable_aborts_with_fail_policy(fake_state):
    """`yb.failback.onPhase2Timeout=fail` means "don't fail back onto a primary we
    can't confirm has caught up"; an unreachable YBA is that case. The
    failback aborts, routing stays on the healthy secondary."""
    group, client = _failback_group(
        fake_state, [YBAClientError("net down")] * 3, "fail",
    )

    trigger_drain(group, "primary", HealthResult.HEALTHY, drain_timeout_s=0)

    assert len(client.calls) == 3
    assert group.primary_status == HealthResult.UNHEALTHY
    assert group.last_phase2_timeout_ts > 0
    assert group.dispatch_paused is False


def test_failback_yba_unreachable_proceeds_with_proceed_policy(fake_state):
    group, client = _failback_group(
        fake_state, [YBAClientError("net down")] * 3, "proceed",
    )

    trigger_drain(group, "primary", HealthResult.HEALTHY, drain_timeout_s=0)

    assert group.primary_status == HealthResult.HEALTHY


def test_failover_yba_unreachable_proceeds_even_with_fail_policy(fake_state):
    group, client = _make_group_with_yba(
        fake_state, yba_responses=[YBAClientError("net down")] * 3,
        threshold_ms=0, lag_wait_timeout_s=30,
    )
    group.failback_on_phase2_timeout = "fail"

    trigger_drain(group, "primary", HealthResult.UNHEALTHY, drain_timeout_s=0)

    assert group.primary_status == HealthResult.UNHEALTHY
    assert group.dispatch_paused is False


# ------------------------------------------------------- unexpected error + yb.failback.onPhase2Timeout=fail

def test_failback_unexpected_yba_error_aborts_with_fail_policy(fake_state):
    """Review #8 (Amogh): a non-YBAClientError from the lag source took
    the "unexpected YBA error — proceeding with flip" path and failed
    back with no lag confirmation, whatever the policy. Under `fail` the
    failback aborts there too."""
    group, client = _failback_group(
        fake_state, [RuntimeError("boom")], "fail",
    )

    trigger_drain(group, "primary", HealthResult.HEALTHY, drain_timeout_s=0)

    assert group.primary_status == HealthResult.UNHEALTHY
    assert group.last_phase2_timeout_ts > 0
    assert group.dispatch_paused is False


def test_failback_unexpected_yba_error_proceeds_with_proceed_policy(fake_state):
    group, client = _failback_group(
        fake_state, [RuntimeError("boom")], "proceed",
    )

    trigger_drain(group, "primary", HealthResult.HEALTHY, drain_timeout_s=0)

    assert group.primary_status == HealthResult.HEALTHY


def test_failover_unexpected_yba_error_proceeds_even_with_fail_policy(fake_state):
    group, client = _make_group_with_yba(
        fake_state, yba_responses=[RuntimeError("boom")],
        threshold_ms=0, lag_wait_timeout_s=30,
    )
    group.failback_on_phase2_timeout = "fail"

    trigger_drain(group, "primary", HealthResult.UNHEALTHY, drain_timeout_s=0)

    assert group.primary_status == HealthResult.UNHEALTHY
    assert group.dispatch_paused is False
