"""
Repro + regression tests for DB-25061 (PR-H8).

Shubham reported (#proj-smartdriver-xclusterfailover, Oct 26 2026): a
second ``bootstrap_failover_group`` call for the same universe pair
with ``yb.failover.drainTimeoutSecs=15`` silently returned the
FailoverGroup created by the first DSN with ``drainTimeoutSecs=10``.
No error, no warning. 1 of 1 cycles (100%).

The registry dedups groups by ``universe_uuid`` by design — one group
per pair per process — but it silently discarded the second caller's
knobs so their drain / lag / cooldown settings never took effect.

Fix: compare the incoming yb_params to the stored group's
bootstrap_params on every cache-hit path; raise
``ConflictingFailoverConfigError`` naming the conflicting keys so the
operator decides (align or process-split) rather than guessing which
caller wins.
"""

# Copyright (C) 2026 Yugabyte

from __future__ import annotations

import threading

import pytest

from psycopg.yb import ConflictingFailoverConfigError
from psycopg.yb.health import HealthResult
from psycopg.yb.registry import (
    FailoverGroup,
    _assert_bootstrap_params_match,
    _diff_failover_params,
)


pytestmark = pytest.mark.yb_unit


class _Params:
    """Tiny stand-in for YBParams — holds just the fields the diff
    helper inspects. Lets tests parametrize per-field without pulling
    the whole YBParams construction path (which validates many knobs
    and is tested separately in test_params_failover.py)."""

    def __init__(self, **kw):
        # Set defaults for all failover-governing fields so tests only
        # need to override the one they care about.
        for name in (
            "secondary_cluster_hosts",
            "replication_name",
            "failback_replication_name",
            "on_phase2_timeout",
        ):
            setattr(self, name, "")
        for name in (
            "drain_timeout_s",
            "lag_wait_timeout_s",
            "threshold_replication_lag_ms",
            "cooldown_s",
            "check_timeout_s",
            "refresh_interval_s",
            "failback_drain_timeout_s",
            "failback_lag_wait_timeout_s",
            "failback_threshold_replication_lag_ms",
            "global_time_budget_s",
        ):
            setattr(self, name, 10)
        self.auto_failback_enabled = True
        self.failback_global_time_budget_secs = None
        self.secondary_cluster_hosts = ("127.0.0.4", "127.0.0.5", "127.0.0.6")
        for name, value in kw.items():
            setattr(self, name, value)


def _make_group(fake_state, params) -> FailoverGroup:
    p = fake_state(("p1", "aws", "us-west", "us-west-1a", "primary"), uuid="P")
    s = fake_state(("s1", "aws", "us-east", "us-east-1a", "primary"), uuid="S")
    return FailoverGroup(
        primary=p, secondary=s, lock=threading.Lock(),
        primary_status=HealthResult.HEALTHY,
        secondary_status=HealthResult.HEALTHY,
        cooldown_s=params.cooldown_s,
        bootstrap_params=params,
    )


# ------------------------------------------------------------ diff helper

def test_diff_identical_params_is_empty():
    """Two identical YBParams: no conflicts."""
    a = _Params()
    b = _Params()
    assert _diff_failover_params(a, b) == []


def test_diff_detects_drain_timeout_conflict():
    """The exact scenario from DB-25061 report: 10 vs 15 on drain_timeout_s."""
    a = _Params(drain_timeout_s=10)
    b = _Params(drain_timeout_s=15)
    diffs = _diff_failover_params(a, b)
    assert len(diffs) == 1
    assert diffs[0] == ("drain_timeout_s", 10, 15)


def test_diff_detects_multiple_conflicts():
    a = _Params(drain_timeout_s=10, cooldown_s=60, lag_wait_timeout_s=30)
    b = _Params(drain_timeout_s=15, cooldown_s=120, lag_wait_timeout_s=30)
    names = {d[0] for d in _diff_failover_params(a, b)}
    assert names == {"drain_timeout_s", "cooldown_s"}


def test_diff_ignores_non_failover_fields():
    """A field not in the governing list must not trigger a conflict,
    even when it differs. Non-failover-affecting knobs (log level, etc.)
    stay outside the allowlist."""
    a = _Params()
    b = _Params()
    a.log_level = "DEBUG"
    b.log_level = "INFO"
    a.some_random_attr = 1
    b.some_random_attr = 2
    assert _diff_failover_params(a, b) == []


def test_diff_detects_yba_endpoint_conflict():
    """agent-K PR #20:193. YBA knobs govern whether Phase 2 lag wait
    runs: first DSN w/o YBA creates a group with yba_client=None →
    Phase 2 no-op; a second DSN with YBA wired would silently inherit
    the groupless-YBA group and Phase 2 would never run for its
    traffic. Treat YBA knobs as governing."""
    a = _Params()
    b = _Params()
    a.yba_endpoint = ""
    b.yba_endpoint = "https://yba.local"
    diffs = _diff_failover_params(a, b)
    names = {d[0] for d in diffs}
    assert "yba_endpoint" in names


def test_diff_detects_secondary_hosts_conflict():
    """Different secondary hosts means the two DSNs are routing to
    DIFFERENT secondary clusters — must raise, not silently route one
    DSN's traffic to the other's secondary."""
    a = _Params(secondary_cluster_hosts=("127.0.0.4", "127.0.0.5", "127.0.0.6"))
    b = _Params(secondary_cluster_hosts=("10.0.0.1", "10.0.0.2", "10.0.0.3"))
    diffs = _diff_failover_params(a, b)
    assert len(diffs) == 1
    assert diffs[0][0] == "secondary_cluster_hosts"


# ------------------------------------------------------------ assert helper

def test_assert_matches_on_identical_params_is_noop(fake_state):
    """Second bootstrap with identical knobs is the legitimate
    "two threads opened the same DSN" case — must not raise."""
    group = _make_group(fake_state, _Params(drain_timeout_s=10))
    _assert_bootstrap_params_match(group, _Params(drain_timeout_s=10))


def test_assert_raises_on_drain_timeout_conflict_db25061(fake_state):
    """The exact DB-25061 repro: group bootstrapped with
    drain_timeout_s=10, second call brings drain_timeout_s=15, must
    raise ConflictingFailoverConfigError naming the key."""
    group = _make_group(fake_state, _Params(drain_timeout_s=10))

    with pytest.raises(ConflictingFailoverConfigError) as exc_info:
        _assert_bootstrap_params_match(group, _Params(drain_timeout_s=15))

    msg = str(exc_info.value)
    assert "drain_timeout_s" in msg
    assert "10" in msg
    assert "15" in msg
    # The error must name the specific group so an operator with
    # several FailoverGroups in the process knows WHICH one conflicted.
    assert "primary_uuid=P" in msg


def test_assert_no_stored_params_is_tolerant(fake_state):
    """A FailoverGroup created directly (test fixture, legacy shim)
    with ``bootstrap_params=None`` must not raise — nothing to compare
    against. Verifies the H8 fix is backwards-compatible with existing
    integration tests that build groups via ``FailoverGroup(...)``
    directly."""
    group = _make_group(fake_state, _Params())
    group.bootstrap_params = None

    _assert_bootstrap_params_match(group, _Params(drain_timeout_s=999))


def test_assert_names_multiple_conflicts(fake_state):
    """When several knobs disagree, the error must list all of them so
    the operator doesn't have to fix-rerun-fix-rerun."""
    group = _make_group(fake_state, _Params(
        drain_timeout_s=10, cooldown_s=60, lag_wait_timeout_s=30,
    ))

    with pytest.raises(ConflictingFailoverConfigError) as exc_info:
        _assert_bootstrap_params_match(group, _Params(
            drain_timeout_s=15, cooldown_s=120, lag_wait_timeout_s=30,
        ))

    msg = str(exc_info.value)
    assert "drain_timeout_s" in msg
    assert "cooldown_s" in msg
    # lag_wait_timeout_s was equal — must NOT appear.
    assert "lag_wait_timeout_s" not in msg


def test_assert_subclasses_operational_error(fake_state):
    """ConflictingFailoverConfigError inherits from OperationalError
    so applications' generic retry-on-OperationalError wrappers can
    catch it with their existing code path."""
    from psycopg.errors import OperationalError
    group = _make_group(fake_state, _Params(drain_timeout_s=10))

    with pytest.raises(OperationalError):
        _assert_bootstrap_params_match(group, _Params(drain_timeout_s=15))
