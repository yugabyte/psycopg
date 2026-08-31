"""
Unit tests for xCluster failover conninfo parsing.

These tests cover the ``yb.failover.*`` parameters the driver still owns:

* ``yb.failover.secondaryClusterHosts`` (string, comma-separated host list)
* ``yb.failover.cooldownSecs`` (int, default 1500)
* ``yb.failover.drainTimeoutSecs`` (int, default 10, sentinels: -1, 0, N)
* ``yb.failover.checkTimeoutSecs`` (int)

The ``trackerTableTablets`` and ``maxUpdateFailuresAllowed`` params were
removed when the default CB was moved to ``demo/samples/`` — they were
tracker-table-specific knobs, and custom CBs configure themselves via
their own constructors.

Plus the derived ``xcluster_enabled`` property — both
``load_balance_hosts=true`` AND a non-empty secondary host list must be set
before the failover machinery becomes active.

Auto-tagged as ``yb_unit`` via the filename prefix (see conftest.py).
"""

# Copyright (C) 2026 Yugabyte

from __future__ import annotations

import pytest

from psycopg.yb.params import (
    DEFAULT_COOLDOWN_SEC,
    YBParams,
    extract_yb_params,
)


# ----------------------------------------------------------------- defaults

def test_defaults_when_no_failover_params():
    yb, _, _ = extract_yb_params("host=h1", {})
    assert yb.secondary_cluster_hosts == []
    assert yb.cooldown_s == DEFAULT_COOLDOWN_SEC  # 1500
    assert yb.xcluster_enabled is False


# ----------------------------------------------------------------- key forms (dotted, snake, kebab)

@pytest.mark.parametrize("key_form", [
    "yb.failover.secondaryClusterHosts",
    "yb_failover_secondaryClusterHosts",
    "yb-failover-secondaryClusterHosts",
    "yb.failover.secondary_cluster_hosts",
    "yb_failover_secondary_cluster_hosts",
    "yb-failover-secondary-cluster-hosts",
    "yb.failover.secondary-cluster-hosts",
    # Case-insensitivity sanity
    "YB.FAILOVER.SECONDARYCLUSTERHOSTS",
    "YB.Failover.SecondaryClusterHosts",
])
def test_secondary_hosts_accepts_all_key_forms(key_form):
    conninfo = f"host=h1 load_balance_hosts=true {key_form}=s1,s2,s3"
    yb, cleaned, _ = extract_yb_params(conninfo, {})
    assert yb.secondary_cluster_hosts == ["s1", "s2", "s3"]
    # The key must be stripped from cleaned conninfo regardless of form.
    assert "failover" not in cleaned.lower()
    assert "s1" not in cleaned  # value also gone with the key
    # Original host parameter survives.
    assert "host=h1" in cleaned


@pytest.mark.parametrize("key_form", [
    "yb.failover.cooldownSecs",
    "yb_failover_cooldown_secs",
    "yb-failover-cooldown-secs",
])
def test_cooldown_secs_accepts_all_key_forms(key_form):
    conninfo = f"host=h1 {key_form}=600"
    yb, _, _ = extract_yb_params(conninfo, {})
    assert yb.cooldown_s == 600


# ----------------------------------------------------------------- value parsing

def test_secondary_hosts_single_host():
    yb, _, _ = extract_yb_params(
        "host=h1 load_balance_hosts=true "
        "yb.failover.secondaryClusterHosts=secondary1",
        {},
    )
    assert yb.secondary_cluster_hosts == ["secondary1"]


def test_secondary_hosts_strips_whitespace_and_drops_empties():
    yb, _, _ = extract_yb_params(
        "host=h1 load_balance_hosts=true "
        "yb.failover.secondaryClusterHosts='  h1,  h2 ,h3,,'",
        {},
    )
    assert yb.secondary_cluster_hosts == ["h1", "h2", "h3"]


def test_quoted_value_strips_quotes():
    yb, _, _ = extract_yb_params(
        'host=h1 load_balance_hosts=true '
        'yb.failover.secondaryClusterHosts="s1,s2"',
        {},
    )
    assert yb.secondary_cluster_hosts == ["s1", "s2"]


# ----------------------------------------------------------------- xcluster_enabled gating

def test_xcluster_enabled_requires_both_lb_and_secondary_hosts():
    # Neither — off.
    yb, _, _ = extract_yb_params("host=h1", {})
    assert yb.xcluster_enabled is False

    # load_balance_hosts only — off.
    yb, _, _ = extract_yb_params("host=h1 load_balance_hosts=true", {})
    assert yb.xcluster_enabled is False

    # secondaryClusterHosts only — off.
    yb, _, _ = extract_yb_params(
        "host=h1 yb.failover.secondaryClusterHosts=s1", {}
    )
    assert yb.xcluster_enabled is False

    # Both — on.
    yb, _, _ = extract_yb_params(
        "host=h1 load_balance_hosts=true "
        "yb.failover.secondaryClusterHosts=s1",
        {},
    )
    assert yb.xcluster_enabled is True


def test_xcluster_enabled_false_when_load_balance_hosts_is_disable():
    """`load_balance_hosts=disable` is a libpq pass-through, not opt-in to
    the smart driver. xCluster must stay off even with secondary hosts set."""
    yb, _, _ = extract_yb_params(
        "host=h1 load_balance_hosts=disable "
        "yb.failover.secondaryClusterHosts=s1",
        {},
    )
    assert yb.smart_driver_enabled is False
    assert yb.xcluster_enabled is False


# ----------------------------------------------------------------- kwargs (both string and spread forms)

def test_secondary_hosts_via_kwargs_underscored_form():
    yb, _, cleaned_kwargs = extract_yb_params(
        "host=h1 load_balance_hosts=true",
        {"yb_failover_secondary_cluster_hosts": "s1,s2"},
    )
    assert yb.secondary_cluster_hosts == ["s1", "s2"]
    # The kwarg is consumed; libpq must not see it.
    assert "yb_failover_secondary_cluster_hosts" not in cleaned_kwargs


def test_secondary_hosts_via_kwargs_dotted_form_via_spread():
    """Python identifiers can't contain `.`, but users can pass the dotted
    form via dict-spread: `psycopg.connect(**{"yb.failover.X": ...})`."""
    yb, _, cleaned_kwargs = extract_yb_params(
        "host=h1 load_balance_hosts=true",
        {"yb.failover.secondaryClusterHosts": "s1,s2"},
    )
    assert yb.secondary_cluster_hosts == ["s1", "s2"]
    assert "yb.failover.secondaryClusterHosts" not in cleaned_kwargs


def test_kwargs_override_conninfo():
    """Per the existing convention (mirrored from libpq), kwargs win on collision."""
    yb, _, _ = extract_yb_params(
        "host=h1 load_balance_hosts=true "
        "yb.failover.secondaryClusterHosts=conninfo_host",
        {"yb.failover.secondaryClusterHosts": "kwarg_host"},
    )
    assert yb.secondary_cluster_hosts == ["kwarg_host"]


def test_xcluster_kwargs_via_spread():
    yb, _, cleaned_kwargs = extract_yb_params(
        "host=h1 load_balance_hosts=true",
        {
            "yb.failover.secondaryClusterHosts": "s1,s2",
            "yb.failover.cooldownSecs": "600",
        },
    )
    assert yb.secondary_cluster_hosts == ["s1", "s2"]
    assert yb.cooldown_s == 600
    # None of them leaked into the cleaned kwargs.
    assert cleaned_kwargs == {}


# ----------------------------------------------------------------- clamps

def test_cooldown_secs_clamps_to_minimum_0():
    yb, _, _ = extract_yb_params(
        "host=h1 yb.failover.cooldownSecs=-1", {}
    )
    assert yb.cooldown_s == 0


def test_cooldown_secs_no_upper_clamp():
    """Long cool-downs are legitimate; no upper bound."""
    yb, _, _ = extract_yb_params(
        "host=h1 yb.failover.cooldownSecs=99999", {}
    )
    assert yb.cooldown_s == 99999


# ----------------------------------------------------------------- conninfo cleaning

def test_failover_keys_stripped_from_cleaned_conninfo():
    conninfo = (
        "host=h1,h2 port=5433 user=u dbname=d "
        "load_balance_hosts=true "
        "yb.failover.secondaryClusterHosts=s1,s2,s3 "
        "yb.failover.cooldownSecs=300"
    )
    yb, cleaned, _ = extract_yb_params(conninfo, {})
    # Non-YB keys preserved.
    assert "host=h1,h2" in cleaned
    assert "port=5433" in cleaned
    assert "user=u" in cleaned
    assert "dbname=d" in cleaned
    # YB keys stripped.
    assert "failover" not in cleaned.lower()
    assert "load_balance_hosts" not in cleaned
    # No stray equals or dangling tokens.
    assert "= " not in cleaned
    assert not cleaned.endswith("=")
    # Sanity: the YBParams object got the values.
    assert yb.secondary_cluster_hosts == ["s1", "s2", "s3"]
    assert yb.cooldown_s == 300


def test_failover_key_in_middle_of_conninfo_is_stripped_cleanly():
    """Whitespace collapse must not leave dangling tokens or merge adjacent params."""
    conninfo = (
        "host=h1 yb.failover.cooldownSecs=42 port=5433"
    )
    yb, cleaned, _ = extract_yb_params(conninfo, {})
    assert yb.cooldown_s == 42
    assert cleaned == "host=h1 port=5433"


# ----------------------------------------------------------------- coexistence with existing params

def test_xcluster_does_not_disturb_topology_keys():
    yb, _, _ = extract_yb_params(
        "host=h1 load_balance_hosts=true "
        "topology_keys=aws.us-west.us-west-1a,aws.us-west.us-west-1b "
        "yb.failover.secondaryClusterHosts=s1,s2",
        {},
    )
    assert len(yb.topology_keys) == 2
    assert yb.secondary_cluster_hosts == ["s1", "s2"]
    assert yb.smart_driver_enabled is True
    assert yb.xcluster_enabled is True


def test_xcluster_does_not_disturb_refresh_interval_or_delay():
    yb, _, _ = extract_yb_params(
        "host=h1 load_balance_hosts=true "
        "yb_servers_refresh_interval=60 "
        "failed_host_reconnect_delay_secs=10 "
        "yb.failover.cooldownSecs=99",
        {},
    )
    assert yb.refresh_interval_s == 60
    assert yb.failed_host_reconnect_delay_s == 10
    assert yb.cooldown_s == 99


# ----------------------------------------------------------------- YBA lag-wait phase (K.2)

def test_yba_params_default_lag_wait_disabled():
    """Without any yba.* keys, lag_wait_enabled must be False even if
    xcluster is otherwise enabled."""
    yb, _, _ = extract_yb_params(
        "host=h1 load_balance_hosts=true "
        "yb.failover.secondaryClusterHosts=s1",
        {},
    )
    assert yb.xcluster_enabled is True
    assert yb.lag_wait_enabled is False
    assert yb.yba_endpoint == ""
    assert yb.yba_api_token == ""
    assert yb.replication_name == ""
    assert yb.threshold_replication_lag_ms == 0
    assert yb.lag_wait_timeout_s == 30


def test_yba_params_full_camel_case():
    yb, _, _ = extract_yb_params(
        "host=h1 load_balance_hosts=true "
        "yb.failover.secondaryClusterHosts=s1 "
        "yb.failover.ybaEndpoint=https://yba.example.com "
        "yb.failover.ybaApiToken=tok-12345 "
        "yb.failover.replicationName=rep-src-to-target "
        "yb.failover.thresholdReplicationLagMs=250 "
        "yb.failover.lagWaitTimeoutSecs=45",
        {},
    )
    assert yb.lag_wait_enabled is True
    assert yb.yba_endpoint == "https://yba.example.com"
    assert yb.yba_api_token == "tok-12345"
    assert yb.replication_name == "rep-src-to-target"
    assert yb.threshold_replication_lag_ms == 250
    assert yb.lag_wait_timeout_s == 45


def test_yba_params_snake_and_kebab_forms():
    """Mixed separators — regex + normalisation should fold to canonical
    snake_case params."""
    yb, _, _ = extract_yb_params(
        "host=h1 load_balance_hosts=true "
        "yb.failover.secondaryClusterHosts=s1 "
        "yb-failover-yba-endpoint=https://yba "
        "yb_failover_yba_api_token=t "
        "yb.failover.replication_name=repname "
        "yb-failover-threshold-replication-lag-ms=100 "
        "yb.failover.lag_wait_timeout_secs=15",
        {},
    )
    assert yb.lag_wait_enabled is True
    assert yb.yba_endpoint == "https://yba"
    assert yb.threshold_replication_lag_ms == 100
    assert yb.lag_wait_timeout_s == 15


def test_yba_params_endpoint_alone_not_enough():
    """Presence of ybaEndpoint alone doesn't enable — token + replication
    name are ALSO required."""
    yb, _, _ = extract_yb_params(
        "host=h1 load_balance_hosts=true "
        "yb.failover.secondaryClusterHosts=s1 "
        "yb.failover.ybaEndpoint=https://yba.example.com",
        {},
    )
    assert yb.lag_wait_enabled is False


def test_yba_params_stripped_from_cleaned_conninfo():
    """Cleaned conninfo must not carry the yb.* keys — libpq must not see
    the YBA credentials."""
    dsn = (
        "host=h1 load_balance_hosts=true "
        "yb.failover.secondaryClusterHosts=s1 "
        "yb.failover.ybaEndpoint=https://yba.example.com "
        "yb.failover.ybaApiToken=secret-tok "
        "yb.failover.replicationName=rep-abc"
    )
    yb, cleaned, _ = extract_yb_params(dsn, {})
    assert "yba" not in cleaned.lower()
    assert "secret-tok" not in cleaned
    assert "host=h1" in cleaned


# ============================================================
# globalTimeBudgetSecs (P1.1) — Amex ask, design doc §3.7
# ============================================================

def _dsn_with_secondary(*extras: str) -> str:
    """Build a valid xCluster-enabled DSN with additional params."""
    return " ".join([
        "host=h1 load_balance_hosts=true",
        "yb.failover.secondaryClusterHosts=s1",
        *extras,
    ])


def test_no_budget_uses_existing_defaults():
    """Unset globalTimeBudgetSecs — parser leaves drain/lag_wait at their
    existing defaults. Fully backward-compatible."""
    yb, _, _ = extract_yb_params(_dsn_with_secondary(), {})
    assert yb.global_time_budget_secs is None
    assert yb.drain_timeout_s == 10           # DEFAULT_DRAIN_TIMEOUT_SEC
    assert yb.lag_wait_timeout_s == 30        # DEFAULT_LAG_WAIT_TIMEOUT_SEC
    assert yb.yba_connect_timeout_s == 2.0    # DEFAULT_YBA_CONNECT_TIMEOUT_S
    assert yb.yba_read_timeout_s == 3.0       # DEFAULT_YBA_READ_TIMEOUT_S


def test_budget_only_derives_drain_and_lag_wait():
    """Budget set, no explicit phase knobs — derive 40/60 split."""
    yb, _, _ = extract_yb_params(
        _dsn_with_secondary("yb.failover.globalTimeBudgetSecs=10"), {},
    )
    assert yb.global_time_budget_secs == 10
    assert yb.drain_timeout_s == 4     # round(10 * 0.40)
    assert yb.lag_wait_timeout_s == 6  # round(10 * 0.60)


@pytest.mark.parametrize("budget,expected_drain,expected_lag", [
    (5,  2, 3),   # 40% = 2, 60% = 3
    (10, 4, 6),
    (20, 8, 12),
    (30, 12, 18),
])
def test_budget_derivation_various_sizes(budget, expected_drain, expected_lag):
    yb, _, _ = extract_yb_params(
        _dsn_with_secondary(f"yb.failover.globalTimeBudgetSecs={budget}"), {},
    )
    assert yb.drain_timeout_s == expected_drain
    assert yb.lag_wait_timeout_s == expected_lag


def test_budget_plus_explicit_drain_within_share_keeps_ratio_lag_wait():
    """User provides budget AND drainTimeoutSecs at or below its 40%
    share — driver leaves drain as-is and derives lag_wait via the 60%
    ratio. Sum ≤ budget."""
    yb, _, _ = extract_yb_params(
        _dsn_with_secondary(
            "yb.failover.globalTimeBudgetSecs=10",
            "yb.failover.drainTimeoutSecs=3",
        ),
        {},
    )
    assert yb.drain_timeout_s == 3
    assert yb.lag_wait_timeout_s == 6  # round(10 * 0.60) — ratio still applies
    # Sum = 9s < 10s budget: 1s of slack, that's fine.


def test_budget_plus_explicit_drain_over_share_raises():
    """User sets drain above its 40% share — ratio-derived lag_wait
    would push the sum over budget, so the parser rejects."""
    with pytest.raises(ValueError, match="globalTimeBudgetSecs"):
        extract_yb_params(
            _dsn_with_secondary(
                "yb.failover.globalTimeBudgetSecs=10",
                "yb.failover.drainTimeoutSecs=7",   # > 40% share of 4s
            ),
            {},
        )


def test_budget_plus_explicit_lag_wait_within_share_keeps_ratio_drain():
    yb, _, _ = extract_yb_params(
        _dsn_with_secondary(
            "yb.failover.globalTimeBudgetSecs=10",
            "yb.failover.lagWaitTimeoutSecs=5",  # < 60% share of 6s
        ),
        {},
    )
    assert yb.lag_wait_timeout_s == 5
    assert yb.drain_timeout_s == 4  # round(10 * 0.40)


def test_budget_plus_explicit_lag_wait_over_share_raises():
    with pytest.raises(ValueError, match="globalTimeBudgetSecs"):
        extract_yb_params(
            _dsn_with_secondary(
                "yb.failover.globalTimeBudgetSecs=10",
                "yb.failover.lagWaitTimeoutSecs=8",   # > 60% share of 6s
            ),
            {},
        )


def test_over_budget_raises():
    """Sum of explicit knobs > budget — parser rejects at bootstrap."""
    with pytest.raises(ValueError, match="globalTimeBudgetSecs"):
        extract_yb_params(
            _dsn_with_secondary(
                "yb.failover.globalTimeBudgetSecs=10",
                "yb.failover.drainTimeoutSecs=8",
                "yb.failover.lagWaitTimeoutSecs=5",
            ),
            {},
        )


def test_budget_zero_rejected():
    with pytest.raises(ValueError, match="positive"):
        extract_yb_params(
            _dsn_with_secondary("yb.failover.globalTimeBudgetSecs=0"), {},
        )


def test_budget_negative_rejected():
    with pytest.raises(ValueError, match="positive"):
        extract_yb_params(
            _dsn_with_secondary("yb.failover.globalTimeBudgetSecs=-5"), {},
        )


def test_drain_sentinel_not_counted_against_budget():
    """drain_timeout_s = -1 (wait forever) or 0 (immediate) are sentinels,
    not real wall-clock — they must not count against the budget."""
    yb, _, _ = extract_yb_params(
        _dsn_with_secondary(
            "yb.failover.globalTimeBudgetSecs=10",
            "yb.failover.drainTimeoutSecs=0",
            "yb.failover.lagWaitTimeoutSecs=10",
        ),
        {},
    )
    assert yb.drain_timeout_s == 0
    assert yb.lag_wait_timeout_s == 10  # allowed: 0 + 10 == budget


def test_yba_client_timeouts_scale_with_budget():
    """When budget is set, YBAClient inner timeouts shrink to fit inside
    a single Phase 2 poll iteration."""
    yb, _, _ = extract_yb_params(
        _dsn_with_secondary("yb.failover.globalTimeBudgetSecs=10"), {},
    )
    # lag_wait = 6s; connect = min(2.0, 6 * 0.25) = 1.5; read = min(3.0, 6 * 0.35) = 2.1
    assert yb.yba_connect_timeout_s == pytest.approx(1.5)
    assert yb.yba_read_timeout_s == pytest.approx(2.1)


def test_yba_client_timeouts_capped_by_default():
    """Very large budget → YBA timeouts do not exceed the defaults
    (min-based capping)."""
    yb, _, _ = extract_yb_params(
        _dsn_with_secondary("yb.failover.globalTimeBudgetSecs=100"), {},
    )
    # lag_wait = 60s; connect = min(2.0, 60 * 0.25=15) = 2.0; read = min(3.0, 60 * 0.35=21) = 3.0
    assert yb.yba_connect_timeout_s == 2.0
    assert yb.yba_read_timeout_s == 3.0


@pytest.mark.parametrize("key_form", [
    "yb.failover.globalTimeBudgetSecs",
    "yb_failover_global_time_budget_secs",
    "yb-failover-global-time-budget-secs",
    "yb.failover.global_time_budget_secs",
    "YB.FAILOVER.GLOBALTIMEBUDGETSECS",
])
def test_budget_key_forms(key_form):
    yb, _, _ = extract_yb_params(
        _dsn_with_secondary(f"{key_form}=10"), {},
    )
    assert yb.global_time_budget_secs == 10


def test_budget_stripped_from_cleaned_conninfo():
    dsn = _dsn_with_secondary("yb.failover.globalTimeBudgetSecs=10")
    _, cleaned, _ = extract_yb_params(dsn, {})
    assert "globaltimebudget" not in cleaned.lower()
    assert "host=h1" in cleaned



# ============================================================
# autoFailbackEnabled (P1.2)
# ============================================================

def test_auto_failback_default_true():
    """Unset — driver preserves v1 auto-failback behaviour."""
    yb, _, _ = extract_yb_params(_dsn_with_secondary(), {})
    assert yb.auto_failback_enabled is True


@pytest.mark.parametrize("truthy", ["true", "True", "TRUE", "1", "yes", "on", "YES"])
def test_auto_failback_truthy(truthy):
    yb, _, _ = extract_yb_params(
        _dsn_with_secondary(f"yb.failover.autoFailbackEnabled={truthy}"), {},
    )
    assert yb.auto_failback_enabled is True


@pytest.mark.parametrize("falsy", ["false", "False", "FALSE", "0", "no", "off", "NO"])
def test_auto_failback_falsy(falsy):
    yb, _, _ = extract_yb_params(
        _dsn_with_secondary(f"yb.failover.autoFailbackEnabled={falsy}"), {},
    )
    assert yb.auto_failback_enabled is False


def test_auto_failback_invalid_raises():
    with pytest.raises(ValueError, match="autoFailbackEnabled"):
        extract_yb_params(
            _dsn_with_secondary("yb.failover.autoFailbackEnabled=maybe"), {},
        )


@pytest.mark.parametrize("key_form", [
    "yb.failover.autoFailbackEnabled",
    "yb_failover_auto_failback_enabled",
    "yb-failover-auto-failback-enabled",
    "yb.failover.auto_failback_enabled",
    "YB.FAILOVER.AUTOFAILBACKENABLED",
])
def test_auto_failback_key_forms(key_form):
    yb, _, _ = extract_yb_params(
        _dsn_with_secondary(f"{key_form}=false"), {},
    )
    assert yb.auto_failback_enabled is False


def test_auto_failback_stripped_from_cleaned_conninfo():
    dsn = _dsn_with_secondary("yb.failover.autoFailbackEnabled=false")
    _, cleaned, _ = extract_yb_params(dsn, {})
    assert "autofailback" not in cleaned.lower()
    assert "host=h1" in cleaned



# ============================================================
# yb.failback.* namespace (P1.3) — per-direction overrides
# ============================================================

def test_failback_defaults_to_failover_values_when_unset():
    """No yb.failback.* set — failback params equal failover params
    (except replicationName which has no fallback)."""
    yb, _, _ = extract_yb_params(
        _dsn_with_secondary(
            "yb.failover.drainTimeoutSecs=7",
            "yb.failover.lagWaitTimeoutSecs=25",
            "yb.failover.thresholdReplicationLagMs=100",
        ),
        {},
    )
    assert yb.failback_drain_timeout_s == 7
    assert yb.failback_lag_wait_timeout_s == 25
    assert yb.failback_threshold_replication_lag_ms == 100
    assert yb.failback_global_time_budget_secs is None
    # replicationName has NO fallback.
    assert yb.failback_replication_name == ""


def test_failback_replication_name_no_fallback_from_failover():
    """Only yb.failover.replicationName set — failback_replication_name
    remains empty (must be set explicitly for the reverse direction)."""
    yb, _, _ = extract_yb_params(
        _dsn_with_secondary(
            "yb.failover.replicationName=rep-a-to-b",
        ),
        {},
    )
    assert yb.replication_name == "rep-a-to-b"
    assert yb.failback_replication_name == ""


def test_failback_replication_name_set_explicitly():
    yb, _, _ = extract_yb_params(
        _dsn_with_secondary(
            "yb.failover.replicationName=rep-a-to-b",
            "yb.failback.replicationName=rep-b-to-a",
        ),
        {},
    )
    assert yb.replication_name == "rep-a-to-b"
    assert yb.failback_replication_name == "rep-b-to-a"


def test_failback_drain_overrides_failover():
    yb, _, _ = extract_yb_params(
        _dsn_with_secondary(
            "yb.failover.drainTimeoutSecs=10",
            "yb.failback.drainTimeoutSecs=3",
        ),
        {},
    )
    assert yb.drain_timeout_s == 10
    assert yb.failback_drain_timeout_s == 3


def test_failback_lag_wait_overrides_failover():
    yb, _, _ = extract_yb_params(
        _dsn_with_secondary(
            "yb.failover.lagWaitTimeoutSecs=30",
            "yb.failback.lagWaitTimeoutSecs=5",
        ),
        {},
    )
    assert yb.lag_wait_timeout_s == 30
    assert yb.failback_lag_wait_timeout_s == 5


def test_failback_threshold_overrides_failover():
    """The asymmetric-threshold Case 2 pattern: loose on failover, strict
    on failback."""
    yb, _, _ = extract_yb_params(
        _dsn_with_secondary(
            "yb.failover.thresholdReplicationLagMs=1000",
            "yb.failback.thresholdReplicationLagMs=0",
        ),
        {},
    )
    assert yb.threshold_replication_lag_ms == 1000
    assert yb.failback_threshold_replication_lag_ms == 0


def test_failback_budget_derives_failback_drain_and_lag_wait():
    """Failback-side budget, no explicit failback phase knobs — driver
    derives failback drain / lag_wait via 40/60 ratio just like the
    failover side."""
    yb, _, _ = extract_yb_params(
        _dsn_with_secondary(
            "yb.failback.globalTimeBudgetSecs=10",
        ),
        {},
    )
    assert yb.failback_global_time_budget_secs == 10
    assert yb.failback_drain_timeout_s == 4  # 40%
    assert yb.failback_lag_wait_timeout_s == 6  # 60%


def test_failback_budget_inherits_from_failover_when_unset():
    """No failback-specific budget → falls back to the failover budget.
    Failback drain/lag_wait derived from the inherited budget."""
    yb, _, _ = extract_yb_params(
        _dsn_with_secondary(
            "yb.failover.globalTimeBudgetSecs=10",
        ),
        {},
    )
    assert yb.global_time_budget_secs == 10
    assert yb.failback_global_time_budget_secs == 10
    # Both sides derive from the same budget.
    assert yb.drain_timeout_s == 4
    assert yb.failback_drain_timeout_s == 4


def test_failback_budget_asymmetric_from_failover():
    """Different budgets per direction — Case 2 pattern."""
    yb, _, _ = extract_yb_params(
        _dsn_with_secondary(
            "yb.failover.globalTimeBudgetSecs=10",
            "yb.failback.globalTimeBudgetSecs=20",
        ),
        {},
    )
    assert yb.global_time_budget_secs == 10
    assert yb.failback_global_time_budget_secs == 20
    assert yb.drain_timeout_s == 4              # 40% of 10
    assert yb.failback_drain_timeout_s == 8     # 40% of 20
    assert yb.lag_wait_timeout_s == 6           # 60% of 10
    assert yb.failback_lag_wait_timeout_s == 12  # 60% of 20


def test_failback_over_budget_raises():
    """Sum of failback drain + lag_wait > failback budget → parser rejects."""
    with pytest.raises(ValueError, match="failback"):
        extract_yb_params(
            _dsn_with_secondary(
                "yb.failback.globalTimeBudgetSecs=10",
                "yb.failback.drainTimeoutSecs=8",     # > 40% share of 4s
            ),
            {},
        )


@pytest.mark.parametrize("key_form", [
    "yb.failback.replicationName",
    "yb_failback_replication_name",
    "yb-failback-replication-name",
    "yb.failback.replication_name",
    "YB.FAILBACK.REPLICATIONNAME",
])
def test_failback_key_forms(key_form):
    yb, _, _ = extract_yb_params(
        _dsn_with_secondary(f"{key_form}=rep-b-to-a"), {},
    )
    assert yb.failback_replication_name == "rep-b-to-a"


def test_failback_params_stripped_from_cleaned_conninfo():
    dsn = _dsn_with_secondary(
        "yb.failback.replicationName=rep-b-to-a",
        "yb.failback.thresholdReplicationLagMs=0",
    )
    _, cleaned, _ = extract_yb_params(dsn, {})
    assert "failback" not in cleaned.lower()
    assert "rep-b-to-a" not in cleaned

