"""
Unit tests for xCluster failover conninfo parsing (Phase 1 of the xCluster
failover implementation plan).

These tests cover the four new ``yb.failover.*`` parameters:

* ``yb.failover.secondaryClusterHosts`` (string, comma-separated host list)
* ``yb.failover.trackerTableTablets`` (int, default 9)
* ``yb.failover.maxUpdateFailuresAllowed`` (int, default 0)
* ``yb.failover.cooldownSecs`` (int, default 1500)

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
    DEFAULT_MAX_UPDATE_FAILURES_ALLOWED,
    DEFAULT_TRACKER_TABLE_TABLETS,
    YBParams,
    extract_yb_params,
)


# ----------------------------------------------------------------- defaults

def test_defaults_when_no_failover_params():
    yb, _, _ = extract_yb_params("host=h1", {})
    assert yb.secondary_cluster_hosts == []
    assert yb.tracker_table_tablets == DEFAULT_TRACKER_TABLE_TABLETS  # 9
    assert yb.max_update_failures_allowed == DEFAULT_MAX_UPDATE_FAILURES_ALLOWED  # 0
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


@pytest.mark.parametrize("key_form,expected_value", [
    ("yb.failover.trackerTableTablets", 27),
    ("yb_failover_tracker_table_tablets", 27),
    ("yb-failover-tracker-table-tablets", 27),
])
def test_tracker_table_tablets_accepts_all_key_forms(key_form, expected_value):
    conninfo = f"host=h1 {key_form}={expected_value}"
    yb, _, _ = extract_yb_params(conninfo, {})
    assert yb.tracker_table_tablets == expected_value


@pytest.mark.parametrize("key_form", [
    "yb.failover.maxUpdateFailuresAllowed",
    "yb_failover_max_update_failures_allowed",
    "yb-failover-max-update-failures-allowed",
])
def test_max_update_failures_allowed_accepts_all_key_forms(key_form):
    conninfo = f"host=h1 {key_form}=5"
    yb, _, _ = extract_yb_params(conninfo, {})
    assert yb.max_update_failures_allowed == 5


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


def test_all_four_xcluster_kwargs_via_spread():
    yb, _, cleaned_kwargs = extract_yb_params(
        "host=h1 load_balance_hosts=true",
        {
            "yb.failover.secondaryClusterHosts": "s1,s2",
            "yb.failover.trackerTableTablets": "42",
            "yb.failover.maxUpdateFailuresAllowed": "3",
            "yb.failover.cooldownSecs": "600",
        },
    )
    assert yb.secondary_cluster_hosts == ["s1", "s2"]
    assert yb.tracker_table_tablets == 42
    assert yb.max_update_failures_allowed == 3
    assert yb.cooldown_s == 600
    # None of them leaked into the cleaned kwargs.
    assert cleaned_kwargs == {}


# ----------------------------------------------------------------- clamps

def test_tracker_table_tablets_clamps_to_minimum_1():
    """Zero / negative tablet count is meaningless; clamp to 1."""
    yb, _, _ = extract_yb_params(
        "host=h1 yb.failover.trackerTableTablets=0", {}
    )
    assert yb.tracker_table_tablets == 1

    yb, _, _ = extract_yb_params(
        "host=h1 yb.failover.trackerTableTablets=-5", {}
    )
    assert yb.tracker_table_tablets == 1


def test_max_update_failures_allowed_clamps_to_minimum_0():
    yb, _, _ = extract_yb_params(
        "host=h1 yb.failover.maxUpdateFailuresAllowed=-1", {}
    )
    assert yb.max_update_failures_allowed == 0


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
        "yb.failover.trackerTableTablets=9 "
        "yb.failover.maxUpdateFailuresAllowed=0 "
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
    assert yb.tracker_table_tablets == 9
    assert yb.max_update_failures_allowed == 0
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
