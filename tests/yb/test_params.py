"""
Unit tests for `psycopg.yb.params.extract_yb_params`.

These tests are pure-Python (no DB). They lock in the conninfo-parsing
contract documented in the design doc and the plan: which keys we strip
before libpq sees them, which values we pass through, how clamps work, etc.
"""

# Copyright (C) 2026 Yugabyte

from __future__ import annotations

import pytest

from psycopg.yb.node import Placement
from psycopg.yb.params import (
    DEFAULT_FAILED_HOST_RECONNECT_DELAY_SEC,
    DEFAULT_REFRESH_INTERVAL_SEC,
    MAX_FAILED_HOST_RECONNECT_DELAY_SEC,
    MAX_REFRESH_INTERVAL_SEC,
    extract_yb_params,
)


# --------------------------------------------------------------------- smart_driver_enabled


def test_smart_driver_enabled_true():
    yb, _, _ = extract_yb_params("load_balance_hosts=true", {})
    assert yb.smart_driver_enabled is True


def test_smart_driver_enabled_false():
    yb, _, _ = extract_yb_params("load_balance_hosts=false", {})
    assert yb.smart_driver_enabled is False


def test_smart_driver_default_off_when_absent():
    yb, _, _ = extract_yb_params("host=h1 dbname=foo", {})
    assert yb.smart_driver_enabled is False


def test_libpq_value_disable_passes_through():
    """`load_balance_hosts=disable` must remain in the conninfo string so libpq
    handles it (existing PG16+ behaviour). Smart driver stays off."""
    yb, cleaned, _ = extract_yb_params("host=h1 load_balance_hosts=disable", {})
    assert yb.smart_driver_enabled is False
    assert "load_balance_hosts=disable" in cleaned


def test_libpq_value_random_passes_through():
    yb, cleaned, _ = extract_yb_params("host=h1 load_balance_hosts=random", {})
    assert yb.smart_driver_enabled is False
    assert "load_balance_hosts=random" in cleaned


def test_our_true_stripped_from_cleaned():
    """`load_balance_hosts=true` is *our* extension and must not reach libpq."""
    _, cleaned, _ = extract_yb_params("host=h1 load_balance_hosts=true", {})
    assert "load_balance_hosts" not in cleaned
    assert "true" not in cleaned.split()  # not a free-floating word either


def test_our_false_stripped_from_cleaned():
    _, cleaned, _ = extract_yb_params("host=h1 load_balance_hosts=false", {})
    assert "load_balance_hosts" not in cleaned


# --------------------------------------------------------------------- kwargs paths


def test_kwargs_take_precedence_over_conninfo():
    yb, _, _ = extract_yb_params(
        "load_balance_hosts=false",
        {"load_balance_hosts": "true"},
    )
    assert yb.smart_driver_enabled is True


def test_kwargs_keys_removed():
    yb, _, kw = extract_yb_params(
        "host=h1",
        {
            "load_balance_hosts": "true",
            "topology_keys": "aws.us-west.us-west-1a",
            "yb_servers_refresh_interval": "60",
        },
    )
    assert "load_balance_hosts" not in kw
    assert "topology_keys" not in kw
    assert "yb_servers_refresh_interval" not in kw


def test_kwargs_libpq_value_passes_through():
    """`load_balance_hosts=random` in kwargs is a libpq value; keep it in cleaned_kwargs."""
    _, _, kw = extract_yb_params("host=h1", {"load_balance_hosts": "random"})
    assert kw.get("load_balance_hosts") == "random"


def test_unrelated_kwargs_preserved():
    _, _, kw = extract_yb_params(
        "host=h1",
        {"load_balance_hosts": "true", "user": "alice", "dbname": "db1"},
    )
    assert kw["user"] == "alice"
    assert kw["dbname"] == "db1"


# --------------------------------------------------------------------- topology_keys


def test_topology_keys_single_entry():
    yb, _, _ = extract_yb_params(
        "load_balance_hosts=true topology_keys=aws.us-west.us-west-1a",
        {},
    )
    assert yb.topology_keys == [Placement("aws", "us-west", "us-west-1a")]


def test_topology_keys_multiple_entries():
    yb, _, _ = extract_yb_params(
        "load_balance_hosts=true topology_keys=aws.us-west.1a,aws.us-east.2a",
        {},
    )
    assert yb.topology_keys == [
        Placement("aws", "us-west", "1a"),
        Placement("aws", "us-east", "2a"),
    ]


def test_topology_keys_zone_wildcard_allowed():
    yb, _, _ = extract_yb_params(
        "load_balance_hosts=true topology_keys=aws.us-west.*",
        {},
    )
    assert yb.topology_keys == [Placement("aws", "us-west", "*")]


def test_topology_keys_cloud_wildcard_rejected():
    with pytest.raises(ValueError, match="cloud or region"):
        extract_yb_params(
            "load_balance_hosts=true topology_keys=*.us-west.1a",
            {},
        )


def test_topology_keys_region_wildcard_rejected():
    with pytest.raises(ValueError, match="cloud or region"):
        extract_yb_params(
            "load_balance_hosts=true topology_keys=aws.*.1a",
            {},
        )


def test_topology_keys_too_few_parts_rejected():
    with pytest.raises(ValueError, match="cloud.region.zone"):
        extract_yb_params(
            "load_balance_hosts=true topology_keys=aws.uswest",
            {},
        )


def test_topology_keys_empty_when_unset():
    yb, _, _ = extract_yb_params("load_balance_hosts=true", {})
    assert yb.topology_keys == []


def test_topology_keys_stripped_from_cleaned():
    _, cleaned, _ = extract_yb_params(
        "host=h1 load_balance_hosts=true topology_keys=aws.us-west.1a",
        {},
    )
    assert "topology_keys" not in cleaned


# --------------------------------------------------------------------- refresh interval


def test_refresh_interval_default():
    yb, _, _ = extract_yb_params("load_balance_hosts=true", {})
    assert yb.refresh_interval_s == DEFAULT_REFRESH_INTERVAL_SEC


def test_refresh_interval_custom():
    yb, _, _ = extract_yb_params(
        "load_balance_hosts=true yb_servers_refresh_interval=120",
        {},
    )
    assert yb.refresh_interval_s == 120


def test_refresh_interval_clamped_at_max():
    yb, _, _ = extract_yb_params(
        "load_balance_hosts=true yb_servers_refresh_interval=9999",
        {},
    )
    assert yb.refresh_interval_s == MAX_REFRESH_INTERVAL_SEC


def test_refresh_interval_zero_allowed():
    """Setting refresh_interval=0 means 'always refresh' — used by some tests."""
    yb, _, _ = extract_yb_params(
        "load_balance_hosts=true yb_servers_refresh_interval=0",
        {},
    )
    assert yb.refresh_interval_s == 0


# --------------------------------------------------------------------- reconnect delay


def test_reconnect_delay_default():
    yb, _, _ = extract_yb_params("load_balance_hosts=true", {})
    assert yb.failed_host_reconnect_delay_s == DEFAULT_FAILED_HOST_RECONNECT_DELAY_SEC


def test_reconnect_delay_custom():
    yb, _, _ = extract_yb_params(
        "load_balance_hosts=true failed_host_reconnect_delay_secs=20",
        {},
    )
    assert yb.failed_host_reconnect_delay_s == 20


def test_reconnect_delay_clamped_at_max():
    yb, _, _ = extract_yb_params(
        "load_balance_hosts=true failed_host_reconnect_delay_secs=999",
        {},
    )
    assert yb.failed_host_reconnect_delay_s == MAX_FAILED_HOST_RECONNECT_DELAY_SEC


# --------------------------------------------------------------------- naming forms


def test_dashed_key_form_accepted():
    yb, _, _ = extract_yb_params("load-balance-hosts=true", {})
    assert yb.smart_driver_enabled is True


def test_dashed_topology_form_accepted():
    yb, _, _ = extract_yb_params(
        "load-balance-hosts=true topology-keys=aws.us-west.1a",
        {},
    )
    assert yb.topology_keys == [Placement("aws", "us-west", "1a")]


def test_case_insensitive_key():
    yb, _, _ = extract_yb_params("Load_Balance_Hosts=true", {})
    assert yb.smart_driver_enabled is True


# --------------------------------------------------------------------- value forms


def test_quoted_value_single_quotes():
    yb, _, _ = extract_yb_params(
        "load_balance_hosts=true topology_keys='aws.us-west.1a'",
        {},
    )
    assert yb.topology_keys == [Placement("aws", "us-west", "1a")]


def test_quoted_value_double_quotes():
    yb, _, _ = extract_yb_params(
        'load_balance_hosts=true topology_keys="aws.us-west.1a"',
        {},
    )
    assert yb.topology_keys == [Placement("aws", "us-west", "1a")]


def test_other_libpq_keys_left_alone():
    """Non-YB conninfo content (host, dbname, etc.) must not be touched."""
    _, cleaned, _ = extract_yb_params(
        "host=h1,h2,h3 port=5433 dbname=yugabyte user=alice load_balance_hosts=true",
        {},
    )
    assert "host=h1,h2,h3" in cleaned
    assert "port=5433" in cleaned
    assert "dbname=yugabyte" in cleaned
    assert "user=alice" in cleaned
