"""Unit tests for the public bootstrap helpers.

``psycopg.yb.bootstrap_failover_group`` and its async sibling
``abootstrap_failover_group`` glue ``extract_yb_params`` +
``ClusterRegistry.(a)get_or_bootstrap_failover_group``. The lower-level
plumbing is tested elsewhere (test_params_failover.py,
test_registry_failover.py). These tests cover only what the wrapper
itself contributes: DSN passthrough, the xcluster-enabled guard, and
the return type.

Auto-tagged ``yb_unit`` via the filename rule in conftest.
"""

# Copyright (C) 2026 Yugabyte

from __future__ import annotations

import pytest

from psycopg.yb import abootstrap_failover_group, bootstrap_failover_group
from psycopg.yb.registry import ClusterRegistry, FailoverGroup


# This file doesn't match any of the auto-marker prefixes in conftest, so
# tag every test in it explicitly. Unit tests: no cluster required.
pytestmark = pytest.mark.yb_unit


# --------------------------------------------------------------------- sync


def test_sync_returns_failover_group(fresh_registry, fake_state, monkeypatch):
    """Happy path: DSN opts into xCluster, helper returns a FailoverGroup
    from the same registry instance."""
    p = fake_state(("p1", "aws", "us-west", "us-west-1a", "primary"), uuid="P")
    s = fake_state(("s1", "aws", "us-east", "us-east-1a", "primary"), uuid="S")

    def fake_bootstrap(key, conninfo, kwargs):
        return s if "s1" in str(kwargs.get("host", "")) else p

    monkeypatch.setattr(fresh_registry, "get_or_bootstrap", fake_bootstrap)
    monkeypatch.setattr(
        fresh_registry, "_param_dict_for_key",
        lambda conninfo, kwargs: {"host": str(kwargs.get("host", ""))},
    )
    monkeypatch.setattr(fresh_registry, "_start_probe", lambda *a, **k: None)

    dsn = (
        "host=p1 dbname=yugabyte user=yugabyte "
        "load_balance_hosts=true "
        "yb.failover.secondaryClusterHosts=s1"
    )
    group = bootstrap_failover_group(dsn)

    assert isinstance(group, FailoverGroup)
    assert group.primary.uuid == "P"
    assert group.secondary.uuid == "S"


def test_sync_raises_when_dsn_lacks_secondary_hosts(fresh_registry):
    """The `yb.failover.secondaryClusterHosts` param is required — if the
    DSN opts out of xCluster, the helper refuses to bootstrap."""
    with pytest.raises(ValueError, match="xCluster DSN"):
        bootstrap_failover_group(
            "host=p1 dbname=yugabyte user=yugabyte load_balance_hosts=true"
        )


def test_sync_raises_when_load_balance_hosts_disabled(fresh_registry):
    """Even with secondary hosts, if load_balance_hosts is disabled the
    smart driver is off and xCluster is off with it."""
    with pytest.raises(ValueError, match="xCluster DSN"):
        bootstrap_failover_group(
            "host=p1 dbname=yugabyte user=yugabyte "
            "load_balance_hosts=disable "
            "yb.failover.secondaryClusterHosts=s1"
        )


def test_sync_accepts_kwargs_style_dsn(fresh_registry, fake_state, monkeypatch):
    """The helper mirrors psycopg.connect's calling shape — kwargs merge
    over the conninfo string. Same rule as libpq."""
    p = fake_state(("p1", "aws", "us-west", "us-west-1a", "primary"), uuid="P")
    s = fake_state(("s1", "aws", "us-east", "us-east-1a", "primary"), uuid="S")

    def fake_bootstrap(key, conninfo, kwargs):
        return s if "s1" in str(kwargs.get("host", "")) else p

    monkeypatch.setattr(fresh_registry, "get_or_bootstrap", fake_bootstrap)
    monkeypatch.setattr(
        fresh_registry, "_param_dict_for_key",
        lambda conninfo, kwargs: {"host": str(kwargs.get("host", ""))},
    )
    monkeypatch.setattr(fresh_registry, "_start_probe", lambda *a, **k: None)

    group = bootstrap_failover_group(
        host="p1",
        dbname="yugabyte",
        user="yugabyte",
        load_balance_hosts="true",
        **{"yb.failover.secondaryClusterHosts": "s1"},
    )
    assert group.primary.uuid == "P"


def test_sync_result_matches_subsequent_registry_lookup(
    fresh_registry, fake_state, monkeypatch,
):
    """The point of the helper: after bootstrap, a subsequent registry
    lookup by primary uuid returns the SAME group. So any custom CB
    attached to the returned group is what future connections see."""
    p = fake_state(("p1", "aws", "us-west", "us-west-1a", "primary"), uuid="P")
    s = fake_state(("s1", "aws", "us-east", "us-east-1a", "primary"), uuid="S")

    def fake_bootstrap(key, conninfo, kwargs):
        return s if "s1" in str(kwargs.get("host", "")) else p

    monkeypatch.setattr(fresh_registry, "get_or_bootstrap", fake_bootstrap)
    monkeypatch.setattr(
        fresh_registry, "_param_dict_for_key",
        lambda conninfo, kwargs: {"host": str(kwargs.get("host", ""))},
    )
    monkeypatch.setattr(fresh_registry, "_start_probe", lambda *a, **k: None)

    dsn = (
        "host=p1 dbname=yugabyte user=yugabyte "
        "load_balance_hosts=true "
        "yb.failover.secondaryClusterHosts=s1"
    )
    group1 = bootstrap_failover_group(dsn)
    group2 = ClusterRegistry.instance().get_failover_group_by_uuid(
        group1.primary.uuid
    )
    assert group1 is group2


# --------------------------------------------------------------------- async


@pytest.mark.anyio
async def test_async_returns_failover_group(
    fresh_registry, fake_state, monkeypatch,
):
    p = fake_state(("p1", "aws", "us-west", "us-west-1a", "primary"), uuid="P")
    s = fake_state(("s1", "aws", "us-east", "us-east-1a", "primary"), uuid="S")

    async def fake_abootstrap(key, conninfo, kwargs):
        return s if "s1" in str(kwargs.get("host", "")) else p

    monkeypatch.setattr(fresh_registry, "aget_or_bootstrap", fake_abootstrap)
    monkeypatch.setattr(
        fresh_registry, "_param_dict_for_key",
        lambda conninfo, kwargs: {"host": str(kwargs.get("host", ""))},
    )
    monkeypatch.setattr(fresh_registry, "_start_probe", lambda *a, **k: None)

    dsn = (
        "host=p1 dbname=yugabyte user=yugabyte "
        "load_balance_hosts=true "
        "yb.failover.secondaryClusterHosts=s1"
    )
    group = await abootstrap_failover_group(dsn)
    assert isinstance(group, FailoverGroup)
    assert group.primary.uuid == "P"


@pytest.mark.anyio
async def test_async_raises_when_dsn_lacks_secondary_hosts(fresh_registry):
    with pytest.raises(ValueError, match="xCluster DSN"):
        await abootstrap_failover_group(
            "host=p1 dbname=yugabyte user=yugabyte load_balance_hosts=true"
        )
