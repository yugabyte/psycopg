"""
Unit tests for the dual-CB state machine (Phase A of the xCluster
failover implementation plan).

Verifies:

  * ``psycopg.yb.state.active_cluster(group)`` — the routing decision from
    (primary_status, secondary_status). Design doc §4 state-machine table.
  * ``psycopg.yb.pool.xcluster_check`` — raises ``NoViableClusterError``
    when both CBs report UNHEALTHY.
  * ``NoViableClusterError`` — subclass of ``OperationalError`` so
    generic-retry code still catches it.

All metadata-only — no real DB.
"""

# Copyright (C) 2026 Yugabyte

from __future__ import annotations

import threading

import pytest

import psycopg
from psycopg.yb import NoViableClusterError
from psycopg.yb.health import HealthResult
from psycopg.yb.pool import xcluster_check
from psycopg.yb.registry import FailoverGroup
from psycopg.yb.state import active_cluster, active_state


pytestmark = pytest.mark.yb_unit


def _make_group(fake_state, primary_status, secondary_status):
    p = fake_state(("p1", "aws", "us-west", "us-west-1a", "primary"), uuid="P")
    s = fake_state(("s1", "aws", "us-east", "us-east-1a", "primary"), uuid="S")
    return FailoverGroup(
        primary=p,
        secondary=s,
        lock=threading.Lock(),
        primary_status=primary_status,
        secondary_status=secondary_status,
    )


# --------------------------------------------------------- state machine table

@pytest.mark.parametrize(
    "primary,secondary,expected",
    [
        (HealthResult.HEALTHY, HealthResult.HEALTHY, "primary"),
        (HealthResult.HEALTHY, HealthResult.UNHEALTHY, "primary"),
        (HealthResult.UNHEALTHY, HealthResult.HEALTHY, "secondary"),
        (HealthResult.UNHEALTHY, HealthResult.UNHEALTHY, None),
    ],
)
def test_active_cluster_state_machine(fake_state, primary, secondary, expected):
    """Design doc §4 state-machine row-by-row."""
    group = _make_group(fake_state, primary, secondary)
    assert active_cluster(group) == expected


@pytest.mark.parametrize(
    "primary,secondary,expected_uuid",
    [
        (HealthResult.HEALTHY, HealthResult.HEALTHY, "P"),
        (HealthResult.HEALTHY, HealthResult.UNHEALTHY, "P"),
        (HealthResult.UNHEALTHY, HealthResult.HEALTHY, "S"),
    ],
)
def test_active_state_returns_matching_cluster_state(
    fake_state, primary, secondary, expected_uuid,
):
    group = _make_group(fake_state, primary, secondary)
    state = active_state(group)
    assert state is not None
    assert state.uuid == expected_uuid


def test_active_state_is_none_when_no_viable(fake_state):
    group = _make_group(
        fake_state, HealthResult.UNHEALTHY, HealthResult.UNHEALTHY
    )
    assert active_state(group) is None


# --------------------------------------------------------- NoViableClusterError

def test_no_viable_is_subclass_of_operational_error():
    """Generic ``except OperationalError`` handlers must still catch the
    no-viable-cluster case — otherwise applications relying on standard
    retry patterns would silently break."""
    assert issubclass(NoViableClusterError, psycopg.OperationalError)


def test_pool_check_raises_no_viable_when_both_unhealthy(
    fresh_registry, fake_state,
):
    """When both CBs report UNHEALTHY, ``xcluster_check`` raises the
    specific ``NoViableClusterError`` so callers can distinguish
    "route to the other cluster" from "there is no other cluster"."""
    group = _make_group(
        fake_state, HealthResult.UNHEALTHY, HealthResult.UNHEALTHY,
    )
    fresh_registry._clusters[group.primary.uuid] = group.primary
    fresh_registry._clusters[group.secondary.uuid] = group.secondary
    fresh_registry._failover_groups[group.primary.uuid] = group

    class _Conn:
        _yb_uuid = "P"

        def close(self):
            pass

    with pytest.raises(NoViableClusterError):
        xcluster_check(_Conn())
