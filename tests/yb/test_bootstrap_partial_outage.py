"""
Integration-level repro test for the bootstrap-partial-outage fix in PR-H5.

Shubham reported on #proj-smartdriver-xclusterfailover (Oct 23 2026):

  "With every yb-tserver and yb-master process stopped on universe A,
  `bootstrap_failover_group(dsn)` raised OperationalError from A's hosts
  and the application never started. With B stopped instead, A
  bootstrapped and the same call raised from B's hosts 3 ms later."

Root cause: `registry.get_or_bootstrap_failover_group` called
`get_or_bootstrap` for primary (line 716) and secondary (line 734)
without wrapping either; one unreachable universe aborts the whole
bootstrap. The feature exists to survive an outage; failing bootstrap
during an outage defeats its own purpose.

Fix (this PR): tolerate partial outage. If one side is unreachable at
bootstrap, build the FailoverGroup with that side's status set to
UNHEALTHY and a placeholder ClusterState. The probe thread's next tick
will reconnect it when the cluster is back. If BOTH sides are
unreachable → raise `_PartialOutageBootstrapError` (nothing to route
to).

These tests mirror Shubham's cycle 9 (A down) and cycle 10 (B down).
We use a mock bootstrap path that raises OperationalError the way
`Connection._connect_plain` would on a real "server closed the
connection unexpectedly" error.
"""

# Copyright (C) 2026 Yugabyte

from __future__ import annotations

import threading
from unittest.mock import patch

import pytest

from psycopg import errors as e
from psycopg.yb.health import HealthResult
from psycopg.yb.registry import (
    ClusterRegistry,
    ClusterState,
    _PartialOutageBootstrapError,
    _is_placeholder_uuid,
    _make_placeholder_state,
    _PLACEHOLDER_UUID_PREFIX,
)
from psycopg.yb.state import active_cluster
from psycopg.yb.params import extract_yb_params


pytestmark = pytest.mark.yb_unit


# ===================================================================
# Helpers
# ===================================================================

def _yb_params_with_secondary(secondary_hosts: list[str]):
    """Build the YBParams the registry's bootstrap methods expect, with
    the given secondary host list."""
    dsn = (
        f"host=p1,p2,p3 port=5433 dbname=yugabyte user=yugabyte "
        f"load_balance_hosts=true "
        f"yb.failover.secondaryClusterHosts={','.join(secondary_hosts)} "
    )
    params, cleaned_conninfo, cleaned_kwargs = extract_yb_params(dsn, {})
    return params, cleaned_conninfo, cleaned_kwargs


def _make_real_state(uuid: str, host: str, fake_state):
    """Build a real (non-placeholder) ClusterState for one side."""
    return fake_state(
        (host, "aws", "us-west", "us-west-1a", "primary"),
        uuid=uuid,
    )


# ===================================================================
# Direct unit tests on the placeholder construction
# ===================================================================

def test_placeholder_state_has_pending_uuid_prefix():
    """Placeholder uuids are prefixed with `pending:` so later code can
    tell them apart from real cluster uuids."""
    state = _make_placeholder_state(
        conninfo="",
        kwargs={"port": 5433},
        hosts=["host1", "host2"],
        tag="primary",
    )
    assert state.uuid.startswith(_PLACEHOLDER_UUID_PREFIX)
    assert _is_placeholder_uuid(state.uuid)


def test_placeholder_state_is_deterministic():
    """Same host list + tag → same placeholder uuid. Lets the registry
    deduplicate repeated bootstraps against the same down cluster."""
    state1 = _make_placeholder_state("", {}, ["a", "b"], "primary")
    state2 = _make_placeholder_state("", {}, ["b", "a"], "primary")
    state3 = _make_placeholder_state("", {}, ["a", "b"], "secondary")
    assert state1.uuid == state2.uuid, "same hosts (any order) → same uuid"
    assert state1.uuid != state3.uuid, "different tag → different uuid"


def test_placeholder_state_seeds_nodes_from_hosts():
    """Probe thread needs the host list to retry; placeholder populates
    NodeInfo with the DSN hosts so `_ensure_control_sync` has candidates."""
    state = _make_placeholder_state(
        conninfo="",
        kwargs={"port": 5433},
        hosts=["h1", "h2", "h3"],
        tag="primary",
    )
    assert set(state.nodes.keys()) == {"h1", "h2", "h3"}
    for node in state.nodes.values():
        assert node.is_down is True
        assert node.port == 5433


def test_placeholder_state_remembers_bootstrap_params():
    """When the cluster comes back, the probe retries using
    bootstrap_conninfo/kwargs. Placeholder must preserve them."""
    state = _make_placeholder_state(
        conninfo="host=a,b dbname=yb",
        kwargs={"host": "a,b", "port": 5433, "dbname": "yb"},
        hosts=["a", "b"],
        tag="primary",
    )
    assert state.bootstrap_conninfo == "host=a,b dbname=yb"
    assert state.bootstrap_kwargs["port"] == 5433
    assert state.bootstrap_kwargs["dbname"] == "yb"


# ===================================================================
# Bootstrap-time partial-outage behavior (sync)
# ===================================================================

def test_db25xxx_primary_down_bootstrap_succeeds_with_placeholder(
    fresh_registry, fake_state,
):
    """Shubham's cycle 9: primary unreachable at bootstrap. App should
    still come up on the secondary. Under the old code `bootstrap_
    failover_group` raised OperationalError and the app died."""
    params, conninfo, kwargs = _yb_params_with_secondary(["s1", "s2"])

    real_secondary = _make_real_state("SECONDARY-B", "s1", fake_state)

    def fake_get_or_bootstrap(key, conninfo, kw):
        # Hosts can be in kwargs["host"] (secondary call sets it via
        # _make_secondary_kwargs) OR embedded in the conninfo (primary
        # call keeps kwargs empty).
        host_src = str(kw.get("host") or "") + " " + conninfo
        if "s1" in host_src or "s2" in host_src:
            return real_secondary
        raise e.OperationalError(
            'connection failed: connection to server at "p1", port 5433 '
            'failed: server closed the connection unexpectedly'
        )

    with patch.object(
        ClusterRegistry, "get_or_bootstrap",
        side_effect=fake_get_or_bootstrap,
    ):
        group = ClusterRegistry.instance().get_or_bootstrap_failover_group(
            params, conninfo, kwargs,
        )

    # Group was constructed, not raised.
    assert group is not None
    # Primary is a placeholder marked UNHEALTHY.
    assert _is_placeholder_uuid(group.primary.uuid), (
        f"primary should be placeholder, got {group.primary.uuid}"
    )
    assert group.primary_status is HealthResult.UNHEALTHY
    # Secondary is real + HEALTHY.
    assert group.secondary.uuid == "SECONDARY-B"
    assert group.secondary_status is HealthResult.HEALTHY
    # Routing goes to the healthy side.
    assert active_cluster(group) == "secondary", (
        "app should route to secondary when primary is down at bootstrap"
    )


def test_db25xxx_secondary_down_bootstrap_succeeds_with_placeholder(
    fresh_registry, fake_state,
):
    """Shubham's cycle 10: secondary unreachable, primary succeeds. App
    comes up on primary; secondary marked UNHEALTHY (no failover target
    for now, until the secondary comes back)."""
    params, conninfo, kwargs = _yb_params_with_secondary(["s1", "s2"])

    real_primary = _make_real_state("PRIMARY-A", "p1", fake_state)

    def fake_get_or_bootstrap(key, conninfo, kw):
        # Secondary call sets kw["host"] via _make_secondary_kwargs;
        # primary call keeps the hosts in conninfo.
        if "s1" in (kw.get("host") or "") or "s2" in (kw.get("host") or ""):
            raise e.OperationalError(
                'connection failed: connection to server at "s1", port 5433 '
                'failed: server closed the connection unexpectedly'
            )
        return real_primary

    with patch.object(
        ClusterRegistry, "get_or_bootstrap",
        side_effect=fake_get_or_bootstrap,
    ):
        group = ClusterRegistry.instance().get_or_bootstrap_failover_group(
            params, conninfo, kwargs,
        )

    assert group.primary.uuid == "PRIMARY-A"
    assert group.primary_status is HealthResult.HEALTHY
    assert _is_placeholder_uuid(group.secondary.uuid)
    assert group.secondary_status is HealthResult.UNHEALTHY
    assert active_cluster(group) == "primary"


def test_db25xxx_both_down_bootstrap_raises(
    fresh_registry,
):
    """BOTH unreachable → raise. The feature exists to tolerate ONE
    outage, not nothing-is-up."""
    params, conninfo, kwargs = _yb_params_with_secondary(["s1", "s2"])

    def fake_get_or_bootstrap(key, conninfo, kw):
        raise e.OperationalError("connection failed: everything is down")

    with patch.object(
        ClusterRegistry, "get_or_bootstrap",
        side_effect=fake_get_or_bootstrap,
    ):
        with pytest.raises(_PartialOutageBootstrapError) as excinfo:
            ClusterRegistry.instance().get_or_bootstrap_failover_group(
                params, conninfo, kwargs,
            )

    # Error carries both underlying exceptions so the operator can
    # distinguish "primary down" from "secondary down".
    assert excinfo.value.primary_exc is not None
    assert excinfo.value.secondary_exc is not None
    assert isinstance(excinfo.value.primary_exc, e.OperationalError)
    assert isinstance(excinfo.value.secondary_exc, e.OperationalError)


def test_db25xxx_both_healthy_still_works_regression(
    fresh_registry, fake_state,
):
    """Regression: when both sides come up normally, we must still
    produce the pre-fix shape (both real, both HEALTHY)."""
    params, conninfo, kwargs = _yb_params_with_secondary(["s1", "s2"])

    real_primary = _make_real_state("PRIMARY-A", "p1", fake_state)
    real_secondary = _make_real_state("SECONDARY-B", "s1", fake_state)

    def fake_get_or_bootstrap(key, conninfo, kw):
        host = kw.get("host") or ""
        if "s1" in host or "s2" in host:
            return real_secondary
        return real_primary

    with patch.object(
        ClusterRegistry, "get_or_bootstrap",
        side_effect=fake_get_or_bootstrap,
    ):
        group = ClusterRegistry.instance().get_or_bootstrap_failover_group(
            params, conninfo, kwargs,
        )

    assert group.primary.uuid == "PRIMARY-A"
    assert group.secondary.uuid == "SECONDARY-B"
    assert group.primary_status is HealthResult.HEALTHY
    assert group.secondary_status is HealthResult.HEALTHY
    assert not _is_placeholder_uuid(group.primary.uuid)
    assert not _is_placeholder_uuid(group.secondary.uuid)


# ===================================================================
# Async path mirrors the sync behavior
# ===================================================================

@pytest.mark.anyio
async def test_db25xxx_primary_down_async_bootstrap_succeeds_with_placeholder(
    fresh_registry, fake_state,
):
    params, conninfo, kwargs = _yb_params_with_secondary(["s1", "s2"])
    real_secondary = _make_real_state("SECONDARY-B", "s1", fake_state)

    async def fake_aget_or_bootstrap(key, conninfo, kw):
        host_src = str(kw.get("host") or "") + " " + conninfo
        if "s1" in host_src or "s2" in host_src:
            return real_secondary
        raise e.OperationalError("primary unreachable")

    with patch.object(
        ClusterRegistry, "aget_or_bootstrap",
        side_effect=fake_aget_or_bootstrap,
    ):
        group = await ClusterRegistry.instance().aget_or_bootstrap_failover_group(
            params, conninfo, kwargs,
        )

    assert _is_placeholder_uuid(group.primary.uuid)
    assert group.primary_status is HealthResult.UNHEALTHY
    assert group.secondary.uuid == "SECONDARY-B"
    assert group.secondary_status is HealthResult.HEALTHY
    assert active_cluster(group) == "secondary"


@pytest.mark.anyio
async def test_db25xxx_both_down_async_bootstrap_raises(
    fresh_registry,
):
    params, conninfo, kwargs = _yb_params_with_secondary(["s1", "s2"])

    async def fake_aget_or_bootstrap(key, conninfo, kw):
        raise e.OperationalError("down")

    with patch.object(
        ClusterRegistry, "aget_or_bootstrap",
        side_effect=fake_aget_or_bootstrap,
    ):
        with pytest.raises(_PartialOutageBootstrapError):
            await ClusterRegistry.instance().aget_or_bootstrap_failover_group(
                params, conninfo, kwargs,
            )


# ===================================================================
# Operator-visible verification: app can connect after partial-outage bootstrap
# ===================================================================

def test_db25xxx_app_can_look_up_group_by_secondary_uuid_when_primary_is_placeholder(
    fresh_registry, fake_state,
):
    """After partial-outage bootstrap, the group is registered under the
    PLACEHOLDER primary uuid in `_failover_groups`. Pool conns tagged with
    the real secondary uuid use `get_failover_group_by_uuid` which walks
    all groups searching either side — this must still find our group."""
    params, conninfo, kwargs = _yb_params_with_secondary(["s1", "s2"])
    real_secondary = _make_real_state("SECONDARY-B", "s1", fake_state)

    def fake_get_or_bootstrap(key, cfo, kw):
        # agent-K PR #17:346. Hosts live in the conninfo STRING (``p1,
        # p2, p3``), not kwargs, so ``kw.get("host")`` is empty on
        # BOTH primary and secondary calls. The pre-fix test checked
        # kw["host"], matched neither, and both calls returned
        # real_secondary → the group was built with primary=secondary
        # and primary.uuid=="SECONDARY-B". The lookup trivially
        # succeeded while the placeholder path was never exercised.
        # Dispatch on whether the kwargs' secondary override is set
        # (set only for the secondary call via _make_secondary_kwargs).
        is_secondary_call = any(
            h in str(kw.get("host") or cfo) for h in ("s1", "s2")
        )
        if not is_secondary_call:
            raise e.OperationalError("primary down")
        return real_secondary

    with patch.object(
        ClusterRegistry, "get_or_bootstrap",
        side_effect=fake_get_or_bootstrap,
    ):
        group = ClusterRegistry.instance().get_or_bootstrap_failover_group(
            params, conninfo, kwargs,
        )

    # Confirm the primary IS a placeholder — the whole point of the
    # test. Pre-fix this assertion was missing and the group had a real
    # SECONDARY uuid on both sides.
    assert _is_placeholder_uuid(group.primary.uuid), (
        f"primary should be placeholder but got {group.primary.uuid!r}"
    )
    # Lookup by secondary uuid finds the group — critical for pool
    # xcluster_check which identifies conns by their uuid.
    found = ClusterRegistry.instance().get_failover_group_by_uuid("SECONDARY-B")
    assert found is group
