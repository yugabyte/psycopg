"""
Unit tests for `psycopg.yb.policy`.

These tests run against synthetic `ClusterState` objects (built via the
`fake_state` fixture). No DB, no I/O. The goal is to lock in the
selection contract:

  * least-loaded with random tie-break (cluster-aware)
  * placement filter applied first, same picker after (topology-aware)
  * down-node TTL semantics
  * attempted-set respected
  * returns None when no eligible node remains
  * single-pass (one iteration over the node dict per call)
"""

# Copyright (C) 2026 Yugabyte

from __future__ import annotations

import random
import time

import pytest

from psycopg.yb.node import Placement
from psycopg.yb.params import YBParams
from psycopg.yb.policy import (
    ClusterAwarePolicy,
    TopologyAwarePolicy,
    build_policy,
)


TTL = 5.0


# --------------------------------------------------------------------- ClusterAwarePolicy: basic


def test_picks_unique_min(fake_state):
    state = fake_state(
        ("a", "aws", "r", "z1", "primary", 5),
        ("b", "aws", "r", "z2", "primary", 2),
        ("c", "aws", "r", "z3", "primary", 3),
    )
    got = ClusterAwarePolicy().get_least_loaded_server(state, set(), TTL)
    assert got.host == "b"


def test_random_tie_break_distribution(fake_state):
    """When all eligible nodes are tied, the picker chooses uniformly at
    random. Test statistically: 3000 picks across 3 tied nodes should be
    within ~5% of perfectly even."""
    state = fake_state(
        ("a", "aws", "r", "z1", "primary", 2),
        ("b", "aws", "r", "z2", "primary", 2),
        ("c", "aws", "r", "z3", "primary", 2),
    )
    policy = ClusterAwarePolicy()
    random.seed(0xC0DED)

    counts = {"a": 0, "b": 0, "c": 0}
    for _ in range(3000):
        got = policy.get_least_loaded_server(state, set(), TTL)
        counts[got.host] += 1

    for host, n in counts.items():
        assert 900 <= n <= 1100, f"{host} skewed: {n}/3000"


def test_skips_attempted(fake_state):
    state = fake_state(
        ("a", "aws", "r", "z1", "primary", 0),
        ("b", "aws", "r", "z2", "primary", 0),
        ("c", "aws", "r", "z3", "primary", 1),  # unique min after a/b excluded
    )
    got = ClusterAwarePolicy().get_least_loaded_server(
        state, attempted={"a", "b"}, ttl=TTL
    )
    # Wait — with a/b excluded, c is the only eligible node, so the result is c.
    # But a and b would have been preferred otherwise. This confirms attempted works.
    assert got.host == "c"


def test_returns_none_when_no_eligible(fake_state):
    state = fake_state(
        ("a", "aws", "r", "z1", "primary", 0),
        ("b", "aws", "r", "z2", "primary", 0),
    )
    got = ClusterAwarePolicy().get_least_loaded_server(
        state, attempted={"a", "b"}, ttl=TTL
    )
    assert got is None


def test_returns_none_when_state_has_no_nodes(fake_state):
    state = fake_state()  # zero nodes
    got = ClusterAwarePolicy().get_least_loaded_server(state, set(), TTL)
    assert got is None


# --------------------------------------------------------------------- ClusterAwarePolicy: down nodes


def test_skips_down_within_ttl(fake_state):
    state = fake_state(
        ("a", "aws", "r", "z1", "primary", 0, True),   # down
        ("b", "aws", "r", "z2", "primary", 5, False),  # up but loaded
    )
    # `a` is down with is_down_since=now (set by fake_state); within TTL=5s.
    got = ClusterAwarePolicy().get_least_loaded_server(state, set(), ttl=5.0)
    assert got.host == "b"


def test_reconsiders_down_after_ttl(fake_state):
    state = fake_state(
        ("a", "aws", "r", "z1", "primary", 0, True),
        ("b", "aws", "r", "z2", "primary", 5, False),
    )
    # Backdate the failure so it's outside the TTL window.
    state.nodes["a"].is_down_since = time.monotonic() - 60.0
    got = ClusterAwarePolicy().get_least_loaded_server(state, set(), ttl=10.0)
    # a is eligible again (TTL expired), has count=0 vs b's 5 → a wins.
    assert got.host == "a"


def test_zero_ttl_means_always_eligible_when_down(fake_state):
    state = fake_state(
        ("a", "aws", "r", "z1", "primary", 0, True),
        ("b", "aws", "r", "z2", "primary", 5, False),
    )
    # With ttl=0, any down node whose is_down_since < now is past the window.
    # Sleep a beat so monotonic advances.
    time.sleep(0.01)
    got = ClusterAwarePolicy().get_least_loaded_server(state, set(), ttl=0.0)
    assert got.host == "a"


# --------------------------------------------------------------------- TopologyAwarePolicy


def test_topology_filter_exact(fake_state):
    state = fake_state(
        ("a", "aws", "us-west", "1a", "primary", 0),
        ("b", "aws", "us-west", "1b", "primary", 0),
        ("c", "aws", "us-east", "2a", "primary", 0),
    )
    policy = TopologyAwarePolicy([Placement("aws", "us-west", "1a")])
    random.seed(0)
    counts = {"a": 0, "b": 0, "c": 0}
    for _ in range(500):
        got = policy.get_least_loaded_server(state, set(), TTL)
        counts[got.host] += 1
    assert counts["a"] == 500
    assert counts["b"] == 0
    assert counts["c"] == 0


def test_topology_filter_wildcard_zone(fake_state):
    state = fake_state(
        ("a", "aws", "us-west", "1a", "primary", 0),
        ("b", "aws", "us-west", "1b", "primary", 0),
        ("c", "aws", "us-east", "2a", "primary", 0),
    )
    policy = TopologyAwarePolicy([Placement("aws", "us-west", "*")])
    random.seed(0xBEEF)
    counts = {"a": 0, "b": 0, "c": 0}
    for _ in range(2000):
        got = policy.get_least_loaded_server(state, set(), TTL)
        counts[got.host] += 1
    # Both us-west zones share the load; us-east doesn't get any.
    assert counts["c"] == 0
    assert 850 <= counts["a"] <= 1150
    assert 850 <= counts["b"] <= 1150


def test_topology_multiple_keys_union(fake_state):
    """When multiple topology keys are given, nodes matching ANY of them are eligible."""
    state = fake_state(
        ("a", "aws", "us-west", "1a", "primary", 0),
        ("b", "aws", "us-east", "2a", "primary", 0),
        ("c", "gcp", "us-central", "1c", "primary", 0),
    )
    policy = TopologyAwarePolicy([
        Placement("aws", "us-west", "1a"),
        Placement("aws", "us-east", "2a"),
    ])
    random.seed(0)
    counts = {"a": 0, "b": 0, "c": 0}
    for _ in range(1000):
        got = policy.get_least_loaded_server(state, set(), TTL)
        counts[got.host] += 1
    assert counts["c"] == 0
    assert counts["a"] > 0
    assert counts["b"] > 0


def test_topology_no_match_returns_none(fake_state):
    state = fake_state(
        ("a", "aws", "us-west", "1a", "primary", 0),
        ("b", "aws", "us-east", "2a", "primary", 0),
    )
    policy = TopologyAwarePolicy([Placement("gcp", "us-central", "1c")])
    got = policy.get_least_loaded_server(state, set(), TTL)
    assert got is None


def test_topology_respects_down_state(fake_state):
    state = fake_state(
        ("a", "aws", "us-west", "1a", "primary", 0, True),   # down
        ("b", "aws", "us-west", "1b", "primary", 5, False),
    )
    policy = TopologyAwarePolicy([Placement("aws", "us-west", "*")])
    got = policy.get_least_loaded_server(state, set(), ttl=5.0)
    assert got.host == "b"


# --------------------------------------------------------------------- single-pass invariant


def test_single_pass_over_node_dict(fake_state):
    """The policy must iterate the node dict at most once per call.

    `dict.values` is read-only on the built-in dict, so we swap the dict for
    a subclass that counts how many times `values()` is invoked.
    """
    state = fake_state(
        ("a", "aws", "r", "z1", "primary", 0),
        ("b", "aws", "r", "z2", "primary", 0),
        ("c", "aws", "r", "z3", "primary", 0),
    )

    class CountingDict(dict):
        values_calls = 0
        def values(self):
            CountingDict.values_calls += 1
            return super().values()

    state.nodes = CountingDict(state.nodes)
    ClusterAwarePolicy().get_least_loaded_server(state, set(), TTL)
    assert CountingDict.values_calls == 1


# --------------------------------------------------------------------- build_policy factory


def test_build_policy_cluster_aware_when_no_topology():
    p = build_policy(YBParams(smart_driver_enabled=True, topology_keys=[]))
    assert isinstance(p, ClusterAwarePolicy)
    assert not isinstance(p, TopologyAwarePolicy)


def test_build_policy_topology_aware_when_keys_set():
    p = build_policy(YBParams(
        smart_driver_enabled=True,
        topology_keys=[Placement("aws", "us-west", "1a")],
    ))
    assert isinstance(p, TopologyAwarePolicy)
    # TopologyAwarePolicy *is also* a ClusterAwarePolicy (inheritance)
    assert isinstance(p, ClusterAwarePolicy)
