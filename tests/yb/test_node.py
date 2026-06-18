"""
Unit tests for `psycopg.yb.node` (NodeInfo, Placement).

Placement matching is the only behaviour worth covering — NodeInfo is a
plain dataclass. The matcher is intentionally non-commutative (the right-hand
side is the *filter*) so we exercise that property explicitly.
"""

# Copyright (C) 2026 Yugabyte

from __future__ import annotations

import pytest

from psycopg.yb.node import NodeInfo, Placement


# --------------------------------------------------------------------- Placement.matches


def test_exact_match():
    a = Placement("aws", "us-west", "us-west-1a")
    assert a.matches(Placement("aws", "us-west", "us-west-1a"))


def test_zone_mismatch():
    node = Placement("aws", "us-west", "us-west-1a")
    assert not node.matches(Placement("aws", "us-west", "us-west-1b"))


def test_region_mismatch():
    node = Placement("aws", "us-west", "us-west-1a")
    assert not node.matches(Placement("aws", "us-east", "us-west-1a"))


def test_cloud_mismatch():
    node = Placement("aws", "us-west", "us-west-1a")
    assert not node.matches(Placement("gcp", "us-west", "us-west-1a"))


def test_zone_wildcard_matches_any_zone():
    node = Placement("aws", "us-west", "us-west-1c")
    assert node.matches(Placement("aws", "us-west", "*"))


def test_zone_wildcard_still_requires_region_match():
    node = Placement("aws", "us-west", "us-west-1c")
    assert not node.matches(Placement("aws", "us-east", "*"))


def test_matches_is_directional():
    """The right-hand side is the *filter*. Wildcards on `self` are NOT
    interpreted as wildcards."""
    wildcard_as_self = Placement("aws", "us-west", "*")
    real_node = Placement("aws", "us-west", "us-west-1a")
    # `*` as the node's zone equals nothing in particular, so it matches
    # `Placement(..., "*")` literally, but not a specific zone.
    assert wildcard_as_self.matches(Placement("aws", "us-west", "*"))
    assert not wildcard_as_self.matches(real_node)


def test_case_sensitive():
    """Placement matching is case-sensitive, mirroring pgjdbc-yb behaviour."""
    assert not Placement("AWS", "us-west", "us-west-1a").matches(
        Placement("aws", "us-west", "us-west-1a")
    )


# --------------------------------------------------------------------- NodeInfo


def test_nodeinfo_defaults():
    n = NodeInfo(
        host="h1",
        public_ip=None,
        port=5433,
        placement=Placement("aws", "us-west", "us-west-1a"),
        node_type="primary",
    )
    assert n.connection_count == 0
    assert n.is_down is False
    assert n.is_down_since == 0.0


def test_nodeinfo_node_type_string():
    """node_type is a string ('primary' or 'read_replica'), not an enum.
    Stored verbatim — v1 doesn't filter on it but v2's role-based modes will."""
    n = NodeInfo(
        host="h1",
        public_ip="1.2.3.4",
        port=5433,
        placement=Placement("aws", "us-west", "us-west-1a"),
        node_type="read_replica",
    )
    assert n.node_type == "read_replica"


def test_placement_is_hashable():
    """Placement is frozen → hashable. Tests don't need this directly, but
    callers wanting to dedupe placement sets do."""
    p1 = Placement("aws", "us-west", "us-west-1a")
    p2 = Placement("aws", "us-west", "us-west-1a")
    assert hash(p1) == hash(p2)
    assert {p1, p2} == {p1}
