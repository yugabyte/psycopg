"""
Load-balancing policies for the YugabyteDB smart driver.

`build_policy(yb_params)` is the factory the connection dispatcher uses:
it returns a `ClusterAwarePolicy` for plain smart-driver mode (no topology
keys) and a `TopologyAwarePolicy` when topology keys are set. Future role-
based modes (`only-primary`, `prefer-rr`) slot in here as new branches.
"""

# Copyright (C) 2026 Yugabyte

from __future__ import annotations

from ..params import YBParams
from .base import LoadBalancingPolicy
from .cluster_aware import ClusterAwarePolicy
from .topology_aware import TopologyAwarePolicy


__all__ = [
    "LoadBalancingPolicy",
    "ClusterAwarePolicy",
    "TopologyAwarePolicy",
    "build_policy",
]


def build_policy(yb_params: YBParams) -> LoadBalancingPolicy:
    if yb_params.topology_keys:
        return TopologyAwarePolicy(yb_params.topology_keys)
    return ClusterAwarePolicy()
