"""
Topology-aware load-balancing policy: cluster-aware with a placement filter.

In v1 all topology-key entries are equal — no priority levels, no
cluster-wide fallback. Connections distribute evenly across the set of nodes
whose placement matches any of the user's topology keys. If no matching node
is eligible, returns None and the caller surfaces an error.

Priority levels and the cluster-wide fallback flag are designed-in but live
in v2 (see plan §6). When they land, this class adds level iteration around
the existing `get_least_loaded_server` call.
"""

# Copyright (C) 2026 Yugabyte

from __future__ import annotations

from typing import TYPE_CHECKING

from .cluster_aware import ClusterAwarePolicy

if TYPE_CHECKING:
    from ..node import NodeInfo, Placement


class TopologyAwarePolicy(ClusterAwarePolicy):

    def __init__(self, topology_keys: list["Placement"]) -> None:
        self._keys = topology_keys

    def _is_eligible(
        self,
        node: "NodeInfo",
        attempted: set[str],
        now: float,
        ttl: float,
    ) -> bool:
        if not super()._is_eligible(node, attempted, now, ttl):
            return False
        return any(node.placement.matches(k) for k in self._keys)
