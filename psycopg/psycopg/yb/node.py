"""
Per-node identity and placement for YugabyteDB smart-driver load balancing.

All types here are I/O-free pure data. Mutations of `NodeInfo` are serialised
by the owning `ClusterState`'s lock; the dataclass itself carries no lock.
"""

# Copyright (C) 2026 Yugabyte

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Placement:
    """A cloud/region/zone triple, as reported by yb_servers().

    `zone` may be the literal '*' (wildcard) when used in topology-key filters;
    `cloud` and `region` cannot — those wildcards are rejected at parse time.
    """

    cloud: str
    region: str
    zone: str

    def matches(self, other: "Placement") -> bool:
        """True if this placement satisfies the filter `other`.

        Symmetry note: `other` is the filter (potentially wildcard); `self` is
        the node's actual placement. `Placement.matches` is not commutative.
        """
        if self.cloud != other.cloud or self.region != other.region:
            return False
        if other.zone == "*":
            return True
        return self.zone == other.zone


@dataclass
class NodeInfo:
    """One row of `SELECT * FROM yb_servers()` plus our runtime bookkeeping.

    `node_type` is stored but not yet filtered on; the read-replica milestone
    (see design doc §6) is a pure policy-layer change that begins consulting
    this field.
    """

    host: str
    public_ip: str | None
    port: int
    placement: Placement
    node_type: str  # "primary" | "read_replica"

    # Mutated only under ClusterState.lock.
    connection_count: int = 0
    is_down: bool = False
    is_down_since: float = 0.0  # time.monotonic() at failure
