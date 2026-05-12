"""
Load-balancing policy ABC.

Method name `get_least_loaded_server` matches pgjdbc-yb's
`LoadBalancer.getLeastLoadedServer` so anyone moving between drivers sees
the same surface.
"""

# Copyright (C) 2026 Yugabyte

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..node import NodeInfo
    from ..registry import ClusterState


class LoadBalancingPolicy(ABC):
    @abstractmethod
    def get_least_loaded_server(
        self, state: "ClusterState", attempted: set[str], ttl: float
    ) -> "NodeInfo | None":
        """Return the next node to try, or None if no eligible node remains.

        `attempted` is the set of hosts the caller has already tried during
        this connection request (so we don't loop). `ttl` is the
        `failed_host_reconnect_delay_secs` knob — nodes marked down whose
        downtime has exceeded `ttl` become eligible again.
        """
        raise NotImplementedError


def is_eligible(
    node: "NodeInfo", attempted: set[str], now: float, ttl: float
) -> bool:
    """Common eligibility check shared by both concrete policies.

    A node is eligible when it isn't in the current request's failed list and
    either is healthy or has been down for longer than the configured TTL.
    Subclasses layer additional filters (e.g. placement match) on top.
    """
    if node.host in attempted:
        return False
    if node.is_down and (now - node.is_down_since) <= ttl:
        return False
    return True
