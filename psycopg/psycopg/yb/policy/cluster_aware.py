"""
Cluster-aware load-balancing policy: least-connections with random tie-break.

`get_least_loaded_server` atomically chooses the least-loaded eligible node
AND increments its counter, under state.lock, so concurrent callers see
each other's increments and distribute exactly. The caller rolls back the
reservation on connect failure (decrement + mark_failed).

This diverges from pgjdbc-yb's increment-on-success pattern, which admits
a race where N concurrent connects can all pick the same "min count" node
before any of them has finished the TCP handshake. With reservation built
in, that race can't happen.

Lock discipline: `state.lock` is held for the dict iteration, the tie-break
random pick, and the counter increment. The TCP connect happens after the
lock is released.
"""

# Copyright (C) 2026 Yugabyte

from __future__ import annotations

import logging
import random
import sys
import time
from typing import TYPE_CHECKING

from .. import TRACE
from .base import LoadBalancingPolicy, is_eligible

if TYPE_CHECKING:
    from ..node import NodeInfo
    from ..registry import ClusterState

logger = logging.getLogger(__name__)


class ClusterAwarePolicy(LoadBalancingPolicy):

    def get_least_loaded_server(
        self, state: "ClusterState", attempted: set[str], ttl: float
    ) -> "NodeInfo | None":
        """Atomically pick the least-loaded eligible node and reserve it.

        On a successful return, the chosen NodeInfo's `connection_count` has
        already been incremented. The caller MUST roll back the reservation
        if the subsequent connect attempt fails — either by calling
        `ClusterRegistry.decrement(uuid, host)` or `mark_failed` (which zeros
        the count). Otherwise the reservation leaks.

        Returns None when no eligible node remains.
        """
        now = time.monotonic()
        with state.lock:
            min_count = sys.maxsize
            tied: list["NodeInfo"] = []
            for n in state.nodes.values():
                if not self._is_eligible(n, attempted, now, ttl):
                    logger.log(TRACE, "policy filter: skip %s (count=%d, is_down=%s)",
                               n.host, n.connection_count, n.is_down)
                    continue
                if n.connection_count < min_count:
                    min_count = n.connection_count
                    tied = [n]
                elif n.connection_count == min_count:
                    tied.append(n)
            if not tied:
                logger.debug(
                    "policy: no eligible node (attempted=%s, total_nodes=%d)",
                    sorted(attempted), len(state.nodes),
                )
                return None
            chosen = random.choice(tied)
            chosen.connection_count += 1
            if len(tied) > 1:
                logger.debug(
                    "policy pick: %s (count: %d→%d) via tie-break across %d hosts",
                    chosen.host, min_count, chosen.connection_count, len(tied),
                )
            else:
                logger.debug(
                    "policy pick: %s (count: %d→%d), unique minimum",
                    chosen.host, min_count, chosen.connection_count,
                )
            return chosen

    # Overridable for subclasses (TopologyAwarePolicy adds the placement filter).
    def _is_eligible(
        self,
        node: "NodeInfo",
        attempted: set[str],
        now: float,
        ttl: float,
    ) -> bool:
        return is_eligible(node, attempted, now, ttl)
