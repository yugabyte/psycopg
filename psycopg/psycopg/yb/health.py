"""
Cluster-health decision surface for xCluster failover.

This module defines the contract — ``HealthResult`` and the single function
``cluster_status_check(group)`` — that the ``HealthProbe`` background thread
calls on each tick to decide whether the primary cluster should be marked
HEALTHY or UNHEALTHY.

The function is implemented in this revision as an always-HEALTHY **stub**.
When the tracker-table-based detection logic from the spec lands (the
"Phase 9" follow-on), only the body of ``cluster_status_check`` changes —
the signature, callers, and surrounding plumbing stay identical.

See /tmp/xcluster_failover_design.html §6 for the full design and the
phase-9 sketch.
"""

# Copyright (C) 2026 Yugabyte

from __future__ import annotations

import logging
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .registry import FailoverGroup

logger = logging.getLogger(__name__)


class HealthResult(Enum):
    """Binary cluster-health outcome. Matches the spec's ``CLUSTER_STATUS``
    flag — only two states; no INCONCLUSIVE in this revision."""

    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"


def cluster_status_check(group: "FailoverGroup") -> HealthResult:
    """Decide whether ``group.primary`` is currently usable.

    **STUB IMPLEMENTATION** — always returns ``HealthResult.HEALTHY``.

    The signature is the stable contract. The eventual tracker-table-based
    check (spec §"Circuit Breaker") replaces only the body:

      * runs ``UPDATE yb_cluster_health_tracker SET last_updated = NOW()``
        on ``group.primary.control_sync``
      * recreates the table on TABLE-NOT-FOUND
      * filters auth / TLS errors out of the failure counter
      * returns UNHEALTHY after
        ``group.max_update_failures_allowed + 1`` consecutive failures

    Until that lands, integration tests drive failover via
    ``FailoverGroup.force_status`` (Phase 7 §14 Tier 1).
    """
    return HealthResult.HEALTHY
