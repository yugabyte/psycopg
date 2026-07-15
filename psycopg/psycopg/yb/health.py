"""
Cluster-health decision surface for xCluster failover.

This module defines the contract — ``HealthResult`` — and the two thin
delegating helpers the ``HealthProbe`` thread calls on each tick to
decide whether each cluster should be marked HEALTHY or UNHEALTHY:

  * ``check_primary_cluster(group)`` — delegates to
    ``group.primary_circuit_breaker.check(group)``
  * ``check_secondary_cluster(group)`` — delegates to
    ``group.secondary_circuit_breaker.check(group)``

Both fall back to HEALTHY if the corresponding circuit breaker slot is
``None`` — this is what tests get when they build a ``FailoverGroup``
without attaching breakers (the default at bootstrap is a
``TrackerTableCircuitBreaker`` per cluster, but unit tests often skip
that).

See docs/xcluster_failover_design.html §4 for the full design and the
per-cluster CB rationale.
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


def check_primary_cluster(group: "FailoverGroup") -> HealthResult:
    """Decide whether ``group.primary`` is currently usable.

    Thin dispatcher — delegates to ``group.primary_circuit_breaker.check(group)``.
    The default at bootstrap is a ``TrackerTableCircuitBreaker`` pointed at
    the primary cluster; tests can swap it by assigning to
    ``group.primary_circuit_breaker``.

    Returns HEALTHY if the slot is ``None`` — matches the pre-CB stub
    semantics and keeps the contract well-defined for unit tests that
    build a group without attaching a breaker.
    """
    cb = group.primary_circuit_breaker
    if cb is None:
        return HealthResult.HEALTHY
    return cb.check(group)


def check_secondary_cluster(group: "FailoverGroup") -> HealthResult:
    """Symmetric to :func:`check_primary_cluster` for the secondary cluster.
    Delegates to ``group.secondary_circuit_breaker.check(group)`` and
    falls back to HEALTHY if the slot is ``None``."""
    cb = group.secondary_circuit_breaker
    if cb is None:
        return HealthResult.HEALTHY
    return cb.check(group)
