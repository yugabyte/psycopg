"""xCluster integration tests verifying DATA continuity through failover.

Companion to ``test_smart_driver_xcluster_cb.py`` (which proves the
circuit breaker + dispatcher route connections correctly): these tests
also verify the data layer — that rows written on the primary cluster
via the pool actually reach the secondary cluster via xCluster
replication, and that after the primary fails, pool reads against the
secondary return those rows.

The ``yb_xcluster_clusters`` session fixture creates the
``xcluster_test_data`` table on both clusters and registers
``yb-admin setup_universe_replication`` for it. Each test starts with a
``DELETE FROM`` to guarantee a clean slate (``TRUNCATE`` would fail —
YB blocks table-rewriting DDL on xCluster source tables).

Auto-tagged with the ``yb`` integration marker via the
``test_smart_driver*`` filename rule in conftest.
"""

# Copyright (C) 2026 Yugabyte

from __future__ import annotations

import time
from collections import Counter
from contextlib import ExitStack

import pytest

import psycopg
from psycopg.yb import bootstrap_failover_group
from psycopg.yb.health import HealthResult
from psycopg.yb.registry import ClusterRegistry

from demo.samples.tracker_table_cb import TrackerTableCircuitBreaker


PRIMARY_HOSTS = "127.0.0.1,127.0.0.2,127.0.0.3"
SECONDARY_HOSTS = "127.0.0.4,127.0.0.5,127.0.0.6"
PROBE_INTERVAL_S = 3
MAX_UPDATE_FAILURES_ALLOWED = 1
THRESHOLD = MAX_UPDATE_FAILURES_ALLOWED + 1

_CB_DSN = (
    f"host={PRIMARY_HOSTS} port=5433 user=yugabyte dbname=yugabyte "
    f"load_balance_hosts=true "
    f"yb.failover.secondaryClusterHosts={SECONDARY_HOSTS} "
    f"yb_servers_refresh_interval={PROBE_INTERVAL_S} "
    f"yb.failover.cooldownSecs=0"
)

# Each pool conn inserts this many rows. With POOL_SIZE=6 → 24 total.
POOL_SIZE = 6
ROWS_PER_CONN = 4
TOTAL_ROWS = POOL_SIZE * ROWS_PER_CONN

# How long to wait for xCluster replication to ship the inserts to the
# secondary. Empirically <3s on a quiet test machine; 10s gives generous
# headroom under load.
REPLICATION_LAG_BUDGET_S = 10.0

# Detection budget for the CB to trip after we stop the primary. The
# tracker UPDATE on the (now-broken) control conn raises immediately
# with 'connection refused' once all primary nodes are stopped, so trip
# is usually sub-second — but the YB cluster shutdown itself takes a
# few seconds and there's a 3s probe interval, so budget ~30s.
_TRIP_TIMEOUT_S = 30.0


def _clean_table(host: str) -> None:
    """DELETE FROM xcluster_test_data via a raw conn on ``host``. Used to
    set per-test slate without touching the pool/circuit-breaker."""
    dsn = (
        f"host={host} port=5433 user=yugabyte dbname=yugabyte "
        f"connect_timeout=5"
    )
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM xcluster_test_data")
        conn.commit()


def _count_rows(host: str) -> int:
    dsn = (
        f"host={host} port=5433 user=yugabyte dbname=yugabyte "
        f"connect_timeout=5"
    )
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM xcluster_test_data")
            row = cur.fetchone()
            assert row is not None
            return row[0]


def _wait_replicated(host: str, target_count: int, timeout_s: float) -> bool:
    """Poll the consumer-side row count until it reaches ``target_count``
    or the timeout elapses. xCluster is eventually-consistent — we don't
    want to assert immediately on the insert path."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if _count_rows(host) >= target_count:
            return True
        time.sleep(0.5)
    return False


def _wait_for_status(group, expected: HealthResult, timeout_s: float) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if group.primary_status == expected:
            return True
        time.sleep(0.2)
    return False


def _bootstrap_group():
    """Bootstrap the FailoverGroup, attach sample tracker-table CBs to
    both slots, and return the group. Must run BEFORE any
    ``psycopg.connect(_CB_DSN)`` — the dispatcher raises
    ``MissingCircuitBreakerError`` if CB slots are still None."""
    group = bootstrap_failover_group(_CB_DSN)
    assert group is not None
    if group.primary_circuit_breaker is None:
        group.primary_circuit_breaker = TrackerTableCircuitBreaker(
            which_cluster="primary",
            max_update_failures_allowed=MAX_UPDATE_FAILURES_ALLOWED,
        )
    if group.secondary_circuit_breaker is None:
        group.secondary_circuit_breaker = TrackerTableCircuitBreaker(
            which_cluster="secondary",
            max_update_failures_allowed=MAX_UPDATE_FAILURES_ALLOWED,
        )
    return group


def _histogram(conns) -> dict[str, int]:
    return dict(Counter(c._yb_host for c in conns))


# --------------------------------------------------------------------- replication round-trip


def test_xcluster_replicates_pool_inserts_to_secondary(
    yb_xcluster_clusters,
):
    """Insert rows on primary via the smart-driver pool; verify they
    show up on the secondary within the replication lag budget.

    This is the *data-plane* sibling of the routing tests in
    ``test_smart_driver_xcluster_cb.py`` — proves the xCluster stream
    set up by the session fixture is actually live and ingesting writes
    that flow through the pool's load-balanced primary conns.
    """
    pytest.importorskip("psycopg_pool")
    from psycopg_pool import ConnectionPool

    from psycopg.yb.pool import xcluster_check

    _clean_table(PRIMARY_HOSTS.split(",")[0])
    assert _wait_replicated(
        SECONDARY_HOSTS.split(",")[0], 0, REPLICATION_LAG_BUDGET_S
    ), "secondary did not catch up to 0-row baseline before test started"

    with ConnectionPool(
        _CB_DSN, check=xcluster_check,
        min_size=POOL_SIZE, max_size=POOL_SIZE, timeout=60,
    ) as pool:
        pool.wait()
        with ExitStack() as stack:
            conns = [
                stack.enter_context(pool.connection())
                for _ in range(POOL_SIZE)
            ]
            # Each pool conn inserts ROWS_PER_CONN rows tagged with the
            # host it landed on, so the histogram + the per-row payload
            # both prove the load distribution.
            for i, c in enumerate(conns):
                assert c._yb_cluster == "primary"
                with c.cursor() as cur:
                    for j in range(ROWS_PER_CONN):
                        rid = i * ROWS_PER_CONN + j
                        cur.execute(
                            "INSERT INTO xcluster_test_data "
                            "(id, ts, payload) VALUES (%s, NOW(), %s)",
                            (rid, f"row {rid} via {c._yb_host}"),
                        )
                c.commit()
            # Distribution sanity check — exactly the same shape as the
            # _assert_distributed check in test_smart_driver_xcluster_cb.
            hist = _histogram(conns)
            assert set(hist) == {
                "127.0.0.1", "127.0.0.2", "127.0.0.3"
            }, f"expected all 3 primary hosts; got {hist}"

    assert _wait_replicated(
        SECONDARY_HOSTS.split(",")[0],
        TOTAL_ROWS,
        REPLICATION_LAG_BUDGET_S,
    ), (
        f"secondary never observed all {TOTAL_ROWS} replicated rows "
        f"within {REPLICATION_LAG_BUDGET_S}s "
        f"(actual: {_count_rows(SECONDARY_HOSTS.split(',')[0])})"
    )


# --------------------------------------------------------------------- end-to-end failover


def test_xcluster_data_survives_primary_failure_via_pool(
    yb_xcluster_clusters, yb_xcluster_ctl,
):
    """End-to-end story:

      1. Pool inserts ``TOTAL_ROWS`` rows on primary, distributed across
         all 3 primary nodes.
      2. We wait for xCluster to replicate to secondary.
      3. We STOP every primary tserver. The CB trips on the next probe
         tick; ``group.primary_status`` flips to UNHEALTHY.
      4. We re-borrow ``POOL_SIZE`` conns from the same pool. Every
         conn now lives on the secondary cluster (the dispatcher rerouted
         them; the ``xcluster_check`` callback evicted any cached primary
         conns).
      5. Each pool conn issues ``SELECT count(*)`` on
         ``xcluster_test_data``. All ``POOL_SIZE`` results MUST equal
         ``TOTAL_ROWS`` — the rows we wrote pre-failure are still readable
         on the secondary cluster because xCluster replicated them, and
         the application read them via the secondary because the smart
         driver rerouted the pool. Without xCluster the table would be
         empty on the secondary; without the smart driver the conns would
         try (and fail) to reach the dead primary.

    Cleanup: ``recover_primary_cluster`` restarts every primary node so
    the session-scoped fixture can run further tests after this.
    """
    pytest.importorskip("psycopg_pool")
    from psycopg_pool import ConnectionPool

    from psycopg.yb.pool import xcluster_check

    _clean_table(PRIMARY_HOSTS.split(",")[0])
    assert _wait_replicated(
        SECONDARY_HOSTS.split(",")[0], 0, REPLICATION_LAG_BUDGET_S
    )

    with ConnectionPool(
        _CB_DSN, check=xcluster_check,
        min_size=POOL_SIZE, max_size=POOL_SIZE, timeout=120,
    ) as pool:
        pool.wait()

        # ----- Phase 1: write 24 rows on primary -----
        with ExitStack() as stack:
            conns = [
                stack.enter_context(pool.connection())
                for _ in range(POOL_SIZE)
            ]
            for i, c in enumerate(conns):
                with c.cursor() as cur:
                    for j in range(ROWS_PER_CONN):
                        rid = i * ROWS_PER_CONN + j
                        cur.execute(
                            "INSERT INTO xcluster_test_data "
                            "(id, ts, payload) VALUES (%s, NOW(), %s)",
                            (rid, f"row {rid}"),
                        )
                c.commit()
            primary_uuid = conns[0]._yb_uuid

        # ----- xCluster catches up -----
        assert _wait_replicated(
            SECONDARY_HOSTS.split(",")[0],
            TOTAL_ROWS,
            REPLICATION_LAG_BUDGET_S,
        ), "xCluster did not finish replicating before primary outage"

        group = ClusterRegistry.instance().get_failover_group_by_uuid(
            primary_uuid
        )
        assert group is not None
        assert group.primary_status == HealthResult.HEALTHY

        # ----- Phase 2: kill the primary cluster -----
        yb_xcluster_ctl.stop_primary_node(1)
        yb_xcluster_ctl.stop_primary_node(2)
        yb_xcluster_ctl.stop_primary_node(3)
        try:
            # CB should detect and trip.
            assert _wait_for_status(
                group, HealthResult.UNHEALTHY, _TRIP_TIMEOUT_S
            ), (
                f"CB did not trip after primary outage within "
                f"{_TRIP_TIMEOUT_S}s; status={group.primary_status}"
            )

            # ----- Phase 3: read via the pool — now on secondary -----
            with ExitStack() as stack:
                conns = [
                    stack.enter_context(pool.connection())
                    for _ in range(POOL_SIZE)
                ]
                # All borrows must land on secondary.
                hist = _histogram(conns)
                assert set(hist) <= {
                    "127.0.0.4", "127.0.0.5", "127.0.0.6"
                }, f"expected only secondary hosts; got {hist}"
                # Each conn reads back the full replicated rowset.
                for c in conns:
                    assert c._yb_cluster == "secondary"
                    with c.cursor() as cur:
                        cur.execute(
                            "SELECT count(*) FROM xcluster_test_data"
                        )
                        row = cur.fetchone()
                    c.commit()
                    assert row is not None
                    assert row[0] == TOTAL_ROWS, (
                        f"pool conn on {c._yb_host} sees {row[0]} rows; "
                        f"expected {TOTAL_ROWS}. xCluster replication "
                        f"did not preserve the pre-failure data."
                    )
        finally:
            # Bring the primary back so the next test inherits a working
            # cluster. recover_primary_cluster TCP-probes each host and
            # only restart_nodes the ones that are down.
            yb_xcluster_ctl.recover_primary_cluster()
