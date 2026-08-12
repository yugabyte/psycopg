#!/usr/bin/env python3
"""xCluster failover demo — full data + connection story.

What this shows
---------------
1. A ``ConnectionPool(check=xcluster_check)`` opens 6 conns on the PRIMARY
   cluster and INSERTs rows through them (load-balanced across .1/.2/.3).
2. A side-channel SELECT on the secondary cluster proves those rows have
   replicated over via xCluster (unidirectional primary -> secondary).
3. The operator stops every primary node. The smart driver's circuit
   breaker detects the outage in seconds and flips ``group.status`` to
   UNHEALTHY.
4. The next time the application borrows from the pool, the
   ``xcluster_check`` callback evicts every stale primary-tagged conn and
   the pool opens replacements against the SECONDARY cluster.
5. The application's SELECTs on the secondary return all the rows it
   originally wrote — the failover preserved both the *connection path*
   (driver-side) and the *data* (xCluster-side).
6. Phase 4 holds the pool open so the operator can verify on /rpcz and via
   ysqlsh; press ENTER once to release and exit.

Prereq: ``demo/setup.sh`` has been run. That script creates the two
clusters, the ``failover_demo`` table on each, and the unidirectional
xCluster replication stream.

Run alongside ``demo/watch_rpcz.py`` in a second terminal to see the
``client backend`` count migrate from .1/.2/.3 to .4/.5/.6 in real time.
"""

from __future__ import annotations

import logging
import sys
import time
from collections import Counter
from contextlib import ExitStack

import psycopg
from psycopg_pool import ConnectionPool

from psycopg.yb import bootstrap_failover_group
from psycopg.yb.health import HealthResult
from psycopg.yb.pool import xcluster_check
from psycopg.yb.registry import ClusterRegistry

# Reference sample CB, kept alongside this demo file. The driver ships
# no default — every xCluster deployment MUST attach a CircuitBreaker
# to both group.primary_circuit_breaker and group.secondary_circuit_breaker
# after bootstrap; otherwise psycopg.connect() raises MissingCircuitBreakerError.
from demo.samples.tracker_table_cb import TrackerTableCircuitBreaker

# The smart driver's probe thread keeps trying to reach the (dead)
# primary cluster every refresh interval while we hold open Phase 4 —
# each refused reopen emits a WARNING from psycopg.yb.registry. That's
# routine probe noise once the CB has already declared UNHEALTHY, so
# silence it. The CB's own transition WARNING (psycopg.yb.circuit_breaker)
# is left at default level so the demo audience can still see the
# `primary unhealthy: ... crossed threshold` event.
logging.getLogger("psycopg.yb.registry").setLevel(logging.ERROR)


PRIMARY_HOSTS = "127.0.0.1,127.0.0.2,127.0.0.3"
SECONDARY_HOSTS = "127.0.0.4,127.0.0.5,127.0.0.6"

POOL_SIZE = 6
# Each pool conn inserts this many rows during Phase 1 → POOL_SIZE × this
# is the total row count we'll then read back from the secondary in Phase 3.
ROWS_PER_CONN = 4
TOTAL_ROWS = POOL_SIZE * ROWS_PER_CONN

# Detection-latency tuning for a demo target of "trip within ~10s of
# primary going down":
#
#   * yb_servers_refresh_interval=2 — CB ticks every 2s.
#   * max_update_failures_allowed=0 on the CB (threshold of 1, i.e. ONE
#     failed UPDATE flips status). Faster trip at the cost of being
#     sensitive to single-tablet leader-election blips during normal
#     operation; that's an acceptable demo trade-off because we're
#     showing the failover path, not a long-running production loop.
#   * keepalives_idle=2 + keepalives_interval=1 + keepalives_count=2 —
#     TCP keepalive probes start 2s after the socket goes idle so a
#     half-open connection (postmaster died with no FIN) surfaces
#     within ~4s on Linux. macOS only honors keepalives_idle (it ignores
#     the interval/count knobs), but yb-ctl `stop` triggers a clean
#     postmaster shutdown that sends FIN, so the in-flight send fails
#     immediately and keepalive is rarely needed in practice.
#   * connect_timeout=3 — bounds any libpq connect attempt, so the
#     post-trip failover to secondary can't hang waiting for primary
#     connects we know will fail.
#   * yb.failover.cooldownSecs=0 — no rate-limit on status transitions.
PROBE_INTERVAL_S = 2
MAX_UPDATE_FAILURES_ALLOWED = 0
XCLUSTER_REPLICATION_LAG_BUDGET_S = 3.0

DSN = (
    f"host={PRIMARY_HOSTS} port=5433 user=yugabyte dbname=yugabyte "
    f"load_balance_hosts=true "
    f"connect_timeout=3 "
    f"keepalives=1 keepalives_idle=2 keepalives_interval=1 keepalives_count=2 "
    f"yb.failover.secondaryClusterHosts={SECONDARY_HOSTS} "
    f"yb_servers_refresh_interval={PROBE_INTERVAL_S} "
    f"yb.failover.cooldownSecs=0"
)


def _attach_cbs() -> None:
    """Bootstrap the FailoverGroup and attach the reference tracker-table
    CBs to both slots. Idempotent — safe to call multiple times."""
    group = bootstrap_failover_group(DSN)
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


def histogram(conns) -> dict[str, int]:
    return dict(Counter(c._yb_host for c in conns))


def wait_for_status(group, expected: HealthResult, timeout_s: float) -> bool:
    """Poll group.status with a per-second in-place heartbeat so the demo
    doesn't look frozen while the CB ticks."""
    deadline = time.monotonic() + timeout_s
    last_print = 0.0
    while time.monotonic() < deadline:
        if group.status == expected:
            return True
        if time.monotonic() - last_print >= 1.0:
            elapsed = time.monotonic() - (deadline - timeout_s)
            print(
                f"    ... {elapsed:5.1f}s elapsed; "
                f"group.status={group.status.value}",
                end="\r",
                flush=True,
            )
            last_print = time.monotonic()
        time.sleep(0.1)
    print()
    return False


def banner(title: str) -> None:
    bar = "=" * 70
    print(f"\n{bar}\n  {title}\n{bar}")


def main() -> int:
    banner("xCluster failover demo")
    print(f"  primary    : {PRIMARY_HOSTS}")
    print(f"  secondary  : {SECONDARY_HOSTS}")
    print(f"  pool size  : {POOL_SIZE}")
    print(f"  rows / conn: {ROWS_PER_CONN}  (total {TOTAL_ROWS} rows inserted)")
    print(f"  probe      : every {PROBE_INTERVAL_S}s")
    print(f"  threshold  : {MAX_UPDATE_FAILURES_ALLOWED + 1} consecutive failures")
    print(
        f"\n  prereq: demo/setup.sh has been run (creates the two clusters,\n"
        f"          the failover_demo table on each, and the xCluster\n"
        f"          replication stream primary -> secondary)."
    )

    # MUST happen before any psycopg.connect / ConnectionPool — the
    # dispatcher raises MissingCircuitBreakerError otherwise.
    _attach_cbs()

    with ConnectionPool(
        DSN,
        check=xcluster_check,
        min_size=POOL_SIZE,
        max_size=POOL_SIZE,
        timeout=120,
    ) as pool:
        pool.wait()

        # Clean slate — DELETE so reruns don't hit primary-key conflicts.
        # We can't TRUNCATE: YB blocks table-rewriting DDL on tables that
        # participate in xCluster replication (see yugabyte-db #16625).
        # DELETE is a regular DML that replicates to the secondary cleanly.
        with pool.connection() as c:
            with c.cursor() as cur:
                cur.execute("DELETE FROM failover_demo")
            c.commit()
            primary_uuid = c._yb_uuid

        # ---------- Phase 1: write data on PRIMARY via the pool ----------
        banner(f"Phase 1 — INSERT {TOTAL_ROWS} rows on PRIMARY via the pool")
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
                            "INSERT INTO failover_demo (id, ts, payload) "
                            "VALUES (%s, NOW(), %s)",
                            (rid, f"row {rid} written via {c._yb_host}"),
                        )
                c.commit()
            print("per-host distribution of pool conns:")
            for host, n in sorted(histogram(conns).items()):
                print(
                    f"  {host:<14} {n} pool conn(s)  -> "
                    f"{n * ROWS_PER_CONN} rows inserted via this host"
                )
            print(
                f"\nWatch terminal should now show ~{POOL_SIZE // 3} 'client\n"
                f"backend' entries on each of .1/.2/.3 and zero on .4-.6."
            )
        # Conns now idle in the pool.

        # Brief wait for xCluster to replicate.
        time.sleep(XCLUSTER_REPLICATION_LAG_BUDGET_S)

        # Side-channel SELECT against secondary proves the data is there
        # BEFORE failover. Uses raw psycopg (no smart driver, no pool)
        # so we're sure the read landed on .4 directly.
        print("\nverifying xCluster has replicated to secondary (side-channel SELECT):")
        with psycopg.connect(
            "host=127.0.0.4 port=5433 user=yugabyte dbname=yugabyte "
            "connect_timeout=5"
        ) as side:
            with side.cursor() as cur:
                cur.execute("SELECT count(*) FROM failover_demo")
                row = cur.fetchone()
                assert row is not None
                replicated = row[0]
        if replicated == TOTAL_ROWS:
            print(
                f"  ✓ secondary now holds {replicated}/{TOTAL_ROWS} rows. "
                f"xCluster is replicating live."
            )
        else:
            print(
                f"  WARNING: secondary has {replicated}/{TOTAL_ROWS} rows; "
                f"either replication is lagging or setup is incomplete."
            )

        group = ClusterRegistry.instance().get_failover_group_by_uuid(
            primary_uuid
        )
        if group is None:
            print("ERROR: no FailoverGroup attached — did secondaryClusterHosts parse?")
            return 1

        # ---------- Phase 2: operator stops the primary cluster ----------
        banner("Phase 2 — STOP every node of the PRIMARY cluster")
        print("Run something like:")
        print("    yb-ctl --data_dir=~/yb-primary stop")
        print("Or stop_node 1, 2, 3 in turn.\n")
        input("Press ENTER once the primary is fully down: ")

        print("\nwaiting for the circuit breaker to trip ...")
        t0 = time.monotonic()
        if not wait_for_status(group, HealthResult.UNHEALTHY, timeout_s=120):
            print("  ERROR: CB did not trip within 120s")
            return 1
        elapsed = time.monotonic() - t0
        print(
            f"  ✓ circuit breaker tripped after {elapsed:.1f}s "
            f"(group.status=UNHEALTHY)"
        )

        # ---------- Phase 3: read via pool — now routed to SECONDARY ----------
        banner(f"Phase 3 — SELECT via the pool (now on SECONDARY)")
        with ExitStack() as stack:
            conns = [
                stack.enter_context(pool.connection())
                for _ in range(POOL_SIZE)
            ]
            print("per-host distribution of pool conns:")
            for host, n in sorted(histogram(conns).items()):
                print(f"  {host:<14} {n} pool conn(s)")

            print(
                "\nEach pool conn now runs SELECT count(*) FROM failover_demo:"
            )
            counts: list[tuple[str, int]] = []
            for c in conns:
                with c.cursor() as cur:
                    cur.execute("SELECT count(*) FROM failover_demo")
                    row = cur.fetchone()
                    assert row is not None
                    counts.append((c._yb_host, row[0]))
                c.commit()
            for host, n in counts:
                print(f"  {host:<14} sees {n} rows")

            distinct = {n for _, n in counts}
            if distinct == {TOTAL_ROWS}:
                print(
                    f"\n  ✓ all {len(counts)} pool conns landed on the "
                    f"SECONDARY cluster\n"
                    f"    and read back all {TOTAL_ROWS} rows. xCluster\n"
                    f"    preserved the data; the smart driver routed the\n"
                    f"    SELECT here automatically."
                )
            else:
                print(f"\n  WARNING: row counts disagree: {distinct}")

            # ---------- Phase 4: hold open while operator inspects ----------
            banner("Phase 4 — demo complete")
            print("The pool's 6 conns are pinned open on the secondary cluster.")
            print("Verify in a side shell with:")
            print(
                "    ysqlsh -h 127.0.0.4 -U yugabyte -d yugabyte "
                "-c 'SELECT * FROM failover_demo ORDER BY id LIMIT 5;'"
            )
            input("\nPress ENTER to release the conns and exit: ")

    print("demo done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
