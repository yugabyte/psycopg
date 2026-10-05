#!/usr/bin/env python3
"""xCluster failover demo.

One app. Three invocations:

  # Invocation 1: no CB attached — app fails to start.
  python3 demo/xcluster_drain_timing_demo.py

  # Invocation 2: CB attached, drain = 0. Operator brings nodes down
  # one at a time in another terminal; app fails over and back.
  ATTACH_CB=1 DRAIN_TIMEOUT_S=0 \\
      python3 demo/xcluster_drain_timing_demo.py

  # Invocation 3: CB attached, drain = 15. Operator brings 2 nodes
  # down at once; app fails over after the drain window.
  ATTACH_CB=1 DRAIN_TIMEOUT_S=15 \\
      python3 demo/xcluster_drain_timing_demo.py

The app runs a small pool of long-running-transaction workers and
prints a heartbeat + driver INFO logs (CB transitions, drain start /
complete). Ctrl-C to stop; a summary is printed at exit.

Custom implementation used: ``TrackerTableCircuitBreaker`` under
``demo/samples/`` with ``tracker_table_tablets=50`` .
The driver ships no default CB; ``psycopg.connect()`` raises
``MissingCircuitBreakerError`` if the slots are unset — that is
invocation #1.

Prereq: ``demo/setup.sh`` has been run — creates two RF=3 5-node
clusters + the xCluster replication stream for the ``failover_demo``
table.
"""

from __future__ import annotations

import logging
import os
import signal
import sys
import threading
import time
from typing import Optional

# Put the repo root on sys.path so `from demo.samples...` resolves whether
# the demo is invoked from any cwd.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import psycopg
from psycopg_pool import ConnectionPool

from psycopg.yb import (
    MissingCircuitBreakerError,
    bootstrap_failover_group,
)
from psycopg.yb.health import HealthResult
from psycopg.yb.pool import xcluster_check
from psycopg.yb.registry import ClusterRegistry

from demo.samples.tracker_table_cb import TrackerTableCircuitBreaker


# --------------------------------------------------------------------- config

# 5-node RF=3 primary + 5-node RF=3 secondary. The DSN only needs to
# list ONE host per side as a seed; the smart driver discovers the
# rest via ``yb_servers()``.
PRIMARY_HOSTS = "127.0.0.1,127.0.0.2,127.0.0.3,127.0.0.4,127.0.0.5"
SECONDARY_HOSTS = "127.0.0.6,127.0.0.7,127.0.0.8,127.0.0.9,127.0.0.10"

# 50 tablets on the tracker table.
TRACKER_TABLE_TABLETS = 50

# Threshold: trip on the FIRST failed UPDATE.
# Election-window resilience is provided by the tracker CB's own
# per-UPDATE statement_timeout (10s — see demo/samples/tracker_table_cb.py),
# which is longer than YB's ~3s Raft election window. So a single-node
# stop → UPDATE waits through the election → succeeds → no trip.
MAX_UPDATE_FAILURES_ALLOWED = 0

# Fast probe so the demo doesn't wait minutes to detect the stop.
PROBE_INTERVAL_S = 4

# CB check wall-clock cap. Must exceed the CB's actual `check()`
# duration during a real failure — YB's default RPC timeout is 7s,
# so the tracker UPDATE against a partially-dead cluster can block
# ~5-7s. Default derivation (refresh_s // 2 = 2s) is too short and
# the probe would cut its own check off before the CB could return
# UNHEALTHY. 15s is a safe bound.
CHECK_TIMEOUT_S = 15

# Whether to attach the CB. Invocation #1 leaves this unset → app
# fails at first psycopg.connect() with MissingCircuitBreakerError.
ATTACH_CB = os.environ.get("ATTACH_CB") == "1"

# Drain window. 
DRAIN_TIMEOUT_S = int(os.environ.get("DRAIN_TIMEOUT_S", "0"))

WORKERS = int(os.environ.get("WORKERS", "30"))
WORKER_SLEEP_S = float(os.environ.get("WORKER_SLEEP_S", "10"))
POOL_SIZE = WORKERS

# Server-side statement_timeout on every pool conn. YB does NOT cancel
# a running query when the client socket closes (yugabyte-db#28983,
# #29379). Without a statement_timeout, drain's `conn.close()` closes
# the client socket but the backend keeps executing the query — and if
# that query is blocked on a broken tablet (post-failover), it never
# returns, so the backend never notices the socket is gone. Result:
# stale "active" client_backend entries linger on primary's rpcz for
# minutes. Setting statement_timeout has YB itself cancel the stuck
# query; the backend then hits its socket-read loop, sees the closed
# socket, and exits.
#
# Must be > (WORKER_SLEEP_S * 1000) so normal txns complete. Ceiling
# controls how quickly stale backends drop off the outgoing cluster
# post-failover. 15000ms with WORKER_SLEEP_S=10 gives 5s of headroom
# for the INSERT + COMMIT wrapping the sleep.
STATEMENT_TIMEOUT_MS = int(
    os.environ.get("STATEMENT_TIMEOUT_MS", str(int(WORKER_SLEEP_S * 1000) + 5000))
)

DSN = (
    f"host={PRIMARY_HOSTS} port=5433 user=yugabyte dbname=yugabyte "
    f"load_balance_hosts=true "
    f"connect_timeout=3 "
    f"options='-c statement_timeout={STATEMENT_TIMEOUT_MS}' "
    f"yb.failover.secondaryClusterHosts={SECONDARY_HOSTS} "
    f"yb_servers_refresh_interval={PROBE_INTERVAL_S} "
    f"yb.failover.checkTimeoutSecs={CHECK_TIMEOUT_S} "
    f"yb.failover.drainTimeoutSecs={DRAIN_TIMEOUT_S} "
    f"yb.failover.cooldownSecs=0"
)


# --------------------------------------------------------------------- logging

# Keep the log surface tight: only one demo-owned line announces the
# failover / failback ("primary unhealthy → switched to secondary in
# X.Xs"). Everything else the driver/CB/pool would log at WARNING
# during transition is muted so the operator has a single line to read.
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)-7s] %(name)-30s %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
for name in (
    "psycopg.yb.registry",
    "psycopg.yb.health_probe",
    "psycopg.yb",
    "psycopg.pool",
):
    logging.getLogger(name).setLevel(logging.ERROR)
# Tracker CB stays at WARNING so its "UPDATE failed / UPDATE succeeded"
# transition lines print — one for detection, one for recovery.
logging.getLogger("demo.samples.tracker_table_cb").setLevel(logging.WARNING)
# `psycopg.yb.drain` stays at WARNING so the operator sees the driver's
# "force-closed N in-flight conn(s)" line when drain fires — that's the
# receipt that in-flight conns actually got killed.
logging.getLogger("psycopg.yb.drain").setLevel(logging.WARNING)


# --------------------------------------------------------------------- worker

class Worker(threading.Thread):
    """Long-running-transaction worker. Loops forever: BEGIN;
    INSERT/UPSERT; sleep; COMMIT — one txn at a time, per worker.
    Prints the host it landed on and per-txn timing so failover / failback
    are visible in the app logs."""

    def __init__(self, worker_id: int, pool: ConnectionPool,
                 stop_evt: threading.Event) -> None:
        super().__init__(daemon=True)
        self.worker_id = worker_id
        self.pool = pool
        self.stop_evt = stop_evt
        self.txn_count = 0
        self.error_count = 0
        self.last_host: Optional[str] = None
        self.last_cluster: Optional[str] = None

    def run(self) -> None:
        while not self.stop_evt.is_set():
            t0 = time.monotonic()
            try:
                with self.pool.connection() as c:
                    self.last_host = c._yb_host
                    self.last_cluster = c._yb_cluster
                    with c.cursor() as cur:
                        cur.execute("BEGIN")
                        cur.execute(
                            "INSERT INTO failover_demo (id, ts, payload) "
                            "VALUES (%s, NOW(), %s) "
                            "ON CONFLICT (id) DO UPDATE SET ts = NOW()",
                            (self.worker_id, f"w{self.worker_id}"),
                        )
                        cur.execute(f"SELECT pg_sleep({WORKER_SLEEP_S})")
                        cur.execute("COMMIT")
                elapsed_ms = (time.monotonic() - t0) * 1000
                self.txn_count += 1
                print(
                    f"{time.strftime('%H:%M:%S')}  "
                    f"worker#{self.worker_id} ✓ txn #{self.txn_count} on "
                    f"{self.last_host} ({self.last_cluster}) "
                    f"in {elapsed_ms:.0f}ms",
                    flush=True,
                )
            except Exception as e:
                self.error_count += 1
                elapsed_ms = (time.monotonic() - t0) * 1000
                print(
                    f"{time.strftime('%H:%M:%S')}  "
                    f"worker#{self.worker_id} ✗ txn error after "
                    f"{elapsed_ms:.0f}ms on {self.last_host}: "
                    f"{type(e).__name__}: {str(e)[:100]}",
                    flush=True,
                )
                time.sleep(0.5)


# --------------------------------------------------------------------- app

def start_app() -> None:
    ClusterRegistry.instance().clear()

    # bootstrap ALWAYS runs; the CB attach is what's gated by ATTACH_CB.
    group = bootstrap_failover_group(DSN)

    if ATTACH_CB:
        group.primary_circuit_breaker = TrackerTableCircuitBreaker(
            which_cluster="primary",
            tracker_table_tablets=TRACKER_TABLE_TABLETS,
            max_update_failures_allowed=MAX_UPDATE_FAILURES_ALLOWED,
        )
        group.secondary_circuit_breaker = TrackerTableCircuitBreaker(
            which_cluster="secondary",
            tracker_table_tablets=TRACKER_TABLE_TABLETS,
            max_update_failures_allowed=MAX_UPDATE_FAILURES_ALLOWED,
        )
        print(
            f"attached TrackerTableCircuitBreaker to both slots "
            f"(tablets={TRACKER_TABLE_TABLETS}, "
            f"threshold={MAX_UPDATE_FAILURES_ALLOWED + 1}) "
            f"drain_timeout_s={DRAIN_TIMEOUT_S}",
            flush=True,
        )
    else: 
        # psycopg.connect() should raise MissingCircuitBreakerError.
        print(
            "\nATTACH_CB is not set — CB slots left as None.\n"
            "Next: any psycopg.connect() against this DSN will raise\n"
            "MissingCircuitBreakerError.\n",
            flush=True,
        )

    start_ts = time.monotonic()
    stop_evt = threading.Event()

    def _on_sigint(signum, frame):
        stop_evt.set()

    prev_handler = signal.signal(signal.SIGINT, _on_sigint)

    workers: list[Worker] = []
    try:
        # Fail-fast pre-check: try a single direct connect BEFORE
        # opening the pool. The pool wraps connect errors in retry
        # logic (up to `timeout` seconds) and would eventually raise
        # PoolTimeout — which hides the real cause. A direct connect
        # surfaces MissingCircuitBreakerError immediately.
        psycopg.connect(DSN).close()

        with ConnectionPool(
            DSN, check=xcluster_check,
            min_size=POOL_SIZE, max_size=POOL_SIZE, timeout=60,
        ) as pool:
            pool.wait()

            # Clean slate — DELETE (not TRUNCATE; YB blocks
            # table-rewrite DDL on xCluster source tables).
            with pool.connection() as c:
                with c.cursor() as cur:
                    cur.execute("DELETE FROM failover_demo")
                c.commit()

            for i in range(WORKERS):
                w = Worker(worker_id=i, pool=pool, stop_evt=stop_evt)
                w.start()
                workers.append(w)

            print(
                f"\nworkload running. group.primary_status="
                f"{group.primary_status.value}. Ctrl-C to stop.\n",
                flush=True,
            )

            # Main thread: watch group.primary_status / secondary_status
            # and log every transition with a wall-clock stamp so
            # "time taken from the logs" is right here.
            last_pri = group.primary_status
            while not stop_evt.is_set():
                if group.primary_status != last_pri:
                    cb = group.primary_circuit_breaker
                    now = time.monotonic()
                    # Time from CB's internal decision to the observable
                    # routing switch. That's exactly the drain window
                    # (drainTimeoutSecs), which is the number that
                    # varies across demo runs.
                    ts = time.strftime('%H:%M:%S')
                    if group.primary_status == HealthResult.UNHEALTHY:
                        drain_s = (
                            now - cb.last_trip_time
                            if cb and cb.last_trip_time else 0.0
                        )
                        print(
                            f"\n{ts}  ⚡ primary unhealthy — switched to "
                            f"secondary in {drain_s:.1f}s\n",
                            flush=True,
                        )
                    else:
                        drain_s = (
                            now - cb.last_recovery_time
                            if cb and cb.last_recovery_time else 0.0
                        )
                        print(
                            f"\n{ts}  ⚡ primary healthy — switched back to "
                            f"primary in {drain_s:.1f}s\n",
                            flush=True,
                        )
                    last_pri = group.primary_status
                time.sleep(0.25)
    except MissingCircuitBreakerError as e:
        print(f"\n✗ {e}\n", flush=True)
        return
    finally:
        signal.signal(signal.SIGINT, prev_handler)
        stop_evt.set()
        for w in workers:
            w.join(timeout=15)
        ClusterRegistry.instance().clear()


# --------------------------------------------------------------------- entry

def main() -> int:
    bar = "=" * 72
    print(f"{bar}\n  xCluster failover demo\n{bar}", flush=True)
    print(f"  primary        : {PRIMARY_HOSTS}", flush=True)
    print(f"  secondary      : {SECONDARY_HOSTS}", flush=True)
    print(f"  attach_cb      : {ATTACH_CB}", flush=True)
    print(f"  drain_timeout_s: {DRAIN_TIMEOUT_S}", flush=True)
    print(f"  workers        : {WORKERS}, sleep {WORKER_SLEEP_S}s per txn",
          flush=True)
    print(
        f"\n  prereq: demo/setup.sh has been run (2× RF=3 5-node clusters).",
        flush=True,
    )
    start_app()
    return 0


if __name__ == "__main__":
    sys.exit(main())
