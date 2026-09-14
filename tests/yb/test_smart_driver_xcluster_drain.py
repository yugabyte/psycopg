"""xCluster integration tests for the barrier-with-timeout drain
(Phase E of the implementation plan).

Complements ``test_smart_driver_xcluster_data.py`` (proves data replicates
across the pair) and ``test_smart_driver_xcluster_cb.py`` (proves the CB
trips on real cluster failure). The tests here exercise the specific
Phase E promise: **no dual-writes across a failover barrier**.

Novel primitive — the dual-write detector:
  * Seed a table with N deterministic rows.
  * Client-side workload: continuous ``BEGIN; UPDATE t SET val=val+1
    WHERE id=?; COMMIT``. Every successful commit is logged with
    ``(row_id, conn._yb_cluster, wall_time)``.
  * Trigger failover mid-workload via ``force_primary_status``.
  * Post-drain, verify the per-row commit log has NO overlap: for every
    row, primary-tagged commits all precede secondary-tagged commits.
    A single out-of-order pair proves the driver routed to primary AFTER
    it had committed to serving secondary — the exact silent-corruption
    class §2 identifies.

Auto-tagged ``yb`` (integration) via the ``test_smart_driver*`` filename
rule in conftest.
"""

# Copyright (C) 2026 Yugabyte

from __future__ import annotations

import random
import threading
import time
from dataclasses import dataclass

import pytest

import psycopg
from psycopg.yb.health import HealthResult
from psycopg.yb.registry import ClusterRegistry


PRIMARY_HOSTS = "127.0.0.1,127.0.0.2,127.0.0.3"
SECONDARY_HOSTS = "127.0.0.4,127.0.0.5,127.0.0.6"
PROBE_INTERVAL_S = 3

_DSN = (
    f"host={PRIMARY_HOSTS} port=5433 user=yugabyte dbname=yugabyte "
    f"load_balance_hosts=true "
    f"yb.failover.secondaryClusterHosts={SECONDARY_HOSTS} "
    f"yb_servers_refresh_interval={PROBE_INTERVAL_S} "
    f"yb.failover.cooldownSecs=0 "
    f"yb.failover.drainTimeoutSecs=3 "
    # Statement timeout aligned with drain (§3.4) so any in-flight
    # server-side query is bounded before the drain deadline.
    "options='-c statement_timeout=3000'"
)

# Table dedicated to this suite. Kept separate from the shared
# xcluster_test_data so tests don't step on each other's data.
_ROWS = 10   # seed 10 rows, ids 0..9
_WORKLOAD_DURATION_S = 6.0
_DRAIN_TIMEOUT_S = 3
_WORKERS = 6


@dataclass
class _CommitLog:
    row_id: int
    cluster: str        # "primary" | "secondary"
    ts: float           # time.monotonic() at commit


def _raw_dsn_for(host: str) -> str:
    return (
        f"host={host} port=5433 user=yugabyte dbname=yugabyte "
        f"connect_timeout=5"
    )


def _reset_table(host: str) -> None:
    """(Re)create + seed the workload table on ``host``. Uses raw psycopg
    (no smart driver) so setup isn't tangled with the CB path.

    Note: YB blocks TABLE-rewriting DDL on xCluster source tables. We
    keep the table across runs and reset via DELETE + re-INSERT to work
    within that constraint (same pattern as test_smart_driver_xcluster_data)."""
    with psycopg.connect(_raw_dsn_for(host)) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "CREATE TABLE IF NOT EXISTS xcluster_drain_test ("
                "  id  INT PRIMARY KEY,"
                "  val BIGINT NOT NULL DEFAULT 0"
                ")"
            )
        conn.commit()
        with conn.cursor() as cur:
            cur.execute("DELETE FROM xcluster_drain_test")
            for i in range(_ROWS):
                cur.execute(
                    "INSERT INTO xcluster_drain_test (id, val) VALUES (%s, 0)",
                    (i,),
                )
        conn.commit()


def _wait_replicated_baseline(host: str, timeout_s: float = 20.0) -> None:
    """Wait for the secondary to see the 10-row baseline."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        with psycopg.connect(_raw_dsn_for(host)) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT count(*) FROM xcluster_drain_test")
                row = cur.fetchone()
                if row is not None and row[0] == _ROWS:
                    return
        time.sleep(0.5)
    raise AssertionError(
        f"secondary did not observe baseline {_ROWS} rows within {timeout_s}s"
    )


def _bootstrap_group():
    """Open a throwaway conn to force FailoverGroup bootstrap; return
    the group handle."""
    conn = psycopg.connect(_DSN)
    try:
        group = ClusterRegistry.instance().get_failover_group_by_uuid(
            conn._yb_uuid
        )
        assert group is not None
        return group
    finally:
        conn.close()


def _wait_for_dispatch_pointing_at(group, cluster: str, timeout_s: float) -> bool:
    """Wait for ``active_cluster(group)`` to return ``cluster``. Used
    after ``force_primary_status`` to sync the workload with the flip."""
    from psycopg.yb.state import active_cluster
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        with group.lock:
            paused = group.dispatch_paused
        if not paused and active_cluster(group) == cluster:
            return True
        time.sleep(0.05)
    return False


# --------------------------------------------------------------------- workload

def _run_workload(
    stop_event: threading.Event,
    log: list[_CommitLog],
    log_lock: threading.Lock,
) -> None:
    """One worker thread: continuously opens a conn, increments a random
    row, commits, logs. On OperationalError (drain force-close, no
    viable cluster during transient windows) just retries — the test
    proves the LOG's invariant, not that every attempt succeeded."""
    while not stop_event.is_set():
        row_id = random.randrange(_ROWS)
        try:
            with psycopg.connect(_DSN) as conn:
                cluster = conn._yb_cluster
                with conn.cursor() as cur:
                    cur.execute(
                        "UPDATE xcluster_drain_test SET val = val + 1 "
                        "WHERE id = %s",
                        (row_id,),
                    )
                conn.commit()
            # Log AFTER a successful commit — capturing the cluster tag
            # observed at connect time, and the wall time at commit.
            with log_lock:
                log.append(_CommitLog(
                    row_id=row_id,
                    cluster=cluster or "unknown",
                    ts=time.monotonic(),
                ))
        except psycopg.OperationalError:
            # Force-close during drain, or transient no-viable window.
            # Real apps would surface this; the test just retries.
            continue
        except Exception:
            # Any other error — bail out of this worker (test framework
            # will pick it up if it matters).
            continue


# --------------------------------------------------------------------- the test

def test_no_dual_writes_across_forced_failover(
    yb_xcluster_clusters, yb_xcluster_ctl,   # yb_xcluster_ctl gates yb-ctl availability
):
    """Dual-write detector, the novel primitive of Phase E.

    Runs a concurrent UPDATE-increment workload against the primary,
    force-flips ``primary_status`` to UNHEALTHY partway through, and
    proves the per-row commit log has NO temporal overlap between
    primary-tagged and secondary-tagged writes on the same row.
    """
    primary_host = PRIMARY_HOSTS.split(",")[0]
    secondary_host = SECONDARY_HOSTS.split(",")[0]

    _reset_table(primary_host)
    _wait_replicated_baseline(secondary_host)

    group = _bootstrap_group()

    log: list[_CommitLog] = []
    log_lock = threading.Lock()
    stop_event = threading.Event()

    workers = [
        threading.Thread(
            target=_run_workload, args=(stop_event, log, log_lock),
            daemon=True,
        )
        for _ in range(_WORKERS)
    ]
    for w in workers:
        w.start()

    try:
        # Let the workload warm up so we get a meaningful primary commit
        # burst before flipping.
        time.sleep(_WORKLOAD_DURATION_S / 3)

        flip_start = time.monotonic()
        # Trigger failover via the test hook. In production this happens
        # via a CB tick + trigger_drain — same drain path, same
        # dispatch_paused barrier.
        from psycopg.yb.drain import trigger_drain
        trigger_drain(
            group,
            which_cluster="primary",
            new_status=HealthResult.UNHEALTHY,
            drain_timeout_s=_DRAIN_TIMEOUT_S,
        )
        flip_end = time.monotonic()

        # Sync the workload — wait for the dispatcher to actually route
        # to secondary. dispatch_paused is already cleared by trigger_drain.
        assert _wait_for_dispatch_pointing_at(
            group, "secondary", timeout_s=5.0,
        ), "dispatcher didn't flip to secondary within 5s"

        # Continue the workload past the flip so we accumulate secondary
        # writes.
        time.sleep(_WORKLOAD_DURATION_S / 2)
    finally:
        stop_event.set()
        for w in workers:
            w.join(timeout=5.0)

    # --------- Analysis
    #
    # Bucket per-row commits by cluster. The invariant: for every row,
    # the last primary commit's timestamp is less than the first
    # secondary commit's timestamp. Equivalently, there's no
    # interleaving.
    per_row_primary: dict[int, list[float]] = {}
    per_row_secondary: dict[int, list[float]] = {}
    for entry in log:
        (per_row_primary if entry.cluster == "primary"
         else per_row_secondary).setdefault(entry.row_id, []).append(entry.ts)

    assert log, "no commits logged — workload did not run"

    violations: list[str] = []
    for row_id in range(_ROWS):
        pri = sorted(per_row_primary.get(row_id, []))
        sec = sorted(per_row_secondary.get(row_id, []))
        if not pri or not sec:
            continue   # row saw only one side; no invariant to check
        last_primary = pri[-1]
        first_secondary = sec[0]
        if last_primary > first_secondary:
            # Any primary commit AFTER any secondary commit on the same
            # row is a dual-write. Reported for context; fail below.
            violations.append(
                f"row {row_id}: last primary commit at "
                f"{last_primary:.3f}s vs first secondary commit at "
                f"{first_secondary:.3f}s "
                f"(primary count={len(pri)}, secondary count={len(sec)})"
            )

    assert not violations, (
        f"dual-write invariant violated on {len(violations)} row(s): "
        f"{violations}"
    )

    # Additional sanity: at least some rows should have seen writes on
    # both clusters (otherwise the test didn't exercise the flip).
    both_sides = sum(
        1 for row_id in range(_ROWS)
        if row_id in per_row_primary and row_id in per_row_secondary
    )
    assert both_sides >= 1, (
        f"no row was written on both clusters — flip may not have "
        f"happened; log length={len(log)}, primary rows="
        f"{sorted(per_row_primary)}, secondary rows="
        f"{sorted(per_row_secondary)}"
    )
