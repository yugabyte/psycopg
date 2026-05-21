"""
Pool integration tests for the YugabyteDB smart driver (sync surface).

These verify that connections opened by `psycopg_pool.ConnectionPool` go
through the smart-driver dispatcher exactly the same way as direct
`psycopg.connect(...)` calls. The pool sits *above* the dispatcher — it
just calls `psycopg.connect(dsn)` whenever it needs a new connection — so
the per-host distribution, topology filtering, and counter lifecycle all
have to behave identically.

The shape of each test mirrors the corresponding direct-connect test in
`test_smart_driver.py`. Where a direct test opens N connections in a loop
and asserts a per-host count, the pool variant configures the pool with
`min_size=max_size=N` and lets the pool open them. The driver-side
expectations are the same.

Async pool variants live in `test_smart_driver_pool_async.py`.
"""

# Copyright (C) 2026 Yugabyte

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

import pytest

import psycopg

# Skip the whole file if psycopg-pool isn't available. The smart driver
# doesn't require it; these tests do.
psycopg_pool = pytest.importorskip("psycopg_pool")
from psycopg_pool import ConnectionPool  # noqa: E402


# --------------------------------------------------------------------- pass-through


def test_pool_load_balance_false_is_passthrough(yb_cluster):
    """`load_balance_hosts=false` on the pool's conninfo → no smart-driver
    code touched, registry stays empty even after the pool opens its
    minimum connections."""
    from psycopg.yb.registry import ClusterRegistry
    registry = ClusterRegistry.instance()
    assert len(registry._clusters) == 0

    with ConnectionPool(
        yb_cluster + " load_balance_hosts=false",
        min_size=3, max_size=3,
    ) as pool:
        pool.wait()  # block until min_size conns are open
        with pool.connection() as conn:
            assert conn._yb_uuid is None

    assert len(registry._clusters) == 0


# --------------------------------------------------------------------- basic balance


def test_pool_smart_driver_distributes_evenly(yb_cluster, assert_balanced):
    """Pool of size 12 over a 3-node cluster: the 12 connections the pool
    opens must distribute exactly 4/4/4. `assert_balanced` cross-checks
    the driver counter and `/rpcz` (with control-conn accounting)."""
    with ConnectionPool(
        yb_cluster + " load_balance_hosts=true",
        min_size=12, max_size=12,
    ) as pool:
        pool.wait()
        # Borrow one to grab the uuid; smart driver tagging works inside
        # pool-managed connections too.
        with pool.connection() as conn:
            uuid = conn._yb_uuid
            assert uuid is not None
            assert conn._yb_host in ("127.0.0.1", "127.0.0.2", "127.0.0.3")
        assert_balanced(uuid, {
            "127.0.0.1": 4,
            "127.0.0.2": 4,
            "127.0.0.3": 4,
        })


# --------------------------------------------------------------------- topology


def test_pool_topology_exact_match(yb_multi_zone_cluster, assert_balanced):
    """Pool against the multi-zone cluster (2 zoneA + 1 zoneB), with
    `topology_keys=zoneA`. The pool's 12 conns must distribute exactly
    6/6 across the two zoneA nodes; the zoneB node must receive zero
    traffic. This is the real topology-filter test for the pool path —
    proves the dispatcher's topology filter runs underneath the pool."""
    dsn = (yb_multi_zone_cluster
           + " load_balance_hosts=true topology_keys=cloud1.datacenter1.zoneA")
    with ConnectionPool(dsn, min_size=12, max_size=12) as pool:
        pool.wait()
        with pool.connection() as conn:
            uuid = conn._yb_uuid
        assert_balanced(uuid, {
            "127.0.0.1": 6,
            "127.0.0.2": 6,
            "127.0.0.3": 0,
        })


def test_pool_topology_no_match_fails(yb_cluster):
    """Pool with `topology_keys` matching no nodes: the pool's attempt to
    open its initial connections must fail. We assert that the pool either
    errors at open() or that `wait()` raises — either is acceptable; what
    matters is that the no-eligible-node error from the dispatcher surfaces
    through the pool."""
    dsn = yb_cluster + " load_balance_hosts=true topology_keys=aws.us-west.us-west-1a"
    # Use a short wait timeout so the test doesn't hang if the failure
    # mode is "blocked forever waiting for a healthy conn".
    pool = ConnectionPool(dsn, min_size=1, max_size=1, open=False)
    with pytest.raises(Exception):
        pool.open(wait=True, timeout=10)
    try:
        pool.close()
    except Exception:
        pass


# --------------------------------------------------------------------- borrow + return


def test_pool_borrow_does_not_change_counter(yb_cluster):
    """Borrowing a connection from the pool and returning it must NOT mutate
    the per-host counter. The counter tracks open-on-server connections, not
    in-flight checkouts. Open the pool with 12 conns; borrow + release a few
    times; assert counts are unchanged."""
    from psycopg.yb.registry import ClusterRegistry
    registry = ClusterRegistry.instance()

    with ConnectionPool(
        yb_cluster + " load_balance_hosts=true",
        min_size=12, max_size=12,
    ) as pool:
        pool.wait()
        with pool.connection() as conn:
            uuid = conn._yb_uuid
        before = {
            h: registry.get_load(uuid, h)
            for h in ("127.0.0.1", "127.0.0.2", "127.0.0.3")
        }
        # Borrow + release a few times.
        for _ in range(20):
            with pool.connection() as conn:
                conn.execute("SELECT 1")
        after = {
            h: registry.get_load(uuid, h)
            for h in ("127.0.0.1", "127.0.0.2", "127.0.0.3")
        }
        assert before == after, (
            f"borrow/return should not mutate counters; before={before}, "
            f"after={after}"
        )


# --------------------------------------------------------------------- close lifecycle


def test_pool_close_drains_counter(yb_cluster):
    """When the pool closes, every pool-managed connection's close path
    runs, which the smart driver hooks to decrement. After the pool's
    context-manager exit all per-host counts must be zero again."""
    from psycopg.yb.registry import ClusterRegistry
    registry = ClusterRegistry.instance()

    uuid: str | None = None
    with ConnectionPool(
        yb_cluster + " load_balance_hosts=true",
        min_size=9, max_size=9,
    ) as pool:
        pool.wait()
        with pool.connection() as conn:
            uuid = conn._yb_uuid
        # Mid-life: 9 conns alive, exactly 3 per host.
        mid = {
            h: registry.get_load(uuid, h)
            for h in ("127.0.0.1", "127.0.0.2", "127.0.0.3")
        }
        assert mid == {"127.0.0.1": 3, "127.0.0.2": 3, "127.0.0.3": 3}

    # Pool has been closed by the `with` block — every conn went through
    # the smart driver's close path. Counters must be back to zero.
    after = {
        h: registry.get_load(uuid, h)
        for h in ("127.0.0.1", "127.0.0.2", "127.0.0.3")
    }
    assert after == {"127.0.0.1": 0, "127.0.0.2": 0, "127.0.0.3": 0}, (
        f"pool close should drain all counters back to 0; got {after}"
    )


# --------------------------------------------------------------------- growth


def test_pool_growth_distributes_via_smart_driver(yb_cluster, assert_balanced):
    """Pool starts at `min_size=3` and is allowed to grow to `max_size=12`.
    Concurrent borrowers force the pool to grow; the new connections the
    pool opens must go through the dispatcher and distribute exactly. End
    state: 12 conns, 4/4/4 across the 3 nodes."""
    def hold_one(pool, barrier, release_after_s=2.0):
        with pool.connection() as conn:
            barrier.wait()  # sync so all 12 are checked out simultaneously
            time.sleep(release_after_s)
            return conn._yb_host

    import threading
    barrier = threading.Barrier(12)
    with ConnectionPool(
        yb_cluster + " load_balance_hosts=true",
        min_size=3, max_size=12,
    ) as pool:
        pool.wait()
        # Force pool to grow to max_size by holding 12 conns at once.
        with ThreadPoolExecutor(max_workers=12) as executor:
            futures = [executor.submit(hold_one, pool, barrier) for _ in range(12)]
            for f in futures:
                f.result()
        # All 12 conns are now back in the pool, still alive (max_size==12,
        # min_size==3 but the pool only shrinks lazily; on this fast test
        # the shrink hasn't run yet).
        with pool.connection() as conn:
            uuid = conn._yb_uuid
        # Driver counter sums to 12 (the open conns).
        from psycopg.yb.registry import ClusterRegistry
        registry = ClusterRegistry.instance()
        total = sum(
            registry.get_load(uuid, h)
            for h in ("127.0.0.1", "127.0.0.2", "127.0.0.3")
        )
        assert total == 12
        assert_balanced(uuid, {
            "127.0.0.1": 4,
            "127.0.0.2": 4,
            "127.0.0.3": 4,
        })


# --------------------------------------------------------------------- mixed pool + direct


def test_pool_and_direct_connect_share_registry(yb_cluster):
    """A pool-managed connection and a direct `psycopg.connect()` to the same
    cluster from the same process share one process-wide `ClusterRegistry`.
    Mutations from either side affect the same per-host counters."""
    from psycopg.yb.registry import ClusterRegistry
    registry = ClusterRegistry.instance()

    with ConnectionPool(
        yb_cluster + " load_balance_hosts=true",
        min_size=3, max_size=3,
    ) as pool:
        pool.wait()
        with pool.connection() as conn:
            uuid = conn._yb_uuid
        snapshot_after_pool = {
            h: registry.get_load(uuid, h)
            for h in ("127.0.0.1", "127.0.0.2", "127.0.0.3")
        }
        assert sum(snapshot_after_pool.values()) == 3

        # Now a direct connect against the same cluster: must hit the same
        # ClusterState and bump exactly one host's counter by exactly 1.
        direct = psycopg.connect(yb_cluster + " load_balance_hosts=true")
        try:
            assert direct._yb_uuid == uuid
            chosen = direct._yb_host
            snapshot_after_both = {
                h: registry.get_load(uuid, h)
                for h in ("127.0.0.1", "127.0.0.2", "127.0.0.3")
            }
            for h in ("127.0.0.1", "127.0.0.2", "127.0.0.3"):
                expected = snapshot_after_pool[h] + (1 if h == chosen else 0)
                assert snapshot_after_both[h] == expected, (
                    f"{h}: counter is {snapshot_after_both[h]}, expected "
                    f"{expected}. Direct conn landed on {chosen}; before="
                    f"{snapshot_after_pool}, after={snapshot_after_both}"
                )
        finally:
            direct.close()
