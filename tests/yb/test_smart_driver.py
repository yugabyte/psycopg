"""
Integration tests for the YugabyteDB smart driver (sync surface).

Requires a real cluster, reached via `PSYCOPG_YB_TEST_DSN`. The `yb_cluster`
fixture confirms `yb_servers()` is available before any test runs.

Test cluster on this machine is 3 nodes all in `cloud1.datacenter1.rack1`,
so topology-aware tests use either that exact placement or a non-matching
one to verify the filter behaviour. Tests requiring multi-zone placements
are gated on cluster shape — they skip rather than fail when the cluster
doesn't support them.

Async-only scenarios (cancellation, gather, mixed sync+async) live in
`test_smart_driver_async.py`.
"""

# Copyright (C) 2026 Yugabyte

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

import pytest

import psycopg


# --------------------------------------------------------------------- pass-through paths
#
# NOTE: `load_balance_hosts=disable` and `load_balance_hosts=random` are
# libpq parameters (added in PostgreSQL 16). On older libpq (14, common in
# current YB builds) those values are rejected at parse time. v1 sticks to
# `load_balance_hosts=false` and "no param" for the off-semantic, both of
# which we strip before libpq sees them. Whether to ALSO strip `disable`/
# `random` for libpq-version-independent UX is an open design question —
# see the libpq-compat discussion in the plan file.


def test_load_balance_false_is_passthrough(yb_cluster):
    """`load_balance_hosts=false` (our extension value meaning "off") behaves
    identically to upstream's `disable`. Stripped from conninfo before libpq."""
    from psycopg.yb.registry import ClusterRegistry
    registry = ClusterRegistry.instance()
    assert len(registry._clusters) == 0

    with psycopg.connect(yb_cluster + " load_balance_hosts=false") as conn:
        assert conn._yb_uuid is None

    assert len(registry._clusters) == 0


def test_load_balance_absent_is_passthrough(yb_cluster):
    """No `load_balance_hosts` param at all → upstream behaviour, no smart driver."""
    with psycopg.connect(yb_cluster) as conn:
        assert conn._yb_uuid is None


# --------------------------------------------------------------------- smart driver: basic balance


def test_smart_driver_distributes_evenly(yb_cluster, assert_balanced):
    """24 connections to a 3-node cluster with `load_balance_hosts=true`
    should land 8 per host. Both driver-side counter and `/rpcz` agree."""
    conns = []
    try:
        for _ in range(24):
            conns.append(psycopg.connect(yb_cluster + " load_balance_hosts=true"))

        # All conns share a single cluster uuid; pull it from the first.
        uuid = conns[0]._yb_uuid
        assert uuid is not None
        assert_balanced(uuid, {
            "127.0.0.1": 8,
            "127.0.0.2": 8,
            "127.0.0.3": 8,
        })
    finally:
        for c in conns:
            c.close()


def test_smart_driver_tags_connection_with_uuid_and_host(yb_cluster):
    """A successful smart-driver connect must tag the instance with both uuid
    and host so `close()` can decrement the right counter."""
    with psycopg.connect(yb_cluster + " load_balance_hosts=true") as conn:
        assert conn._yb_uuid is not None
        assert conn._yb_host in ("127.0.0.1", "127.0.0.2", "127.0.0.3")


# --------------------------------------------------------------------- single-host bootstrap


def test_single_host_bootstrap(yb_cluster, assert_balanced):
    """A single contact point is sufficient — once it answers we discover
    the rest of the cluster via `yb_servers()` and load-balance across all."""
    # Strip multi-host config (if any) to one contact point.
    dsn = "host=127.0.0.1 port=5433 user=yugabyte dbname=yugabyte load_balance_hosts=true"
    conns = []
    try:
        for _ in range(12):
            conns.append(psycopg.connect(dsn))
        uuid = conns[0]._yb_uuid
        # 12 conns / 3 nodes = exact 4/4/4 — atomic reservation under the
        # per-cluster lock guarantees no drift on the driver side; `assert_balanced`
        # accounts for the control conn's +1 on whichever host hosts it.
        assert_balanced(uuid, {
            "127.0.0.1": 4,
            "127.0.0.2": 4,
            "127.0.0.3": 4,
        })
    finally:
        for c in conns:
            c.close()


# --------------------------------------------------------------------- topology


def test_topology_exact_match(yb_multi_zone_cluster, assert_balanced):
    """Multi-zone cluster: 2 nodes in zoneA (127.0.0.1, .2), 1 in zoneB (.3).
    With `topology_keys=cloud1.datacenter1.zoneA`, only the two zoneA nodes
    are eligible. 12 conns must distribute exactly 6/6 across them; the
    zoneB node must receive ZERO traffic.

    This is the real topology-filter test — it would FAIL (with ~4/4/4)
    if the filter were a no-op."""
    dsn = (yb_multi_zone_cluster
           + " load_balance_hosts=true topology_keys=cloud1.datacenter1.zoneA")
    conns = []
    try:
        for _ in range(12):
            conns.append(psycopg.connect(dsn))
        uuid = conns[0]._yb_uuid
        assert_balanced(uuid, {
            "127.0.0.1": 6,
            "127.0.0.2": 6,
            "127.0.0.3": 0,
        })
    finally:
        for c in conns:
            c.close()


def test_topology_wildcard_zone(yb_multi_zone_cluster, assert_balanced):
    """Multi-zone cluster as above. `topology_keys=cloud1.datacenter1.*`
    wildcard-matches BOTH zoneA and zoneB, so all three nodes are eligible
    and 12 conns distribute 4/4/4. Proves the wildcard parsing path runs
    and that wildcards genuinely match multiple zones (not just one).

    Compare with `test_topology_exact_match` (above): same cluster, same
    conn count, but the exact-zone filter rejects zoneB → 6/6/0; the
    wildcard matches both zones → 4/4/4."""
    dsn = (yb_multi_zone_cluster
           + " load_balance_hosts=true topology_keys=cloud1.datacenter1.*")
    conns = []
    try:
        for _ in range(12):
            conns.append(psycopg.connect(dsn))
        uuid = conns[0]._yb_uuid
        assert_balanced(uuid, {
            "127.0.0.1": 4,
            "127.0.0.2": 4,
            "127.0.0.3": 4,
        })
    finally:
        for c in conns:
            c.close()


def test_topology_no_match_fails(yb_cluster):
    """When no live node matches the topology keys, smart-driver connect must
    raise OperationalError (no cluster-wide fallback in v1)."""
    dsn = yb_cluster + " load_balance_hosts=true topology_keys=aws.us-west.us-west-1a"
    with pytest.raises(psycopg.OperationalError, match="no eligible"):
        psycopg.connect(dsn)


def test_topology_invalid_cloud_wildcard_rejected_at_parse(yb_cluster):
    """`*.region.zone` is rejected at conninfo-parse time (cloud wildcard
    isn't allowed). We surface this as a ValueError from extract_yb_params."""
    dsn = yb_cluster + " load_balance_hosts=true topology_keys=*.foo.bar"
    with pytest.raises(ValueError, match="cloud or region"):
        psycopg.connect(dsn)


# --------------------------------------------------------------------- close / counter lifecycle


def test_close_decrements_counter(yb_cluster):
    """A successful close on a smart-driver connection must decrement the
    per-host counter to its pre-connect value."""
    from psycopg.yb.registry import ClusterRegistry
    registry = ClusterRegistry.instance()

    conn = psycopg.connect(yb_cluster + " load_balance_hosts=true")
    host = conn._yb_host
    uuid = conn._yb_uuid
    # On a fresh cluster the connect we just opened is the only one on that
    # host, so the counter must be exactly 1.
    assert registry.get_load(uuid, host) == 1

    conn.close()
    assert registry.get_load(uuid, host) == 0


def test_close_on_passthrough_doesnt_touch_registry(yb_cluster):
    """Closing a connection that never went through the smart-driver path
    must not call decrement (guarded by `if self._yb_uuid`). No crash, no
    counter mutation."""
    from psycopg.yb.registry import ClusterRegistry
    # Open a smart-driver conn first so a cluster state exists in the registry.
    sentinel = psycopg.connect(yb_cluster + " load_balance_hosts=true")
    uuid = sentinel._yb_uuid
    host = sentinel._yb_host
    baseline = ClusterRegistry.instance().get_load(uuid, host)

    # Open and close a PASS-THROUGH connection. Its host happens to be the
    # same physical machine but it's not tagged with _yb_uuid.
    plain = psycopg.connect(yb_cluster)
    assert plain._yb_uuid is None
    plain.close()

    # The passthrough close must not have touched any counter.
    assert ClusterRegistry.instance().get_load(uuid, host) == baseline
    sentinel.close()


# --------------------------------------------------------------------- /rpcz cross-check


def test_rpcz_vs_driver_counter_agreement(yb_cluster, assert_balanced):
    """Cross-check per-host driver counters against /rpcz "client backend"
    counts — both EXACT, per-host. 9 conns over a 3-node cluster must split
    3/3/3 on the driver side AND show 3/3/3 on /rpcz (plus a single +1 on
    the host hosting the control connection — `assert_balanced` accounts
    for that explicitly, no tolerance).

    This is the same technique pgjdbc-yb uses in `verifyOn()`, applied
    per-host: distribution must be uniform AND the driver counter must be
    a faithful proxy for what the server actually sees.
    """
    conns = [psycopg.connect(yb_cluster + " load_balance_hosts=true") for _ in range(9)]
    try:
        uuid = conns[0]._yb_uuid
        assert_balanced(uuid, {
            "127.0.0.1": 3,
            "127.0.0.2": 3,
            "127.0.0.3": 3,
        })
    finally:
        for c in conns:
            c.close()


# --------------------------------------------------------------------- concurrency


def test_concurrent_connection_creation(yb_cluster, assert_balanced):
    """30 connections opened concurrently must distribute across the 3 nodes.
    Stress test for the locking + counter increment paths."""
    def open_one():
        return psycopg.connect(yb_cluster + " load_balance_hosts=true")

    conns: list[psycopg.Connection] = []
    try:
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(open_one) for _ in range(30)]
            for f in futures:
                conns.append(f.result())

        uuid = conns[0]._yb_uuid
        # Exact 10/10/10 on both sides: `pick_and_reserve` is atomic under
        # the per-cluster lock so concurrent threads can't desync the driver
        # counter; `assert_balanced` accounts for the control conn's +1 on
        # the host hosting it.
        assert_balanced(uuid, {
            "127.0.0.1": 10,
            "127.0.0.2": 10,
            "127.0.0.3": 10,
        })
    finally:
        for c in conns:
            c.close()
