"""
Pytest configuration and fixtures for the YugabyteDB smart-driver tests.

Three marker tiers, mirroring the markers we register here:

  * `yb_unit`  — pure-Python, no database, runs in milliseconds.
  * `yb`       — requires a real YugabyteDB cluster reachable via the
                 PSYCOPG_YB_TEST_DSN env var (or --yb-test-dsn option).
  * `perf`     — slow benchmark, opt-in via `pytest -m perf`.

Test files in this folder are auto-tagged with the appropriate marker through
`pytest_collection_modifyitems` so individual files don't need decorators.

For server-side ground-truth verification (cross-checking the driver's
connection-count map against what the cluster actually sees), the
`rpcz_count` helper scrapes `http://<host>:13000/rpcz` and counts
"client backend" occurrences — the same technique pgjdbc-yb's
`FallbackOptionsLBTest.verifyOn` uses in the JDBC suite.
"""

# Copyright (C) 2026 Yugabyte

from __future__ import annotations

import os
import sys
import shlex
import subprocess
import threading
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from typing import Any, Iterator

import pytest

# Re-export the anyio_backend fixture from the root conftest so async tests
# under tests/yb/ run on the same loop as the rest of the suite.
from ..conftest import asyncio_options

# The driver no longer ships default CB implementations — reference
# CBs live under demo/samples/. Put the repo root on sys.path so tests
# and integration harnesses can do `from demo.samples.tracker_table_cb
# import TrackerTableCircuitBreaker` without an editable install of the
# demo/ tree.
_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


# --------------------------------------------------------------------- markers
#
# Registering here means the markers are visible to `pytest --markers` and
# don't trip `filterwarnings = ["error"]` on unknown-marker warnings.

def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "yb_unit: smart-driver unit test (pure Python, no database)",
    )
    config.addinivalue_line(
        "markers",
        "yb: smart-driver integration test (requires PSYCOPG_YB_TEST_DSN)",
    )
    config.addinivalue_line(
        "markers",
        "perf: slow benchmark; opt-in via `pytest -m perf`",
    )


def pytest_collection_modifyitems(items):
    """Auto-mark tests by file naming convention.

    Files under `tests/yb/` that don't already carry an explicit marker get
    one inferred from their filename:

      test_params.py, test_node.py, test_policy.py, test_registry.py → yb_unit
      test_smart_driver*.py                                          → yb
      test_perf*.py                                                  → perf

    Note: we match against the FILENAME, not the full nodeid. The earlier
    version of this hook substring-matched the nodeid tail which includes
    the test function name — so a unit test function named
    `test_smart_driver_enabled_true` (legitimately living in test_params.py)
    got misclassified as an integration test and skipped under -m yb_unit.
    """
    for item in items:
        if "/yb/" not in item.nodeid:
            continue
        # `item.path.name` is the filename only, e.g. "test_params.py".
        filename = item.path.name
        if filename.startswith("test_perf"):
            item.add_marker(pytest.mark.perf)
        elif filename.startswith("test_smart_driver"):
            item.add_marker(pytest.mark.yb)
        elif filename.startswith(("test_params", "test_node", "test_policy",
                                  "test_registry")):
            item.add_marker(pytest.mark.yb_unit)


def pytest_addoption(parser):
    parser.addoption(
        "--yb-test-dsn",
        default=None,
        help=(
            "Connection string for a YugabyteDB cluster. Overrides "
            "PSYCOPG_YB_TEST_DSN env var. Required for tests marked `yb`."
        ),
    )


# --------------------------------------------------------------------- async


@pytest.fixture(
    params=[pytest.param(("asyncio", asyncio_options.copy()), id="asyncio")],
    scope="session",
)
def anyio_backend(request):
    backend, options = request.param
    if request.config.option.loop == "uvloop":
        options["use_uvloop"] = True
    return backend, options


# --------------------------------------------------------------------- DSN


@pytest.fixture(scope="session")
def yb_dsn(request):
    """Connection string for a real YugabyteDB cluster.

    Source priority: --yb-test-dsn option > PSYCOPG_YB_TEST_DSN env var
    > default (127.0.0.1:5433, the address the yb-ctl-created cluster
    listens on). Tests that take this fixture should be ones that don't
    create their own cluster; tests using `yb_cluster` / `yb_multi_zone_cluster`
    get a freshly-created cluster's DSN directly from those fixtures.
    """
    return (
        request.config.getoption("--yb-test-dsn")
        or os.environ.get("PSYCOPG_YB_TEST_DSN")
        or "host=127.0.0.1 port=5433 user=yugabyte dbname=yugabyte"
    )


_DEFAULT_DSN = "host=127.0.0.1 port=5433 user=yugabyte dbname=yugabyte"

# Single-zone baseline: 3 nodes, all in the same zone (one placement entry
# applies to all `--rf` nodes). Non-destructive integration tests use this.
SINGLE_ZONE_PLACEMENT = "cloud1.datacenter1.rack1"
SINGLE_ZONE_RF = 3

# Multi-zone shape: 2 nodes in zoneA, 1 in zoneB. yb-ctl assigns the Nth
# placement entry to the Nth node. The two-in-zoneA arrangement lets the
# topology-matrix tests stop a node in the bound topology and still observe
# traffic redistributing to the remaining in-topology node.
MULTI_ZONE_PLACEMENT = (
    "cloud1.datacenter1.zoneA,"
    "cloud1.datacenter1.zoneA,"
    "cloud1.datacenter1.zoneB"
)
MULTI_ZONE_RF = 3

# yb-ctl `create` is fast (~6s) and `destroy` is faster (~4s) on this machine,
# so we do destroy + create per test rather than reuse a long-lived cluster.
# That's the JDBC pattern (yb-ctl create per test method) — measured to be only
# ~10s/cycle, far cheaper than the reset bookkeeping it would replace.

CREATE_TIMEOUT = 120
DESTROY_TIMEOUT = 60
# After create returns, the cluster may take a couple more seconds before
# ysql is listening on 5433. Poll-until-connect to absorb that.
WAIT_FOR_READY_TIMEOUT = 30


_YB_DATA_DIR = os.path.expanduser("~/yugabyte-data")

# Two-cluster (xCluster Tier 2) data directories. The default `_YB_DATA_DIR`
# is shared by all yb-ctl invocations that don't pass `--data_dir`; we
# separate the two xCluster clusters by passing per-cluster `--data_dir`s
# so they can run concurrently without colliding on metadata.
_YB_PRIMARY_DATA_DIR = os.path.expanduser("~/yugabyte-data-xcluster-primary")
_YB_SECONDARY_DATA_DIR = os.path.expanduser("~/yugabyte-data-xcluster-secondary")
# IP ranges. yb-ctl's `--ip_start N` makes the cluster's tservers bind to
# 127.0.0.N, 127.0.0.N+1, 127.0.0.N+2 (for RF=3). Loopback aliases for the
# secondary range (.4-.6) must be set up via sudo ifconfig before tests run.
_XCLUSTER_PRIMARY_IP_START = 1     # 127.0.0.1 / .2 / .3
_XCLUSTER_SECONDARY_IP_START = 4   # 127.0.0.4 / .5 / .6
_XCLUSTER_PRIMARY_HOSTS = ("127.0.0.1", "127.0.0.2", "127.0.0.3")
_XCLUSTER_SECONDARY_HOSTS = ("127.0.0.4", "127.0.0.5", "127.0.0.6")

# Used by tests that need a real xCluster-replicated user table to write
# rows on the primary and read them on the secondary. Created on both
# clusters at session-fixture startup and registered with
# `setup_universe_replication` so primary writes flow to secondary.
_XCLUSTER_TEST_TABLE = "xcluster_test_data"
_XCLUSTER_REPL_GROUP_ID = "xcluster_test_repl"
_XCLUSTER_PRIMARY_MASTERS = ",".join(
    f"{h}:7100" for h in _XCLUSTER_PRIMARY_HOSTS
)
_XCLUSTER_SECONDARY_MASTERS = ",".join(
    f"{h}:7100" for h in _XCLUSTER_SECONDARY_HOSTS
)


def _ybctl_destroy_silent():
    """Destroy any existing cluster. Best-effort with escalating force:

    1. Kill any leftover yb-ctl processes from previous runs.
    2. Try `yb-ctl destroy` with the normal timeout.
    3. If destroy times out or fails, kill yb-tserver and yb-master processes
       directly and wipe the data directory.

    Test cluster state is disposable so we don't need to be careful about
    losing data — only need to ensure the next `create` starts clean.
    """
    import shutil as _shutil
    import subprocess as _sp

    # Step 1: kill stuck yb-ctl from previous runs.
    _sp.run(["pkill", "-9", "-f", "yb-ctl "], capture_output=True)

    # Step 2: try the polite destroy.
    try:
        _run_yb_ctl(["destroy"], timeout=DESTROY_TIMEOUT)
    except _sp.TimeoutExpired:
        pass
    except Exception:
        pass

    # Step 3: ALWAYS hard-kill anything left over and wipe the data dir.
    # yb-ctl's destroy can leave node-N directories around after add_node
    # operations and a subsequent destroy. A leftover directory makes the
    # next create fail. The data dir is regenerated by create.
    _sp.run(["pkill", "-9", "yb-tserver"], capture_output=True)
    _sp.run(["pkill", "-9", "yb-master"], capture_output=True)
    # Give the OS a moment to release sockets.
    time.sleep(1)
    if os.path.isdir(_YB_DATA_DIR):
        try:
            _shutil.rmtree(_YB_DATA_DIR)
        except Exception:
            pass


def _ybctl_create(placement: str, rf: int):
    """Create a fresh cluster with the given placement_info and replication
    factor. `rf` controls the number of nodes; `placement_info` controls how
    they're distributed across cloud.region.zone tuples (round-robin, with
    repetition when len(placement) > rf is not a thing — placements must be
    <= rf and either equal length or 1 entry that applies to all rf nodes).
    """
    result = _run_yb_ctl(
        ["create", "--rf", str(rf), "--placement_info", placement],
        timeout=CREATE_TIMEOUT,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"yb-ctl create failed (placement={placement!r}, rf={rf}): "
            f"{result.stderr}"
        )


def _wait_for_cluster_ready(dsn: str, expected_nodes: int = 3):
    """Block until ``yb_servers()`` reports ``expected_nodes`` tservers.

    Just-TCP-reachable isn't enough for tests that assert per-host
    distribution: a node whose tserver is up but hasn't yet registered
    with the master is invisible to ``yb_servers()`` and therefore won't
    be in the dispatcher's node list. Waiting for the full topology
    eliminates a class of test-order flakiness.
    """
    import psycopg
    import time as _time
    deadline = _time.monotonic() + WAIT_FOR_READY_TIMEOUT
    last_err: Exception | None = None
    while _time.monotonic() < deadline:
        try:
            with psycopg.connect(dsn + " connect_timeout=3") as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT count(*) FROM yb_servers()")
                    (n,) = cur.fetchone()
                    if n >= expected_nodes:
                        return
        except Exception as exc:
            last_err = exc
        _time.sleep(1.0)
    raise RuntimeError(
        f"cluster did not reach {expected_nodes} tservers within "
        f"{WAIT_FOR_READY_TIMEOUT}s: last error: {last_err!r}"
    )


def _clear_registry_singleton():
    """Reset the smart-driver registry so each test starts from a known state."""
    try:
        from psycopg.yb.registry import ClusterRegistry
        ClusterRegistry.instance().clear()
        ClusterRegistry._instance = None
    except Exception:
        pass


def _settle_until_rpcz_stable(
    hosts=("127.0.0.1", "127.0.0.2", "127.0.0.3"),
    *,
    max_wait_s: float = 8.0,
    poll_interval_s: float = 0.3,
    stable_streak: int = 3,
):
    """Wait until per-host /rpcz "client backend" counts stop changing.

    YB doesn't synchronously remove a "client backend" entry from /rpcz when
    a libpq client disconnects — there's a brief reaping window. Tests that
    assert /rpcz counts exactly need the cluster's view to have settled
    before they start. We poll until we see `stable_streak` consecutive
    identical readings across all hosts, or `max_wait_s` elapses (last
    reading wins; tests will fail loudly if the count is still off).
    """
    last = None
    streak = 0
    deadline = time.monotonic() + max_wait_s
    while time.monotonic() < deadline:
        try:
            cur = tuple(rpcz_count(h) for h in hosts)
        except RuntimeError:
            # Endpoint not up yet; retry.
            time.sleep(poll_interval_s)
            continue
        if cur == last:
            streak += 1
            if streak >= stable_streak:
                return
        else:
            streak = 1
            last = cur
        time.sleep(poll_interval_s)


@pytest.fixture
def yb_cluster():
    """Per-test fresh single-zone cluster.

    Destroys any existing cluster, creates a 3-node single-zone one, waits for
    it to accept connections, yields the DSN, then destroys it again. The
    smart-driver registry singleton is reset before and after.

    Skips the test (rather than failing) if `yb-ctl` isn't available.

    A previous version of this fixture just verified that an external cluster
    was reachable. We switched to per-test create/destroy after running into
    state-leak problems between tests; cluster cycle time is ~10s on a warm
    machine which is small enough not to worry about.
    """
    # Skip if yb-ctl missing.
    probe = subprocess.run(
        [YB_CTL_PATH, "--help"], capture_output=True, text=True
    )
    if probe.returncode != 0 and "Usage" not in (probe.stdout + probe.stderr):
        pytest.skip(f"yb-ctl not usable: {probe.stderr.strip()}")

    _clear_registry_singleton()
    _ybctl_destroy_silent()
    _ybctl_create(SINGLE_ZONE_PLACEMENT, SINGLE_ZONE_RF)
    _wait_for_cluster_ready(_DEFAULT_DSN)
    # `_wait_for_cluster_ready` opens + closes one probe connection. YB needs
    # a moment to reap the backend on its side before /rpcz stops reporting
    # it. Without this settle, `assert_balanced`'s exact /rpcz check would
    # see a stale "client backend" counted from the readiness probe.
    _settle_until_rpcz_stable()
    try:
        yield _DEFAULT_DSN
    finally:
        _clear_registry_singleton()
        _ybctl_destroy_silent()


@pytest.fixture
def yb_multi_zone_cluster():
    """Per-test fresh multi-zone cluster.

    Shape: 2 nodes in cloud1.datacenter1.zoneA, 1 in cloud1.datacenter1.zoneB.
    The two-in-zoneA arrangement lets the topology-matrix tests stop a node
    in the bound topology and still observe traffic re-distributing to the
    remaining in-topology node.

    Same fast destroy+create lifecycle as `yb_cluster`.
    """
    probe = subprocess.run(
        [YB_CTL_PATH, "--help"], capture_output=True, text=True
    )
    if probe.returncode != 0 and "Usage" not in (probe.stdout + probe.stderr):
        pytest.skip(f"yb-ctl not usable: {probe.stderr.strip()}")

    _clear_registry_singleton()
    _ybctl_destroy_silent()
    _ybctl_create(MULTI_ZONE_PLACEMENT, MULTI_ZONE_RF)
    _wait_for_cluster_ready(_DEFAULT_DSN)
    _settle_until_rpcz_stable()
    try:
        yield _DEFAULT_DSN
    finally:
        _clear_registry_singleton()
        _ybctl_destroy_silent()


# --------------------------------------------------------------------- xCluster Tier 2: two real clusters


def _ybctl_destroy_at(data_dir: str, ip_start: int):
    """Best-effort destroy of a specific cluster identified by its data_dir.
    Mirrors `_ybctl_destroy_silent` but scoped to one cluster, leaving the
    other untouched. Also kills any leftover processes whose IP matches the
    range owned by this cluster."""
    import shutil as _shutil
    import subprocess as _sp

    # Polite destroy first.
    try:
        _run_yb_ctl(["destroy"], timeout=DESTROY_TIMEOUT, data_dir=data_dir)
    except _sp.TimeoutExpired:
        pass
    except Exception:
        pass

    # Hard-kill any leftover yb-tserver / yb-master processes bound to
    # the IPs owned by this cluster (so we don't disturb a sibling cluster
    # still running on the other IP range).
    for offset in range(3):  # RF=3 assumed
        ip = f"127.0.0.{ip_start + offset}"
        _sp.run(
            ["pkill", "-9", "-f", f"--webserver_interface {ip}"],
            capture_output=True,
        )
    time.sleep(1)
    if os.path.isdir(data_dir):
        try:
            _shutil.rmtree(data_dir)
        except Exception:
            pass


def _ybctl_create_at(
    data_dir: str,
    ip_start: int,
    placement: str = SINGLE_ZONE_PLACEMENT,
    rf: int = SINGLE_ZONE_RF,
):
    """Create a cluster scoped to `data_dir` with tservers bound to
    127.0.0.<ip_start> .. 127.0.0.<ip_start + rf - 1>."""
    result = _run_yb_ctl(
        [
            "create",
            "--rf", str(rf),
            "--placement_info", placement,
            "--ip_start", str(ip_start),
        ],
        timeout=CREATE_TIMEOUT,
        data_dir=data_dir,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"yb-ctl create failed (data_dir={data_dir}, "
            f"ip_start={ip_start}, placement={placement!r}): "
            f"{result.stderr}"
        )


@pytest.fixture(scope="session")
def yb_xcluster_clusters():
    """Session-scoped: two yb-ctl clusters + REAL xCluster replication.

    Returns ``(primary_dsn, secondary_dsn)`` where each is a libpq conninfo
    string. The two clusters have distinct ``universe_uuid``s AND the
    fixture sets up unidirectional xCluster replication from primary to
    secondary on the ``xcluster_test_data`` table (created on both at
    session startup). Tests that just exercise routing/pool semantics
    via ``FailoverGroup.force_status`` can ignore the table; tests in
    ``test_smart_driver_xcluster_data.py`` use it to verify data
    continuity across a primary failure.

    Session-scoped because the tests don't mutate cluster state — failover
    is driven by ``FailoverGroup.force_status``, not by stopping real
    nodes (with the exception of the suites in
    ``test_smart_driver_xcluster_cb.py`` and
    ``test_smart_driver_xcluster_data.py``, which DO stop nodes and
    restart them in their own finally blocks). This amortises the ~20s
    two-cluster setup plus the ~5s xCluster setup over the whole xCluster
    integration suite.

    Loopback aliases for 127.0.0.4 / .5 / .6 must be present (macOS):

        sudo ifconfig lo0 alias 127.0.0.4/32 up
        sudo ifconfig lo0 alias 127.0.0.5/32 up
        sudo ifconfig lo0 alias 127.0.0.6/32 up

    Without those, the secondary cluster's tservers fail to bind. The
    fixture detects this at create time and skips.
    """
    probe = subprocess.run(
        [YB_CTL_PATH, "--help"], capture_output=True, text=True
    )
    if probe.returncode != 0 and "Usage" not in (probe.stdout + probe.stderr):
        pytest.skip(f"yb-ctl not usable: {probe.stderr.strip()}")

    # Quick loopback-alias check — fail fast if the secondary IPs aren't
    # routable. Save time vs. waiting for yb-ctl to time out at startup.
    import socket
    for ip in _XCLUSTER_SECONDARY_HOSTS:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            try:
                s.bind((ip, 0))
            except OSError as exc:
                pytest.skip(
                    f"loopback alias for {ip} is missing — run "
                    f"`sudo ifconfig lo0 alias {ip}/32 up` "
                    f"(xCluster tests need .4-.6 aliased). Got: {exc}"
                )
        finally:
            s.close()

    # Defensive: destroy any previous artifact of these clusters before we
    # build them fresh, in case a previous run was interrupted.
    _ybctl_destroy_at(_YB_PRIMARY_DATA_DIR, _XCLUSTER_PRIMARY_IP_START)
    _ybctl_destroy_at(_YB_SECONDARY_DATA_DIR, _XCLUSTER_SECONDARY_IP_START)

    _ybctl_create_at(_YB_PRIMARY_DATA_DIR, _XCLUSTER_PRIMARY_IP_START)
    try:
        _ybctl_create_at(_YB_SECONDARY_DATA_DIR, _XCLUSTER_SECONDARY_IP_START)
    except Exception:
        # If secondary fails, clean up the primary we already started.
        _ybctl_destroy_at(_YB_PRIMARY_DATA_DIR, _XCLUSTER_PRIMARY_IP_START)
        raise

    primary_dsn = (
        "host="
        + ",".join(_XCLUSTER_PRIMARY_HOSTS)
        + " port=5433 user=yugabyte dbname=yugabyte"
    )
    secondary_dsn = (
        "host="
        + ",".join(_XCLUSTER_SECONDARY_HOSTS)
        + " port=5433 user=yugabyte dbname=yugabyte"
    )
    try:
        _wait_for_cluster_ready(primary_dsn)
        _wait_for_cluster_ready(secondary_dsn)
    except Exception:
        _ybctl_destroy_at(_YB_PRIMARY_DATA_DIR, _XCLUSTER_PRIMARY_IP_START)
        _ybctl_destroy_at(_YB_SECONDARY_DATA_DIR, _XCLUSTER_SECONDARY_IP_START)
        raise

    try:
        _setup_xcluster_replication()
    except Exception:
        _ybctl_destroy_at(_YB_PRIMARY_DATA_DIR, _XCLUSTER_PRIMARY_IP_START)
        _ybctl_destroy_at(_YB_SECONDARY_DATA_DIR, _XCLUSTER_SECONDARY_IP_START)
        raise

    try:
        yield (primary_dsn, secondary_dsn)
    finally:
        # Best-effort: delete the replication stream first, then destroy
        # the clusters. If primary is already stopped (a stop_node test
        # left it that way and recover_primary_cluster's restart_node
        # failed), delete_universe_replication still works on the consumer
        # side because we run it against the secondary's masters.
        _teardown_xcluster_replication()
        _ybctl_destroy_at(_YB_PRIMARY_DATA_DIR, _XCLUSTER_PRIMARY_IP_START)
        _ybctl_destroy_at(_YB_SECONDARY_DATA_DIR, _XCLUSTER_SECONDARY_IP_START)


# --------------------------------------------------------------------- registry


@pytest.fixture
def fresh_registry():
    """Yield a fresh `ClusterRegistry` instance for the duration of the test.

    The singleton is replaced before the test runs and restored after, so
    unit tests don't leak state into each other through the process-global
    registry. Use this fixture for `yb_unit` tests that exercise the
    registry directly.
    """
    from psycopg.yb.registry import ClusterRegistry

    saved = ClusterRegistry._instance
    fresh = ClusterRegistry()
    ClusterRegistry._instance = fresh
    try:
        yield fresh
    finally:
        ClusterRegistry._instance = saved


@pytest.fixture
def fake_state():
    """Factory for building synthetic `ClusterState` objects.

    Usage:

        def test_pick(fake_state):
            state = fake_state(
                ("a", "aws", "us-west", "us-west-1a", "primary", 2),
                ("b", "aws", "us-west", "us-west-1b", "primary", 5),
                ("c", "aws", "us-east", "us-east-2a", "read_replica", 0),
            )
            ...

    Each tuple is `(host, cloud, region, zone, node_type, connection_count)`.
    `is_down` defaults to False; supply via the optional 7th element.
    """
    from psycopg.yb.node import NodeInfo, Placement
    from psycopg.yb.registry import ClusterState

    def _factory(*nodes, uuid: str = "test-cluster") -> "ClusterState":
        node_map = {}
        for spec in nodes:
            host, cloud, region, zone, node_type = spec[:5]
            count = spec[5] if len(spec) > 5 else 0
            down = spec[6] if len(spec) > 6 else False
            node_map[host] = NodeInfo(
                host=host,
                public_ip=None,
                port=5433,
                placement=Placement(cloud=cloud, region=region, zone=zone),
                node_type=node_type,
                connection_count=count,
                is_down=down,
                is_down_since=time.monotonic() if down else 0.0,
            )
        return ClusterState(
            uuid=uuid,
            nodes=node_map,
            lock=threading.Lock(),
            last_refresh=time.monotonic(),
        )

    return _factory


# --------------------------------------------------------------------- xcluster failover


@pytest.fixture
def yb_failover_group(fresh_registry, fake_state):
    """Build a synthetic ``FailoverGroup`` (primary + secondary) and install
    it directly into the registry, bypassing real-cluster bootstrap.

    Used by ``test_xcluster_failover.py`` to exercise routing, pool
    eviction, cool-down, and concurrent-flip scenarios without spinning up
    two real clusters. The fresh_registry fixture handles singleton
    isolation; teardown clears the registry which stops any probe threads
    started by tests.

    The cooldown defaults to ``0`` so tests can flip status repeatedly
    without time-warping. Set ``group.cooldown_s = N`` directly if a test
    needs to exercise cool-down behaviour.
    """
    from psycopg.yb.circuit_breaker import AlwaysHealthyCircuitBreaker
    from psycopg.yb.health import HealthResult
    from psycopg.yb.registry import FailoverGroup

    primary = fake_state(
        ("p1", "aws", "us-west", "us-west-1a", "primary"),
        ("p2", "aws", "us-west", "us-west-1b", "primary"),
        uuid="primary-uuid",
    )
    secondary = fake_state(
        ("s1", "aws", "us-east", "us-east-1a", "primary"),
        ("s2", "aws", "us-east", "us-east-1b", "primary"),
        uuid="secondary-uuid",
    )
    # Attach inert CBs so the dispatcher's fail-fast check passes;
    # these tests drive status via ``force_*_status`` directly.
    group = FailoverGroup(
        primary=primary,
        secondary=secondary,
        lock=threading.Lock(),
        primary_status=HealthResult.HEALTHY,
        secondary_status=HealthResult.HEALTHY,
        cooldown_s=0,
        primary_circuit_breaker=AlwaysHealthyCircuitBreaker(),
        secondary_circuit_breaker=AlwaysHealthyCircuitBreaker(),
    )
    fresh_registry._clusters[primary.uuid] = primary
    fresh_registry._clusters[secondary.uuid] = secondary
    fresh_registry._failover_groups[primary.uuid] = group
    return group


@pytest.fixture
def flip_to_unhealthy_after(yb_failover_group):
    """Returns a ``flip(delay_s, status=UNHEALTHY, which="primary")`` callable.

    The callable schedules a background daemon thread that sleeps
    ``delay_s`` seconds then calls ``yb_failover_group.force_primary_status``
    or ``yb_failover_group.force_secondary_status`` depending on ``which``.
    Multiple flips can be scheduled per test. All threads are stopped
    cleanly at teardown via a shared ``threading.Event``.

    Matches the spec sketch in design doc §14 — used to exercise
    "concurrent flip while connecting" and "delayed-flip during test"
    scenarios.
    """
    from psycopg.yb.health import HealthResult

    stop_event = threading.Event()
    threads: list[threading.Thread] = []

    def flip(
        delay_s: float,
        status: "HealthResult" = HealthResult.UNHEALTHY,
        which: str = "primary",
    ):
        setter = getattr(yb_failover_group, f"force_{which}_status")

        def run():
            if stop_event.wait(timeout=delay_s):
                return  # teardown happened first; abort
            setter(status)

        t = threading.Thread(target=run, daemon=True)
        t.start()
        threads.append(t)
        return t

    try:
        yield flip
    finally:
        stop_event.set()
        for t in threads:
            t.join(timeout=2.0)


# --------------------------------------------------------------------- /rpcz
#
# Server-side ground-truth verification: scrape the YB-TServer's HTTP debug
# page and count "client backend" occurrences. This is the same technique
# pgjdbc-yb's FallbackOptionsLBTest.verifyOn uses.
#
# We use urllib (stdlib) instead of httpx/requests to keep the test fixture
# free of extra dependencies.


def rpcz_count(host: str, port: int = 13000, timeout: float = 5.0) -> int:
    """Count `client backend` substrings in the tserver's /rpcz HTTP output.

    Raises `RuntimeError` if the tserver is unreachable; tests should let
    that propagate (a working integration setup must have /rpcz available
    on every tserver).
    """
    url = f"http://{host}:{port}/rpcz"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, OSError) as exc:
        raise RuntimeError(f"failed to reach {url}: {exc}") from exc
    return body.count("client backend")


@pytest.fixture
def rpcz():
    """Expose the `rpcz_count` helper as a fixture (matches JDBC's verifyOn)."""
    return rpcz_count


@pytest.fixture
def assert_balanced(rpcz):
    """Cross-check driver-side and server-side per-host load — EXACT.

    Both sides are asserted with `==`. No tolerance knobs.

    Driver side: `get_least_loaded_server` reserves+increments atomically
    under the per-cluster lock, so concurrent connects can't desync the
    counts. The driver counter is the authoritative statement of load.

    Server side (`/rpcz` "client backend" count): the smart driver keeps
    one long-lived control connection per flavor (sync, async) pinned to a
    specific host — that's the ONLY out-of-band backend on a freshly-created
    per-test cluster. The fixture identifies which host(s) host the control
    connections by reading `state.control_sync.info.host` and
    `state.control_async.info.host`, then adds +1 to the expected /rpcz for
    each control-conn host. The remaining /rpcz must match the driver
    counter exactly.

    Pre-test cluster startup also opens a brief readiness-probe connection;
    `_settle_until_rpcz_stable` (invoked by the `yb_cluster` fixture) waits
    for YB to reap that backend before the test body runs.
    """
    from psycopg.yb.registry import ClusterRegistry

    def _control_hosts(state):
        sync_host = async_host = None
        try:
            if state.control_sync is not None and not state.control_sync.closed:
                sync_host = state.control_sync.info.host
        except Exception:
            pass
        try:
            if state.control_async is not None and not state.control_async.closed:
                async_host = state.control_async.info.host
        except Exception:
            pass
        return sync_host, async_host

    def _check(uuid, expected_per_host):
        registry = ClusterRegistry.instance()
        state = registry._clusters.get(uuid)
        sync_host, async_host = _control_hosts(state) if state is not None else (None, None)

        problems = []
        for host, want in expected_per_host.items():
            got_driver = registry.get_load(uuid, host)
            if got_driver != want:
                problems.append(
                    f"{host}: driver counter is {got_driver}, expected exactly {want}"
                )
            extra = (1 if host == sync_host else 0) + (1 if host == async_host else 0)
            expected_server = want + extra
            try:
                got_server = rpcz(host)
            except RuntimeError as exc:
                problems.append(f"{host}: {exc}")
                continue
            if got_server != expected_server:
                detail = f" (= {want} workload"
                if extra:
                    detail += f" + {extra} control conn"
                detail += ")"
                problems.append(
                    f"{host}: /rpcz reports {got_server}, expected exactly "
                    f"{expected_server}{detail}"
                )
        if problems:
            raise AssertionError("\n  ".join(["balance mismatch:"] + problems))

    return _check


# --------------------------------------------------------------------- yb-ctl


YB_CTL_PATH = os.environ.get("YB_CTL", "yb-ctl")

# yb-admin lives next to yb-ctl in a standard YB install. Tests that
# configure xCluster replication need it; if YB_ADMIN is explicitly set
# in the environment we honor that, otherwise we derive it from YB_CTL.
YB_ADMIN_PATH = os.environ.get("YB_ADMIN") or (
    os.path.join(os.path.dirname(YB_CTL_PATH), "yb-admin")
    if os.path.dirname(YB_CTL_PATH)
    else "yb-admin"
)


def _run_yb_admin(
    args: list[str], *, timeout: int = 60
) -> subprocess.CompletedProcess:
    """Invoke yb-admin. Tests use this for `setup_universe_replication`
    + `delete_universe_replication`; the fixture handles the lifecycle."""
    cmd = [YB_ADMIN_PATH, *args]
    try:
        return subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout
        )
    except FileNotFoundError:
        pytest.skip(
            f"yb-admin not found at {YB_ADMIN_PATH!r}; set YB_ADMIN env var "
            f"(usually lives next to yb-ctl)"
        )


def _setup_xcluster_replication() -> None:
    """Create the test table on both clusters and establish unidirectional
    xCluster replication primary -> secondary. Idempotent — `CREATE TABLE
    IF NOT EXISTS` + a fixed replication group id. Called from
    `yb_xcluster_clusters` once both clusters are ready."""
    import psycopg as _ps

    for host in (_XCLUSTER_PRIMARY_HOSTS[0], _XCLUSTER_SECONDARY_HOSTS[0]):
        dsn = (
            f"host={host} port=5433 user=yugabyte dbname=yugabyte "
            f"connect_timeout=5"
        )
        with _ps.connect(dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"CREATE TABLE IF NOT EXISTS {_XCLUSTER_TEST_TABLE} "
                    "(id INT PRIMARY KEY, ts TIMESTAMP, payload TEXT)"
                )
            conn.commit()

    # Look up the producer's table id via `yb-admin list_tables`.
    r = _run_yb_admin(
        ["-master_addresses", _XCLUSTER_PRIMARY_MASTERS,
         "list_tables", "include_table_id"],
    )
    if r.returncode != 0:
        raise RuntimeError(
            f"yb-admin list_tables failed: {r.stderr or r.stdout}"
        )
    table_id = None
    for line in r.stdout.splitlines():
        # Format: "<namespace>.<name> <table_id>"
        parts = line.split()
        if (len(parts) == 2
                and parts[0] == f"yugabyte.{_XCLUSTER_TEST_TABLE}"):
            table_id = parts[1]
            break
    if not table_id:
        raise RuntimeError(
            f"could not find table id for {_XCLUSTER_TEST_TABLE} in "
            f"yb-admin list_tables output: {r.stdout[:500]}"
        )

    # Run setup_universe_replication against the CONSUMER (secondary)
    # masters, pointing at the PRODUCER (primary) masters and table id.
    r = _run_yb_admin(
        ["-master_addresses", _XCLUSTER_SECONDARY_MASTERS,
         "setup_universe_replication",
         _XCLUSTER_REPL_GROUP_ID,
         _XCLUSTER_PRIMARY_MASTERS,
         table_id],
        timeout=60,
    )
    if r.returncode != 0:
        # If the stream already exists from an earlier interrupted run,
        # delete it and try once more. Otherwise propagate.
        if "already exists" in (r.stderr + r.stdout):
            _teardown_xcluster_replication()
            r = _run_yb_admin(
                ["-master_addresses", _XCLUSTER_SECONDARY_MASTERS,
                 "setup_universe_replication",
                 _XCLUSTER_REPL_GROUP_ID,
                 _XCLUSTER_PRIMARY_MASTERS,
                 table_id],
                timeout=60,
            )
        if r.returncode != 0:
            raise RuntimeError(
                f"setup_universe_replication failed: "
                f"{r.stderr or r.stdout}"
            )


def _teardown_xcluster_replication() -> None:
    """Delete the test replication stream. Best-effort: a stopped primary
    won't have a reachable master, but the consumer side can still drop
    the local stream config."""
    _run_yb_admin(
        ["-master_addresses", _XCLUSTER_SECONDARY_MASTERS,
         "delete_universe_replication",
         _XCLUSTER_REPL_GROUP_ID,
         "ignore-errors"],
        timeout=60,
    )


def _run_yb_ctl(
    args: list[str], *, timeout: int = 30, data_dir: str | None = None
) -> subprocess.CompletedProcess:
    """Invoke yb-ctl. `data_dir` (if given) is prepended as a global
    `--data_dir` option so xCluster Tier 2 tests can manage two
    concurrent clusters with independent metadata."""
    cmd = [YB_CTL_PATH]
    if data_dir is not None:
        cmd.extend(["--data_dir", data_dir])
    cmd.extend(args)
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError:
        pytest.skip(f"yb-ctl not found at {YB_CTL_PATH!r}; set YB_CTL env var")


class YBCtl:
    """Thin wrapper around `yb-ctl` for failure-injection in integration tests.

    Mirrors the pgjdbc-yb test pattern: real `yb-ctl stop_node`/`start_node`,
    no socket trickery, no SQL marking. Tests typically pair this with a
    `Thread.sleep`/`time.sleep` longer than the smart driver's
    `yb_servers_refresh_interval` so the refresh picks up the change.
    """

    # Generous per-command timeouts. `stop_node` is quick (kill a process);
    # `start_node` and `add_node` need to wait for the tserver to register
    # with the master, which can take ~30s on a warm machine and longer on a
    # cold one. Tests are expected to be slow when they touch yb-ctl.
    STOP_TIMEOUT = 30
    START_TIMEOUT = 90
    ADD_TIMEOUT = 180

    @staticmethod
    def stop_node(n: int):
        result = _run_yb_ctl(["stop_node", str(n)], timeout=YBCtl.STOP_TIMEOUT)
        if result.returncode != 0:
            raise RuntimeError(f"yb-ctl stop_node {n} failed: {result.stderr}")

    @staticmethod
    def start_node(n: int, placement_info: str | None = None):
        args = ["start_node", str(n)]
        if placement_info:
            args.extend(["--placement_info", placement_info])
        result = _run_yb_ctl(args, timeout=YBCtl.START_TIMEOUT)
        if result.returncode != 0:
            raise RuntimeError(f"yb-ctl start_node {n} failed: {result.stderr}")

    @staticmethod
    def add_node(placement_info: str | None = None):
        args = ["add_node"]
        if placement_info:
            args.extend(["--placement_info", placement_info])
        result = _run_yb_ctl(args, timeout=YBCtl.ADD_TIMEOUT)
        if result.returncode != 0:
            raise RuntimeError(f"yb-ctl add_node failed: {result.stderr}")


@pytest.fixture
def yb_ctl():
    """Provide the `YBCtl` helper. Skips the test if `yb-ctl` isn't available."""
    # Probe once. If absent, skip cleanly.
    probe = subprocess.run(
        [YB_CTL_PATH, "--help"], capture_output=True, text=True
    )
    if probe.returncode != 0 and "Usage" not in (probe.stdout + probe.stderr):
        pytest.skip(f"yb-ctl not usable: {probe.stderr.strip()}")
    return YBCtl()


class _XClusterCtl:
    """yb-ctl wrappers scoped to the two-cluster xCluster fixture.

    The default ``YBCtl`` operates on the single-cluster ``_YB_DATA_DIR``;
    the xCluster Tier 2 tests need to target either the primary or the
    secondary cluster by passing the matching ``--data_dir``. This class
    bundles those four operations so tests don't have to thread the
    data_dir through every call.
    """

    @staticmethod
    def stop_primary_node(n: int):
        r = _run_yb_ctl(
            ["stop_node", str(n)],
            timeout=YBCtl.STOP_TIMEOUT,
            data_dir=_YB_PRIMARY_DATA_DIR,
        )
        if r.returncode != 0:
            raise RuntimeError(
                f"yb-ctl stop_node {n} (primary) failed: {r.stderr}"
            )

    @staticmethod
    def start_primary_node(n: int):
        r = _run_yb_ctl(
            ["start_node", str(n)],
            timeout=YBCtl.START_TIMEOUT,
            data_dir=_YB_PRIMARY_DATA_DIR,
        )
        if r.returncode != 0:
            raise RuntimeError(
                f"yb-ctl start_node {n} (primary) failed: {r.stderr}"
            )

    @staticmethod
    def stop_secondary_node(n: int):
        r = _run_yb_ctl(
            ["stop_node", str(n)],
            timeout=YBCtl.STOP_TIMEOUT,
            data_dir=_YB_SECONDARY_DATA_DIR,
        )
        if r.returncode != 0:
            raise RuntimeError(
                f"yb-ctl stop_node {n} (secondary) failed: {r.stderr}"
            )

    @staticmethod
    def start_secondary_node(n: int):
        r = _run_yb_ctl(
            ["start_node", str(n)],
            timeout=YBCtl.START_TIMEOUT,
            data_dir=_YB_SECONDARY_DATA_DIR,
        )
        if r.returncode != 0:
            raise RuntimeError(
                f"yb-ctl start_node {n} (secondary) failed: {r.stderr}"
            )

    @staticmethod
    def restart_primary_node(n: int):
        """``restart_node`` against the primary cluster. Unlike ``start_node``,
        this works whether yb-ctl thinks the node is running or not — important
        for recovering from the master-quorum-loss cascade, where stopping two
        of three masters causes the third tserver's postmaster to also exit
        but yb-ctl still reports it as "running"."""
        r = _run_yb_ctl(
            ["restart_node", str(n)],
            timeout=YBCtl.START_TIMEOUT,
            data_dir=_YB_PRIMARY_DATA_DIR,
        )
        if r.returncode != 0:
            raise RuntimeError(
                f"yb-ctl restart_node {n} (primary) failed: {r.stderr}"
            )

    @staticmethod
    def recover_primary_cluster():
        """Best-effort restore: ensure every primary node accepts TCP
        connects on port 5433. Healthy nodes are skipped (fast — ~1 ms
        probe); only actually-down nodes get ``restart_node`` (slow —
        ~30 s per restart).

        Catches two failure modes:
          * Nodes the test explicitly stopped.
          * Node 1's postmaster exiting via master-quorum-loss cascade
            when masters 2+3 are killed — yb-ctl still reports it as
            "running" so ``start_node 1`` would error out, but
            ``restart_node 1`` works.

        Errors per node are swallowed so a partial cleanup doesn't mask
        the original test failure."""
        import socket as _socket
        for n in (1, 2, 3):
            host = f"127.0.0.{n}"
            s = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
            s.settimeout(1.0)
            reachable = False
            try:
                s.connect((host, 5433))
                reachable = True
            except (OSError, _socket.timeout):
                pass
            finally:
                s.close()
            if reachable:
                continue
            try:
                _XClusterCtl.restart_primary_node(n)
            except Exception:
                pass

        # After restart_node returns the tserver PROCESS is up, but full
        # convergence of the cluster state across all three masters AND
        # all three tservers can lag a few more seconds. We require:
        #   (a) every host individually answers a SELECT
        #   (b) yb_servers() returns 3 rows from EACH host (master state
        #       is consistent across the cluster, not just on one master)
        # Without (b), a subsequent test's bootstrap might land on a
        # contact host whose master replica still reports 2 tservers; the
        # dispatcher then picks from a 2-host subset and a distribution
        # assertion fails. After the join check passes we settle 2 more
        # seconds because YB's master-to-master gossip can briefly admit
        # a still-registering tserver before all tservers see it.
        import psycopg as _ps

        def _hosts_all_see_three(timeout_s: float) -> bool:
            deadline_local = time.monotonic() + timeout_s
            while time.monotonic() < deadline_local:
                ok = True
                for probe_host in _XCLUSTER_PRIMARY_HOSTS:
                    try:
                        with _ps.connect(
                            f"host={probe_host} port=5433 user=yugabyte "
                            f"dbname=yugabyte connect_timeout=2"
                        ) as conn:
                            with conn.cursor() as cur:
                                cur.execute(
                                    "SELECT count(*) FROM yb_servers()"
                                )
                                row = cur.fetchone()
                                if row is None or row[0] < 3:
                                    ok = False
                                    break
                    except Exception:
                        ok = False
                        break
                if ok:
                    return True
                time.sleep(1.0)
            return False

        if _hosts_all_see_three(60):
            # Give the cluster's last-mover gossip a moment to settle.
            time.sleep(2)


@pytest.fixture
def yb_xcluster_ctl():
    """``_XClusterCtl`` helper for failure injection on either xCluster.

    Skips the test if ``yb-ctl`` isn't available. Pair with the
    ``yb_xcluster_clusters`` session fixture so the two clusters exist
    before the test starts stopping nodes.
    """
    probe = subprocess.run(
        [YB_CTL_PATH, "--help"], capture_output=True, text=True
    )
    if probe.returncode != 0 and "Usage" not in (probe.stdout + probe.stderr):
        pytest.skip(f"yb-ctl not usable: {probe.stderr.strip()}")
    return _XClusterCtl()


# --------------------------------------------------------------------- cleanup


@pytest.fixture(autouse=True)
def _clear_registry_after_integration_tests(request):
    """For `yb`-marked tests, clear the ClusterRegistry singleton after the test.

    Integration tests that talk to a real cluster mutate the global registry;
    leaving counters non-zero would interfere with subsequent tests' balance
    assertions. Unit tests (with `fresh_registry`) handle isolation themselves.
    """
    yield
    if request.node.get_closest_marker("yb"):
        # Late import to avoid import cost for tests that don't need it.
        from psycopg.yb.registry import ClusterRegistry
        ClusterRegistry.instance().clear()
