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


def _wait_for_cluster_ready(dsn: str):
    """Block until the cluster accepts ysql connections."""
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
                    if n > 0:
                        return
        except Exception as exc:
            last_err = exc
        _time.sleep(1.0)
    raise RuntimeError(
        f"cluster did not become ready within {WAIT_FOR_READY_TIMEOUT}s: "
        f"last error: {last_err!r}"
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


def _run_yb_ctl(args: list[str], *, timeout: int = 30) -> subprocess.CompletedProcess:
    cmd = [YB_CTL_PATH, *args]
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
