"""
Unit tests for `psycopg.yb.registry`.

These tests don't connect to a real database. They manipulate the registry's
internal state directly to verify:

  * Singleton identity
  * `ClusterKey` semantics (sort-invariant, hashable, password-free)
  * Counter primitives (increment / decrement / mark_failed)
  * Cross-key dedup (different ClusterKeys → same uuid → same ClusterState)
  * `get_load` / `clear` public test hooks
  * Counter thread-safety under concurrent.futures stress
"""

# Copyright (C) 2026 Yugabyte

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from psycopg.yb.node import NodeInfo, Placement
from psycopg.yb.registry import (
    ClusterKey,
    ClusterRegistry,
    ClusterState,
)


# --------------------------------------------------------------------- ClusterKey


def test_cluster_key_sorts_hosts():
    a = ClusterKey.from_params({"host": "h1,h2,h3", "port": 5433})
    b = ClusterKey.from_params({"host": "h3,h2,h1"})
    assert a == b
    assert hash(a) == hash(b)


def test_cluster_key_lowercases_hosts():
    a = ClusterKey.from_params({"host": "H1.example.COM"})
    b = ClusterKey.from_params({"host": "h1.example.com"})
    assert a == b


def test_cluster_key_strips_whitespace_in_hosts():
    a = ClusterKey.from_params({"host": "h1 , h2 , h3"})
    b = ClusterKey.from_params({"host": "h1,h2,h3"})
    assert a == b


def test_cluster_key_hashable_and_dict_keyed():
    k = ClusterKey.from_params({"host": "h1"})
    d = {k: "value"}
    same = ClusterKey.from_params({"host": "h1"})
    assert d[same] == "value"


def test_cluster_key_omits_password():
    """Password is intentionally not part of the cluster identity."""
    a = ClusterKey.from_params({"host": "h1", "password": "secret-1"})
    b = ClusterKey.from_params({"host": "h1", "password": "secret-2"})
    assert a == b


def test_cluster_key_omits_sslmode():
    """SSL mode is connection-config, not cluster identity."""
    a = ClusterKey.from_params({"host": "h1", "sslmode": "require"})
    b = ClusterKey.from_params({"host": "h1", "sslmode": "disable"})
    assert a == b


def test_cluster_key_distinguishes_dbname():
    """Different databases on the same hosts are different cluster identities
    for our purposes — they have distinct connection pools at the YB level."""
    a = ClusterKey.from_params({"host": "h1", "dbname": "yugabyte"})
    b = ClusterKey.from_params({"host": "h1", "dbname": "production"})
    assert a != b


def test_cluster_key_distinguishes_user():
    a = ClusterKey.from_params({"host": "h1", "user": "alice"})
    b = ClusterKey.from_params({"host": "h1", "user": "bob"})
    assert a != b


def test_cluster_key_default_port_5433():
    """YugabyteDB's default port is 5433, not Postgres's 5432."""
    k = ClusterKey.from_params({"host": "h1"})
    assert k.port == 5433


# --------------------------------------------------------------------- singleton


def test_singleton_returns_same_instance():
    a = ClusterRegistry.instance()
    b = ClusterRegistry.instance()
    assert a is b


# --------------------------------------------------------------------- counter primitives


def _install_state(reg: ClusterRegistry, hosts: list[str], uuid: str = "X"):
    """Test helper: install a synthetic ClusterState bypassing bootstrap."""
    nodes = {
        h: NodeInfo(
            host=h, public_ip=None, port=5433,
            placement=Placement("aws", "r", f"z{i}"),
            node_type="primary",
        )
        for i, h in enumerate(hosts)
    }
    state = ClusterState(
        uuid=uuid,
        nodes=nodes,
        lock=threading.Lock(),
        last_refresh=time.monotonic(),
    )
    reg._clusters[uuid] = state
    return state


def test_increment_basic(fresh_registry):
    _install_state(fresh_registry, ["h1", "h2"])
    fresh_registry.increment("X", "h1")
    fresh_registry.increment("X", "h1")
    fresh_registry.increment("X", "h2")
    assert fresh_registry.get_load("X", "h1") == 2
    assert fresh_registry.get_load("X", "h2") == 1


def test_decrement_clamps_at_zero(fresh_registry):
    _install_state(fresh_registry, ["h1"])
    fresh_registry.decrement("X", "h1")  # was 0; should stay 0, not go negative
    assert fresh_registry.get_load("X", "h1") == 0


def test_increment_unknown_uuid_is_noop(fresh_registry):
    fresh_registry.increment("does-not-exist", "h1")
    assert fresh_registry.get_load("does-not-exist", "h1") == 0


def test_increment_unknown_host_is_noop(fresh_registry):
    _install_state(fresh_registry, ["h1"])
    fresh_registry.increment("X", "ghost")
    assert fresh_registry.get_load("X", "h1") == 0
    assert fresh_registry.get_load("X", "ghost") == 0


def test_mark_failed_sets_state(fresh_registry):
    state = _install_state(fresh_registry, ["h1", "h2"])
    fresh_registry.increment("X", "h1")
    fresh_registry.increment("X", "h1")
    assert state.nodes["h1"].connection_count == 2

    before = time.monotonic()
    fresh_registry.mark_failed("X", "h1")
    after = time.monotonic()

    assert state.nodes["h1"].is_down is True
    assert before <= state.nodes["h1"].is_down_since <= after
    assert state.nodes["h1"].connection_count == 0
    assert state.force_refresh is True


# --------------------------------------------------------------------- thread safety


def test_increment_decrement_thread_safe(fresh_registry):
    """Stress: 10 threads × 1000 inc/dec each on the same node.
    Final count must be exactly 0 if no atomicity bug."""
    _install_state(fresh_registry, ["h1"])

    def worker():
        for _ in range(1000):
            fresh_registry.increment("X", "h1")
            fresh_registry.decrement("X", "h1")

    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = [pool.submit(worker) for _ in range(10)]
        for f in futures:
            f.result()

    assert fresh_registry.get_load("X", "h1") == 0


# --------------------------------------------------------------------- cross-key dedup


def test_cross_key_dedup_via_uuid(fresh_registry):
    """Two ClusterKeys that resolve to the same universe_uuid must share state.

    Bootstrap dedups against the uuid: if `_clusters[uuid]` already exists,
    the redundant ClusterState is dropped and the new key is installed as
    an alias pointing at the existing state.
    """
    # Simulate two callers, each having bootstrapped a cluster (or thinking
    # they have) and discovered the SAME universe_uuid.
    k1 = ClusterKey.from_params({"host": "h1"})
    k2 = ClusterKey.from_params({"host": "h2"})

    # First caller's "bootstrap result": fresh state installed at uuid X.
    state_first = _install_state(fresh_registry, ["h1", "h2", "h3"], uuid="X")
    fresh_registry._key_to_uuid[k1] = "X"

    # Second caller's "bootstrap result": same uuid is discovered, so the
    # caller installs the alias and reuses the existing state. Simulate by
    # adding the alias.
    fresh_registry._key_to_uuid[k2] = "X"

    # Both keys must resolve to the same ClusterState.
    state_via_k1 = fresh_registry._clusters[fresh_registry._key_to_uuid[k1]]
    state_via_k2 = fresh_registry._clusters[fresh_registry._key_to_uuid[k2]]
    assert state_via_k1 is state_via_k2

    # Increments through either alias mutate the shared state.
    fresh_registry.increment("X", "h1")  # caller 1's action
    fresh_registry.increment("X", "h1")  # caller 2's action
    assert fresh_registry.get_load("X", "h1") == 2
    assert state_first.nodes["h1"].connection_count == 2


# --------------------------------------------------------------------- clear


def test_clear_drops_all_state(fresh_registry):
    _install_state(fresh_registry, ["h1", "h2"], uuid="X")
    _install_state(fresh_registry, ["h3", "h4"], uuid="Y")
    fresh_registry._key_to_uuid[ClusterKey.from_params({"host": "h1"})] = "X"
    fresh_registry._key_to_uuid[ClusterKey.from_params({"host": "h3"})] = "Y"
    fresh_registry.increment("X", "h1")
    fresh_registry.increment("Y", "h3")

    fresh_registry.clear()

    assert fresh_registry._clusters == {}
    assert fresh_registry._key_to_uuid == {}
    assert fresh_registry.get_load("X", "h1") == 0


# --------------------------------------------------------------------- get_load on missing


def test_get_load_missing_uuid_returns_zero(fresh_registry):
    assert fresh_registry.get_load("never-seen", "h1") == 0


def test_get_load_missing_host_returns_zero(fresh_registry):
    _install_state(fresh_registry, ["h1"])
    assert fresh_registry.get_load("X", "ghost") == 0


# --------------------------------------------------------------------- control reopen

class _FakeConn:
    """Stand-in for a psycopg Connection in control-reopen tests."""
    def __init__(self, host: str) -> None:
        self.host = host
        self.closed = False

    def close(self) -> None:
        self.closed = True


def _state_with_bootstrap(reg: ClusterRegistry, hosts: list[str], down: set[str] = frozenset()):
    state = _install_state(reg, hosts)
    state.bootstrap_conninfo = ""
    state.bootstrap_kwargs = {"dbname": "yugabyte", "user": "yugabyte"}
    state.control_sync = None
    for h in down:
        state.nodes[h].is_down = True
    return state


def test_ensure_control_sync_returns_existing_when_present(fresh_registry):
    state = _state_with_bootstrap(fresh_registry, ["h1", "h2"])
    existing = _FakeConn("h1")
    state.control_sync = existing
    assert fresh_registry._ensure_control_sync(state) is existing


def test_ensure_control_sync_opens_against_first_live_node(fresh_registry, monkeypatch):
    state = _state_with_bootstrap(fresh_registry, ["h1", "h2", "h3"])
    attempted: list[str] = []

    def fake_connect(conninfo, **kwargs):
        attempted.append(kwargs["host"])
        return _FakeConn(kwargs["host"])

    from psycopg.connection import Connection
    monkeypatch.setattr(Connection, "_connect_plain", classmethod(lambda cls, *a, **k: fake_connect(*a, **k)))

    ctrl = fresh_registry._ensure_control_sync(state)
    assert ctrl is not None
    assert ctrl.host == attempted[0]
    assert state.control_sync is ctrl


def test_ensure_control_sync_skips_down_nodes(fresh_registry, monkeypatch):
    # h1 and h2 are flagged down; only h3 is eligible.
    state = _state_with_bootstrap(fresh_registry, ["h1", "h2", "h3"], down={"h1", "h2"})

    def fake_connect(conninfo, **kwargs):
        return _FakeConn(kwargs["host"])

    from psycopg.connection import Connection
    monkeypatch.setattr(Connection, "_connect_plain", classmethod(lambda cls, *a, **k: fake_connect(*a, **k)))

    ctrl = fresh_registry._ensure_control_sync(state)
    assert ctrl is not None
    assert ctrl.host == "h3"


def test_ensure_control_sync_falls_over_to_next_when_first_refuses(
    fresh_registry, monkeypatch
):
    state = _state_with_bootstrap(fresh_registry, ["h1", "h2", "h3"])
    # h1 refuses, h2 accepts, h3 is never asked.
    refused = {"h1"}

    def fake_connect(conninfo, **kwargs):
        if kwargs["host"] in refused:
            raise OSError("connection refused")
        return _FakeConn(kwargs["host"])

    from psycopg.connection import Connection
    monkeypatch.setattr(Connection, "_connect_plain", classmethod(lambda cls, *a, **k: fake_connect(*a, **k)))

    ctrl = fresh_registry._ensure_control_sync(state)
    assert ctrl is not None
    assert ctrl.host != "h1"


def test_ensure_control_sync_marks_refusing_nodes_failed(
    fresh_registry, monkeypatch
):
    """A host that refuses the control-conn reopen is also marked failed so
    the dispatcher's next pick doesn't waste a real connect on it."""
    state = _state_with_bootstrap(fresh_registry, ["h1", "h2", "h3"])
    refused = {"h1"}

    def fake_connect(conninfo, **kwargs):
        if kwargs["host"] in refused:
            raise OSError("connection refused")
        return _FakeConn(kwargs["host"])

    from psycopg.connection import Connection
    monkeypatch.setattr(Connection, "_connect_plain", classmethod(lambda cls, *a, **k: fake_connect(*a, **k)))

    fresh_registry._ensure_control_sync(state)
    # h1 refused, so it's now flagged down.
    assert state.nodes["h1"].is_down is True
    # h2 accepted, so it's untouched.
    assert state.nodes["h2"].is_down is False
    # h3 was never tried (h2 won first), so it's untouched.
    assert state.nodes["h3"].is_down is False


def test_ensure_control_sync_returns_none_when_all_nodes_refuse(
    fresh_registry, monkeypatch
):
    state = _state_with_bootstrap(fresh_registry, ["h1", "h2"])

    from psycopg.connection import Connection
    def fake_connect(*a, **k):
        raise OSError("connection refused")
    monkeypatch.setattr(Connection, "_connect_plain", classmethod(lambda cls, *a, **k: fake_connect(*a, **k)))

    assert fresh_registry._ensure_control_sync(state) is None
    assert state.control_sync is None


def test_ensure_control_sync_returns_none_when_all_nodes_down(fresh_registry):
    state = _state_with_bootstrap(fresh_registry, ["h1", "h2"], down={"h1", "h2"})
    # No live candidates; never invokes _connect_plain.
    assert fresh_registry._ensure_control_sync(state) is None


def test_claim_refresh_slot_honours_interval(fresh_registry):
    state = _install_state(fresh_registry, ["h1"])
    state.last_refresh = time.monotonic()  # just refreshed
    # Within interval, force_refresh is False → no claim.
    assert fresh_registry._claim_refresh_slot(state, interval_s=300) is False


def test_claim_refresh_slot_force_refresh_wins(fresh_registry):
    state = _install_state(fresh_registry, ["h1"])
    state.last_refresh = time.monotonic()
    state.force_refresh = True
    assert fresh_registry._claim_refresh_slot(state, interval_s=300) is True
    # Claim clears force_refresh and updates last_refresh.
    assert state.force_refresh is False


def test_claim_refresh_slot_single_winner(fresh_registry):
    """Only one of N concurrent callers claims the slot per interval."""
    state = _install_state(fresh_registry, ["h1"])
    state.last_refresh = 0.0  # well past any interval
    state.force_refresh = False

    wins = 0
    for _ in range(50):
        if fresh_registry._claim_refresh_slot(state, interval_s=300):
            wins += 1
    # First call wins; subsequent calls in the same interval don't.
    assert wins == 1


def test_drop_control_sync_clears_and_flags_force_refresh(fresh_registry):
    state = _state_with_bootstrap(fresh_registry, ["h1", "h2"])
    state.control_sync = _FakeConn("h1")
    fresh_registry._drop_control_sync(state)
    assert state.control_sync is None
    assert state.force_refresh is True


def test_refresh_if_stale_reopens_after_drop(fresh_registry, monkeypatch):
    """End-to-end: control conn dies during refresh, NEXT refresh tick reopens
    against a survivor instead of going blind."""
    state = _state_with_bootstrap(fresh_registry, ["h1", "h2", "h3"])
    state.last_refresh = 0.0  # past any interval

    # First refresh: control_sync is None → _ensure_control_sync opens against h1.
    # That fetch then raises (simulating a half-dead host); _drop_control_sync
    # nulls control_sync and re-flags force_refresh.
    # Second refresh: control_sync is None → _ensure_control_sync opens against
    # h2 (because we mark h1 as refusing). fetch succeeds.

    refused: set[str] = set()
    fetch_fails: set[str] = set()
    discovered_nodes = [
        NodeInfo(
            host=h, public_ip=None, port=5433,
            placement=Placement("aws", "r", f"z{i}"),
            node_type="primary",
        ) for i, h in enumerate(["h1", "h2", "h3"])
    ]

    def fake_connect(conninfo, **kwargs):
        host = kwargs["host"]
        if host in refused:
            raise OSError("refused")
        return _FakeConn(host)

    def fake_fetch(conn):
        if conn.host in fetch_fails:
            raise RuntimeError("control conn died")
        return discovered_nodes, "X"

    from psycopg.connection import Connection
    from psycopg.yb import discovery
    monkeypatch.setattr(Connection, "_connect_plain", classmethod(lambda cls, *a, **k: fake_connect(*a, **k)))
    monkeypatch.setattr(discovery, "fetch_servers_sync", fake_fetch)

    # First refresh: h1 wins selection, but the fetch on h1 dies.
    fetch_fails.add("h1")
    fresh_registry.refresh_if_stale(state, interval_s=0)
    assert state.control_sync is None
    assert state.force_refresh is True

    # Second refresh: h1 now refuses (it's down). h2 accepts, fetch succeeds.
    refused.add("h1")
    fresh_registry.refresh_if_stale(state, interval_s=0)
    assert state.control_sync is not None
    assert state.control_sync.host == "h2"
    assert state.force_refresh is False


def test_refresh_if_stale_keeps_force_refresh_when_all_nodes_refuse(
    fresh_registry, monkeypatch
):
    state = _state_with_bootstrap(fresh_registry, ["h1", "h2"])
    state.last_refresh = 0.0
    state.force_refresh = False

    from psycopg.connection import Connection
    monkeypatch.setattr(
        Connection, "_connect_plain",
        classmethod(lambda cls, *a, **k: (_ for _ in ()).throw(OSError("refused"))),
    )

    fresh_registry.refresh_if_stale(state, interval_s=0)
    # No fresh control conn means no fetch happened; the next caller should
    # retry IMMEDIATELY, not wait another full interval.
    assert state.force_refresh is True
    assert state.control_sync is None
