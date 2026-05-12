"""
Process-wide registry of YugabyteDB clusters this driver has talked to.

Holds the connection-count map, the control connections used for
`yb_servers()` refresh, and the lookup index from ClusterKey → universe_uuid.
Mirrors pgjdbc-yb's `LoadBalanceService` static singleton, with two notable
adaptations for Python:

  * Two control connections per cluster (sync and async). Python's
    `Connection` and `AsyncConnection` are different types, so we can't share
    one across both worlds the way the JVM-side does.
  * `threading.Lock` (not `asyncio.Lock`) so sync and async callers share
    state. Critical sections never do I/O — they're a few dict lookups and
    integer increments. Holding `threading.Lock` blocks the event loop for
    microseconds, which is acceptable.
"""

# Copyright (C) 2026 Yugabyte

from __future__ import annotations

import time
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar

from . import discovery
from .node import NodeInfo

if TYPE_CHECKING:
    from ..connection import Connection
    from ..connection_async import AsyncConnection


@dataclass(frozen=True)
class ClusterKey:
    """Pre-bootstrap fingerprint derived from the user's conninfo.

    The registry's primary key is `universe_uuid` (which only the server can
    issue). `ClusterKey` is a fast-path cache: hit means "we've seen this
    cluster, skip bootstrap", miss means "open a bootstrap connection".

    Two different `ClusterKey`s can resolve to the same `universe_uuid`
    (e.g. one user specifies `host=h1`, another specifies `host=h2,h3`).
    Bootstrap dedups against the uuid the moment it has one; both keys then
    map to the same `ClusterState`.

    Password and SSL mode are deliberately excluded — neither is part of
    cluster identity, and keeping the password out of long-lived memory is
    a small hardening win.
    """

    hosts: tuple[str, ...]  # sorted, lowercased
    port: int
    dbname: str
    user: str

    @classmethod
    def from_params(cls, pg_kwargs: dict[str, object]) -> "ClusterKey":
        host_str = str(pg_kwargs.get("host", ""))
        hosts = tuple(sorted(
            h.strip().lower() for h in host_str.split(",") if h.strip()
        ))
        return cls(
            hosts=hosts,
            port=int(pg_kwargs.get("port", 5433) or 5433),
            dbname=str(pg_kwargs.get("dbname", "yugabyte") or "yugabyte"),
            user=str(pg_kwargs.get("user", "yugabyte") or "yugabyte"),
        )


@dataclass
class ClusterState:
    """All registry state pertaining to one YugabyteDB cluster.

    `nodes` is the host → connection-count map. Mutated only under `lock`.
    `control_sync` and `control_async` are long-lived connections kept open
    for `yb_servers()` refresh — see design doc §4.3 callout.

    `bootstrap_conninfo` and `bootstrap_kwargs` capture the original connect
    parameters (cleaned of YB-only keys) so the registry can re-open a fresh
    control connection against any live node when the cached one breaks.
    Without these, a dead control host would leave us blind to topology
    changes until the entire process restarts.
    """

    uuid: str
    nodes: dict[str, NodeInfo]
    lock: threading.Lock
    last_refresh: float
    force_refresh: bool = False
    # Connections are typed loosely to avoid a hard import dependency.
    control_sync: "Connection | None" = None
    control_async: "AsyncConnection | None" = None
    bootstrap_conninfo: str = ""
    bootstrap_kwargs: dict = field(default_factory=dict)


class ClusterRegistry:
    """Process-wide singleton holding all cluster state.

    Public API:

      * `instance()` — double-checked-locking accessor
      * `get_or_bootstrap` / `aget_or_bootstrap` — sync / async bootstrap
      * `refresh_if_stale` / `arefresh_if_stale` — lazy refresh
      * `increment` / `decrement` — per-connect bookkeeping
      * `mark_failed` — quarantine a node after connect failure

    Test hooks (mirroring pgjdbc-yb's `LoadBalanceService.getLoad/clear`):

      * `get_load(uuid, host)` — read current count without mutating
      * `clear()` — drop all state (closes sync control conns best-effort)
    """

    _instance: ClassVar["ClusterRegistry | None"] = None
    _instance_lock: ClassVar[threading.Lock] = threading.Lock()

    def __init__(self) -> None:
        # Protects the two maps (their identity, not their values' identity).
        # Per-cluster ClusterState.lock protects each cluster's nodes dict.
        self._lock = threading.Lock()
        self._clusters: dict[str, ClusterState] = {}
        self._key_to_uuid: dict[ClusterKey, str] = {}

    @classmethod
    def instance(cls) -> "ClusterRegistry":
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    # ------------------------------------------------------------------ bootstrap
    #
    # The pattern in each flavour:
    #   1. Check the cache under the registry lock. Hit → return.
    #   2. Miss → release the lock and open a bootstrap connection (I/O outside
    #      the lock so we never block other callers).
    #   3. Run `yb_servers()` + `yb_universe_uuid()` on the new connection.
    #   4. Re-acquire the registry lock. If `_clusters[uuid]` already exists
    #      (another caller beat us to it), install the alias key and either
    #      attach our conn as the missing control connection or close it.
    #      Otherwise, build the new `ClusterState` and install.

    def get_or_bootstrap(
        self, key: ClusterKey, conninfo: str, kwargs: dict[str, object]
    ) -> ClusterState:
        with self._lock:
            uuid = self._key_to_uuid.get(key)
            if uuid is not None:
                return self._clusters[uuid]

        # Late import to break the cycle (connection_async imports from yb/).
        # The async version of this helper is `_aconnect_plain`; the sync sibling
        # is renamed to `_connect_plain` by async_to_sync's names_map.
        from ..connection import Connection
        conn = Connection._connect_plain(conninfo, **kwargs)
        try:
            nodes, uuid = discovery.fetch_servers_sync(conn)
        except Exception:
            try:
                conn.close()
            except Exception:
                pass
            raise

        with self._lock:
            existing = self._clusters.get(uuid)
            if existing is not None:
                self._key_to_uuid[key] = uuid
                if existing.control_sync is None:
                    existing.control_sync = conn
                else:
                    try:
                        conn.close()
                    except Exception:
                        pass
                return existing

            state = ClusterState(
                uuid=uuid,
                nodes={n.host: n for n in nodes},
                lock=threading.Lock(),
                last_refresh=time.monotonic(),
                control_sync=conn,
                bootstrap_conninfo=conninfo,
                bootstrap_kwargs=dict(kwargs),
            )
            self._clusters[uuid] = state
            self._key_to_uuid[key] = uuid
            return state

    async def aget_or_bootstrap(
        self, key: ClusterKey, conninfo: str, kwargs: dict[str, object]
    ) -> ClusterState:
        with self._lock:
            uuid = self._key_to_uuid.get(key)
            if uuid is not None:
                return self._clusters[uuid]

        from ..connection_async import AsyncConnection
        conn = await AsyncConnection._aconnect_plain(conninfo, **kwargs)
        try:
            nodes, uuid = await discovery.fetch_servers_async(conn)
        except Exception:
            try:
                await conn.close()
            except Exception:
                pass
            raise

        with self._lock:
            existing = self._clusters.get(uuid)
            if existing is not None:
                self._key_to_uuid[key] = uuid
                if existing.control_async is None:
                    existing.control_async = conn
                else:
                    try:
                        await conn.close()
                    except Exception:
                        pass
                return existing

            state = ClusterState(
                uuid=uuid,
                nodes={n.host: n for n in nodes},
                lock=threading.Lock(),
                last_refresh=time.monotonic(),
                control_async=conn,
                bootstrap_conninfo=conninfo,
                bootstrap_kwargs=dict(kwargs),
            )
            self._clusters[uuid] = state
            self._key_to_uuid[key] = uuid
            return state

    # ------------------------------------------------------------------ refresh

    def refresh_if_stale(self, state: ClusterState, interval_s: int) -> None:
        if not self._claim_refresh_slot(state, interval_s):
            return
        if not self._do_refresh_sync(state):
            # No refresh succeeded (no live control conn, or fetch failed on
            # every attempt). Re-flag so the next caller retries immediately
            # instead of waiting a full interval.
            with state.lock:
                state.force_refresh = True

    async def arefresh_if_stale(self, state: ClusterState, interval_s: int) -> None:
        if not self._claim_refresh_slot(state, interval_s):
            return
        if not await self._ado_refresh_async(state):
            with state.lock:
                state.force_refresh = True

    def _do_refresh_sync(self, state: ClusterState) -> bool:
        """Attempt the fetch+merge once. If the cached control conn fails,
        drop it, re-open against a survivor, and retry once more in the SAME
        refresh tick. Returns True iff a merge happened.

        Single-cycle recovery (vs the older "drop now, reopen next tick"
        pattern) means the very first connect after a control-host loss
        sees the refreshed topology and `is_down` state — so the policy's
        first pick can already skip the dead host and discover newly-added
        nodes, without needing a second connect to drive the reopen.
        """
        for _ in range(2):
            ctrl = self._ensure_control_sync(state)
            if ctrl is None:
                return False
            try:
                new_nodes, _ = discovery.fetch_servers_sync(ctrl)
            except Exception:
                self._drop_control_sync(state)
                continue
            self._merge_new_nodes(state, new_nodes)
            return True
        return False

    async def _ado_refresh_async(self, state: ClusterState) -> bool:
        for _ in range(2):
            ctrl = await self._aensure_control_async(state)
            if ctrl is None:
                return False
            try:
                new_nodes, _ = await discovery.fetch_servers_async(ctrl)
            except Exception:
                await self._drop_control_async(state)
                continue
            self._merge_new_nodes(state, new_nodes)
            return True
        return False

    def _claim_refresh_slot(self, state: ClusterState, interval_s: int) -> bool:
        """Return True iff this caller has won the right to run a refresh.

        Only one caller succeeds per interval window. Resets `last_refresh`
        and clears `force_refresh` as part of the claim — losers wait.
        """
        now = time.monotonic()
        with state.lock:
            if not state.force_refresh and (now - state.last_refresh) < interval_s:
                return False
            state.last_refresh = now
            state.force_refresh = False
            return True

    def _ensure_control_sync(
        self, state: ClusterState
    ) -> "Connection | None":
        """Return a working control conn, opening a fresh one if needed.

        Tries `state.bootstrap_kwargs` (overriding host/port) against each
        non-down node until one accepts. The bootstrap host being dead is
        irrelevant — we use the cached node list to fail over.
        """
        with state.lock:
            ctrl = state.control_sync
        if ctrl is not None:
            return ctrl
        if not state.bootstrap_kwargs and not state.bootstrap_conninfo:
            return None
        with state.lock:
            candidates = [n for n in state.nodes.values() if not n.is_down]

        from ..connection import Connection
        for node in candidates:
            per_host = {
                **state.bootstrap_kwargs,
                "host": node.host,
                "port": node.port,
            }
            try:
                conn = Connection._connect_plain(state.bootstrap_conninfo, **per_host)
            except Exception:
                # We just discovered this host refuses connections. Mark it
                # failed so the dispatcher's next pick doesn't waste a real
                # connect attempt on it.
                self.mark_failed(state.uuid, node.host)
                continue
            with state.lock:
                if state.control_sync is None:
                    state.control_sync = conn
                    return conn
                winner = state.control_sync
            try:
                conn.close()
            except Exception:
                pass
            return winner
        return None

    async def _aensure_control_async(
        self, state: ClusterState
    ) -> "AsyncConnection | None":
        with state.lock:
            ctrl = state.control_async
        if ctrl is not None:
            return ctrl
        if not state.bootstrap_kwargs and not state.bootstrap_conninfo:
            return None
        with state.lock:
            candidates = [n for n in state.nodes.values() if not n.is_down]

        from ..connection_async import AsyncConnection
        for node in candidates:
            per_host = {
                **state.bootstrap_kwargs,
                "host": node.host,
                "port": node.port,
            }
            try:
                conn = await AsyncConnection._aconnect_plain(
                    state.bootstrap_conninfo, **per_host
                )
            except Exception:
                # Refusing host → mark it failed so the dispatcher's next
                # pick skips it for the TTL window.
                self.mark_failed(state.uuid, node.host)
                continue
            with state.lock:
                if state.control_async is None:
                    state.control_async = conn
                    return conn
                winner = state.control_async
            try:
                await conn.close()
            except Exception:
                pass
            return winner
        return None

    def _merge_new_nodes(
        self, state: ClusterState, new_nodes: list[NodeInfo]
    ) -> None:
        """Replace state.nodes with the freshly-discovered list, preserving counters.

        Called only on the success path of `refresh_if_stale`, so this is
        the "we just refreshed" moment. Clear `force_refresh` here — any
        flag set during the refresh itself (e.g. `mark_failed` triggered by
        `_ensure_control_sync` refusing a dead host) is now stale.
        """
        with state.lock:
            new_dict: dict[str, NodeInfo] = {}
            for n in new_nodes:
                if (existing := state.nodes.get(n.host)) is not None:
                    n.connection_count = existing.connection_count
                    n.is_down = existing.is_down
                    n.is_down_since = existing.is_down_since
                new_dict[n.host] = n
            state.nodes = new_dict
            state.force_refresh = False

    def _drop_control_sync(self, state: ClusterState) -> None:
        with state.lock:
            ctrl = state.control_sync
            state.control_sync = None
            state.force_refresh = True
        if ctrl is not None:
            try:
                ctrl.close()
            except Exception:
                pass

    async def _drop_control_async(self, state: ClusterState) -> None:
        with state.lock:
            ctrl = state.control_async
            state.control_async = None
            state.force_refresh = True
        if ctrl is not None:
            try:
                await ctrl.close()
            except Exception:
                pass

    # ------------------------------------------------------------------ counts

    def increment(self, uuid: str, host: str) -> None:
        state = self._clusters.get(uuid)
        if state is None:
            return
        with state.lock:
            if (ni := state.nodes.get(host)) is not None:
                ni.connection_count += 1

    def decrement(self, uuid: str, host: str) -> None:
        state = self._clusters.get(uuid)
        if state is None:
            return
        with state.lock:
            if (ni := state.nodes.get(host)) is not None:
                ni.connection_count = max(0, ni.connection_count - 1)

    def mark_failed(self, uuid: str, host: str) -> None:
        state = self._clusters.get(uuid)
        if state is None:
            return
        now = time.monotonic()
        with state.lock:
            if (ni := state.nodes.get(host)) is not None:
                ni.is_down = True
                ni.is_down_since = now
                ni.connection_count = 0
            state.force_refresh = True

    # ------------------------------------------------------------------ test hooks
    #
    # Match pgjdbc-yb's LoadBalanceService.getLoad / clear public surface so
    # tests asserting per-host load can read the same values the policy reads.

    def get_load(self, uuid: str, host: str) -> int:
        state = self._clusters.get(uuid)
        if state is None:
            return 0
        with state.lock:
            ni = state.nodes.get(host)
            return ni.connection_count if ni is not None else 0

    def clear(self) -> None:
        """Drop all state. Best-effort closes any control connections held.

        Async control connections are closed by force-finishing the underlying
        libpq handle directly (bypassing the async wrapper) — this lets us
        clean up cleanly from a sync teardown without needing an event loop,
        while still preventing `BaseConnection.__del__` from emitting the
        "deleted while still open" ResourceWarning at GC time.
        """
        with self._lock:
            clusters = list(self._clusters.values())
            self._clusters.clear()
            self._key_to_uuid.clear()
        for state in clusters:
            if state.control_sync is not None:
                try:
                    state.control_sync.close()
                except Exception:
                    pass
            if state.control_async is not None:
                # Force-finish the libpq handle and flag the wrapper as closed.
                # We cannot `await close()` from this sync context, but the
                # underlying socket is a libpq object that doesn't need the
                # event loop to be finished.
                try:
                    state.control_async.pgconn.finish()
                    state.control_async._closed = True
                except Exception:
                    pass

    async def aclear(self) -> None:
        """Async sibling of `clear`. Prefer this in async tests so the
        control_async wrapper's `close()` runs through the event loop cleanly."""
        with self._lock:
            clusters = list(self._clusters.values())
            self._clusters.clear()
            self._key_to_uuid.clear()
        for state in clusters:
            if state.control_sync is not None:
                try:
                    state.control_sync.close()
                except Exception:
                    pass
            if state.control_async is not None:
                try:
                    await state.control_async.close()
                except Exception:
                    pass
