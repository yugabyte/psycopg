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

import logging
import time
import threading
import weakref
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar

from . import TRACE, discovery
from .node import NodeInfo

logger = logging.getLogger(__name__)


def _safe_host(conn) -> str:
    """Best-effort `conn.info.host`. Returns "?" if info isn't available
    (e.g. mocked connection in unit tests, or after the conn was force-
    finished by `clear()`)."""
    try:
        return conn.info.host
    except Exception:
        return "?"

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
    # Weakref set of direct-connect conns opened against this cluster
    # (Phase D — see design doc §3.1). Populated by the dispatcher on
    # every successful ``psycopg.connect(dsn)``; pool-borrowed conns are
    # NOT added here because the pool already holds strong references.
    # The drain walk (Phase E) reads this set to find in-flight
    # transactions on the outgoing cluster.
    tracked_conns: "weakref.WeakSet" = field(default_factory=weakref.WeakSet)


@dataclass
class FailoverGroup:
    """A primary + secondary cluster pair for xCluster failover.

    See docs/xcluster_failover_design.html §3–§4 (design doc v2) for the
    full design. Each cluster has its own circuit breaker + status flag;
    the dispatcher reads both via ``state.active_cluster(group)`` on every
    connect to pick the target cluster.

    Per-cluster health state — written under ``lock``. The probe thread
    writes via each CB's ``check()``; test code can flip directly via
    ``force_primary_status`` / ``force_secondary_status``:

      * ``primary_status`` / ``secondary_status`` — HEALTHY or UNHEALTHY
      * ``primary_last_transition_time`` / ``secondary_last_transition_time``
        — wall-clock (monotonic) of the last transition. ``0.0`` at
        construction so a newly-spawned process can transition on its very
        first probe tick — no "fresh-app starvation" window.
      * ``primary_circuit_breaker`` / ``secondary_circuit_breaker`` — the CB
        instances. Default at bootstrap is a ``TrackerTableCircuitBreaker``
        pointed at each cluster; tests can assign their own. If either is
        ``None``, ``check_primary_cluster`` / ``check_secondary_cluster``
        falls back to HEALTHY.

    ``cooldown_s`` applies symmetrically to both — the probe consults
    ``can_transition_primary`` / ``can_transition_secondary`` before writing
    a status change (§7.3 Q1: symmetric-parameters default).

    ``tracker_table_tablets`` and ``max_update_failures_allowed`` are the
    per-cluster CB knobs, kept on the group so synthetic test groups can
    inspect them without reaching into the CBs.

    ``primary_probe`` / ``secondary_probe`` are the per-cluster
    ``HealthProbe`` daemons. Each polls its own cluster and refreshes the
    topology on the same connection. Lifecycle is managed by
    ``ClusterRegistry`` — created at bootstrap, stopped by ``clear()``.
    """

    primary: ClusterState
    secondary: ClusterState
    lock: threading.Lock
    # Per-cluster status. See class docstring.
    primary_status: "HealthResult"
    secondary_status: "HealthResult"
    primary_last_transition_time: float = 0.0
    secondary_last_transition_time: float = 0.0
    cooldown_s: int = 1500
    # Tracker-table config — consumed by the default
    # TrackerTableCircuitBreaker instances (one per cluster).
    tracker_table_tablets: int = 9
    max_update_failures_allowed: int = 0
    # Per-cluster probes. Type kept loose to avoid a circular import on
    # health_probe.HealthProbe.
    primary_probe: "object | None" = None
    secondary_probe: "object | None" = None
    # The per-cluster CircuitBreaker instances. Set during bootstrap by
    # `get_or_bootstrap_failover_group`. Tests can assign their own. If
    # left `None`, the `check_primary_cluster` / `check_secondary_cluster`
    # delegating helpers return HEALTHY.
    primary_circuit_breaker: "object | None" = None
    secondary_circuit_breaker: "object | None" = None
    # Dispatch pause flag (Phase D — see design doc §3). When True, new
    # connection opens block on ``dispatch_paused_condition`` until the
    # drain sequence flips it back to False. Existing conns are NOT
    # affected — only new opens gate on this. Written under ``lock``.
    dispatch_paused: bool = False
    # Condition bound to ``lock`` for pause/resume notify. Constructed in
    # ``__post_init__`` so it wraps whatever lock was passed in (or
    # created by default).
    dispatch_paused_condition: "threading.Condition | None" = field(
        default=None, repr=False,
    )

    def __post_init__(self) -> None:
        # Bind the condition to `lock` — so wait/notify releases and re-
        # acquires the same lock the rest of the group's state changes
        # under. Skip if the caller already provided one (rare — some
        # tests may want to inject a mock).
        if self.dispatch_paused_condition is None:
            self.dispatch_paused_condition = threading.Condition(self.lock)

    def pause_dispatch(self) -> None:
        """Set ``dispatch_paused = True``. Called from the drain sequence
        (Phase E) at the start of a failover. Every ``group.lock``-held
        write must set-and-notify together, so we expose this as a
        method rather than relying on callers to remember the notify."""
        with self.lock:
            self.dispatch_paused = True

    def resume_dispatch(self) -> None:
        """Set ``dispatch_paused = False`` and notify all waiters. Called
        from the drain sequence when the barrier releases."""
        with self.lock:
            self.dispatch_paused = False
            assert self.dispatch_paused_condition is not None
            self.dispatch_paused_condition.notify_all()

    def force_primary_status(self, status: "HealthResult") -> None:
        """TEST HOOK. Manually set ``primary_status``.

        In production, this is set only by the probe thread running the
        primary CB's ``check()``. Integration tests use this to drive
        failover deterministically without spinning up a real cluster.
        """
        now = time.monotonic()
        with self.lock:
            self.primary_status = status
            self.primary_last_transition_time = now

    def force_secondary_status(self, status: "HealthResult") -> None:
        """TEST HOOK. Manually set ``secondary_status``. Symmetric with
        :meth:`force_primary_status`."""
        now = time.monotonic()
        with self.lock:
            self.secondary_status = status
            self.secondary_last_transition_time = now

    def can_transition_primary(self, now: float) -> bool:
        """Return True iff the cool-down window has elapsed since the last
        primary transition. ``primary_last_transition_time = 0.0`` at
        construction → always True on first call."""
        return (now - self.primary_last_transition_time) >= self.cooldown_s

    def can_transition_secondary(self, now: float) -> bool:
        """Symmetric with :meth:`can_transition_primary` for secondary."""
        return (now - self.secondary_last_transition_time) >= self.cooldown_s


class ClusterRegistry:
    """Process-wide singleton holding all cluster state.

    Public API:

      * `instance()` — double-checked-locking accessor
      * `get_or_bootstrap` / `aget_or_bootstrap` — sync / async bootstrap
      * `refresh_if_stale` / `arefresh_if_stale` — lazy refresh
      * `increment` / `decrement` — per-connect bookkeeping
      * `mark_failed` — quarantine a node after connect failure

    xCluster failover (when `yb_params.xcluster_enabled` is True):

      * `get_or_bootstrap_failover_group` / `aget_or_bootstrap_failover_group`
        — pair primary + secondary cluster states under one `FailoverGroup`
      * `get_failover_group(primary_uuid)` — by primary uuid
      * `get_failover_group_by_uuid(any_uuid)` — searches both primary AND
        secondary uuids (used by `xcluster_check` pool callback)
      * `reset_failover_group(primary_uuid)` — operator failback API:
        flips status back to HEALTHY immediately
      * `get_failover_status(primary_uuid)` — read current status + last
        transition timestamp

    Test hooks (mirroring pgjdbc-yb's `LoadBalanceService.getLoad/clear`):

      * `get_load(uuid, host)` — read current count without mutating
      * `clear()` — drop all state (closes sync control conns best-effort,
        stops all probe threads)
    """

    _instance: ClassVar["ClusterRegistry | None"] = None
    _instance_lock: ClassVar[threading.Lock] = threading.Lock()

    def __init__(self) -> None:
        # Protects the three maps (their identity, not their values' identity).
        # Per-cluster ClusterState.lock protects each cluster's nodes dict.
        # Per-group FailoverGroup.lock protects each group's status flag.
        self._lock = threading.Lock()
        self._clusters: dict[str, ClusterState] = {}
        self._key_to_uuid: dict[ClusterKey, str] = {}
        # xCluster failover: keyed by PRIMARY uuid (one group per primary cluster).
        self._failover_groups: dict[str, "FailoverGroup"] = {}

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
                logger.debug("cluster cache hit: key=%s → uuid=%s", key, uuid)
                return self._clusters[uuid]

        # Late import to break the cycle (connection_async imports from yb/).
        # The async version of this helper is `_aconnect_plain`; the sync sibling
        # is renamed to `_connect_plain` by async_to_sync's names_map.
        from ..connection import Connection
        logger.debug("bootstrapping cluster via contact host(s): %s", kwargs.get("host"))
        conn = Connection._connect_plain(conninfo, **kwargs)
        try:
            nodes, uuid = discovery.fetch_servers_sync(conn)
        except Exception:
            logger.warning("bootstrap fetch_servers failed; closing contact conn")
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
                    logger.debug(
                        "bootstrap race: aliasing key=%s to existing uuid=%s; "
                        "attaching our conn as the sync control connection",
                        key, uuid,
                    )
                    existing.control_sync = conn
                else:
                    logger.debug(
                        "bootstrap race: aliasing key=%s to existing uuid=%s; "
                        "discarding redundant contact conn",
                        key, uuid,
                    )
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
            logger.info(
                "cluster bootstrapped: uuid=%s, %d nodes (%s); sync control conn on %s",
                uuid, len(nodes),
                ",".join(n.host for n in nodes),
                _safe_host(conn),
            )
            for n in nodes:
                logger.log(TRACE, "  discovered node: %s", n)
            return state

    async def aget_or_bootstrap(
        self, key: ClusterKey, conninfo: str, kwargs: dict[str, object]
    ) -> ClusterState:
        with self._lock:
            uuid = self._key_to_uuid.get(key)
            if uuid is not None:
                logger.debug("cluster cache hit (async): key=%s → uuid=%s", key, uuid)
                return self._clusters[uuid]

        from ..connection_async import AsyncConnection
        logger.debug(
            "bootstrapping cluster (async) via contact host(s): %s",
            kwargs.get("host"),
        )
        conn = await AsyncConnection._aconnect_plain(conninfo, **kwargs)
        try:
            nodes, uuid = await discovery.fetch_servers_async(conn)
        except Exception:
            logger.warning("bootstrap fetch_servers (async) failed; closing contact conn")
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
                    logger.debug(
                        "bootstrap race (async): aliasing key=%s to existing uuid=%s; "
                        "attaching our conn as the async control connection",
                        key, uuid,
                    )
                    existing.control_async = conn
                else:
                    logger.debug(
                        "bootstrap race (async): aliasing key=%s to existing uuid=%s; "
                        "discarding redundant contact conn",
                        key, uuid,
                    )
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
            logger.info(
                "cluster bootstrapped (async): uuid=%s, %d nodes (%s); "
                "async control conn on %s",
                uuid, len(nodes),
                ",".join(n.host for n in nodes),
                _safe_host(conn),
            )
            for n in nodes:
                logger.log(TRACE, "  discovered node: %s", n)
            return state

    # ------------------------------------------------------------------ xCluster failover
    #
    # `get_or_bootstrap_failover_group` (and its async sibling) is the entry
    # point the dispatcher calls when `yb_params.xcluster_enabled` is True.
    # The pattern:
    #   1. Bootstrap the primary (re-uses `get_or_bootstrap` — cached on
    #      repeat calls with the same ClusterKey).
    #   2. Check `_failover_groups[primary.uuid]` — return if already present.
    #   3. Bootstrap the secondary using the same kwargs but with the host
    #      list overridden to `yb_params.secondary_cluster_hosts`. The
    #      secondary becomes its own ClusterState in `_clusters` under its
    #      own uuid.
    #   4. Construct the `FailoverGroup` and install it (race-checked).
    #   5. Start the `HealthProbe` thread on the group.

    def _make_secondary_kwargs(
        self, kwargs: dict[str, object], secondary_hosts: list[str]
    ) -> dict[str, object]:
        """Return a copy of `kwargs` with `host` replaced by the secondary
        cluster's host list. All other params (port, user, dbname,
        ssl-related, topology, refresh interval, ...) carry over verbatim —
        spec §5 says existing smart-driver params apply to both clusters."""
        sec_kwargs = dict(kwargs)
        sec_kwargs["host"] = ",".join(secondary_hosts)
        return sec_kwargs

    def get_or_bootstrap_failover_group(
        self,
        yb_params,                                  # YBParams; lazy-typed to dodge cycle
        conninfo: str,
        kwargs: dict[str, object],
    ) -> "FailoverGroup":
        """Sync bootstrap of a primary + secondary `FailoverGroup`. Idempotent
        on primary uuid. Spec §5 / design §3."""
        from .health import HealthResult            # late import — health.py imports FailoverGroup forward-ref

        # 1. Bootstrap primary (cached on repeats).
        primary_key = ClusterKey.from_params(
            self._param_dict_for_key(conninfo, kwargs)
        )
        primary = self.get_or_bootstrap(primary_key, conninfo, kwargs)

        # 2. Already paired?
        with self._lock:
            existing = self._failover_groups.get(primary.uuid)
        if existing is not None:
            logger.debug(
                "failover group cache hit: primary_uuid=%s", primary.uuid
            )
            return existing

        # 3. Bootstrap secondary.
        sec_kwargs = self._make_secondary_kwargs(
            kwargs, yb_params.secondary_cluster_hosts
        )
        secondary_key = ClusterKey.from_params(
            self._param_dict_for_key(conninfo, sec_kwargs)
        )
        secondary = self.get_or_bootstrap(secondary_key, conninfo, sec_kwargs)

        # 4. Install (race-checked).
        with self._lock:
            existing = self._failover_groups.get(primary.uuid)
            if existing is not None:
                logger.debug(
                    "failover group race lost; discarding our pairing for "
                    "primary_uuid=%s", primary.uuid,
                )
                return existing
            # Late import — circuit_breaker imports HealthResult from
            # health.py, which forward-refs FailoverGroup → cycle if we
            # import at module top.
            from .circuit_breaker import TrackerTableCircuitBreaker
            group = FailoverGroup(
                primary=primary,
                secondary=secondary,
                lock=threading.Lock(),
                primary_status=HealthResult.HEALTHY,
                secondary_status=HealthResult.HEALTHY,
                cooldown_s=yb_params.cooldown_s,
                tracker_table_tablets=yb_params.tracker_table_tablets,
                max_update_failures_allowed=yb_params.max_update_failures_allowed,
                primary_circuit_breaker=TrackerTableCircuitBreaker(
                    which_cluster="primary",
                    tracker_table_tablets=yb_params.tracker_table_tablets,
                    max_update_failures_allowed=yb_params.max_update_failures_allowed,
                ),
                secondary_circuit_breaker=TrackerTableCircuitBreaker(
                    which_cluster="secondary",
                    tracker_table_tablets=yb_params.tracker_table_tablets,
                    max_update_failures_allowed=yb_params.max_update_failures_allowed,
                ),
            )
            self._failover_groups[primary.uuid] = group

        logger.info(
            "failover group bootstrapped: primary_uuid=%s, secondary_uuid=%s, "
            "secondary_hosts=%s, cooldown_s=%d",
            primary.uuid, secondary.uuid,
            ",".join(yb_params.secondary_cluster_hosts),
            yb_params.cooldown_s,
        )

        # 5. Start the probe thread (Phase 4 wiring).
        self._start_probe(group, yb_params.refresh_interval_s, yb_params.check_timeout_s, yb_params.drain_timeout_s)
        return group

    async def aget_or_bootstrap_failover_group(
        self,
        yb_params,
        conninfo: str,
        kwargs: dict[str, object],
    ) -> "FailoverGroup":
        """Async sibling. Same shape, awaits the async primary + secondary
        bootstraps. The probe thread itself is sync regardless of caller
        (see design doc §10)."""
        from .health import HealthResult

        primary_key = ClusterKey.from_params(
            self._param_dict_for_key(conninfo, kwargs)
        )
        primary = await self.aget_or_bootstrap(primary_key, conninfo, kwargs)

        with self._lock:
            existing = self._failover_groups.get(primary.uuid)
        if existing is not None:
            logger.debug(
                "failover group cache hit (async): primary_uuid=%s", primary.uuid
            )
            return existing

        sec_kwargs = self._make_secondary_kwargs(
            kwargs, yb_params.secondary_cluster_hosts
        )
        secondary_key = ClusterKey.from_params(
            self._param_dict_for_key(conninfo, sec_kwargs)
        )
        secondary = await self.aget_or_bootstrap(
            secondary_key, conninfo, sec_kwargs
        )

        with self._lock:
            existing = self._failover_groups.get(primary.uuid)
            if existing is not None:
                logger.debug(
                    "failover group race lost (async); discarding our pairing "
                    "for primary_uuid=%s", primary.uuid,
                )
                return existing
            # Late import — circuit_breaker imports HealthResult from
            # health.py, which forward-refs FailoverGroup → cycle if we
            # import at module top.
            from .circuit_breaker import TrackerTableCircuitBreaker
            group = FailoverGroup(
                primary=primary,
                secondary=secondary,
                lock=threading.Lock(),
                primary_status=HealthResult.HEALTHY,
                secondary_status=HealthResult.HEALTHY,
                cooldown_s=yb_params.cooldown_s,
                tracker_table_tablets=yb_params.tracker_table_tablets,
                max_update_failures_allowed=yb_params.max_update_failures_allowed,
                primary_circuit_breaker=TrackerTableCircuitBreaker(
                    which_cluster="primary",
                    tracker_table_tablets=yb_params.tracker_table_tablets,
                    max_update_failures_allowed=yb_params.max_update_failures_allowed,
                ),
                secondary_circuit_breaker=TrackerTableCircuitBreaker(
                    which_cluster="secondary",
                    tracker_table_tablets=yb_params.tracker_table_tablets,
                    max_update_failures_allowed=yb_params.max_update_failures_allowed,
                ),
            )
            self._failover_groups[primary.uuid] = group

        logger.info(
            "failover group bootstrapped (async): primary_uuid=%s, "
            "secondary_uuid=%s, secondary_hosts=%s, cooldown_s=%d",
            primary.uuid, secondary.uuid,
            ",".join(yb_params.secondary_cluster_hosts),
            yb_params.cooldown_s,
        )

        self._start_probe(group, yb_params.refresh_interval_s, yb_params.check_timeout_s, yb_params.drain_timeout_s)
        return group

    def _param_dict_for_key(
        self, conninfo: str, kwargs: dict[str, object]
    ) -> dict[str, object]:
        """Build the dict ClusterKey wants for hashing. Mirrors the way the
        dispatcher constructs ClusterKey before calling `get_or_bootstrap`
        — late import of `conninfo_to_dict` to avoid pulling psycopg's
        full conninfo module at registry import time."""
        from ..conninfo import conninfo_to_dict
        return conninfo_to_dict(conninfo, **kwargs)

    def _start_probe(
        self,
        group: "FailoverGroup",
        interval_s: int,
        check_timeout_s: float = 1.0,
        drain_timeout_s: int = 10,
    ) -> None:
        """Spin up one daemon ``HealthProbe`` per cluster (Phase B — see
        design doc §4.1 "topology refresh rides on the same thread").
        Each probe polls its cluster's CB with a wall-clock cap of
        ``check_timeout_s`` and refreshes ``yb_servers()`` on the same
        tick. When a transition fires the probe invokes
        ``drain.trigger_drain`` with ``drain_timeout_s`` (Phase E).
        Idempotent — safe to call twice."""
        from .health_probe import HealthProbe       # late import; cycle-safe
        if group.primary_probe is None:
            p = HealthProbe(
                group, interval_s,
                which_cluster="primary",
                check_timeout_s=check_timeout_s,
                drain_timeout_s=drain_timeout_s,
            )
            group.primary_probe = p
            p.start()
        if group.secondary_probe is None:
            s = HealthProbe(
                group, interval_s,
                which_cluster="secondary",
                check_timeout_s=check_timeout_s,
                drain_timeout_s=drain_timeout_s,
            )
            group.secondary_probe = s
            s.start()

    def get_failover_group(self, primary_uuid: str) -> "FailoverGroup | None":
        """Lookup by primary uuid. Returns None if no group exists for it."""
        with self._lock:
            return self._failover_groups.get(primary_uuid)

    def get_failover_group_by_uuid(
        self, any_uuid: str
    ) -> "FailoverGroup | None":
        """Lookup by EITHER primary or secondary uuid. The pool's
        `xcluster_check` (Phase 6) calls this on a conn whose `_yb_uuid`
        could be either side of the pair."""
        with self._lock:
            for group in self._failover_groups.values():
                if group.primary.uuid == any_uuid or group.secondary.uuid == any_uuid:
                    return group
            return None

    def reset_failover_group(self, primary_uuid: str) -> bool:
        """Operator failback API. Flips BOTH primary and secondary status
        back to HEALTHY immediately. Returns True if a group existed and
        was reset, False otherwise. Equivalent of JDBC's JMX failback
        operation (design doc §8)."""
        from .health import HealthResult
        group = self.get_failover_group(primary_uuid)
        if group is None:
            return False
        group.force_primary_status(HealthResult.HEALTHY)
        group.force_secondary_status(HealthResult.HEALTHY)
        logger.info(
            "failover group reset to HEALTHY by operator: primary_uuid=%s",
            primary_uuid,
        )
        return True

    def get_failover_status(
        self, primary_uuid: str
    ) -> "tuple[HealthResult, HealthResult, float, float] | None":
        """Read current ``(primary_status, secondary_status,
        primary_last_transition_time, secondary_last_transition_time)``
        without mutating. Returns None if no group exists for the uuid."""
        group = self.get_failover_group(primary_uuid)
        if group is None:
            return None
        with group.lock:
            return (
                group.primary_status,
                group.secondary_status,
                group.primary_last_transition_time,
                group.secondary_last_transition_time,
            )

    # ------------------------------------------------------------------ refresh

    def refresh_if_stale(self, state: ClusterState, interval_s: int) -> None:
        if not self._claim_refresh_slot(state, interval_s):
            logger.log(TRACE, "refresh skipped: within interval (%ds)", interval_s)
            return
        logger.debug("refresh start: uuid=%s, interval=%ds", state.uuid, interval_s)
        if not self._do_refresh_sync(state):
            # No refresh succeeded (no live control conn, or fetch failed on
            # every attempt). Re-flag so the next caller retries immediately
            # instead of waiting a full interval.
            logger.warning(
                "refresh failed: no usable control conn for uuid=%s; "
                "re-flagging force_refresh", state.uuid,
            )
            with state.lock:
                state.force_refresh = True
        else:
            logger.debug("refresh complete: uuid=%s, %d nodes",
                         state.uuid, len(state.nodes))

    async def arefresh_if_stale(self, state: ClusterState, interval_s: int) -> None:
        if not self._claim_refresh_slot(state, interval_s):
            logger.log(TRACE, "refresh skipped (async): within interval (%ds)", interval_s)
            return
        logger.debug("refresh start (async): uuid=%s, interval=%ds",
                     state.uuid, interval_s)
        if not await self._ado_refresh_async(state):
            logger.warning(
                "refresh failed (async): no usable control conn for uuid=%s; "
                "re-flagging force_refresh", state.uuid,
            )
            with state.lock:
                state.force_refresh = True
        else:
            logger.debug("refresh complete (async): uuid=%s, %d nodes",
                         state.uuid, len(state.nodes))

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
        for attempt in range(2):
            ctrl = self._ensure_control_sync(state)
            if ctrl is None:
                logger.debug("refresh attempt %d: no control conn available", attempt)
                return False
            try:
                new_nodes, _ = discovery.fetch_servers_sync(ctrl)
            except Exception as exc:
                logger.info(
                    "refresh attempt %d: fetch_servers failed on control host; "
                    "dropping and retrying (%s)", attempt, exc,
                )
                self._drop_control_sync(state)
                continue
            self._merge_new_nodes(state, new_nodes)
            return True
        return False

    async def _ado_refresh_async(self, state: ClusterState) -> bool:
        for attempt in range(2):
            ctrl = await self._aensure_control_async(state)
            if ctrl is None:
                logger.debug("refresh attempt %d (async): no control conn available",
                             attempt)
                return False
            try:
                new_nodes, _ = await discovery.fetch_servers_async(ctrl)
            except Exception as exc:
                logger.info(
                    "refresh attempt %d (async): fetch_servers failed on control "
                    "host; dropping and retrying (%s)", attempt, exc,
                )
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
        # The cached conn might be broken — e.g. the postmaster crashed
        # mid-flight after a quorum loss — in which case `closed` is True
        # but the slot still holds the dead object. Treat that as "no
        # usable conn" and fall through to the reopen path. Without this
        # check the caller would receive the broken conn forever and the
        # circuit breaker would never see recovery.
        if ctrl is not None and not ctrl.closed:
            return ctrl
        if ctrl is not None and ctrl.closed:
            with state.lock:
                if state.control_sync is ctrl:
                    state.control_sync = None
        if not state.bootstrap_kwargs and not state.bootstrap_conninfo:
            return None
        with state.lock:
            candidates = [n for n in state.nodes.values() if not n.is_down]
            # All-marked-down is usually stale state, not ground truth:
            # a quorum-loss cascade marks every node within seconds; the
            # next refresh would clear them but refresh can't run without
            # a control conn → deadlock. Break it by retrying every node;
            # the connect itself authoritatively reports which are up.
            #
            # The policy uses a TTL window to gradually re-eligible
            # marked-down nodes, but `_ensure_control_sync` is the
            # bootstrap-everything-else primitive and can't afford to wait
            # for the TTL to expire — without a control conn, no refresh,
            # so the TTL never matters.
            if not candidates:
                candidates = list(state.nodes.values())
                logger.info(
                    "control conn re-open: every node marked is_down for "
                    "uuid=%s; retrying all %d to break the deadlock",
                    state.uuid, len(candidates),
                )

        from ..connection import Connection
        logger.debug(
            "control conn re-open: trying %d candidate(s) for uuid=%s",
            len(candidates), state.uuid,
        )
        for node in candidates:
            per_host = {
                **state.bootstrap_kwargs,
                "host": node.host,
                "port": node.port,
            }
            logger.log(TRACE, "control conn re-open: trying %s", node.host)
            try:
                conn = Connection._connect_plain(state.bootstrap_conninfo, **per_host)
            except Exception as exc:
                logger.log(TRACE, "control conn re-open: %s refused (%s)",
                           node.host, exc)
                # We just discovered this host refuses connections. Mark it
                # failed so the dispatcher's next pick doesn't waste a real
                # connect attempt on it.
                self.mark_failed(state.uuid, node.host)
                continue
            with state.lock:
                if state.control_sync is None:
                    state.control_sync = conn
                    logger.info("control conn re-opened on %s for uuid=%s",
                                node.host, state.uuid)
                    return conn
                winner = state.control_sync
            try:
                conn.close()
            except Exception:
                pass
            logger.debug("control conn re-open race lost; closing our duplicate")
            return winner
        logger.warning(
            "control conn re-open: every non-down candidate refused for uuid=%s; "
            "smart driver going blind until next refresh attempt", state.uuid,
        )
        return None

    async def _aensure_control_async(
        self, state: ClusterState
    ) -> "AsyncConnection | None":
        with state.lock:
            ctrl = state.control_async
        # Mirror the sync sibling: a broken cached conn (postmaster crash,
        # network drop) must NOT be returned — it would poison every
        # subsequent caller indefinitely. Detect via `closed` and fall
        # through to the reopen path.
        if ctrl is not None and not ctrl.closed:
            return ctrl
        if ctrl is not None and ctrl.closed:
            with state.lock:
                if state.control_async is ctrl:
                    state.control_async = None
        if not state.bootstrap_kwargs and not state.bootstrap_conninfo:
            return None
        with state.lock:
            candidates = [n for n in state.nodes.values() if not n.is_down]
            # See sync sibling: all-marked-down is usually stale; the
            # connect itself is the authoritative liveness signal.
            if not candidates:
                candidates = list(state.nodes.values())
                logger.info(
                    "control conn re-open (async): every node marked "
                    "is_down for uuid=%s; retrying all %d to break the "
                    "deadlock", state.uuid, len(candidates),
                )

        from ..connection_async import AsyncConnection
        logger.debug(
            "control conn re-open (async): trying %d non-down candidate(s) for uuid=%s",
            len(candidates), state.uuid,
        )
        for node in candidates:
            per_host = {
                **state.bootstrap_kwargs,
                "host": node.host,
                "port": node.port,
            }
            logger.log(TRACE, "control conn re-open (async): trying %s", node.host)
            try:
                conn = await AsyncConnection._aconnect_plain(
                    state.bootstrap_conninfo, **per_host
                )
            except Exception as exc:
                logger.log(TRACE, "control conn re-open (async): %s refused (%s)",
                           node.host, exc)
                # Refusing host → mark it failed so the dispatcher's next
                # pick skips it for the TTL window.
                self.mark_failed(state.uuid, node.host)
                continue
            with state.lock:
                if state.control_async is None:
                    state.control_async = conn
                    logger.info("control conn re-opened (async) on %s for uuid=%s",
                                node.host, state.uuid)
                    return conn
                winner = state.control_async
            try:
                await conn.close()
            except Exception:
                pass
            logger.debug(
                "control conn re-open (async) race lost; closing our duplicate"
            )
            return winner
        logger.warning(
            "control conn re-open (async): every non-down candidate refused for "
            "uuid=%s; smart driver going blind until next refresh attempt",
            state.uuid,
        )
        return None

    def _merge_new_nodes(
        self, state: ClusterState, new_nodes: list[NodeInfo]
    ) -> None:
        """Reconcile state.nodes with the freshly-discovered list.

        Called only on the success path of `refresh_if_stale`, so this is
        the "we just refreshed" moment. Clear `force_refresh` here — any
        flag set during the refresh itself (e.g. `mark_failed` triggered by
        `_ensure_control_sync` refusing a dead host) is now stale.

        Reconciliation rules:

        * Nodes that appear in ``new_nodes``: master considers them alive,
          so ``is_down`` is cleared. ``connection_count`` is preserved from
          the existing entry (we own that counter, not the master).
        * Nodes that were previously known but are MISSING from
          ``new_nodes``: kept in ``state.nodes`` and marked ``is_down=True``
          (with ``is_down_since=now`` only if not already down). A transient
          ``yb_servers()`` response missing a tserver that is still
          re-registering with the master should NOT cause us to forget the
          host entirely — the next refresh either restores it or confirms
          it's gone. Wholesale-dropping makes the dispatcher pick from a
          shrunk subset for the entire next refresh interval, which
          empirically manifests as "all conns go to a 2-host subset of a
          3-host cluster" right after failback.
        """
        with state.lock:
            old_hosts = set(state.nodes.keys())
            new_hosts = {n.host for n in new_nodes}
            added = new_hosts - old_hosts
            removed = old_hosts - new_hosts
            new_dict: dict[str, NodeInfo] = {}
            for n in new_nodes:
                if (existing := state.nodes.get(n.host)) is not None:
                    n.connection_count = existing.connection_count
                # `is_down`/`is_down_since` deliberately NOT preserved here.
                # The master returning this host in `yb_servers()` is the
                # canonical alive signal; trusting it over a stale local
                # flag avoids a 5-second TTL stall every time a node
                # transitions down→up.
                new_dict[n.host] = n
            # Carry forward nodes that vanished from this refresh — they
            # might be transient (still re-registering after restart).
            now = time.monotonic()
            for host in removed:
                stale = state.nodes[host]
                if not stale.is_down:
                    stale.is_down = True
                    stale.is_down_since = now
                new_dict[host] = stale
            state.nodes = new_dict
            state.force_refresh = False
        if added or removed:
            logger.info(
                "topology change observed for uuid=%s: added=%s removed=%s "
                "(removed kept as is_down=True until next refresh)",
                state.uuid, sorted(added) or "—", sorted(removed) or "—",
            )

    def _drop_control_sync(self, state: ClusterState) -> None:
        with state.lock:
            ctrl = state.control_sync
            state.control_sync = None
            state.force_refresh = True
        if ctrl is not None:
            logger.debug("control conn dropped for uuid=%s (was on %s)",
                         state.uuid, _safe_host(ctrl))
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
            logger.debug("control conn dropped (async) for uuid=%s (was on %s)",
                         state.uuid, _safe_host(ctrl))
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
                logger.log(TRACE, "increment: %s → %d", host, ni.connection_count)

    def decrement(self, uuid: str, host: str) -> None:
        state = self._clusters.get(uuid)
        if state is None:
            return
        with state.lock:
            if (ni := state.nodes.get(host)) is not None:
                ni.connection_count = max(0, ni.connection_count - 1)
                logger.log(TRACE, "decrement: %s → %d", host, ni.connection_count)

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
        logger.info("host quarantined: %s (uuid=%s) — connect failure", host, uuid)

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

    def _drain_failover_groups(self) -> "list[FailoverGroup]":
        """Pop all failover groups and stop their probe threads. Returns the
        popped groups so callers can do any further cleanup (currently none
        needed beyond the probe-stop). Called from `clear()` / `aclear()`
        with `self._lock` held by the caller."""
        groups = list(self._failover_groups.values())
        self._failover_groups.clear()
        return groups

    def _stop_probes_best_effort(
        self, groups: "list[FailoverGroup]"
    ) -> None:
        """Stop both per-cluster probes on each group. Runs OUTSIDE
        ``self._lock`` — joining a thread while holding a class-level
        lock would deadlock if the probe ever re-enters the registry."""
        for group in groups:
            for probe in (group.primary_probe, group.secondary_probe):
                if probe is None:
                    continue
                try:
                    probe.stop()
                except Exception:
                    logger.warning(
                        "probe stop failed during clear(); ignoring",
                        exc_info=True,
                    )

    def clear(self) -> None:
        """Drop all state. Best-effort closes any control connections held
        and stops all xCluster probe threads.

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
            groups = self._drain_failover_groups()
        # Probes stopped OUTSIDE the registry lock (see _stop_probes_best_effort).
        self._stop_probes_best_effort(groups)
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
            groups = self._drain_failover_groups()
        self._stop_probes_best_effort(groups)
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
