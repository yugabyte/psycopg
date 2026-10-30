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
import os
import time
import threading
import weakref
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar

from . import TRACE, discovery
from .node import NodeInfo
from ..errors import OperationalError as _PsycopgOperationalError

logger = logging.getLogger(__name__)

# Prune a host after this many consecutive refreshes in which `yb_servers()`
# does not list it. One refresh (per `refresh_interval_s`, default 300s) is
# the shortest absence we expect to tolerate transient re-registration; three
# consecutive misses means ≥10 minutes gone at the default cadence and ≥3s
# gone at the fastest xCluster probe cadence — both well past a graceful
# tserver restart.
_MAX_ABSENCE_TICKS = 3


# ------------------------------------------------------------------- fork safety
#
# DB-25060: pre-fork servers (gunicorn --preload, Celery prefork, multiprocessing
# with 'fork' start) are the dominant Python production deployment. If the app
# calls bootstrap_failover_group BEFORE forking (which gunicorn --preload does
# by design), the child inherits:
#
#   * The parent's ClusterRegistry singleton snapshot — including the
#     FailoverGroup objects with their CB attachments.
#   * BUT: only the forking thread survives. Probe threads are gone.
#   * BUT: threading primitives (Lock, Condition, Event) hold inherited state
#     that doesn't apply here. A lock the parent held at fork is now a
#     locked-but-ownerless lock in the child → deadlock on first use.
#   * BUT: control-connection sockets are shared with the parent. Writes race.
#
# Without a hook, the child never follows a failover: it has no probe thread
# running, so routing stays at the status flag it had at fork time; its
# dispatch_paused is its own copy, so the parent's drain never gates its
# connects; and its socket writes stomp on the parent's.
#
# Fix: register a `after_in_child` hook that rebuilds the fork-unsafe state in
# every FailoverGroup and restarts the probe threads in the child.
#
# We register ONCE, lazily, when the first probe starts — so test suites that
# never bootstrap a failover group don't incur the hook at all. On platforms
# where os.register_at_fork isn't available (Windows), registration no-ops.

_FORK_HOOK_REGISTERED = False
_FORK_HOOK_LOCK = threading.Lock()


def _register_fork_hook_once() -> None:
    """Register `_after_fork_in_child` with `os.register_at_fork`, exactly
    once per process. Called from `_start_probe` so test runs that never
    construct a FailoverGroup don't register anything."""
    global _FORK_HOOK_REGISTERED
    if _FORK_HOOK_REGISTERED:
        return
    with _FORK_HOOK_LOCK:
        if _FORK_HOOK_REGISTERED:
            return
        try:
            os.register_at_fork(after_in_child=_after_fork_in_child)
            _FORK_HOOK_REGISTERED = True
            logger.debug("xcluster fork-safety hook registered (DB-25060)")
        except (AttributeError, OSError):
            # Windows / embedded Python without fork support — nothing
            # to do.
            _FORK_HOOK_REGISTERED = True


# ------------------------------------------------------------------ partial outage
#
# DB-25xxx: bootstrap must survive one universe being down at app startup.
# The whole point of the failover feature is to tolerate an outage; the
# feature can't require both universes to be reachable at bootstrap. The
# strategy is to still return a FailoverGroup, with the failed side filled
# in by a placeholder ClusterState marked UNHEALTHY, and let the probe
# thread reconnect it on a later tick.

_PLACEHOLDER_UUID_PREFIX = "pending:"


class _PartialOutageBootstrapError(_PsycopgOperationalError):
    """Raised by `get_or_bootstrap_failover_group` only when BOTH primary
    and secondary are unreachable at bootstrap. Carries both underlying
    exceptions so the operator can distinguish 'everything is down' from
    one-side-down.

    Subclasses :class:`psycopg.errors.OperationalError` (agent-K PR
    #17:107) so applications' generic ``except OperationalError`` or
    ``except psycopg.Error`` handlers catch it with no code change —
    the private underscore name is a scope marker, not a signal that
    callers should catch something exotic.
    """

    def __init__(
        self,
        message: str,
        *,
        primary_exc: "Exception | None" = None,
        secondary_exc: "Exception | None" = None,
    ) -> None:
        super().__init__(message)
        self.primary_exc = primary_exc
        self.secondary_exc = secondary_exc


def _make_placeholder_state(
    conninfo: str,
    kwargs: dict,
    hosts: "list[str]",
    tag: str,
) -> ClusterState:
    """Build a placeholder ClusterState for a cluster that was unreachable
    at bootstrap time. The probe thread will try to open control_sync on
    its next tick via `_ensure_control_sync`, which reads from
    `bootstrap_conninfo` + `bootstrap_kwargs`; when the cluster comes back
    online the real control conn + real uuid will replace the placeholder.
    """
    import hashlib
    # Deterministic placeholder uuid so repeated bootstraps don't create
    # duplicate groups in the registry. Based on the host set because
    # that's what identifies "this cluster" independent of whether it's
    # reachable right now.
    host_signature = ",".join(sorted(hosts))
    uuid = _PLACEHOLDER_UUID_PREFIX + hashlib.sha1(
        f"{tag}:{host_signature}".encode()
    ).hexdigest()[:16]
    # Seed one NodeInfo per DSN host so the probe has something to try.
    # Placement is unknown until yb_servers() returns; use empty strings
    # as sentinels. The policy layer treats these as "match nothing"
    # when filtering by zone, which is fine — we're UNHEALTHY anyway.
    nodes: "dict[str, NodeInfo]" = {}
    placement = __import__(
        "psycopg.yb.node", fromlist=["Placement"]
    ).Placement(cloud="", region="", zone="")
    for host in hosts:
        if not host:
            continue
        nodes[host] = NodeInfo(
            host=host,
            public_ip=None,
            port=int(kwargs.get("port", 5433) or 5433),
            placement=placement,
            node_type="primary",
            connection_count=0,
            is_down=True,                 # it IS down right now
            is_down_since=time.monotonic(),
        )
    return ClusterState(
        uuid=uuid,
        nodes=nodes,
        lock=threading.Lock(),
        last_refresh=0.0,                 # never refreshed yet
        force_refresh=True,               # probe should retry on first tick
        control_sync=None,
        control_async=None,
        bootstrap_conninfo=conninfo,
        bootstrap_kwargs=dict(kwargs),
    )


def _is_placeholder_uuid(uuid: str) -> bool:
    """True iff the ClusterState was placeholder-constructed (and should
    be promoted to real state on the next successful probe tick)."""
    return uuid.startswith(_PLACEHOLDER_UUID_PREFIX)


def _after_fork_in_child() -> None:
    """Called in the child process after `os.fork()`. Rebuilds the state
    inside every FailoverGroup that doesn't survive fork, then restarts
    the probe threads.

    Design doc §DB-25060:
      * KEEP: CB attachments (app installed them on the parent; CBs are
        typically stateless config objects or carry their own fork-safe
        state).
      * KEEP: primary_status / secondary_status — reflect what the probe
        last observed; child continues routing on that until its own
        probe reconfirms. (If parent flipped to secondary before fork,
        child should also start on secondary, not reset to primary.)
      * REBUILD: group.lock, dispatch_paused_condition, async_dispatch_waiters
        — threading primitives aren't fork-safe.
      * REBUILD: per-cluster lock, tracked_conns — fork-copied.
      * DROP:    control_sync / control_async — fork-shared sockets, child
                 must open fresh ones.
      * DROP:    primary_probe / secondary_probe references — the threads
                 are gone, _start_probe will create fresh ones.
      * DROP:    yba_client's persistent HTTP conn (same reason as control
                 conns).
      * RESET:   dispatch_paused — child doesn't inherit parent's drain.
      * RESET:   last_transition_time — child can transition on first tick.
    """
    inst = ClusterRegistry._instance
    if inst is None:
        return
    # Registry-level lock: fork-unsafe.
    inst._lock = threading.Lock()
    # agent-K PR #16:127. Don't wholesale-drop `_clusters` / `_key_to_uuid`
    # — the group's primary/secondary ClusterState objects are the ones
    # the dispatcher routes through, so orphaning them means the next
    # connect bootstraps a NEW ClusterState for the same uuid; the group
    # cache then returns the OLD group whose per-host counters live on
    # the OLD state (which is no longer in `_clusters`), so decrement /
    # mark_failed by uuid never finds it. Keep the entries for states
    # that belong to a live group; drop the ones that don't.
    group_state_ids = set()
    for grp in inst._failover_groups.values():
        for st in (grp.primary, grp.secondary):
            if st is not None:
                group_state_ids.add(id(st))
    inst._clusters = {
        uuid: st for uuid, st in inst._clusters.items()
        if id(st) in group_state_ids
    }
    inst._key_to_uuid = {
        key: uuid for key, uuid in inst._key_to_uuid.items()
        if uuid in inst._clusters
    }
    # Fix up every failover group in place.
    groups = list(inst._failover_groups.values())
    for group in groups:
        try:
            group._reset_for_child()
        except Exception:
            # Defensive: if a group's reset raises for whatever reason,
            # don't blow up the whole fork hook. The group may be in a
            # broken state but at least other groups still get fixed.
            logger.warning(
                "xcluster fork reset failed for group primary_uuid=%s",
                getattr(getattr(group, "primary", None), "uuid", "?"),
                exc_info=True,
            )
    # Restart probes using the stashed per-group params.
    for group in groups:
        try:
            inst._start_probe(
                group,
                group._probe_interval_s,
                group._probe_check_timeout_s,
                group._probe_drain_timeout_s,
            )
        except Exception:
            logger.warning(
                "xcluster probe restart failed for group primary_uuid=%s",
                getattr(getattr(group, "primary", None), "uuid", "?"),
                exc_info=True,
            )


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
    # agent-K PR #12:46. Last exception observed while opening a control
    # connection, so CB callbacks can distinguish "control conn failed to
    # open due to misconfig (auth / missing DB / TLS)" from "cluster
    # genuinely unreachable". ``_ensure_control_sync`` / its async
    # sibling stamp this; CBs read it in their ``conn is None`` branch
    # and apply the same SQLSTATE filter they already use in
    # ``_handle_exception``. Cleared on a successful connect.
    last_control_connect_error: "Exception | None" = None
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
      * ``primary_circuit_breaker`` / ``secondary_circuit_breaker`` — the
        CB instances. The driver does NOT install a default; both slots
        are ``None`` at bootstrap. Applications MUST attach an
        implementation to each slot before opening connections — the
        dispatcher raises ``MissingCircuitBreakerError`` otherwise. See
        ``demo/samples/`` for reference CBs.

    ``cooldown_s`` applies symmetrically to both — the probe consults
    ``can_transition_primary`` / ``can_transition_secondary`` before writing
    a status change.

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
    # Per-cluster probes. Type kept loose to avoid a circular import on
    # health_probe.HealthProbe.
    primary_probe: "object | None" = None
    secondary_probe: "object | None" = None
    # Probe-construction params stashed at bootstrap so the fork-safety
    # hook (DB-25060) can rebuild probes in the child process without
    # re-reading YBParams. Set by ``ClusterRegistry._start_probe``.
    _probe_interval_s: int = 5
    _probe_check_timeout_s: float = 1.0
    _probe_drain_timeout_s: int = 10
    # The per-cluster CircuitBreaker instances. The driver does NOT install
    # a default — applications MUST attach implementations after calling
    # bootstrap_failover_group and before any psycopg.connect() traffic.
    # The dispatcher raises MissingCircuitBreakerError if either slot is
    # still None at connect time. See demo/samples/ for reference CBs.
    primary_circuit_breaker: "object | None" = None
    secondary_circuit_breaker: "object | None" = None
    # Dispatch pause flag (Phase D — see design doc §3). When True, new
    # connection opens block on ``dispatch_paused_condition`` until the
    # drain sequence flips it back to False. Existing conns are NOT
    # affected — only new opens gate on this. Written under ``lock``.
    dispatch_paused: bool = False
    # Condition bound to ``lock`` for pause/resume notify. Constructed in
    # ``__post_init__`` so it wraps whatever lock was passed in (or
    # created by default). Used by sync ``wait_for_dispatch`` callers.
    dispatch_paused_condition: "threading.Condition | None" = field(
        default=None, repr=False,
    )
    # Bumped under ``lock`` every time a drain arms the barrier. A connect
    # records it after passing the barrier and re-checks it once its conn
    # is in ``tracked_conns``: a conn whose TCP connect straddled the
    # arming would otherwise miss every drain sweep and write to the
    # outgoing cluster until the flip (DB-24801, per_txn connects).
    barrier_generation: int = 0
    # Per-waiter async-event list for ``wait_for_dispatch_async``
    # (DB-24838 part 1). The sync Condition would block an asyncio
    # event-loop OS thread for the whole drain window; the async
    # variant registers a per-call ``(loop, asyncio.Event)`` tuple
    # here, and ``resume_dispatch`` fans out from the probe thread via
    # ``loop.call_soon_threadsafe`` so each event loop stays
    # responsive during the drain. Written under ``lock``.
    async_dispatch_waiters: "list" = field(default_factory=list, repr=False)

    # YBA lag-wait phase (Amogh's flow, 2026-08-11). ``yba_client`` is set
    # by bootstrap when ``yba_endpoint`` + ``yba_api_token`` +
    # ``replication_name`` are configured in the DSN; ``None`` means the
    # lag-wait phase is disabled and the drain flip proceeds immediately
    # after the txn drain.
    #
    # ``yba_customer_uuid`` and ``xcluster_config_uuid`` are figured out
    # once at bootstrap by the driver — the user does NOT supply them —
    # and are read on every Phase-2 poll. All fields stay stable for the
    # group's lifetime; no ``lock`` required for reads.
    #
    # Empirically (2026-08-20 against portal.dev): YBA's /metrics only
    # needs xClusterConfigUuid to identify the lag stream. We do NOT
    # cache nodePrefix — it's redundant and its per-side lookup was the
    # main reason we needed the YBA-side universe UUID at bootstrap
    # (which doesn't necessarily match yb_servers().universe_uuid).
    yba_client: "object | None" = None
    yba_customer_uuid: str = ""
    xcluster_config_uuid: str = ""
    threshold_replication_lag_ms: int = 0
    lag_wait_timeout_s: int = 30

    # Failback-direction (B→A) Phase 2 config (design doc §3.5.1).
    # Populated at bootstrap only when the operator set
    # ``yb.failback.replicationName`` — resolved to a distinct xCluster
    # config UUID via the same /universes walk used for the A→B side.
    # An empty ``failback_xcluster_config_uuid`` means "Phase 2 skipped
    # on failback" (fail-open) — same behaviour as YBA being unreachable.
    failback_xcluster_config_uuid: str = ""
    failback_threshold_replication_lag_ms: int = 0
    failback_lag_wait_timeout_s: int = 30

    # Auto-failback toggle (see design doc §3.5.1 / §7.3). When False,
    # the driver stays on secondary even after the primary CB reports
    # HEALTHY. Failback requires operator action (e.g. writing UNHEALTHY
    # then HEALTHY to an ExternalSignalCircuitBreaker's signal table
    # from the operator's side). Default True preserves v1 behaviour.
    auto_failback_enabled: bool = True

    # Phase 2 timeout policy. ``"proceed"`` (default) flips once the
    # lag wait times out. ``"fail"`` keeps routing on the current side
    # when a FAILBACK lag wait times out; failover always proceeds.
    on_phase2_timeout: str = "proceed"

    # Per-stage timestamps (design doc §3.8). All ``time.monotonic()``
    # floats. Zero means "never fired". Written under ``self.lock`` at
    # the stage boundary; safe to read without the lock (float writes
    # are atomic on CPython). Custom CBs (e.g. a debounced secondary CB
    # that reads ``last_failover_complete_ts`` before returning HEALTHY
    # to avoid an immediate failback) rely on these.
    last_cb_trip_ts: float = 0.0
    last_failover_start_ts: float = 0.0
    last_phase1_complete_ts: float = 0.0
    last_phase2_complete_ts: float = 0.0
    last_failover_complete_ts: float = 0.0
    last_failback_complete_ts: float = 0.0
    # agent-K PR #10:81. Stamped the moment xcluster_check (or
    # assert_viable_or_raise) observes both CBs reporting UNHEALTHY.
    # psycopg_pool catches NoViableClusterError inside its check loop
    # and the borrow caller only sees PoolTimeout — operators can read
    # this timestamp to distinguish "no viable cluster" from
    # "transient stale conn" after a PoolTimeout.
    last_no_viable_ts: float = 0.0
    # Set by the drain when a failback Phase 2 lag wait times out and
    # ``on_phase2_timeout == "fail"`` aborts the flip. Operators can
    # scrape this to detect degraded failover; custom CBs can read it
    # to rate-limit their trip signaling.
    last_phase2_timeout_ts: float = 0.0

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
            self.barrier_generation += 1

    def resume_dispatch(self) -> None:
        """Set ``dispatch_paused = False`` and notify all waiters. Called
        from the drain sequence when the barrier releases.

        Fans out to both the sync ``threading.Condition`` (for the sync
        connect path's ``wait_for_dispatch``) AND to every registered
        async waiter's ``asyncio.Event`` via ``call_soon_threadsafe``
        (DB-24838 part 1 — for ``wait_for_dispatch_async`` callers,
        which run on arbitrary event loops in possibly different OS
        threads). Each async loop is addressed on its own thread so no
        loop is held hostage by the probe thread's resume call.
        """
        with self.lock:
            self.dispatch_paused = False
            assert self.dispatch_paused_condition is not None
            self.dispatch_paused_condition.notify_all()
            # Snapshot + clear async waiters under the lock so a new
            # waiter arriving concurrently doesn't miss the resume.
            waiters = list(self.async_dispatch_waiters)
            self.async_dispatch_waiters.clear()
        for loop, event in waiters:
            try:
                loop.call_soon_threadsafe(event.set)
            except RuntimeError:
                # Event loop was closed (app shut down between pause
                # and resume). The waiter's coroutine will see this
                # as a cancelled/stuck wait; nothing we can do here.
                pass

    def wait_for_first_check(self, timeout_s: "float | None" = None) -> bool:
        """Block until each attached CB has completed its first
        ``check()`` — either as a natural probe tick or via an explicit
        synchronous call here.

        Design doc §4.6: prevents the race where an app opens
        connections after ``bootstrap_failover_group()`` returns but
        before the probe has had any actual data about the clusters.
        Call this AFTER attaching CBs and BEFORE opening application
        connections.

        Bounded by ``check_timeout_s`` per CB (enforced inside each
        probe's ``_tick_check``). agent-K PR #10:284: the first-check
        path explicitly does NOT fire ``_maybe_apply`` → ``trigger_drain``
        on the caller's thread, so the outer ``timeout_s`` is now a
        real bound; previously a bootstrap-time first check could
        trigger a Phase 2 lag wait of up to ``lag_wait_timeout_s``
        inside this call. The daemon's next natural tick applies any
        differing verdict normally. If ``timeout_s`` is provided, this
        call gives up after that many seconds and returns ``False``
        (individual probes may still complete afterwards). Returns
        ``True`` when both first checks have fired.

        Idempotent — calling twice with both first-checks already
        complete returns immediately."""
        probes = [
            p for p in (self.primary_probe, self.secondary_probe)
            if p is not None
        ]
        if not probes:
            return True   # no probes wired yet — nothing to wait on
        # Force a synchronous first tick on each probe that hasn't
        # ticked yet. Probes with an already-complete first tick skip
        # this internally (idempotent).
        import time as _time
        deadline = None if timeout_s is None else _time.monotonic() + timeout_s
        for p in probes:
            # `run_first_check_synchronously` returns after the tick
            # completes OR the probe's own check_timeout_s cap fires.
            p.run_first_check_synchronously()
            if deadline is not None and _time.monotonic() >= deadline:
                return all(p.first_check_complete for p in probes)
        return True

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

    def _reset_for_child(self) -> None:
        """DB-25060 fork-safety reset. Called in-child from
        ``_after_fork_in_child``.

        Rebuilds every piece of state that doesn't survive ``os.fork()``:

          * threading.Lock + Condition (parent's hold-state is nonsense in
            the child, so a locked lock in the parent at fork time becomes
            a locked-but-ownerless lock in the child → deadlock)
          * dispatch_paused flag (child doesn't inherit parent's drain)
          * tracked_conns (parent's conns, not ours)
          * control_sync / control_async (fork-shared sockets)
          * primary_probe / secondary_probe references (threads are gone)
          * YBAClient's persistent HTTP connection (fork-shared socket)

        Preserves what should survive:

          * CB attachments
          * primary_status / secondary_status (routing continuity — if the
            parent had failed over to secondary, child should start on
            secondary too, not reset to primary and wait for its first probe
            tick to re-discover the fact)
          * yb_params-derived config (cooldown, auto_failback, thresholds,
            YBA endpoint URL)
          * ``_probe_*_s`` stash so the restart in ``_after_fork_in_child``
            can rebuild probes with the original params
        """
        # Registry-level primitives
        self.lock = threading.Lock()
        self.dispatch_paused_condition = threading.Condition(self.lock)
        self.dispatch_paused = False
        self.async_dispatch_waiters = []
        # agent-K PR #16:515. KEEP the parent's cool-down timestamps.
        # ``time.monotonic()`` is shared across fork on the same host,
        # so the parent's windows are directly comparable in the child.
        # Zeroing them lets the child flip back immediately after a
        # parent-initiated failover; the parent (and other workers)
        # stay on the new side for the remainder of the cooldown,
        # producing exactly the split-brain this ticket is about.
        # Null out probe references — the threads are gone; the registry's
        # fork hook will restart them.
        self.primary_probe = None
        self.secondary_probe = None
        # Per-cluster: fresh lock, dropped sockets, dropped WeakSet.
        for state in (self.primary, self.secondary):
            state.lock = threading.Lock()
            state.tracked_conns = weakref.WeakSet()
            # DROP (don't close) — closing would FIN the socket that is
            # still shared with the parent. Null out the references and
            # let the parent own its own copies.
            state.control_sync = None
            state.control_async = None
            state.force_refresh = True   # next tick re-reads yb_servers()
        # YBAClient: drop its persistent HTTP connection AND rebuild
        # ``_lock``. The fork may have landed with the lock held — a
        # Phase-2 lag poll on the parent takes the lock for the whole
        # HTTP round-trip (yba_client.py:_request), and
        # HealthAndLagCircuitBreaker.check acquires it on every
        # unhealthy tick. A locked-but-ownerless lock in the child
        # wedges every subsequent ``get_committed_lag_sample`` call
        # forever: drain blocks → ``resume_dispatch`` never runs →
        # new child connects hang in ``wait_for_dispatch``. Dropping
        # ``_conn`` alone is not enough — the lock itself must be
        # replaced. Agent-K PR #16:537. We do NOT close() the client:
        # that would FIN the parent's shared socket.
        if self.yba_client is not None:
            try:
                self.yba_client._conn = None    # type: ignore[attr-defined]
            except Exception:
                pass
            try:
                self.yba_client._lock = threading.Lock()  # type: ignore[attr-defined]
            except Exception:
                pass

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
        # agent-K PR #17:G7 — serialize the full
        # ``get_or_bootstrap_failover_group`` sequence per primary
        # ClusterKey. Prevents the one-side-down race where thread A
        # (primary unreachable → placeholder uuid) and thread B
        # (primary reachable → real uuid) each install their own
        # FailoverGroup under different primary.uuid keys for the same
        # cluster. ``_bootstrap_locks_mutex`` guards the dict itself;
        # the per-key Lock serializes bootstraps for that cluster
        # without serializing bootstraps of DIFFERENT clusters.
        self._bootstrap_locks: dict[ClusterKey, threading.Lock] = {}
        # Parallel async-side map; keyed the same way. Separate so sync
        # callers never acquire an asyncio.Lock (which would need a
        # running loop) and async callers never block their event loop
        # on a sync Lock held by another thread.
        self._async_bootstrap_locks: dict[ClusterKey, "object"] = {}
        self._bootstrap_locks_mutex = threading.Lock()

    def _bootstrap_lock_for(self, key: "ClusterKey") -> threading.Lock:
        """Return the per-key sync bootstrap lock, creating it on first use.
        See ``__init__`` for the race this closes (agent-K PR #17:G7)."""
        with self._bootstrap_locks_mutex:
            lock = self._bootstrap_locks.get(key)
            if lock is None:
                lock = threading.Lock()
                self._bootstrap_locks[key] = lock
            return lock

    def _async_bootstrap_lock_for(self, key: "ClusterKey") -> "object":
        """Return the per-key async bootstrap lock (asyncio.Lock) for the
        current loop, creating it on first use. See ``__init__``."""
        import asyncio
        with self._bootstrap_locks_mutex:
            lock = self._async_bootstrap_locks.get(key)
            if lock is None:
                lock = asyncio.Lock()
                self._async_bootstrap_locks[key] = lock
            return lock

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

    def _wire_yba_lag_phase(self, group: "FailoverGroup", yb_params) -> None:
        """Bootstrap-time YBA client setup. Called once per FailoverGroup
        after construction, before probes start.

        Steps (per Amogh's 2026-08-11 spec, revised 2026-08-20):
          1. Instantiate ``YBAClient`` with the DSN's endpoint + token.
          2. Fetch ``customerUUID`` via ``/session_info`` → cache on group.
          3. Resolve ``replicationName`` → ``xClusterConfigUUID`` by
             walking every universe's ``sourceXClusterConfigs`` → cache
             on group.

        Only two YBA identifiers ever leave the bootstrap: customerUUID
        and xClusterConfigUUID. YBA's /metrics endpoint doesn't need a
        node_prefix when the config UUID is set, so we skip the two
        /universes/{uuid} calls that were previously needed. This also
        means we don't need the YBA-side universeUUID at all — which
        avoided a real portability problem (YBA's universeUUID doesn't
        necessarily match yb_servers().universe_uuid).

        Best-effort: any failure (YBA unreachable, bad token, unknown
        replication name) is logged at WARNING and leaves
        ``group.yba_client`` as ``None``. The lag-wait phase in
        ``drain.py`` treats a ``None`` client as "feature disabled" and
        proceeds with the immediate flip after the txn drain. This
        means a broken YBA never blocks failover.
        """
        if not yb_params.lag_wait_enabled:
            return

        from .yba_client import YBAClient, YBAClientError

        # Stash the immutable knobs first — even if the initial YBA
        # probe fails, they're preserved so the drain code can log
        # them consistently.
        group.threshold_replication_lag_ms = yb_params.threshold_replication_lag_ms
        group.lag_wait_timeout_s = yb_params.lag_wait_timeout_s
        # Failback-direction knobs. `failback_xcluster_config_uuid` is
        # populated below only if the operator set
        # yb.failback.replicationName.
        group.failback_threshold_replication_lag_ms = (
            yb_params.failback_threshold_replication_lag_ms
        )
        group.failback_lag_wait_timeout_s = yb_params.failback_lag_wait_timeout_s

        try:
            client = YBAClient(
                endpoint=yb_params.yba_endpoint,
                api_token=yb_params.yba_api_token,
                connect_timeout_s=yb_params.yba_connect_timeout_s,
                read_timeout_s=yb_params.yba_read_timeout_s,
            )
            # 1. Customer UUID. Cached inside the client too, but we
            #    surface it on the group so operators can see it in
            #    diagnostics without poking at the client.
            group.yba_customer_uuid = client.get_session_info()

            # 2. Resolve replication_name → xClusterConfigUUID by
            #    walking all universes' sourceXClusterConfigs. One
            #    /universes call + one /xcluster_configs/{uuid} per
            #    candidate.
            group.xcluster_config_uuid = client.resolve_xcluster_config_uuid(
                yb_params.replication_name,
            )

            # 3. If the operator configured a failback (B→A) replication
            #    name, resolve it to a second xCluster config UUID.
            #    Best-effort: a failure here leaves failback_xcluster_config_uuid
            #    empty, which the drain treats as "no Phase 2 on failback"
            #    (fail-open) but still allows the A→B Phase 2 to work.
            if yb_params.failback_replication_name:
                try:
                    group.failback_xcluster_config_uuid = (
                        client.resolve_xcluster_config_uuid(
                            yb_params.failback_replication_name,
                        )
                    )
                except YBAClientError as exc:
                    logger.warning(
                        "YBA failback Phase 2 disabled — could not resolve "
                        "yb.failback.replicationName=%r: %s. Failover (A→B) "
                        "Phase 2 still active; failback (B→A) will skip "
                        "Phase 2 (fail-open).",
                        yb_params.failback_replication_name, exc,
                    )
        except YBAClientError as exc:
            logger.warning(
                "YBA lag phase disabled — bootstrap probe failed: %s. "
                "Failover drain will flip routing immediately after the "
                "txn drain; no replication-lag wait.",
                exc,
            )
            return
        except Exception:
            logger.warning(
                "YBA lag phase disabled — unexpected error during bootstrap probe. "
                "Failover drain will flip routing immediately after the txn drain.",
                exc_info=True,
            )
            return

        group.yba_client = client
        logger.info(
            "YBA lag phase enabled: endpoint=%s, replication_name=%s, "
            "customer_uuid=%s, xcluster_config_uuid=%s, "
            "threshold=%dms, wait_timeout=%ds",
            yb_params.yba_endpoint,
            yb_params.replication_name,
            group.yba_customer_uuid,
            group.xcluster_config_uuid,
            yb_params.threshold_replication_lag_ms,
            yb_params.lag_wait_timeout_s,
        )

    def get_or_bootstrap_failover_group(
        self,
        yb_params,                                  # YBParams; lazy-typed to dodge cycle
        conninfo: str,
        kwargs: dict[str, object],
    ) -> "FailoverGroup":
        """Sync bootstrap of a primary + secondary `FailoverGroup`. Idempotent
        on primary uuid. Spec §5 / design §3.

        **Partial-outage tolerance (DB-25xxx):** if ONE side is unreachable
        at bootstrap, the group is still created with that side marked
        UNHEALTHY and backed by a placeholder ``ClusterState``; the probe
        thread's next tick will reconnect and promote it to the real
        ClusterState. If BOTH sides are unreachable the call raises — the
        feature exists to survive outages, not to come up with no cluster.
        """
        from .health import HealthResult            # late import — health.py imports FailoverGroup forward-ref

        # agent-K PR #17:G7 — hold the per-key bootstrap lock for the
        # full sequence so concurrent calls for the same cluster can't
        # install competing groups (one placeholder, one real) during
        # a one-side-down window.
        primary_key = ClusterKey.from_params(
            self._param_dict_for_key(conninfo, kwargs)
        )
        with self._bootstrap_lock_for(primary_key):
            return self._do_bootstrap_failover_group(
                yb_params, conninfo, kwargs, primary_key, HealthResult,
            )

    def _do_bootstrap_failover_group(
        self, yb_params, conninfo, kwargs, primary_key, HealthResult,
    ) -> "FailoverGroup":
        # 1. Try primary — tolerate failure.
        primary: "ClusterState | None" = None
        primary_exc: "Exception | None" = None
        try:
            primary = self.get_or_bootstrap(primary_key, conninfo, kwargs)
        except Exception as exc:
            primary_exc = exc
            logger.warning(
                "primary bootstrap failed (%s); will try secondary and "
                "mark primary UNHEALTHY if secondary succeeds",
                exc,
            )

        # 2. If primary came up, check for cached pairing.
        if primary is not None:
            with self._lock:
                existing = self._failover_groups.get(primary.uuid)
            if existing is not None:
                logger.debug(
                    "failover group cache hit: primary_uuid=%s", primary.uuid
                )
                return existing

        # 3. Try secondary — tolerate failure.
        sec_kwargs = self._make_secondary_kwargs(
            kwargs, yb_params.secondary_cluster_hosts
        )
        secondary_key = ClusterKey.from_params(
            self._param_dict_for_key(conninfo, sec_kwargs)
        )
        secondary: "ClusterState | None" = None
        secondary_exc: "Exception | None" = None
        try:
            secondary = self.get_or_bootstrap(
                secondary_key, conninfo, sec_kwargs
            )
        except Exception as exc:
            secondary_exc = exc
            logger.warning(
                "secondary bootstrap failed (%s); will mark secondary "
                "UNHEALTHY if primary succeeded",
                exc,
            )

        # 4. Both-unreachable → raise. There's nothing to route to.
        if primary is None and secondary is None:
            # Prefer the primary error — it's usually what the operator
            # wants to see first. Chain the secondary error through.
            raise _PartialOutageBootstrapError(
                "xCluster bootstrap failed: both primary and secondary "
                "clusters are unreachable",
                primary_exc=primary_exc,
                secondary_exc=secondary_exc,
            ) from primary_exc

        # 5. Fill in whichever side failed with a placeholder — the probe
        # thread will reconnect it on a later tick.
        if primary is None:
            primary = _make_placeholder_state(
                conninfo, kwargs,
                hosts=self._hosts_from_kwargs(kwargs, conninfo=conninfo),
                tag="primary",
            )
            primary_status = HealthResult.UNHEALTHY
            # agent-K PR #17: register placeholder in _clusters and under
            # primary_key, so subsequent connects find the SAME
            # ClusterState object (via the key cache) instead of spawning
            # a fresh bootstrap. Without this, every connect raced a
            # second group creation with None CBs → MissingCircuitBreaker
            # on the next connect.
            with self._lock:
                self._clusters[primary.uuid] = primary
                self._key_to_uuid[primary_key] = primary.uuid
            logger.warning(
                "xCluster failover group bootstrapped WITHOUT primary — "
                "placeholder uuid=%s, status=UNHEALTHY; probe will retry",
                primary.uuid,
            )
        else:
            primary_status = HealthResult.HEALTHY

        if secondary is None:
            secondary = _make_placeholder_state(
                conninfo, sec_kwargs,
                hosts=list(yb_params.secondary_cluster_hosts),
                tag="secondary",
            )
            secondary_status = HealthResult.UNHEALTHY
            with self._lock:
                self._clusters[secondary.uuid] = secondary
                self._key_to_uuid[secondary_key] = secondary.uuid
            logger.warning(
                "xCluster failover group bootstrapped WITHOUT secondary "
                "— placeholder uuid=%s, status=UNHEALTHY; probe will retry",
                secondary.uuid,
            )
        else:
            secondary_status = HealthResult.HEALTHY

        # 6. Install (race-checked).
        with self._lock:
            existing = self._failover_groups.get(primary.uuid)
            if existing is not None:
                logger.debug(
                    "failover group race lost; discarding our pairing for "
                    "primary_uuid=%s", primary.uuid,
                )
                return existing
            # Both CB slots start empty. The application MUST attach
            # implementations after this call returns and before any
            # psycopg.connect() traffic. The dispatcher raises
            # MissingCircuitBreakerError if either slot is still None at
            # connect time. See demo/samples/ for reference CBs.
            group = FailoverGroup(
                primary=primary,
                secondary=secondary,
                lock=threading.Lock(),
                primary_status=primary_status,
                secondary_status=secondary_status,
                cooldown_s=yb_params.cooldown_s,
                primary_circuit_breaker=None,
                secondary_circuit_breaker=None,
                auto_failback_enabled=yb_params.auto_failback_enabled,
                on_phase2_timeout=yb_params.on_phase2_timeout,
            )
            self._failover_groups[primary.uuid] = group

        logger.info(
            "failover group bootstrapped: primary_uuid=%s (%s), "
            "secondary_uuid=%s (%s), secondary_hosts=%s, cooldown_s=%d",
            primary.uuid, primary_status.value,
            secondary.uuid, secondary_status.value,
            ",".join(yb_params.secondary_cluster_hosts),
            yb_params.cooldown_s,
        )

        # 5. YBA lag phase — best-effort. Silently disables if unreachable.
        self._wire_yba_lag_phase(group, yb_params)

        # 6. Start the probe thread (Phase 4 wiring).
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
        (see design doc §10). Partial-outage tolerance mirrors the sync
        path (see `get_or_bootstrap_failover_group`)."""
        from .health import HealthResult

        primary_key = ClusterKey.from_params(
            self._param_dict_for_key(conninfo, kwargs)
        )
        # agent-K PR #17:G7 — per-key async lock, same purpose as sync
        # side. See ``_async_bootstrap_lock_for``.
        async with self._async_bootstrap_lock_for(primary_key):  # type: ignore[attr-defined]
            return await self._ado_bootstrap_failover_group(
                yb_params, conninfo, kwargs, primary_key, HealthResult,
            )

    async def _ado_bootstrap_failover_group(
        self, yb_params, conninfo, kwargs, primary_key, HealthResult,
    ) -> "FailoverGroup":
        primary: "ClusterState | None" = None
        primary_exc: "Exception | None" = None
        try:
            primary = await self.aget_or_bootstrap(
                primary_key, conninfo, kwargs
            )
        except Exception as exc:
            primary_exc = exc
            logger.warning(
                "primary bootstrap failed (async) (%s); will try secondary",
                exc,
            )

        if primary is not None:
            with self._lock:
                existing = self._failover_groups.get(primary.uuid)
            if existing is not None:
                logger.debug(
                    "failover group cache hit (async): primary_uuid=%s",
                    primary.uuid,
                )
                return existing

        sec_kwargs = self._make_secondary_kwargs(
            kwargs, yb_params.secondary_cluster_hosts
        )
        secondary_key = ClusterKey.from_params(
            self._param_dict_for_key(conninfo, sec_kwargs)
        )
        secondary: "ClusterState | None" = None
        secondary_exc: "Exception | None" = None
        try:
            secondary = await self.aget_or_bootstrap(
                secondary_key, conninfo, sec_kwargs
            )
        except Exception as exc:
            secondary_exc = exc
            logger.warning(
                "secondary bootstrap failed (async) (%s)",
                exc,
            )

        if primary is None and secondary is None:
            raise _PartialOutageBootstrapError(
                "xCluster bootstrap failed: both primary and secondary "
                "clusters are unreachable",
                primary_exc=primary_exc,
                secondary_exc=secondary_exc,
            ) from primary_exc

        if primary is None:
            primary = _make_placeholder_state(
                conninfo, kwargs,
                hosts=self._hosts_from_kwargs(kwargs, conninfo=conninfo),
                tag="primary",
            )
            primary_status = HealthResult.UNHEALTHY
            # agent-K PR #17 — mirror the sync path: register in
            # _clusters + _key_to_uuid so the next connect finds the
            # same ClusterState object (via the key cache) rather than
            # spawning a fresh bootstrap that races a 2nd group.
            with self._lock:
                self._clusters[primary.uuid] = primary
                self._key_to_uuid[primary_key] = primary.uuid
            logger.warning(
                "xCluster failover group bootstrapped (async) WITHOUT "
                "primary — placeholder uuid=%s, status=UNHEALTHY",
                primary.uuid,
            )
        else:
            primary_status = HealthResult.HEALTHY

        if secondary is None:
            secondary = _make_placeholder_state(
                conninfo, sec_kwargs,
                hosts=list(yb_params.secondary_cluster_hosts),
                tag="secondary",
            )
            secondary_status = HealthResult.UNHEALTHY
            with self._lock:
                self._clusters[secondary.uuid] = secondary
                self._key_to_uuid[secondary_key] = secondary.uuid
            logger.warning(
                "xCluster failover group bootstrapped (async) WITHOUT "
                "secondary — placeholder uuid=%s, status=UNHEALTHY",
                secondary.uuid,
            )
        else:
            secondary_status = HealthResult.HEALTHY

        with self._lock:
            existing = self._failover_groups.get(primary.uuid)
            if existing is not None:
                logger.debug(
                    "failover group race lost (async); discarding our pairing "
                    "for primary_uuid=%s", primary.uuid,
                )
                return existing
            # Both CB slots start empty. The application MUST attach
            # implementations after this call returns and before any
            # psycopg.connect() traffic. The dispatcher raises
            # MissingCircuitBreakerError if either slot is still None at
            # connect time. See demo/samples/ for reference CBs.
            group = FailoverGroup(
                primary=primary,
                secondary=secondary,
                lock=threading.Lock(),
                primary_status=primary_status,
                secondary_status=secondary_status,
                cooldown_s=yb_params.cooldown_s,
                primary_circuit_breaker=None,
                secondary_circuit_breaker=None,
                auto_failback_enabled=yb_params.auto_failback_enabled,
                on_phase2_timeout=yb_params.on_phase2_timeout,
            )
            self._failover_groups[primary.uuid] = group

        logger.info(
            "failover group bootstrapped (async): primary_uuid=%s (%s), "
            "secondary_uuid=%s (%s), secondary_hosts=%s, cooldown_s=%d",
            primary.uuid, primary_status.value,
            secondary.uuid, secondary_status.value,
            ",".join(yb_params.secondary_cluster_hosts),
            yb_params.cooldown_s,
        )

        self._wire_yba_lag_phase(group, yb_params)
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

    def _hosts_from_kwargs(
        self, kwargs: dict[str, object], conninfo: str = "",
    ) -> "list[str]":
        """Extract the comma-separated host list from psycopg kwargs and/
        or conninfo string. Used when building a placeholder ClusterState
        for a cluster that was unreachable at bootstrap (DB-25011
        partial-outage path).

        Pre-fix (agent-K PR #17:1066) only looked at ``kwargs["host"]``,
        which is empty when the operator put hosts in the conninfo
        STRING instead of kwargs. That left the placeholder with zero
        nodes → probe had no hosts to try → placeholder never recovers
        even after the cluster comes back online. Now we also parse
        from the ``conninfo`` string via ``conninfo_to_dict``.
        """
        hosts: list[str] = []
        raw = kwargs.get("host")
        if raw:
            hosts.extend(h.strip() for h in str(raw).split(",") if h.strip())
        if not hosts and conninfo:
            try:
                from ..conninfo import conninfo_to_dict  # late import
                parsed = conninfo_to_dict(str(conninfo))
                raw2 = parsed.get("host")
                if raw2:
                    hosts.extend(
                        h.strip() for h in str(raw2).split(",") if h.strip()
                    )
            except Exception:
                # Parsing failed — placeholder stays empty, probe will
                # still raise visibly in logs. Not worth crashing the
                # bootstrap over.
                pass
        return hosts

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
        # Stash the construction params so the fork-safety hook
        # (DB-25060) can rebuild probes in the child without replaying
        # YBParams.
        group._probe_interval_s = interval_s
        group._probe_check_timeout_s = check_timeout_s
        group._probe_drain_timeout_s = drain_timeout_s
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
        # Register the fork-safety hook (idempotent via a module-level
        # flag; see `_register_fork_hook_once`).
        _register_fork_hook_once()

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
        last_exc: "Exception | None" = None
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
                last_exc = exc
                # We just discovered this host refuses connections. Mark it
                # failed so the dispatcher's next pick doesn't waste a real
                # connect attempt on it.
                self.mark_failed(state.uuid, node.host)
                continue
            with state.lock:
                if state.control_sync is None:
                    state.control_sync = conn
                    state.last_control_connect_error = None  # clear on success
                    logger.info("control conn re-opened on %s for uuid=%s",
                                node.host, state.uuid)
                    # agent-K PR #17:G6 — surface placeholder promotion.
                    # When the state's uuid is still ``pending:<hash>``
                    # the operator's monitoring will not find this
                    # group by its real universe uuid. A full in-place
                    # uuid upgrade would need to swap keys in
                    # ``_clusters`` / ``_key_to_uuid`` /
                    # ``_failover_groups`` under lock while the probe
                    # thread may be mid-tick — bigger surgery than
                    # this follow-up. Log at WARNING so operators can
                    # correlate ``pending:...`` lines in their logs
                    # with the host that promoted them.
                    if _is_placeholder_uuid(state.uuid):
                        logger.warning(
                            "placeholder cluster uuid=%s was promoted "
                            "via control conn on %s; the uuid stays "
                            "``pending:...`` for the lifetime of this "
                            "group. Monitoring tools keyed on the "
                            "real universe uuid will not find this "
                            "group — re-bootstrap after both universes "
                            "are reachable for a clean uuid.",
                            state.uuid, node.host,
                        )
                    return conn
                winner = state.control_sync
            try:
                conn.close()
            except Exception:
                pass
            logger.debug("control conn re-open race lost; closing our duplicate")
            return winner
        # Every candidate refused — stamp the last error so CB callbacks
        # can distinguish misconfig (auth / missing DB / TLS) from a real
        # outage before counting this toward the failure threshold.
        with state.lock:
            state.last_control_connect_error = last_exc
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
        last_exc: "Exception | None" = None
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
                last_exc = exc
                # Refusing host → mark it failed so the dispatcher's next
                # pick skips it for the TTL window.
                self.mark_failed(state.uuid, node.host)
                continue
            with state.lock:
                if state.control_async is None:
                    state.control_async = conn
                    state.last_control_connect_error = None  # clear on success
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
        with state.lock:
            state.last_control_connect_error = last_exc
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

        * Nodes that appear in ``new_nodes``: ``connection_count`` is
          preserved from the existing entry (we own that counter, not the
          master). ``is_down``/``is_down_since`` are also preserved — the
          policy's ``failed_host_reconnect_delay_secs`` TTL (default 5s)
          re-eligibles the node naturally once the window elapses, so a
          mark_failed set by ``_ensure_control_sync`` or a connect failure
          doesn't get wiped by the next refresh that still lists the host
          (the master keeps a tserver listed through its heartbeat timeout,
          so a dead host reappears in ``yb_servers()`` for several seconds).
          ``absent_refresh_count`` resets to 0 when the host reappears.
        * Nodes that were previously known but are MISSING from
          ``new_nodes``: kept in ``state.nodes`` and marked ``is_down=True``
          (with ``is_down_since=now`` only if not already down). A transient
          ``yb_servers()`` response missing a tserver that is still
          re-registering with the master should NOT cause us to forget the
          host entirely — the next refresh either restores it or confirms
          it's gone. Wholesale-dropping makes the dispatcher pick from a
          shrunk subset for the entire next refresh interval, which
          empirically manifests as "all conns go to a 2-host subset of a
          3-host cluster" right after failback. After
          ``_MAX_ABSENCE_TICKS`` consecutive misses the host is pruned —
          otherwise a decommissioned node stays an eligible candidate
          forever (TTL expires every window, policy picks it, connect fails,
          mark_failed re-arms, lather-rinse-repeat).
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
                    n.is_down = existing.is_down
                    n.is_down_since = existing.is_down_since
                    # Reappeared → reset absence counter.
                    n.absent_refresh_count = 0
                new_dict[n.host] = n
            # Carry forward nodes that vanished from this refresh (they
            # might be transient still re-registering after restart), up to
            # the absence budget — then prune.
            now = time.monotonic()
            pruned: set[str] = set()
            for host in removed:
                stale = state.nodes[host]
                stale.absent_refresh_count += 1
                if stale.absent_refresh_count > _MAX_ABSENCE_TICKS:
                    pruned.add(host)
                    continue
                if not stale.is_down:
                    stale.is_down = True
                    stale.is_down_since = now
                new_dict[host] = stale
            state.nodes = new_dict
            state.force_refresh = False
        if added or removed or pruned:
            logger.info(
                "topology change observed for uuid=%s: added=%s removed=%s "
                "pruned=%s",
                state.uuid, sorted(added) or "—", sorted(removed) or "—",
                sorted(pruned) or "—",
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

        **Caller contract (agent-K PR #10:1375):** this drops every
        ``FailoverGroup`` the registry holds. CBs the application
        attached to those groups via ``bootstrap_failover_group`` are
        dropped alongside — the application's local reference to the
        group becomes orphaned. The next ``psycopg.connect(xcluster_dsn)``
        will bootstrap a FRESH group with empty CB slots and raise
        :class:`MissingCircuitBreakerError`. ``clear()`` is intended as
        a shutdown signal (and used by the test-isolation fixture); if
        the application needs to keep running after calling it, it MUST
        re-call ``bootstrap_failover_group`` and re-attach its CBs
        before the next connect.
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
