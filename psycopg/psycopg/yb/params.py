"""
Parse psycopg-yugabytedb smart-driver parameters out of the user's conninfo
string / kwargs.

Why this lives here and not in `conninfo.py`:

- libpq's parser (which psycopg's `conninfo_to_dict` ultimately calls) rejects
  unknown parameter names. Three of our four params (`topology_keys`,
  `yb_servers_refresh_interval`, `failed_host_reconnect_delay_secs`) are
  unknown to libpq and would error out before we got a chance to read them.

- The fourth, `load_balance_hosts`, is a libpq parameter we *extend* with new
  values (`true`/`false`). libpq accepts the keyword but rejects unknown
  values at connect time. So we strip `true`/`false` and let `disable`/
  `random` flow through.

We use a tolerant regex-based pre-parser sufficient for our parameter shapes
(values never contain whitespace). The cleaned conninfo string and kwargs
returned by `extract_yb_params` are then safe to hand to libpq.
"""

# Copyright (C) 2026 Yugabyte

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from .node import Placement

logger = logging.getLogger(__name__)


# Mirror pgjdbc-yb's clamps. See design doc §3.2.4.
DEFAULT_REFRESH_INTERVAL_SEC = 300
MAX_REFRESH_INTERVAL_SEC = 600
DEFAULT_FAILED_HOST_RECONNECT_DELAY_SEC = 5
MAX_FAILED_HOST_RECONNECT_DELAY_SEC = 60

# xCluster failover defaults — see /tmp/xcluster_failover_design.html §5.
# `trackerTableTablets` and `maxUpdateFailuresAllowed` used to live here;
# they were tracker-table-CB-specific config. Now that the driver ships
# without a default CB, that config moves to the sample implementation
# (see demo/samples/tracker_table_cb.py). Keep only the driver-generic
# knobs here.
DEFAULT_COOLDOWN_SEC = 1500
# `check_timeout_s` default is derived from `refresh_interval_s` at parse
# time — see extract_yb_params. `1` here is the floor when no refresh is set.
DEFAULT_CHECK_TIMEOUT_SEC = 1
# Barrier-with-timeout drain default (design doc §3.3). Sentinel values:
#   -1 → wait indefinitely for in-flight transactions; never force-close.
#    0 → force-close in-flight transactions immediately. No drain window.
#    N > 0 → wait up to N seconds, then force-close survivors.
DEFAULT_DRAIN_TIMEOUT_SEC = 10

# YBA lag-wait phase (Amogh's flow, 2026-08-11). All zero-value / unset by
# default; the phase is opt-in via ``yb.failover.ybaEndpoint``. When
# unset the drain flip is immediate after the txn drain — same as
# pre-lag-phase behaviour.
DEFAULT_THRESHOLD_REPLICATION_LAG_MS = 0
# DB-24483: with the freshness + liveness gates enabled, a planned
# failover's wall-clock floor is `lag_freshness_s + N × scrape_hint`.
# The pre-DB-24483 default of 30s remains the shipped default so
# existing harness profiles that pass lagWaitTimeoutSecs=30 keep
# working. Operators who turn the gate on (lagFreshnessSecs>0 and/or
# lagConfirmationSamples>1) should raise this knob accordingly.
DEFAULT_LAG_WAIT_TIMEOUT_SEC = 30

# DB-24483 — freshness gate. Minimum age `sample_ts - barrier_ts` before
# a YBA lag sample is trusted to describe post-barrier replication. The
# tserver recomputes the lag metric every 15s (cdc_service.cc:101) and
# YBA scrapes every 10s (reference.conf:985); the sum (25s) is the
# wall-clock floor past which a sample is guaranteed to reflect writes
# that landed before the barrier was armed. Setting this to 0 disables
# the gate — the shipped default — because enabling it requires the
# operator to raise lagWaitTimeoutSecs above the derived floor, and
# ratcheting an existing deployment's timeout silently on upgrade would
# break harnesses that assume the pre-DB-24483 ~30s budget. Opt in via
# `yb.failover.lagFreshnessSecs=25` (and `yb.failback.lagFreshnessSecs=25`)
# together with a `lagWaitTimeoutSecs` ≥ floor.
DEFAULT_LAG_FRESHNESS_SEC = 0
# DB-24483 — liveness gate. Number of consecutive post-barrier samples
# at lag ≤ threshold required before the drain flips routing. YBA's
# known pathology (DB-24852-adjacent): when the primary dies it replays
# the last pre-death sample for its full ~5min Prometheus window; one
# "lag=0" post-barrier reading is not enough to tell a true converged
# standby from a replayed stale sample. Defaults to 1 (disabled); turn
# on with `yb.failover.lagConfirmationSamples=2` once the operator has
# budgeted the extra ~10s per confirmation sample.
DEFAULT_LAG_CONFIRMATION_SAMPLES = 1
# DB-24483 — Prometheus scrape interval YBA uses, used only in the
# bootstrap floor calculation (lag_wait_timeout ≥ freshness + N × hint).
# Not currently a DSN knob; follow YBA's reference.conf default. If YBA
# ever exposes a per-universe override, surface it here as a knob.
_LAG_SCRAPE_HINT_S = 10

# Auto-failback toggle. When True (the default), the driver automatically
# returns traffic to the primary once the primary CB flips back to
# HEALTHY. When False, the driver stays on secondary even after primary
# recovers — an operator has to trigger failback via
# ExternalSignalCircuitBreaker (or a future force_recheck API).
DEFAULT_AUTO_FAILBACK_ENABLED = True

# Global failover time budget (Amex ask, Aug 2026). When set, derives
# drainTimeoutSecs and lagWaitTimeoutSecs so the total post-detection
# failover wall-clock ≤ budget. See design doc §3.7 for the semantics
# and the 40/60 split rationale.
#
# The clock starts at CB detection (after ``check()`` returns UNHEALTHY)
# and ends at the routing flip; it does NOT include
# ``yb_servers_refresh_interval`` (detection latency), ``checkTimeoutSecs``
# (CB check cap), or ``cooldownSecs`` (rate limit).
_BUDGET_DRAIN_RATIO = 0.40
_BUDGET_LAG_WAIT_RATIO = 0.60

# YBAClient inner-timeout defaults. When ``globalTimeBudgetSecs`` is set,
# these get lowered so a single blocked YBA call cannot eat the whole
# Phase 2 window:
#     connect_timeout_s = min(2.0, lag_wait_timeout_s * 0.25)
#     read_timeout_s    = min(3.0, lag_wait_timeout_s * 0.35)
DEFAULT_YBA_CONNECT_TIMEOUT_S = 2.0
DEFAULT_YBA_READ_TIMEOUT_S = 3.0
_BUDGET_YBA_CONNECT_RATIO = 0.25
_BUDGET_YBA_READ_RATIO = 0.35

# Conninfo keys that always come out of the string before libpq parses it.
# `load_balance_hosts` is special-cased: libpq-recognised values stay; our
# extension values (true/false) come out.
_PURE_YB_KEYS = frozenset({
    "topology_keys",
    "yb_servers_refresh_interval",
    "failed_host_reconnect_delay_secs",
    # xCluster failover keys (canonical snake_case forms). The regex below
    # accepts camelCase, dotted, dashed, and snake forms; all normalise to
    # these names.
    "yb_failover_secondary_cluster_hosts",
    "yb_failover_cooldown_secs",
    "yb_failover_check_timeout_secs",
    "yb_failover_drain_timeout_secs",
    # YBA lag-wait phase.
    "yb_failover_yba_endpoint",
    "yb_failover_yba_api_token",
    "yb_failover_replication_name",
    "yb_failover_threshold_replication_lag_ms",
    "yb_failover_lag_wait_timeout_secs",
    # DB-24483 — freshness + liveness gates for Phase 2.
    "yb_failover_lag_freshness_secs",
    "yb_failover_lag_confirmation_samples",
    # Global failover time budget — derives drain + lag-wait when set.
    "yb_failover_global_time_budget_secs",
    # Auto-failback toggle (bool; default true).
    "yb_failover_auto_failback_enabled",
    # Phase 2 timeout policy (DB-corruption fix). Values: 'fail' (default)
    # or 'proceed'. On 'fail', when Phase 2 times out before lag converges
    # the drain stays on the current side (does NOT flip to a known-stale
    # target). On 'proceed', the pre-fix behavior is preserved: flip
    # anyway and accept the data-consistency risk.
    "yb_failover_on_phase2_timeout",
    # Failback namespace — optional overrides for the reverse (B→A)
    # direction. See design doc §3.5.1. Fallback rules:
    #   * replicationName: NO fallback (points to a different xCluster
    #     config on YBA). Empty means "Phase 2 skipped on failback".
    #   * Numeric knobs: fall back to the failover value.
    "yb_failback_replication_name",
    "yb_failback_drain_timeout_secs",
    "yb_failback_lag_wait_timeout_secs",
    "yb_failback_threshold_replication_lag_ms",
    "yb_failback_global_time_budget_secs",
    # DB-24483 — same gates in the failback direction.
    "yb_failback_lag_freshness_secs",
    "yb_failback_lag_confirmation_samples",
})
_LOAD_BALANCE_KEY = "load_balance_hosts"
_OUR_LB_VALUES = frozenset({"true", "false"})

# All keys we recognise, with permissive separator handling.
#
# For the xCluster `yb.failover.*` parameters, the spec writes camelCase
# names (`secondaryClusterHosts`, `trackerTableTablets`, ...). We also
# accept underscore and dash forms (`secondary_cluster_hosts`,
# `secondary-cluster-hosts`) for consistency with the existing params.
# Separators between the `yb`, `failover`, and tail tokens accept any of
# `.`, `_`, `-`. `re.IGNORECASE` covers casing variations.
_YB_KEY_RE = re.compile(
    r"(?P<key>\b("
    r"topology[_-]keys"
    r"|yb[_-]servers[_-]refresh[_-]interval"
    r"|failed[_-]host[_-]reconnect[_-]delay[_-]secs"
    r"|load[_-]balance[_-]hosts"
    r"|yb[._-]failover[._-](?:secondaryClusterHosts|secondary[._-]cluster[._-]hosts)"
    r"|yb[._-]failover[._-](?:cooldownSecs|cooldown[._-]secs)"
    r"|yb[._-]failover[._-](?:checkTimeoutSecs|check[._-]timeout[._-]secs)"
    r"|yb[._-]failover[._-](?:drainTimeoutSecs|drain[._-]timeout[._-]secs)"
    r"|yb[._-]failover[._-](?:ybaEndpoint|yba[._-]endpoint)"
    r"|yb[._-]failover[._-](?:ybaApiToken|yba[._-]api[._-]token)"
    r"|yb[._-]failover[._-](?:replicationName|replication[._-]name)"
    r"|yb[._-]failover[._-](?:thresholdReplicationLagMs|threshold[._-]replication[._-]lag[._-]ms)"
    r"|yb[._-]failover[._-](?:lagWaitTimeoutSecs|lag[._-]wait[._-]timeout[._-]secs)"
    r"|yb[._-]failover[._-](?:lagFreshnessSecs|lag[._-]freshness[._-]secs)"
    r"|yb[._-]failover[._-](?:lagConfirmationSamples|lag[._-]confirmation[._-]samples)"
    r"|yb[._-]failover[._-](?:globalTimeBudgetSecs|global[._-]time[._-]budget[._-]secs)"
    r"|yb[._-]failover[._-](?:autoFailbackEnabled|auto[._-]failback[._-]enabled)"
    # agent-K PR #18:119 — ``onPhase2Timeout`` was missing from the key
    # splitter regex, so a DSN containing it bled through to libpq
    # unchanged. Every connect then failed with "invalid connection
    # option 'yb.failover.onphase2timeout'" and the knob was silently
    # ignored. Fix: list it alongside the other failover keys.
    r"|yb[._-]failover[._-](?:onPhase2Timeout|on[._-]phase2[._-]timeout)"
    # Failback namespace — same knob names as yb.failover.* but under
    # the yb.failback.* prefix; each is an optional per-direction override.
    r"|yb[._-]failback[._-](?:replicationName|replication[._-]name)"
    r"|yb[._-]failback[._-](?:drainTimeoutSecs|drain[._-]timeout[._-]secs)"
    r"|yb[._-]failback[._-](?:lagWaitTimeoutSecs|lag[._-]wait[._-]timeout[._-]secs)"
    r"|yb[._-]failback[._-](?:lagFreshnessSecs|lag[._-]freshness[._-]secs)"
    r"|yb[._-]failback[._-](?:lagConfirmationSamples|lag[._-]confirmation[._-]samples)"
    r"|yb[._-]failback[._-](?:thresholdReplicationLagMs|threshold[._-]replication[._-]lag[._-]ms)"
    r"|yb[._-]failback[._-](?:globalTimeBudgetSecs|global[._-]time[._-]budget[._-]secs)"
    r"))\s*=\s*"
    # Value: bare token (no whitespace), or single-quoted, or double-quoted.
    # libpq's wire format technically supports backslash escapes inside quotes;
    # our params never need that, so we keep the parser narrow.
    r"(?P<value>'[^']*'|\"[^\"]*\"|\S+)",
    re.IGNORECASE,
)

# Pre-compile the camelCase → snake_case boundary splitter for _normalize_key.
_CAMEL_BOUNDARY_RE = re.compile(r"([a-z])([A-Z])")

# Prefix check: the camelCase splitter must ONLY run on yb.failover /
# yb.failback keys, because applied unconditionally it mangles existing
# libpq keys like `load_balance_hostS` → `load_balance_host_s` →
# silently dropped since nothing reads that spelling. agent-K finding
# on PR #7:209.
_YB_FAILOVER_KEY_PREFIX_RE = re.compile(
    r"^yb[._-]?fail(over|back)", re.IGNORECASE,
)

# Map post-normalization key shapes to the canonical snake_case form.
#
# Why this exists: `_CAMEL_BOUNDARY_RE` splits ``aB`` into ``a_B``, which is
# correct for mixed-case inputs like ``secondaryClusterHosts``. But ALL-CAPS
# inputs like ``SECONDARYCLUSTERHOSTS`` have no lowercase-uppercase boundary,
# so the regex doesn't split — after lowercasing we get
# ``secondaryclusterhosts`` (one token). Same situation for users passing
# the kwarg as ``**{"yb_failover_secondaryclusterhosts": ...}``. This table
# folds those single-token forms back to the canonical snake_case name.
_FAILOVER_KEY_ALIASES: dict[str, str] = {
    "yb_failover_secondaryclusterhosts":         "yb_failover_secondary_cluster_hosts",
    "yb_failover_cooldownsecs":                  "yb_failover_cooldown_secs",
    "yb_failover_checktimeoutsecs":              "yb_failover_check_timeout_secs",
    "yb_failover_draintimeoutsecs":              "yb_failover_drain_timeout_secs",
    "yb_failover_ybaendpoint":                   "yb_failover_yba_endpoint",
    "yb_failover_ybaapitoken":                   "yb_failover_yba_api_token",
    "yb_failover_replicationname":               "yb_failover_replication_name",
    "yb_failover_thresholdreplicationlagms":     "yb_failover_threshold_replication_lag_ms",
    "yb_failover_lagwaittimeoutsecs":            "yb_failover_lag_wait_timeout_secs",
    "yb_failover_lagfreshnesssecs":              "yb_failover_lag_freshness_secs",
    "yb_failover_lagconfirmationsamples":        "yb_failover_lag_confirmation_samples",
    "yb_failover_globaltimebudgetsecs":          "yb_failover_global_time_budget_secs",
    "yb_failover_autofailbackenabled":           "yb_failover_auto_failback_enabled",
    "yb_failover_onphase2timeout":               "yb_failover_on_phase2_timeout",
    # Failback namespace.
    "yb_failback_replicationname":               "yb_failback_replication_name",
    "yb_failback_draintimeoutsecs":              "yb_failback_drain_timeout_secs",
    "yb_failback_lagwaittimeoutsecs":            "yb_failback_lag_wait_timeout_secs",
    "yb_failback_lagfreshnesssecs":              "yb_failback_lag_freshness_secs",
    "yb_failback_lagconfirmationsamples":        "yb_failback_lag_confirmation_samples",
    "yb_failback_thresholdreplicationlagms":     "yb_failback_threshold_replication_lag_ms",
    "yb_failback_globaltimebudgetsecs":          "yb_failback_global_time_budget_secs",
}


def _normalize_key(k: str) -> str:
    """Map any recognised form to its canonical snake_case name.

    Handles camelCase (``secondaryClusterHosts`` → ``secondary_cluster_hosts``),
    kebab-case (``-`` → ``_``), and dotted (``yb.failover.X`` → ``yb_failover_X``).
    Inputs already in snake_case pass through unchanged after lowercasing.
    All-caps single-token inputs are folded via ``_FAILOVER_KEY_ALIASES``.

    The camelCase split is **only** applied to ``yb.failover.*`` and
    ``yb.failback.*`` keys — otherwise existing libpq keys with
    accidental mixed case (e.g. ``load_balance_hostS``) would get split
    (``load_balance_host_s``) and silently dropped. agent-K PR #7:209.
    """
    if _YB_FAILOVER_KEY_PREFIX_RE.match(k):
        k = _CAMEL_BOUNDARY_RE.sub(r"\1_\2", k)
    k = k.lower().replace("-", "_").replace(".", "_")
    return _FAILOVER_KEY_ALIASES.get(k, k)


def _strip_quotes(v: str) -> str:
    if len(v) >= 2 and v[0] == v[-1] and v[0] in ("'", '"'):
        return v[1:-1]
    return v


@dataclass
class YBParams:
    """Parsed smart-driver knobs."""

    # True when load_balance_hosts=true; gates the entire smart-driver path.
    smart_driver_enabled: bool = False

    # Empty list means cluster-aware (no placement filter). Non-empty triggers
    # the topology-aware policy in `yb/policy/__init__.py:build_policy`.
    topology_keys: list[Placement] = field(default_factory=list)

    refresh_interval_s: int = DEFAULT_REFRESH_INTERVAL_SEC
    failed_host_reconnect_delay_s: int = DEFAULT_FAILED_HOST_RECONNECT_DELAY_SEC

    # xCluster failover parameters. All inert unless `xcluster_enabled` is
    # True (gated on `secondaryClusterHosts` non-empty AND
    # `load_balance_hosts=true`).
    secondary_cluster_hosts: list[str] = field(default_factory=list)
    cooldown_s: int = DEFAULT_COOLDOWN_SEC
    # Wall-clock cap on each CircuitBreaker.check() call, enforced by the
    # probe thread. Ticks that exceed the cap are abandoned; the previous
    # per-cluster status is preserved. Prevents a blocking custom CB from
    # stalling failover. Default: max(1, refresh_interval_s // 2) —
    # computed at parse time in extract_yb_params.
    check_timeout_s: int = DEFAULT_CHECK_TIMEOUT_SEC
    # Barrier-with-timeout drain (design doc §3.3). Sentinel values:
    #   -1 → wait indefinitely; never force-close in-flight transactions.
    #    0 → force-close immediately; no drain window.
    #    N > 0 → wait up to N seconds, then force-close survivors.
    drain_timeout_s: int = DEFAULT_DRAIN_TIMEOUT_SEC

    # YBA lag-wait phase parameters (Amogh's flow, 2026-08-11).
    # Phase 2 opts in only when ALL of these are set:
    # ``yba_endpoint`` + ``yba_api_token`` + ``replication_name`` +
    # ``xcluster_enabled``. Missing any one disables the phase (the
    # test `test_yba_params_endpoint_alone_not_enough` asserts this).
    # When enabled, the driver polls YBA every 1s after the txn drain
    # and waits for replication lag to converge to
    # ``threshold_replication_lag_ms`` before flipping routing. The phase
    # exits early on wait-timeout or 3 consecutive YBA-unreachable polls.
    #
    # The user provides the human-readable ``replication_name`` (as shown
    # in YBA's DR page); the driver resolves it to ``customerUUID`` and
    # ``xClusterConfigUUID`` at bootstrap and caches them on the group.
    yba_endpoint: str = ""
    yba_api_token: str = ""
    replication_name: str = ""
    threshold_replication_lag_ms: int = DEFAULT_THRESHOLD_REPLICATION_LAG_MS
    lag_wait_timeout_s: int = DEFAULT_LAG_WAIT_TIMEOUT_SEC
    # DB-24483 — freshness + liveness gates on YBA lag samples. The
    # freshness gate requires `sample_ts > barrier_ts + lag_freshness_s`
    # so a sample is guaranteed to describe post-barrier replication
    # (tserver update + Prometheus scrape slack). The liveness gate
    # requires N consecutive post-barrier samples at `lag ≤ threshold`
    # to reject YBA's known "replay last sample for ~5 min" pathology.
    # `lag_freshness_s=0` disables the freshness gate (restores pre-fix
    # behaviour, accepts the data-loss risk). `lag_confirmation_samples=1`
    # disables the liveness gate.
    lag_freshness_s: int = DEFAULT_LAG_FRESHNESS_SEC
    lag_confirmation_samples: int = DEFAULT_LAG_CONFIRMATION_SAMPLES

    # Global failover time budget (Amex ask, design doc §3.7). When set,
    # ``drain_timeout_s`` and ``lag_wait_timeout_s`` are derived from it
    # (40/60 split) unless the user set them explicitly. If both budget
    # and phase knobs are user-set, the parser validates that the sum
    # ≤ budget and raises ``ValueError`` otherwise.
    # ``None`` means "no budget — use existing behaviour, fully back-compat".
    global_time_budget_secs: int | None = None

    # YBAClient inner-request timeouts. Not user-facing DSN params;
    # written by the parser either to the ``DEFAULT_YBA_*`` constants
    # (no budget) or to smaller values derived from
    # ``lag_wait_timeout_s`` (budget set). Passed by the registry to
    # ``YBAClient(...)`` at wire time so a single blocked YBA call
    # cannot eat the whole Phase 2 window at small budgets.
    yba_connect_timeout_s: float = DEFAULT_YBA_CONNECT_TIMEOUT_S
    yba_read_timeout_s: float = DEFAULT_YBA_READ_TIMEOUT_S

    # Auto-failback toggle. When True (default), the driver returns
    # traffic to the primary automatically once the primary CB flips
    # back to HEALTHY. When False, the driver stays on secondary even
    # after primary recovers — operator must trigger failback manually
    # (write to the ExternalSignalCircuitBreaker's signal table).
    auto_failback_enabled: bool = DEFAULT_AUTO_FAILBACK_ENABLED

    # Phase 2 timeout policy — determines what happens when the YBA lag
    # wait exceeds ``lag_wait_timeout_s`` without converging.
    #
    # ``"fail"`` (default): the drain does NOT flip routing. Dispatch
    #     unpauses, status stays on the current side, a WARNING is
    #     logged, and the probe thread will try again on its next tick.
    #     Protects against DB corruption from flipping to a target that
    #     is known to be behind.
    #
    # ``"proceed"``: the pre-fix behavior — flip anyway and accept the
    #     data-consistency risk. Operators who want aggressive failover
    #     under lag-source outages can opt in.
    on_phase2_timeout: str = "fail"

    # Failback overrides (design doc §3.5.1). Each is populated at
    # parse time with either the operator's ``yb.failback.<knob>`` value
    # (when set) or the corresponding ``yb.failover.<knob>`` fallback
    # (when the failback variant is unset). The one exception is
    # ``failback_replication_name``: it has NO fallback since failover
    # (A→B) and failback (B→A) use DIFFERENT xCluster configs on YBA.
    # An empty string means "no Phase 2 on failback" (fail-open).
    failback_replication_name: str = ""
    failback_drain_timeout_s: int = DEFAULT_DRAIN_TIMEOUT_SEC
    failback_lag_wait_timeout_s: int = DEFAULT_LAG_WAIT_TIMEOUT_SEC
    failback_threshold_replication_lag_ms: int = DEFAULT_THRESHOLD_REPLICATION_LAG_MS
    failback_global_time_budget_secs: int | None = None
    # DB-24483 — same gates on the failback (B→A) direction. Default to
    # the same values as the failover direction unless overridden.
    failback_lag_freshness_s: int = DEFAULT_LAG_FRESHNESS_SEC
    failback_lag_confirmation_samples: int = DEFAULT_LAG_CONFIRMATION_SAMPLES

    @property
    def lag_wait_enabled(self) -> bool:
        """Lag-wait phase is opt-in: requires xCluster enabled AND all
        three credentials/identifiers present in the DSN. The customer
        UUID and xCluster config UUID are NOT in this check — they're
        resolved at bootstrap from ``replication_name``."""
        return (
            self.xcluster_enabled
            and bool(self.yba_endpoint)
            and bool(self.yba_api_token)
            and bool(self.replication_name)
        )

    @property
    def xcluster_enabled(self) -> bool:
        """xCluster failover is opt-in: requires BOTH ``load_balance_hosts=true``
        AND a non-empty ``secondaryClusterHosts``. Spec §5 configuration rules."""
        return self.smart_driver_enabled and bool(self.secondary_cluster_hosts)


def _parse_placement(s: str) -> Placement:
    parts = s.strip().split(".")
    if len(parts) != 3:
        raise ValueError(
            f"topology_keys entry must be 'cloud.region.zone', got {s!r}"
        )
    cloud, region, zone = parts
    if cloud == "*" or region == "*":
        raise ValueError(
            f"wildcards in cloud or region are not supported: {s!r} "
            "(zone wildcard '*' is allowed)"
        )
    return Placement(cloud=cloud, region=region, zone=zone)


def _parse_topology_keys(s: str) -> list[Placement]:
    return [_parse_placement(p) for p in s.split(",") if p.strip()]


def _parse_host_list(s: str) -> list[str]:
    """Split comma-separated host list, strip whitespace, drop empties."""
    return [h.strip() for h in s.split(",") if h.strip()]


def _clamp(value: int, lo: int, hi: int) -> int:
    return max(lo, min(value, hi))


def _clamp_lo(value: int, lo: int) -> int:
    """One-sided clamp — no upper bound. Used for xCluster timing params where
    very large values are legitimate (long cool-downs)."""
    return max(lo, value)


def extract_yb_params(
    conninfo: str, kwargs: dict[str, object]
) -> tuple["YBParams", str, dict[str, object]]:
    """Read YB-specific params from conninfo + kwargs.

    Returns ``(YBParams, cleaned_conninfo, cleaned_kwargs)``. The cleaned
    conninfo string and kwargs have our smart-driver keys removed and are safe
    to pass to libpq's parser. ``load_balance_hosts`` with libpq-recognised
    values (`disable`, `random`) is left untouched.

    `kwargs` values take precedence over `conninfo` string values when both
    define the same key — same precedence rule as `psycopg.conninfo`.
    """
    raw: dict[str, str] = {}

    # 1. Scan the conninfo string. Build the cleaned version as we go.
    cleaned_parts: list[str] = []
    last_end = 0
    for m in _YB_KEY_RE.finditer(conninfo):
        key = _normalize_key(m.group("key"))
        value = _strip_quotes(m.group("value"))
        if key == _LOAD_BALANCE_KEY and value.lower() not in _OUR_LB_VALUES:
            # libpq-compatible value (disable/random/...) — pass through.
            continue
        raw[key] = value
        cleaned_parts.append(conninfo[last_end : m.start()])
        last_end = m.end()
    cleaned_parts.append(conninfo[last_end:])
    cleaned_conninfo = re.sub(r"\s+", " ", "".join(cleaned_parts)).strip()

    # 2. Walk kwargs and pull our keys out. kwargs override conninfo on collision.
    cleaned_kwargs: dict[str, object] = {}
    for k, v in kwargs.items():
        nk = _normalize_key(k)
        if nk in _PURE_YB_KEYS:
            raw[nk] = str(v)
            continue
        if nk == _LOAD_BALANCE_KEY:
            sv = str(v)
            if sv.lower() in _OUR_LB_VALUES:
                raw[nk] = sv
                continue
            # libpq-compatible value — pass through.
        cleaned_kwargs[k] = v

    # 3. Build YBParams from the collected raw values.
    smart = raw.get(_LOAD_BALANCE_KEY, "").lower() == "true"

    tk: list[Placement] = []
    if (tk_raw := raw.get("topology_keys")):
        tk = _parse_topology_keys(tk_raw)

    refresh = _clamp(
        int(raw.get("yb_servers_refresh_interval", DEFAULT_REFRESH_INTERVAL_SEC)),
        0,
        MAX_REFRESH_INTERVAL_SEC,
    )
    delay = _clamp(
        int(raw.get(
            "failed_host_reconnect_delay_secs",
            DEFAULT_FAILED_HOST_RECONNECT_DELAY_SEC,
        )),
        0,
        MAX_FAILED_HOST_RECONNECT_DELAY_SEC,
    )

    # xCluster failover params (defaults are inert; the `xcluster_enabled`
    # property gates whether downstream code does anything with them).
    #
    # agent-K PR #7:234 / :505: validation on `yb.failover.*` values only
    # fires when xCluster is actually enabled. Otherwise a non-xCluster
    # DSN that happens to carry a stray `yb.failover.*` key (e.g. from a
    # shared config template) would fail to connect with ValueError.
    # Compute the gate before any ValueError paths below.
    secondary_hosts: list[str] = []
    if (sh_raw := raw.get("yb_failover_secondary_cluster_hosts")):
        secondary_hosts = _parse_host_list(sh_raw)
    _xcluster_enabled_at_parse = bool(secondary_hosts) and smart
    cooldown = _clamp_lo(
        int(raw.get("yb_failover_cooldown_secs", DEFAULT_COOLDOWN_SEC)),
        0,
    )
    # Wall-clock cap on each CB check(). Default: max(1, refresh // 2). The
    # `max(1, ...)` guards against `refresh=0` producing a 0-second cap.
    # Explicit user values win; clamp floor is 1 second so probes always
    # get at least some chance to complete.
    check_timeout = _clamp_lo(
        int(raw.get(
            "yb_failover_check_timeout_secs",
            max(1, refresh // 2),
        )),
        1,
    )
    # Global failover time budget (§3.7). Parsed first because drain and
    # lag-wait derivation depend on it. ``None`` = unset = existing behaviour.
    global_time_budget_secs: int | None = None
    if (budget_raw := raw.get("yb_failover_global_time_budget_secs")) is not None:
        global_time_budget_secs = int(budget_raw)
        if global_time_budget_secs <= 0 and _xcluster_enabled_at_parse:
            # agent-K PR #7:234/:505: only raise when xCluster is
            # actually enabled; otherwise a plain non-xCluster DSN that
            # happens to carry this key would fail to connect.
            raise ValueError(
                "yb.failover.globalTimeBudgetSecs must be a positive integer, "
                f"got {global_time_budget_secs}"
            )

    # Track whether the user explicitly set the phase knobs. Distinguishes
    # "user set 10 which happens to be the default" from "user didn't set;
    # use default". Only relevant when a budget is present.
    drain_user_set = "yb_failover_drain_timeout_secs" in raw
    lag_wait_user_set = "yb_failover_lag_wait_timeout_secs" in raw

    # Drain timeout. Sentinels (-1, 0) are meaningful — do NOT clamp them
    # to a positive floor. Anything < -1 collapses to -1 (wait forever).
    if global_time_budget_secs is not None and not drain_user_set:
        # Derive from budget (40% of total).
        drain_timeout_raw = round(global_time_budget_secs * _BUDGET_DRAIN_RATIO)
    else:
        drain_timeout_raw = int(raw.get(
            "yb_failover_drain_timeout_secs", DEFAULT_DRAIN_TIMEOUT_SEC,
        ))
        if drain_timeout_raw < -1:
            drain_timeout_raw = -1

    # YBA lag-wait phase. Strings pass through untouched; numeric knobs
    # get one-sided lo-clamps to keep pathological values benign.
    yba_endpoint = raw.get("yb_failover_yba_endpoint", "").strip()
    yba_api_token = raw.get("yb_failover_yba_api_token", "").strip()
    replication_name = raw.get("yb_failover_replication_name", "").strip()
    threshold_replication_lag_ms = _clamp_lo(
        int(raw.get(
            "yb_failover_threshold_replication_lag_ms",
            DEFAULT_THRESHOLD_REPLICATION_LAG_MS,
        )),
        0,
    )
    if global_time_budget_secs is not None and not lag_wait_user_set:
        # Derive from budget (60% of total).
        lag_wait_timeout = _clamp_lo(
            round(global_time_budget_secs * _BUDGET_LAG_WAIT_RATIO), 0,
        )
    else:
        lag_wait_timeout = _clamp_lo(
            int(raw.get(
                "yb_failover_lag_wait_timeout_secs",
                DEFAULT_LAG_WAIT_TIMEOUT_SEC,
            )),
            0,
        )

    # When a budget is set, each phase knob has an implicit share of the
    # budget (drain: 40%, lag_wait: 60%). If the user explicitly sets a
    # knob above its share, the ratio-derivation of the other knob will
    # push the sum over the budget — reject at bootstrap. Sentinels
    # (-1, 0) on drain are "wait forever" / "kill immediately", not
    # wall-clock time, so they don't count against the budget.
    if global_time_budget_secs is not None and _xcluster_enabled_at_parse:
        drain_for_budget = max(0, drain_timeout_raw)
        effective_sum = drain_for_budget + lag_wait_timeout
        if effective_sum > global_time_budget_secs:
            drain_share = round(global_time_budget_secs * _BUDGET_DRAIN_RATIO)
            lag_wait_share = round(global_time_budget_secs * _BUDGET_LAG_WAIT_RATIO)
            raise ValueError(
                f"yb.failover.globalTimeBudgetSecs={global_time_budget_secs}s "
                f"exceeded — effective phase timeouts sum to {effective_sum}s "
                f"(drain={drain_timeout_raw}s, lag_wait={lag_wait_timeout}s). "
                f"Under this budget the ratio-derived shares are "
                f"drain={drain_share}s (40%) and lag_wait={lag_wait_share}s "
                "(60%); a phase knob above its share leaves no room for the "
                "other. Reduce a phase timeout or raise globalTimeBudgetSecs."
            )

    # DB-24483 — freshness + liveness gate knobs. Floors: freshness ≥ 0
    # (0 disables the gate); confirmation_samples ≥ 1 (1 disables the
    # liveness gate).
    lag_freshness_s = _clamp_lo(
        int(raw.get("yb_failover_lag_freshness_secs", DEFAULT_LAG_FRESHNESS_SEC)),
        0,
    )
    lag_confirmation_samples = _clamp_lo(
        int(raw.get(
            "yb_failover_lag_confirmation_samples",
            DEFAULT_LAG_CONFIRMATION_SAMPLES,
        )),
        1,
    )
    # Bootstrap floor: lag_wait_timeout must fit at least one liveness
    # window on top of the freshness wait, otherwise the gate can never
    # pass and every planned failover times out before converging.
    # Only check when Phase 2 will actually run — xcluster enabled AND
    # YBA credentials wired AND replication_name set. A DSN that leaves
    # Phase 2 disabled is free to carry any lag_wait_timeout_s (the knob
    # is inert in that case). Also skip when the user explicitly
    # disabled the gate (lag_freshness_s=0).
    _phase2_will_run = (
        _xcluster_enabled_at_parse
        and bool(yba_endpoint)
        and bool(yba_api_token)
        and bool(replication_name)
    )
    if _phase2_will_run and lag_wait_timeout > 0 and lag_freshness_s > 0:
        needed = lag_freshness_s + (lag_confirmation_samples * _LAG_SCRAPE_HINT_S)
        if lag_wait_timeout < needed:
            # Downgraded to a WARNING (was ValueError). Crashing bootstrap
            # on a timeout-below-floor is worse than letting the drain
            # run and time out: the operator may have deliberately picked
            # a short lag_wait to exercise the timeout path (stress-test
            # harnesses do this), and the Phase 2 timeout + abort policy
            # already handles the result safely. Operators who want strict
            # rejection can treat this logger at ERROR.
            logger.warning(
                "yb.failover.lagWaitTimeoutSecs=%d is below the floor %ds "
                "imposed by lagFreshnessSecs=%d + lagConfirmationSamples=%d × "
                "scrapeIntervalSecs=%d — Phase 2 will time out on every "
                "planned failover under this config. Raise lagWaitTimeoutSecs "
                "to ≥%d, lower lagFreshnessSecs/lagConfirmationSamples, or "
                "disable the gate with lagFreshnessSecs=0.",
                lag_wait_timeout, needed, lag_freshness_s,
                lag_confirmation_samples, _LAG_SCRAPE_HINT_S, needed,
            )

    # Auto-failback toggle. Default true (backward-compatible). Accepts
    # any of the standard bool forms; anything else raises ValueError.
    auto_failback_raw = raw.get("yb_failover_auto_failback_enabled")
    if auto_failback_raw is None:
        auto_failback_enabled = DEFAULT_AUTO_FAILBACK_ENABLED
    else:
        low = auto_failback_raw.strip().lower()
        if low in ("true", "1", "yes", "on"):
            auto_failback_enabled = True
        elif low in ("false", "0", "no", "off"):
            auto_failback_enabled = False
        elif _xcluster_enabled_at_parse:
            raise ValueError(
                "yb.failover.autoFailbackEnabled must be a boolean "
                f"(true/false), got {auto_failback_raw!r}"
            )
        else:
            auto_failback_enabled = DEFAULT_AUTO_FAILBACK_ENABLED

    # Phase 2 timeout policy. 'fail' (default) means do not flip when
    # lag doesn't converge; 'proceed' preserves pre-fix behavior.
    on_phase2_timeout_raw = raw.get("yb_failover_on_phase2_timeout")
    if on_phase2_timeout_raw is None:
        on_phase2_timeout = "fail"
    else:
        low = on_phase2_timeout_raw.strip().lower()
        if low in ("fail", "proceed"):
            on_phase2_timeout = low
        else:
            raise ValueError(
                "yb.failover.onPhase2Timeout must be 'fail' or 'proceed', "
                f"got {on_phase2_timeout_raw!r}"
            )

    # ==================== Failback (reverse-direction overrides) =====
    # All of these are optional overrides. When absent, the failback
    # side of the drain reuses the failover value — except
    # ``failback_replication_name``, which has no fallback (points to
    # a different xCluster config on YBA).
    failback_replication_name = raw.get(
        "yb_failback_replication_name", ""
    ).strip()
    failback_budget_raw = raw.get("yb_failback_global_time_budget_secs")
    failback_global_time_budget_secs: int | None
    if failback_budget_raw is not None:
        failback_global_time_budget_secs = int(failback_budget_raw)
        if failback_global_time_budget_secs <= 0 and _xcluster_enabled_at_parse:
            raise ValueError(
                "yb.failback.globalTimeBudgetSecs must be positive, "
                f"got {failback_global_time_budget_secs}"
            )
    else:
        # Fall back to the failover budget (may itself be None).
        failback_global_time_budget_secs = global_time_budget_secs

    failback_drain_user_set = "yb_failback_drain_timeout_secs" in raw
    failback_lag_wait_user_set = "yb_failback_lag_wait_timeout_secs" in raw

    if failback_drain_user_set:
        failback_drain_timeout_s = int(raw["yb_failback_drain_timeout_secs"])
        if failback_drain_timeout_s < -1:
            failback_drain_timeout_s = -1
    elif failback_budget_raw is not None:
        # Only when the operator explicitly set a failback-specific
        # budget do we derive drain from its 40% share. Previously this
        # also fired when `failback_global_time_budget_secs` was merely
        # INHERITED from the failover budget — which incorrectly
        # overrode an explicit `drain_timeout_raw` (e.g. -1 "never
        # force-close") with the ratio value, silently flipping the
        # failback-side contract. agent-K finding on PR #7:565.
        failback_drain_timeout_s = round(
            failback_global_time_budget_secs * _BUDGET_DRAIN_RATIO,
        )
    else:
        # No failback-specific budget → inherit the failover value
        # directly (including the -1 / 0 sentinels).
        failback_drain_timeout_s = drain_timeout_raw

    if failback_lag_wait_user_set:
        failback_lag_wait_timeout_s = _clamp_lo(
            int(raw["yb_failback_lag_wait_timeout_secs"]), 0,
        )
    elif failback_budget_raw is not None:
        # Same asymmetry fix as drain above: only an explicit
        # failback-side budget drives the ratio derivation. An
        # inherited failover budget alone must not override the
        # failover lag_wait value.
        failback_lag_wait_timeout_s = _clamp_lo(
            round(failback_global_time_budget_secs * _BUDGET_LAG_WAIT_RATIO),
            0,
        )
    else:
        failback_lag_wait_timeout_s = lag_wait_timeout

    # Validate the failback sum against its budget (same rule as
    # failover): drain sentinel (-1, 0) doesn't count as wall-clock.
    if failback_global_time_budget_secs is not None and (
        failback_drain_user_set or failback_lag_wait_user_set
        or failback_budget_raw is not None
    ):
        fb_drain_for_budget = max(0, failback_drain_timeout_s)
        fb_sum = fb_drain_for_budget + failback_lag_wait_timeout_s
        if fb_sum > failback_global_time_budget_secs:
            fb_drain_share = round(
                failback_global_time_budget_secs * _BUDGET_DRAIN_RATIO,
            )
            fb_lag_share = round(
                failback_global_time_budget_secs * _BUDGET_LAG_WAIT_RATIO,
            )
            raise ValueError(
                f"yb.failback.globalTimeBudgetSecs="
                f"{failback_global_time_budget_secs}s exceeded — effective "
                f"phase timeouts sum to {fb_sum}s "
                f"(drain={failback_drain_timeout_s}s, "
                f"lag_wait={failback_lag_wait_timeout_s}s). "
                f"Under this budget the ratio-derived shares are "
                f"drain={fb_drain_share}s (40%) and lag_wait="
                f"{fb_lag_share}s (60%); reduce a phase timeout or "
                "raise the failback budget."
            )

    if (fb_thresh_raw := raw.get(
        "yb_failback_threshold_replication_lag_ms",
    )) is not None:
        failback_threshold_replication_lag_ms = _clamp_lo(int(fb_thresh_raw), 0)
    else:
        failback_threshold_replication_lag_ms = threshold_replication_lag_ms

    # DB-24483 — failback-direction freshness + liveness gate overrides.
    # Fall back to the failover values when unset so the failback side
    # inherits the same correctness floor.
    if (fb_fresh_raw := raw.get("yb_failback_lag_freshness_secs")) is not None:
        failback_lag_freshness_s = _clamp_lo(int(fb_fresh_raw), 0)
    else:
        failback_lag_freshness_s = lag_freshness_s
    if (fb_conf_raw := raw.get("yb_failback_lag_confirmation_samples")) is not None:
        failback_lag_confirmation_samples = _clamp_lo(int(fb_conf_raw), 1)
    else:
        failback_lag_confirmation_samples = lag_confirmation_samples
    # Same floor check as failover, applied against the failback timeout.
    # Failback Phase 2 only runs when the operator wired a separate
    # yb.failback.replicationName — same gate as the drain uses.
    _fb_phase2_will_run = _phase2_will_run and bool(failback_replication_name)
    if (
        _fb_phase2_will_run
        and failback_lag_wait_timeout_s > 0
        and failback_lag_freshness_s > 0
    ):
        fb_needed = (
            failback_lag_freshness_s
            + failback_lag_confirmation_samples * _LAG_SCRAPE_HINT_S
        )
        if failback_lag_wait_timeout_s < fb_needed:
            # Downgraded to a WARNING; see the failover-side comment above
            # for rationale.
            logger.warning(
                "yb.failback.lagWaitTimeoutSecs=%d is below the floor %ds "
                "imposed by lagFreshnessSecs=%d + lagConfirmationSamples=%d × "
                "scrapeIntervalSecs=%d — failback Phase 2 will time out on "
                "every planned failback under this config. Raise "
                "yb.failback.lagWaitTimeoutSecs to ≥%d, lower the failback "
                "freshness/confirmation knobs, or disable the gate with "
                "yb.failback.lagFreshnessSecs=0.",
                failback_lag_wait_timeout_s, fb_needed, failback_lag_freshness_s,
                failback_lag_confirmation_samples, _LAG_SCRAPE_HINT_S, fb_needed,
            )

    # YBA client inner timeouts. Scale down when a budget is set on
    # EITHER side so a single blocked YBA call cannot eat the whole
    # Phase 2 window — on either failover OR failback. agent-K PR
    # #7:632: previously only `global_time_budget_secs` + `lag_wait_timeout`
    # were considered, so a failback-only budget (or a smaller failback
    # one) left YBA calls on defaults. Fix: scale off the SMALLER of
    # whichever budgets + lag_waits are actually in effect.
    if (
        global_time_budget_secs is not None
        or failback_global_time_budget_secs is not None
    ):
        # Pick the tightest lag_wait that will actually run (an unused
        # side is ignored by using the other's value).
        candidate_waits: list[int] = []
        if global_time_budget_secs is not None:
            candidate_waits.append(lag_wait_timeout)
        if failback_global_time_budget_secs is not None:
            candidate_waits.append(failback_lag_wait_timeout_s)
        tight_wait = min(candidate_waits) if candidate_waits else lag_wait_timeout
        yba_connect_timeout_s = min(
            DEFAULT_YBA_CONNECT_TIMEOUT_S,
            tight_wait * _BUDGET_YBA_CONNECT_RATIO,
        )
        yba_read_timeout_s = min(
            DEFAULT_YBA_READ_TIMEOUT_S,
            tight_wait * _BUDGET_YBA_READ_RATIO,
        )
    else:
        yba_connect_timeout_s = DEFAULT_YBA_CONNECT_TIMEOUT_S
        yba_read_timeout_s = DEFAULT_YBA_READ_TIMEOUT_S

    return (
        YBParams(
            smart_driver_enabled=smart,
            topology_keys=tk,
            refresh_interval_s=refresh,
            failed_host_reconnect_delay_s=delay,
            secondary_cluster_hosts=secondary_hosts,
            cooldown_s=cooldown,
            check_timeout_s=check_timeout,
            drain_timeout_s=drain_timeout_raw,
            yba_endpoint=yba_endpoint,
            yba_api_token=yba_api_token,
            replication_name=replication_name,
            threshold_replication_lag_ms=threshold_replication_lag_ms,
            lag_wait_timeout_s=lag_wait_timeout,
            lag_freshness_s=lag_freshness_s,
            lag_confirmation_samples=lag_confirmation_samples,
            global_time_budget_secs=global_time_budget_secs,
            yba_connect_timeout_s=yba_connect_timeout_s,
            yba_read_timeout_s=yba_read_timeout_s,
            auto_failback_enabled=auto_failback_enabled,
            on_phase2_timeout=on_phase2_timeout,
            failback_replication_name=failback_replication_name,
            failback_drain_timeout_s=failback_drain_timeout_s,
            failback_lag_wait_timeout_s=failback_lag_wait_timeout_s,
            failback_threshold_replication_lag_ms=(
                failback_threshold_replication_lag_ms
            ),
            failback_global_time_budget_secs=failback_global_time_budget_secs,
            failback_lag_freshness_s=failback_lag_freshness_s,
            failback_lag_confirmation_samples=failback_lag_confirmation_samples,
        ),
        cleaned_conninfo,
        cleaned_kwargs,
    )
