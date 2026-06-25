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

import re
from dataclasses import dataclass, field

from .node import Placement


# Mirror pgjdbc-yb's clamps. See design doc §3.2.4.
DEFAULT_REFRESH_INTERVAL_SEC = 300
MAX_REFRESH_INTERVAL_SEC = 600
DEFAULT_FAILED_HOST_RECONNECT_DELAY_SEC = 5
MAX_FAILED_HOST_RECONNECT_DELAY_SEC = 60

# xCluster failover defaults — see /tmp/xcluster_failover_design.html §5.
DEFAULT_TRACKER_TABLE_TABLETS = 9
DEFAULT_MAX_UPDATE_FAILURES_ALLOWED = 0
DEFAULT_COOLDOWN_SEC = 1500

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
    "yb_failover_tracker_table_tablets",
    "yb_failover_max_update_failures_allowed",
    "yb_failover_cooldown_secs",
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
    r"|yb[._-]failover[._-](?:trackerTableTablets|tracker[._-]table[._-]tablets)"
    r"|yb[._-]failover[._-](?:maxUpdateFailuresAllowed|max[._-]update[._-]failures[._-]allowed)"
    r"|yb[._-]failover[._-](?:cooldownSecs|cooldown[._-]secs)"
    r"))\s*=\s*"
    # Value: bare token (no whitespace), or single-quoted, or double-quoted.
    # libpq's wire format technically supports backslash escapes inside quotes;
    # our params never need that, so we keep the parser narrow.
    r"(?P<value>'[^']*'|\"[^\"]*\"|\S+)",
    re.IGNORECASE,
)

# Pre-compile the camelCase → snake_case boundary splitter for _normalize_key.
_CAMEL_BOUNDARY_RE = re.compile(r"([a-z])([A-Z])")

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
    "yb_failover_secondaryclusterhosts":     "yb_failover_secondary_cluster_hosts",
    "yb_failover_trackertabletablets":       "yb_failover_tracker_table_tablets",
    "yb_failover_maxupdatefailuresallowed":  "yb_failover_max_update_failures_allowed",
    "yb_failover_cooldownsecs":              "yb_failover_cooldown_secs",
}


def _normalize_key(k: str) -> str:
    """Map any recognised form to its canonical snake_case name.

    Handles camelCase (``secondaryClusterHosts`` → ``secondary_cluster_hosts``),
    kebab-case (``-`` → ``_``), and dotted (``yb.failover.X`` → ``yb_failover_X``).
    Inputs already in snake_case pass through unchanged after lowercasing.
    All-caps single-token inputs are folded via ``_FAILOVER_KEY_ALIASES``.
    """
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

    # xCluster failover parameters — see /tmp/xcluster_failover_design.html §5.
    # All four are inert defaults unless `xcluster_enabled` is True (gated on
    # `secondaryClusterHosts` non-empty AND `load_balance_hosts=true`).
    secondary_cluster_hosts: list[str] = field(default_factory=list)
    tracker_table_tablets: int = DEFAULT_TRACKER_TABLE_TABLETS
    max_update_failures_allowed: int = DEFAULT_MAX_UPDATE_FAILURES_ALLOWED
    cooldown_s: int = DEFAULT_COOLDOWN_SEC

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
    secondary_hosts: list[str] = []
    if (sh_raw := raw.get("yb_failover_secondary_cluster_hosts")):
        secondary_hosts = _parse_host_list(sh_raw)
    tablets = _clamp_lo(
        int(raw.get(
            "yb_failover_tracker_table_tablets",
            DEFAULT_TRACKER_TABLE_TABLETS,
        )),
        1,
    )
    max_fail = _clamp_lo(
        int(raw.get(
            "yb_failover_max_update_failures_allowed",
            DEFAULT_MAX_UPDATE_FAILURES_ALLOWED,
        )),
        0,
    )
    cooldown = _clamp_lo(
        int(raw.get("yb_failover_cooldown_secs", DEFAULT_COOLDOWN_SEC)),
        0,
    )

    return (
        YBParams(
            smart_driver_enabled=smart,
            topology_keys=tk,
            refresh_interval_s=refresh,
            failed_host_reconnect_delay_s=delay,
            secondary_cluster_hosts=secondary_hosts,
            tracker_table_tablets=tablets,
            max_update_failures_allowed=max_fail,
            cooldown_s=cooldown,
        ),
        cleaned_conninfo,
        cleaned_kwargs,
    )
