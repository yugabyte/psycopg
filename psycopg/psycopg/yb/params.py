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

# Conninfo keys that always come out of the string before libpq parses it.
# `load_balance_hosts` is special-cased: libpq-recognised values stay; our
# extension values (true/false) come out.
_PURE_YB_KEYS = frozenset({
    "topology_keys",
    "yb_servers_refresh_interval",
    "failed_host_reconnect_delay_secs",
})
_LOAD_BALANCE_KEY = "load_balance_hosts"
_OUR_LB_VALUES = frozenset({"true", "false"})

# All keys we recognise, dashed-or-underscore form, case-insensitive.
_YB_KEY_RE = re.compile(
    r"(?P<key>\b("
    r"topology[_-]keys"
    r"|yb[_-]servers[_-]refresh[_-]interval"
    r"|failed[_-]host[_-]reconnect[_-]delay[_-]secs"
    r"|load[_-]balance[_-]hosts"
    r"))\s*=\s*"
    # Value: bare token (no whitespace), or single-quoted, or double-quoted.
    # libpq's wire format technically supports backslash escapes inside quotes;
    # our params never need that, so we keep the parser narrow.
    r"(?P<value>'[^']*'|\"[^\"]*\"|\S+)",
    re.IGNORECASE,
)


def _normalize_key(k: str) -> str:
    return k.lower().replace("-", "_")


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


def _clamp(value: int, lo: int, hi: int) -> int:
    return max(lo, min(value, hi))


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

    return (
        YBParams(
            smart_driver_enabled=smart,
            topology_keys=tk,
            refresh_interval_s=refresh,
            failed_host_reconnect_delay_s=delay,
        ),
        cleaned_conninfo,
        cleaned_kwargs,
    )
