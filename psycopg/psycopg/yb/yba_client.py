"""
Thin YBA (Yugabyte Anywhere) REST client used by the smart driver's
xCluster failover to fetch replication lag.

The failover sequence (see design doc §Failover sequence) polls YBA every
second while a Phase-2 wait is in progress. This module implements only
what the poll loop needs — three endpoints, one metric extraction.

Wire format ported from Bill Chan's tools/yba-python-scripts/xcluster_lags.py:
  * base URL: ``{yba_endpoint}/api/v1``
  * auth header: ``X-AUTH-YW-API-TOKEN: <api_token>``
  * self-signed TLS is common in YBA deployments — ``verify_ssl`` defaults
    to False to match the field behaviour of that script.

Uses only stdlib (``urllib.request``, ``ssl``, ``json``) — no ``requests``
dependency added to the driver.
"""

# Copyright (C) 2026 Yugabyte

from __future__ import annotations

import http.client
import json
import logging
import ssl
import threading
import time
import urllib.parse
from typing import Any, Iterable

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------- constants

# YBA's Prometheus-backed metrics endpoint returns per-tablet lag values.
# Trace names we accept — Bill's script filters on any of these substrings.
_COMMITTED_LAG_TRACE_NAMES = (
    "Committed Lag",
    "Committed Lag (Milliseconds)",
    "async_replication_committed_lag_micros",
)

# YBA reports the tserver lag micros divided by 1000, so units are already
# milliseconds when they come back. When a tablet is caught up, YBA returns
# a "not reported" sentinel around 1e12 — we treat anything at or above
# ``_NOT_REPORTED_THRESHOLD_MS`` as "no data" (None).
_NOT_REPORTED_THRESHOLD_MS = 1.0e12

# YBA emits ~1 ms for caught-up tablets due to internal rounding. Anything
# at or below this ceiling is treated as fully drained (0.0 ms).
_CAUGHT_UP_LAG_MS = 1.0

# The /metrics endpoint requires a start/end window. YBA enforces a minimum
# window; use a window big enough that we always get at least one datapoint.
_MIN_METRICS_LOOKBACK_SEC = 300


# ------------------------------------------------------------------- errors

class YBAClientError(Exception):
    """Raised when a YBA API call fails (HTTP error, timeout, malformed
    response). Callers should treat this as "YBA unreachable this poll" —
    the failover sequence counts consecutive failures and proceeds without
    lag confirmation after N strikes."""


# ------------------------------------------------------------------- client

class YBAClient:
    """Minimal stdlib-only YBA REST client for the smart driver.

    Uses a **persistent** ``http.client.HTTPSConnection`` (or
    ``HTTPConnection``) so consecutive calls skip TCP+TLS handshake. In
    the failover poll loop this reduces per-call latency from ~1s to
    ~285ms against portal.dev — the delta matters when the wait window
    is bounded (default 30s) and we want to detect convergence fast.

    Thread-safety: all methods take ``self._lock`` before touching the
    connection object. The connection isn't safe to share between
    threads unlocked (``http.client``'s state machine assumes single
    caller between ``request()`` and ``getresponse()``).

    Reconnection: a broken pipe / stale-socket error on a request
    triggers one automatic reconnect + retry. Anything else surfaces as
    :class:`YBAClientError`.

    Attributes:
      * ``endpoint`` — YBA base URL (no trailing slash, no ``/api/v1``).
      * ``connect_timeout_s`` / ``read_timeout_s`` — bounded so a
        stalled YBA doesn't extend the 1-second poll interval too far.
    """

    def __init__(
        self,
        endpoint: str,
        api_token: str,
        verify_ssl: bool = False,
        connect_timeout_s: float = 2.0,
        read_timeout_s: float = 3.0,
    ) -> None:
        self.endpoint = _normalize_url(endpoint)
        self._api_token = api_token
        self.verify_ssl = verify_ssl
        self.connect_timeout_s = connect_timeout_s
        self.read_timeout_s = read_timeout_s
        self._customer_uuid: str | None = None

        # Parse the endpoint once — the persistent connection binds to
        # (host, port, scheme), which don't change across calls.
        parsed = urllib.parse.urlsplit(self.endpoint)
        self._host = parsed.hostname or ""
        self._port = parsed.port  # None = default (80/443)
        self._is_https = (parsed.scheme == "https")
        if not self._host:
            raise YBAClientError(
                f"YBAClient: could not parse host from endpoint {endpoint!r}"
            )

        # SSLContext prepared once. Recreated only if verify_ssl flips
        # (not exposed today; construction-time only).
        self._ssl_ctx: ssl.SSLContext | None = None
        if self._is_https:
            self._ssl_ctx = ssl.create_default_context()
            if not verify_ssl:
                # YBA over self-signed cert is the typical field
                # deployment; match Bill's script behaviour by default.
                self._ssl_ctx.check_hostname = False
                self._ssl_ctx.verify_mode = ssl.CERT_NONE

        # Persistent connection state. Lazy-opened on first request so
        # constructor is cheap and can't fail on the network path.
        self._conn: http.client.HTTPConnection | None = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------- HTTP core

    def _open_connection(self) -> http.client.HTTPConnection:
        """Open a fresh HTTP(S) connection. Idempotent — closes any
        existing one first."""
        self._close_connection()

        # http.client uses one wall-clock timeout. Sum of connect + read
        # matches the pattern Bill's script used with requests.
        timeout = self.connect_timeout_s + self.read_timeout_s
        if self._is_https:
            self._conn = http.client.HTTPSConnection(
                self._host,
                port=self._port,
                timeout=timeout,
                context=self._ssl_ctx,
            )
        else:
            self._conn = http.client.HTTPConnection(
                self._host,
                port=self._port,
                timeout=timeout,
            )
        # Force the TCP+TLS handshake up front so the first ``request()``
        # doesn't blend handshake latency into its own timing.
        self._conn.connect()
        return self._conn

    def _close_connection(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None

    def close(self) -> None:
        """Explicitly release the persistent connection. Idempotent.

        Not required for correctness — the connection is torn down when
        the client goes out of scope — but useful for tests and for
        callers who want deterministic teardown.
        """
        with self._lock:
            self._close_connection()

    def _request(
        self,
        method: str,
        path: str,
        body: dict | None = None,
        query: dict | None = None,
    ) -> Any:
        """Execute one HTTP call over the persistent connection. Returns
        parsed JSON on 2xx, raises :class:`YBAClientError` otherwise.

        On stale-socket errors (``ConnectionError`` / ``BadStatusLine`` /
        ``RemoteDisconnected``) reconnects once and retries. Anything
        after that surfaces to the caller — the poll loop's 3-strike
        rule handles it from there.
        """
        request_target = f"/api/v1{path}"
        if query:
            request_target = f"{request_target}?{urllib.parse.urlencode(query)}"

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "X-AUTH-YW-API-TOKEN": self._api_token,
            # Explicit — some proxies don't default to keep-alive; we do.
            "Connection": "keep-alive",
        }
        data = json.dumps(body).encode("utf-8") if body is not None else b""

        with self._lock:
            attempt_reconnect = False
            for attempt in (1, 2):
                if self._conn is None or attempt_reconnect:
                    try:
                        self._open_connection()
                    except (TimeoutError, OSError, ssl.SSLError) as exc:
                        raise YBAClientError(
                            f"YBA {method} {path} connect failed: {exc}"
                        ) from exc
                    attempt_reconnect = False

                try:
                    assert self._conn is not None
                    self._conn.request(method, request_target, body=data, headers=headers)
                    resp = self._conn.getresponse()
                    raw = resp.read()
                    status = resp.status
                except (http.client.RemoteDisconnected,
                        http.client.BadStatusLine,
                        ConnectionResetError,
                        BrokenPipeError,
                        ssl.SSLEOFError,
                        ssl.SSLZeroReturnError) as exc:
                    # Server (or a proxy) closed the keep-alive socket
                    # between calls — either at TCP level (RemoteDisconnected
                    # / ConnectionResetError / BrokenPipeError / BadStatusLine)
                    # or by killing TLS without a close_notify alert
                    # (SSLEOFError / SSLZeroReturnError). All of these are
                    # stale-socket signals; reconnect once and retry.
                    if attempt == 1:
                        logger.debug(
                            "YBA connection stale (%s); reconnecting", exc,
                        )
                        self._close_connection()
                        attempt_reconnect = True
                        continue
                    raise YBAClientError(
                        f"YBA {method} {path} connection error: {exc}"
                    ) from exc
                except (TimeoutError, OSError, ssl.SSLError) as exc:
                    # Force a reconnect on the next call — http.client's
                    # state machine may be wedged after this.
                    self._close_connection()
                    raise YBAClientError(
                        f"YBA {method} {path} timeout: {exc}"
                    ) from exc
                break

        if status >= 400:
            err_body = raw.decode("utf-8", "replace")[:200] if raw else ""
            raise YBAClientError(
                f"YBA {method} {path} → HTTP {status}: {err_body}"
            )

        if not raw:
            return None
        try:
            return json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise YBAClientError(
                f"YBA {method} {path} malformed response: {exc}"
            ) from exc

    # ------------------------------------------------------------- API surface

    def get_session_info(self) -> str:
        """Fetch the customer UUID from ``/session_info``. Cached after
        first successful call — subsequent calls return the cached value
        without a network round-trip."""
        if self._customer_uuid is not None:
            return self._customer_uuid
        info = self._request("GET", "/session_info")
        if not isinstance(info, dict):
            raise YBAClientError("/session_info returned non-JSON-object")
        cid = info.get("customerUUID")
        if not cid or not isinstance(cid, str):
            raise YBAClientError("/session_info missing customerUUID")
        self._customer_uuid = cid
        return cid

    def resolve_xcluster_config_uuid(self, replication_name: str) -> str:
        """Resolve a human-readable ``replicationName`` to its xCluster
        config UUID.

        YBA has no bulk list endpoint for xcluster_configs. But it does
        have a bulk ``/universes`` endpoint, and each universe carries
        ``xclusterInfo.sourceXClusterConfigs`` (config UUIDs where it's
        the source). We union those lists across all universes to get
        every xCluster config UUID in the customer's YBA, then probe
        each one's ``name``.

        Called once per FailoverGroup at bootstrap. The result is cached
        on the group; polls in Phase 2 never hit this path.

        Note: does NOT need the driver's YB-internal universe UUID —
        which lets the resolver work even on YBA deployments where the
        YBA-side universeUUID and the YB-side ``universe_uuid`` from
        ``yb_servers()`` differ (as we observed on portal.dev).

        Raises :class:`YBAClientError` when no config with a matching
        name is found, or when multiple configs share the same name.
        """
        cid = self.get_session_info()

        # 1. List all universes; collect every UUID mentioned in any
        #    universe's sourceXClusterConfigs list. One HTTP call.
        universes = self._request("GET", f"/customers/{cid}/universes")
        if not isinstance(universes, list):
            raise YBAClientError(
                "/universes returned non-list; cannot enumerate xCluster configs"
            )
        candidates: list[str] = []
        seen: set[str] = set()
        for u in universes:
            details = (u or {}).get("universeDetails") or {}
            xinfo = details.get("xclusterInfo") or {}
            for uuid in xinfo.get("sourceXClusterConfigs") or []:
                if uuid not in seen:
                    seen.add(uuid)
                    candidates.append(uuid)
        if not candidates:
            raise YBAClientError(
                "no xCluster configurations found under this customer; "
                f"cannot resolve replication_name {replication_name!r}"
            )

        # 2. Probe each config for its name.
        target = replication_name.strip()
        matches: list[str] = []
        for uuid in candidates:
            cfg = self._request(
                "GET", f"/customers/{cid}/xcluster_configs/{uuid}",
                query={"syncWithDB": "false"},
            )
            if not isinstance(cfg, dict):
                continue
            if (cfg.get("name") or "").strip() == target:
                matches.append(uuid)

        if not matches:
            raise YBAClientError(
                f"no xCluster configuration named {replication_name!r} "
                f"found; candidates checked: {candidates}"
            )
        if len(matches) > 1:
            raise YBAClientError(
                f"xCluster name {replication_name!r} is ambiguous; "
                f"matched configs {matches}"
            )
        return matches[0]

    def get_committed_lag_ms(
        self,
        xcluster_config_uuid: str,
        lookback_s: int = _MIN_METRICS_LOOKBACK_SEC,
    ) -> float | None:
        """Return the current max committed replication lag for the given
        xCluster config, in milliseconds. ``None`` means the metric
        returned no usable data (config not reporting, or all traces
        were the caught-up sentinel).

        Empirically confirmed against portal.dev (2026-08-20):
        ``xClusterConfigUuid`` alone is sufficient — YBA does not need
        ``nodePrefix`` when the config UUID is present. This lets us
        skip the two ``/universes/{uuid}`` bootstrap calls that were
        needed to resolve node prefixes.

        Raises :class:`YBAClientError` on transport / auth / parse failures.
        The failover poll loop translates that into "one unreachable poll"
        and moves on — three consecutive unreachables trigger the graceful
        degradation path.
        """
        cid = self.get_session_info()
        end_ts = int(time.time())
        start_ts = end_ts - max(int(lookback_s), _MIN_METRICS_LOOKBACK_SEC)
        body = {
            # YBA UI sends epoch seconds as strings — match that exactly.
            "start": str(start_ts),
            "end": str(end_ts),
            "metrics": ["tserver_async_replication_lag_micros"],
            "xClusterConfigUuid": xcluster_config_uuid,
        }
        resp = self._request("POST", f"/customers/{cid}/metrics", body=body)
        return _extract_committed_lag_ms(resp)


# ------------------------------------------------------------------- helpers

def _normalize_url(url: str) -> str:
    """Prepend ``https://`` if missing, strip trailing slash. Bare hostnames
    become ``https://<host>`` since YBA is almost always TLS."""
    url = url.strip()
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    return url.rstrip("/")


def _latest_y_value(trace: dict) -> float | None:
    """Return the last y-value in a Prometheus trace, or None if empty /
    malformed. Bill's script does the same — using last() gives the most
    recent datapoint in the metric window."""
    y_values = trace.get("y") or []
    if not y_values:
        return None
    try:
        return float(y_values[-1])
    except (TypeError, ValueError):
        return None


def _max_lag_from_traces(
    traces: list, name_hints: Iterable[str] = _COMMITTED_LAG_TRACE_NAMES,
) -> float | None:
    """Aggregate max committed lag across all traces whose name matches one
    of ``name_hints``. Returns:
      * ``None`` — no matching trace had a value (metric not being reported)
      * ``0.0`` — all matching traces were at or below the caught-up ceiling
      * positive float — max ms across matching traces
    """
    latest_values: list[float] = []
    for trace in traces:
        name = (trace.get("name") or trace.get("metricName") or "").strip()
        if name_hints and not any(hint in name for hint in name_hints):
            continue
        v = _latest_y_value(trace)
        if v is None:
            continue
        if v >= _NOT_REPORTED_THRESHOLD_MS or v < 0:
            # YBA "not reported" sentinel — skip this trace.
            continue
        latest_values.append(v)

    if not latest_values:
        return None

    meaningful = [v for v in latest_values if v > _CAUGHT_UP_LAG_MS]
    if meaningful:
        return max(meaningful)
    return 0.0


def _extract_committed_lag_ms(response: Any) -> float | None:
    """Parse the response from ``POST /customers/{cid}/metrics`` and pull
    out the max committed lag. YBA's response is shaped like:

        {"tserver_async_replication_lag_micros":
            {"data": [{"name": "Committed Lag", "y": ["8.4", ...]}, ...]}}
    """
    if not isinstance(response, dict):
        return None
    block = response.get("tserver_async_replication_lag_micros")
    if not isinstance(block, dict):
        return None
    if block.get("error"):
        return None
    data = block.get("data")
    if not isinstance(data, list):
        return None
    return _max_lag_from_traces(data)
