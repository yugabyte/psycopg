"""
Unit tests for ``psycopg.yb.yba_client.YBAClient``.

Runs a stdlib-only in-process HTTP server that mimics YBA's responses.
No external dependencies, no network access to real YBA. Each test
spins up a fresh server on an ephemeral port, points a client at it,
asserts on the observed behaviour.

Auto-tagged ``yb_unit`` — no database, no network beyond loopback.
"""

# Copyright (C) 2026 Yugabyte

from __future__ import annotations

import http.client
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

import pytest

from psycopg.yb.yba_client import (
    YBAClient,
    YBAClientError,
    _extract_committed_lag_ms,
    _max_lag_from_traces,
    _normalize_url,
)


pytestmark = pytest.mark.yb_unit


# ------------------------------------------------------------- mock server

class _MockYBAHandler(BaseHTTPRequestHandler):
    """Serves whatever the current test config on the server dictates.

    The server object owns ``._routes: dict[str, dict]`` — path → response
    spec. Each spec has ``status``, ``body`` (JSON-serialisable), and an
    optional ``expect_header`` for auth-header assertions. Requests are
    recorded on ``._captured: list[tuple[method, path, body_bytes]]``.
    """

    def _respond(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b""
        server = self.server
        server._captured.append((self.command, self.path, body))
        server._captured_headers.append(dict(self.headers))
        route = server._routes.get(self.path.split("?")[0])
        if route is None:
            self.send_response(404)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"error":"not found"}')
            return
        self.send_response(route.get("status", 200))
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        payload = route.get("body")
        if payload is not None:
            self.wfile.write(json.dumps(payload).encode())

    def do_GET(self) -> None:      # noqa: N802
        self._respond()

    def do_POST(self) -> None:     # noqa: N802
        self._respond()

    def log_message(self, *args: Any, **kw: Any) -> None:
        pass  # keep test output clean


class _MockYBAServer:
    """Context manager wrapper around an HTTPServer running on a background
    thread. ``routes`` is mutable — tests can rewrite it between calls."""

    def __init__(self) -> None:
        self.routes: dict[str, dict] = {}
        self.captured: list[tuple[str, str, bytes]] = []
        self.captured_headers: list[dict[str, str]] = []
        self._server: HTTPServer | None = None
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "_MockYBAServer":
        self._server = HTTPServer(("127.0.0.1", 0), _MockYBAHandler)
        self._server._routes = self.routes                          # type: ignore[attr-defined]
        self._server._captured = self.captured                      # type: ignore[attr-defined]
        self._server._captured_headers = self.captured_headers      # type: ignore[attr-defined]
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        assert self._server is not None
        self._server.shutdown()
        self._server.server_close()

    @property
    def endpoint(self) -> str:
        assert self._server is not None
        host, port = self._server.server_address
        return f"http://{host}:{port}"


# ------------------------------------------------------------- fixtures

@pytest.fixture
def yba_server() -> _MockYBAServer:
    """One mock server per test. Routes are empty by default; each test
    sets up whatever endpoints it needs."""
    with _MockYBAServer() as s:
        yield s


# ------------------------------------------------------------- URL normalisation

def test_normalize_url_adds_https_when_missing():
    assert _normalize_url("yba.example.com") == "https://yba.example.com"


def test_normalize_url_strips_trailing_slash():
    assert _normalize_url("https://yba.example.com/") == "https://yba.example.com"


def test_normalize_url_preserves_http_scheme():
    assert _normalize_url("http://yba.example.com") == "http://yba.example.com"


def test_normalize_url_strips_whitespace():
    assert _normalize_url("  https://yba.example.com  ") == "https://yba.example.com"


# ------------------------------------------------------------- session_info

def test_get_session_info_returns_customer_uuid(yba_server):
    yba_server.routes["/api/v1/session_info"] = {
        "body": {"customerUUID": "cust-1234"},
    }
    client = YBAClient(yba_server.endpoint, api_token="tok")
    assert client.get_session_info() == "cust-1234"


def test_get_session_info_caches_after_first_call(yba_server):
    yba_server.routes["/api/v1/session_info"] = {
        "body": {"customerUUID": "cust-5678"},
    }
    client = YBAClient(yba_server.endpoint, api_token="tok")
    client.get_session_info()
    client.get_session_info()
    client.get_session_info()
    # Only ONE GET should have been made — subsequent calls hit the cache.
    session_calls = [c for c in yba_server.captured if c[1] == "/api/v1/session_info"]
    assert len(session_calls) == 1


def test_get_session_info_sends_auth_header(yba_server):
    yba_server.routes["/api/v1/session_info"] = {
        "body": {"customerUUID": "c"},
    }
    client = YBAClient(yba_server.endpoint, api_token="secret-token-xyz")
    client.get_session_info()
    assert yba_server.captured_headers, "no request was captured"
    # urllib normalises header names to Title-Case-With-Dashes on the
    # wire. HTTP headers are case-insensitive per RFC 7230, so YBA will
    # accept whichever casing arrives — verify case-insensitively.
    headers_ci = {k.lower(): v for k, v in yba_server.captured_headers[0].items()}
    assert headers_ci.get("x-auth-yw-api-token") == "secret-token-xyz"


def test_get_session_info_missing_customer_uuid_raises(yba_server):
    yba_server.routes["/api/v1/session_info"] = {"body": {"otherField": "x"}}
    client = YBAClient(yba_server.endpoint, api_token="tok")
    with pytest.raises(YBAClientError, match="missing customerUUID"):
        client.get_session_info()


def test_http_error_wraps_as_yba_client_error(yba_server):
    yba_server.routes["/api/v1/session_info"] = {
        "status": 401,
        "body": {"error": "unauthorised"},
    }
    client = YBAClient(yba_server.endpoint, api_token="bad-tok")
    with pytest.raises(YBAClientError, match="HTTP 401"):
        client.get_session_info()


# ------------------------------------------------------------- committed_lag_ms

def test_get_committed_lag_ms_extracts_max(yba_server):
    yba_server.routes["/api/v1/session_info"] = {"body": {"customerUUID": "c1"}}
    yba_server.routes["/api/v1/customers/c1/metrics"] = {
        "body": {
            "tserver_async_replication_lag_micros": {
                "data": [
                    {"name": "Committed Lag (Milliseconds)", "y": ["3.2", "5.1", "8.4"]},
                    {"name": "Committed Lag (Milliseconds)", "y": ["1.1", "2.2", "3.5"]},
                    {"name": "Sent Lag (Milliseconds)", "y": ["12.0", "14.0"]},   # ignored
                ],
            },
        },
    }
    client = YBAClient(yba_server.endpoint, api_token="tok")
    lag = client.get_committed_lag_ms("xcc-1")
    # Max of last-y across Committed traces: max(8.4, 3.5) = 8.4
    assert lag == pytest.approx(8.4)


def test_get_committed_lag_ms_all_caught_up_returns_zero(yba_server):
    yba_server.routes["/api/v1/session_info"] = {"body": {"customerUUID": "c1"}}
    yba_server.routes["/api/v1/customers/c1/metrics"] = {
        "body": {
            "tserver_async_replication_lag_micros": {
                "data": [
                    {"name": "Committed Lag", "y": ["0.001"]},
                    {"name": "Committed Lag", "y": ["1.0"]},
                ],
            },
        },
    }
    client = YBAClient(yba_server.endpoint, api_token="tok")
    assert client.get_committed_lag_ms("xcc-1") == 0.0


def test_get_committed_lag_ms_no_data_returns_none(yba_server):
    yba_server.routes["/api/v1/session_info"] = {"body": {"customerUUID": "c1"}}
    yba_server.routes["/api/v1/customers/c1/metrics"] = {
        "body": {"tserver_async_replication_lag_micros": {"data": []}},
    }
    client = YBAClient(yba_server.endpoint, api_token="tok")
    assert client.get_committed_lag_ms("xcc-1") is None


def test_get_committed_lag_ms_error_field_returns_none(yba_server):
    yba_server.routes["/api/v1/session_info"] = {"body": {"customerUUID": "c1"}}
    yba_server.routes["/api/v1/customers/c1/metrics"] = {
        "body": {
            "tserver_async_replication_lag_micros": {
                "error": "no data",
                "data": [{"name": "Committed Lag", "y": ["100"]}],
            },
        },
    }
    client = YBAClient(yba_server.endpoint, api_token="tok")
    assert client.get_committed_lag_ms("xcc-1") is None


def test_get_committed_lag_ms_body_includes_expected_fields(yba_server):
    yba_server.routes["/api/v1/session_info"] = {"body": {"customerUUID": "c1"}}
    yba_server.routes["/api/v1/customers/c1/metrics"] = {
        "body": {"tserver_async_replication_lag_micros": {"data": []}},
    }
    client = YBAClient(yba_server.endpoint, api_token="tok")
    client.get_committed_lag_ms("xcc-abc-123")
    # Find the POST /metrics capture
    metrics_calls = [c for c in yba_server.captured if c[0] == "POST"]
    assert len(metrics_calls) == 1
    body = json.loads(metrics_calls[0][2])
    # Confirmed against live YBA (portal.dev 2026-08-20): nodePrefix is
    # NOT required when xClusterConfigUuid is present.
    assert "nodePrefix" not in body
    assert body["xClusterConfigUuid"] == "xcc-abc-123"
    assert body["metrics"] == ["tserver_async_replication_lag_micros"]
    assert "start" in body and "end" in body


def test_get_committed_lag_ms_ignores_not_reported_sentinel(yba_server):
    yba_server.routes["/api/v1/session_info"] = {"body": {"customerUUID": "c1"}}
    yba_server.routes["/api/v1/customers/c1/metrics"] = {
        "body": {
            "tserver_async_replication_lag_micros": {
                "data": [
                    # 1e12 sentinel — "not reported" in YBA's schema
                    {"name": "Committed Lag", "y": ["1000000000000"]},
                    {"name": "Committed Lag", "y": ["5.5"]},
                ],
            },
        },
    }
    client = YBAClient(yba_server.endpoint, api_token="tok")
    assert client.get_committed_lag_ms("xcc-1") == pytest.approx(5.5)


def test_get_committed_lag_ms_all_sentinel_is_no_data(yba_server):
    """Every trace carrying the ~1e12 not-reported sentinel is "no data"
    (None), not caught up (0.0). Pins the behaviour the module comment
    and docstrings describe (review #15)."""
    yba_server.routes["/api/v1/session_info"] = {"body": {"customerUUID": "c1"}}
    yba_server.routes["/api/v1/customers/c1/metrics"] = {
        "body": {
            "tserver_async_replication_lag_micros": {
                "data": [
                    {"name": "Committed Lag", "y": ["1000000000000"]},
                    {"name": "Committed Lag", "y": ["1000000000000"]},
                ],
            },
        },
    }
    client = YBAClient(yba_server.endpoint, api_token="tok")
    assert client.get_committed_lag_ms("xcc-1") is None
    assert client.get_committed_lag_sample("xcc-1") is None


# ------------------------------------------------------------- extractor unit tests

def test_extract_returns_none_for_non_dict():
    assert _extract_committed_lag_ms(None) is None
    assert _extract_committed_lag_ms([]) is None
    assert _extract_committed_lag_ms("not a dict") is None


def test_extract_returns_none_for_missing_metric_key():
    assert _extract_committed_lag_ms({"other_metric": {"data": []}}) is None


def test_max_lag_from_traces_filters_by_name():
    traces = [
        {"name": "Sent Lag", "y": ["100"]},          # ignored
        {"name": "Committed Lag", "y": ["10", "20"]},
        {"name": "Committed Lag", "y": ["30", "5"]},
    ]
    # Last y for each: 20, 5 → max = 20
    assert _max_lag_from_traces(traces) == pytest.approx(20.0)


def test_max_lag_from_traces_empty_returns_none():
    assert _max_lag_from_traces([]) is None
    assert _max_lag_from_traces([{"name": "Sent Lag", "y": ["1"]}]) is None


# ------------------------------------------------------------- resolve_xcluster_config_uuid

def _stub_universes_list(yba_server, universes: list):
    """Stub /session_info + /universes list.

    ``universes`` is a list of dicts with a ``sourceXClusterConfigs`` key
    each — mirrors the shape YBA returns.
    """
    yba_server.routes["/api/v1/session_info"] = {"body": {"customerUUID": "c1"}}
    body = []
    for i, ux in enumerate(universes):
        body.append({
            "universeUUID": f"univ-{i}",
            "universeDetails": {
                "nodePrefix": f"yb-15-u{i}",
                "xclusterInfo": {
                    "sourceXClusterConfigs":
                        ux.get("sourceXClusterConfigs", []),
                },
            },
        })
    yba_server.routes["/api/v1/customers/c1/universes"] = {"body": body}


def test_resolve_xcluster_config_uuid_single_match(yba_server):
    _stub_universes_list(yba_server, [
        {"sourceXClusterConfigs": ["xcc-a", "xcc-b"]},
    ])
    yba_server.routes["/api/v1/customers/c1/xcluster_configs/xcc-a"] = {
        "body": {"name": "east-to-west"},
    }
    yba_server.routes["/api/v1/customers/c1/xcluster_configs/xcc-b"] = {
        "body": {"name": "east-to-north"},
    }
    client = YBAClient(yba_server.endpoint, api_token="tok")
    assert client.resolve_xcluster_config_uuid("east-to-west") == "xcc-a"


def test_resolve_xcluster_config_uuid_finds_across_universes(yba_server):
    """Two universes each have their own configs — resolver should
    walk both and find a match."""
    _stub_universes_list(yba_server, [
        {"sourceXClusterConfigs": ["xcc-a"]},
        {"sourceXClusterConfigs": ["xcc-b"]},
    ])
    yba_server.routes["/api/v1/customers/c1/xcluster_configs/xcc-a"] = {
        "body": {"name": "east-to-west"},
    }
    yba_server.routes["/api/v1/customers/c1/xcluster_configs/xcc-b"] = {
        "body": {"name": "target-name"},
    }
    client = YBAClient(yba_server.endpoint, api_token="tok")
    assert client.resolve_xcluster_config_uuid("target-name") == "xcc-b"


def test_resolve_xcluster_config_uuid_no_configs_raises(yba_server):
    _stub_universes_list(yba_server, [
        {"sourceXClusterConfigs": []},
    ])
    client = YBAClient(yba_server.endpoint, api_token="tok")
    with pytest.raises(YBAClientError, match="no xCluster configurations found"):
        client.resolve_xcluster_config_uuid("east-to-west")


def test_resolve_xcluster_config_uuid_name_not_found_raises(yba_server):
    _stub_universes_list(yba_server, [
        {"sourceXClusterConfigs": ["xcc-a"]},
    ])
    yba_server.routes["/api/v1/customers/c1/xcluster_configs/xcc-a"] = {
        "body": {"name": "east-to-west"},
    }
    client = YBAClient(yba_server.endpoint, api_token="tok")
    with pytest.raises(YBAClientError, match="no xCluster configuration named"):
        client.resolve_xcluster_config_uuid("does-not-exist")


def test_resolve_xcluster_config_uuid_ambiguous_name_raises(yba_server):
    _stub_universes_list(yba_server, [
        {"sourceXClusterConfigs": ["xcc-a", "xcc-b"]},
    ])
    yba_server.routes["/api/v1/customers/c1/xcluster_configs/xcc-a"] = {
        "body": {"name": "dup-name"},
    }
    yba_server.routes["/api/v1/customers/c1/xcluster_configs/xcc-b"] = {
        "body": {"name": "dup-name"},
    }
    client = YBAClient(yba_server.endpoint, api_token="tok")
    with pytest.raises(YBAClientError, match="ambiguous"):
        client.resolve_xcluster_config_uuid("dup-name")


def test_resolve_xcluster_config_uuid_dedupes_across_universes(yba_server):
    """A single xcluster config UUID can appear in multiple universes'
    listings (source ↔ target); the resolver must dedupe to avoid
    a spurious ambiguity error."""
    _stub_universes_list(yba_server, [
        {"sourceXClusterConfigs": ["xcc-a"]},
        {"sourceXClusterConfigs": ["xcc-a"]},   # duplicate
    ])
    yba_server.routes["/api/v1/customers/c1/xcluster_configs/xcc-a"] = {
        "body": {"name": "single-config"},
    }
    client = YBAClient(yba_server.endpoint, api_token="tok")
    assert client.resolve_xcluster_config_uuid("single-config") == "xcc-a"


def test_resolve_xcluster_config_uuid_trims_whitespace(yba_server):
    _stub_universes_list(yba_server, [
        {"sourceXClusterConfigs": ["xcc-a"]},
    ])
    yba_server.routes["/api/v1/customers/c1/xcluster_configs/xcc-a"] = {
        "body": {"name": "  east-to-west  "},
    }
    client = YBAClient(yba_server.endpoint, api_token="tok")
    assert client.resolve_xcluster_config_uuid("east-to-west") == "xcc-a"


def test_close_is_idempotent(yba_server):
    """close() twice must not raise."""
    yba_server.routes["/api/v1/session_info"] = {"body": {"customerUUID": "c1"}}
    client = YBAClient(yba_server.endpoint, api_token="tok")
    client.get_session_info()   # opens the persistent connection
    client.close()
    client.close()   # idempotent — no-op the second time


# ------------------------------------------------------------- stale-socket reconnect

def test_ssl_eof_on_stale_socket_reconnects_and_retries(yba_server):
    """A ``ssl.SSLEOFError`` from the keep-alive socket must trigger one
    reconnect + retry, not a hard failure. This models what happens when
    YBA (or a proxy in front of it) silently kills an idle HTTPS
    connection: the next call sees TCP FIN mid-TLS, Python surfaces
    ``EOF occurred in violation of protocol``, and we must NOT count
    that as one of the poll loop's 3 unreachable strikes."""
    import ssl

    yba_server.routes["/api/v1/session_info"] = {"body": {"customerUUID": "c1"}}
    yba_server.routes["/api/v1/customers/c1/metrics"] = {
        "body": {
            "tserver_async_replication_lag_micros": {
                "data": [
                    {"name": "Committed Lag (Milliseconds)", "y": ["7.5"]},
                ],
            },
        },
    }
    client = YBAClient(yba_server.endpoint, api_token="tok")
    client.get_session_info()   # prime the persistent connection

    # Poison the current _conn so its next .request() raises SSLEOFError.
    # The driver should catch it, close/reconnect, and retry — the retry
    # runs against the real mock server and succeeds.
    def broken_request(*_a, **_kw):
        raise ssl.SSLEOFError("EOF occurred in violation of protocol")
    client._conn.request = broken_request                                # type: ignore[attr-defined]

    lag = client.get_committed_lag_ms("xcc-1")
    assert lag == pytest.approx(7.5)


def test_ssl_zero_return_on_stale_socket_reconnects_and_retries(yba_server):
    """``ssl.SSLZeroReturnError`` — peer sent close_notify then we
    tried to read — is the other half of the stale-keepalive case and
    must be treated the same as SSLEOFError."""
    import ssl

    yba_server.routes["/api/v1/session_info"] = {"body": {"customerUUID": "c1"}}
    yba_server.routes["/api/v1/customers/c1/metrics"] = {
        "body": {
            "tserver_async_replication_lag_micros": {
                "data": [
                    {"name": "Committed Lag (Milliseconds)", "y": ["3.0"]},
                ],
            },
        },
    }
    client = YBAClient(yba_server.endpoint, api_token="tok")
    client.get_session_info()

    def broken_request(*_a, **_kw):
        raise ssl.SSLZeroReturnError("TLS/SSL connection has been closed")
    client._conn.request = broken_request                                # type: ignore[attr-defined]

    lag = client.get_committed_lag_ms("xcc-1")
    assert lag == pytest.approx(3.0)


def test_persistent_ssl_eof_still_raises(yba_server):
    """If reconnect happens but the RETRY also raises SSLEOFError, the
    client must surface YBAClientError — the poll loop's strike counter
    should then advance normally."""
    import ssl

    yba_server.routes["/api/v1/session_info"] = {"body": {"customerUUID": "c1"}}
    client = YBAClient(yba_server.endpoint, api_token="tok")
    client.get_session_info()

    # Break every .request() this client will ever make, on every _conn
    # — including whatever object _open_connection creates for the retry.
    original_open = client._open_connection

    def open_and_poison() -> Any:
        conn = original_open()
        def broken_request(*_a, **_kw):
            raise ssl.SSLEOFError("still broken")
        conn.request = broken_request                                    # type: ignore[attr-defined]
        return conn

    client._open_connection = open_and_poison                            # type: ignore[assignment]
    # Also poison the currently-open connection so attempt #1 fails first.
    def broken_current(*_a, **_kw):
        raise ssl.SSLEOFError("still broken")
    client._conn.request = broken_current                                # type: ignore[attr-defined]

    with pytest.raises(YBAClientError, match="connection error"):
        client.get_committed_lag_ms("xcc-1")
    # The poisoned keep-alive conn stays open otherwise, and the
    # single-threaded mock server's shutdown() blocks on it.
    client.close()


# ------------------------------------------------------------- HTTPException wrapping

def _short_body_server():
    """Raw server that answers every request with a Content-Length
    larger than the body it sends, then closes: the client sees
    ``http.client.IncompleteRead``."""
    import socket

    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(4)

    def serve() -> None:
        while True:
            try:
                conn, _ = listener.accept()
            except OSError:
                return
            try:
                conn.recv(65536)
                conn.sendall(
                    b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
                    b"Content-Length: 100\r\n\r\n{\"cust"
                )
            finally:
                conn.close()

    threading.Thread(target=serve, daemon=True).start()
    host, port = listener.getsockname()
    return listener, f"http://{host}:{port}"


def test_incomplete_read_wraps_as_yba_client_error():
    """Review #8 (Amogh): a truncated body raises IncompleteRead, an
    HTTPException rather than an OSError. It must surface as
    YBAClientError (so the lag wait counts it as unreachable) and the
    connection must be reset."""
    listener, endpoint = _short_body_server()
    try:
        client = YBAClient(endpoint, api_token="tok")
        with pytest.raises(YBAClientError) as ei:
            client.get_session_info()
        assert isinstance(ei.value.__cause__, http.client.IncompleteRead)
        assert ei.value.transient is True
        assert client._conn is None
    finally:
        listener.close()


@pytest.mark.parametrize("exc_factory", [
    lambda: http.client.LineTooLong("header line"),
    lambda: http.client.ResponseNotReady("Idle"),
])
def test_http_exception_from_getresponse_wraps_and_resets(yba_server, exc_factory):
    yba_server.routes["/api/v1/session_info"] = {"body": {"customerUUID": "c1"}}
    client = YBAClient(yba_server.endpoint, api_token="tok")
    client.get_session_info()
    conn = client._conn

    def broken_getresponse(*_a, **_kw):
        raise exc_factory()
    conn.getresponse = broken_getresponse                               # type: ignore[attr-defined]

    with pytest.raises(YBAClientError, match="protocol error"):
        client.get_committed_lag_sample("xcc-1")
    assert client._conn is None


# ------------------------------------------------------------- deadline cut

def test_blocked_metrics_call_returns_at_deadline():
    """A YBA that accepts and never answers holds a call for the full
    socket timeout; under a globalTimeBudgetSecs deadline the call is cut
    when the deadline arrives (stress budget_split overshoot)."""
    import socket
    import time

    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    host, port = listener.getsockname()
    try:
        client = YBAClient(
            f"http://{host}:{port}", api_token="tok",
            connect_timeout_s=2.0, read_timeout_s=3.0,
        )
        client._customer_uuid = "cust-1"
        t0 = time.monotonic()
        with pytest.raises(YBAClientError):
            client.get_committed_lag_sample("xcc-1", deadline=t0 + 0.5)
        assert time.monotonic() - t0 < 1.2
        client.close()
    finally:
        listener.close()


def test_call_past_deadline_fails_without_network():
    client = YBAClient("http://127.0.0.1:9", api_token="tok")
    client._customer_uuid = "cust-1"
    with pytest.raises(YBAClientError):
        client.get_committed_lag_sample("xcc-1", deadline=0.0)


# ------------------------------------------------------------- per-node lag report

def _trace(node, ys, xs, name="Committed Lag"):
    return {"name": name, "instanceName": node, "y": ys, "x": xs}


def test_lag_report_healthy_nodes_caught_up():
    from psycopg.yb.yba_client import _lag_report_from_traces
    xs = ["1000000", "1010000"]
    traces = [
        _trace("n1", ["0.001", "0.001"], xs),
        _trace("n2", ["0", "0.001"], xs),
        _trace("n3", ["0.001", "0"], xs),
        _trace("n1", ["5", "5"], xs, name="Sent Lag"),
    ]
    r = _lag_report_from_traces(traces)
    assert r["lag_ms"] == 0.0 and r["sample_ts"] == 1010.0
    assert r["stale"] == [] and r["leader_reported"] is True
    assert r["values"] == {"n1": 0.001, "n2": 0.001, "n3": 0.0}


def test_lag_report_all_zero_means_no_leader():
    """ClearMetrics stores 0 on a node that leads no tablet; a caught-up
    leader stores 1 us. All zeros: the producer lost its leaders."""
    from psycopg.yb.yba_client import _lag_report_from_traces
    xs = ["1000000", "1010000"]
    r = _lag_report_from_traces([_trace("n1", ["0.001", "0"], xs)])
    assert r["lag_ms"] == 0.0
    assert r["leader_reported"] is False


def test_lag_report_flags_node_whose_series_ended():
    from psycopg.yb.yba_client import _lag_report_from_traces
    r = _lag_report_from_traces([
        _trace("n1", ["0.001", "0.001"], ["1000000", "1010000"]),
        _trace("n2", ["0.001"], ["1000000"]),
    ])
    assert r["stale"] == ["n2"]
    assert r["leader_reported"] is True


def test_get_committed_lag_report_end_to_end(yba_server):
    yba_server.routes["/api/v1/session_info"] = {"body": {"customerUUID": "c1"}}
    yba_server.routes["/api/v1/customers/c1/metrics"] = {"body": {
        "tserver_async_replication_lag_micros": {"data": [
            _trace("n1", ["12.5"], ["1700000000000"]),
            _trace("n2", ["0.001"], ["1700000000000"]),
        ]},
    }}
    client = YBAClient(yba_server.endpoint, api_token="tok")
    r = client.get_committed_lag_report("xcc-1")
    assert r["lag_ms"] == 12.5 and r["sample_ts"] == 1700000000.0
    assert r["leader_reported"] and r["stale"] == []


# ------------------------------------------------------------- non-finite values

def test_nan_latest_value_is_not_caught_up():
    """Review #8 (Amogh): NaN fails every comparison and used to land in
    the caught-up bucket: ["5000", "NaN"] returned (0.0, ts). The NaN is
    skipped and the last finite point (with its own x) is used."""
    from psycopg.yb.yba_client import _max_lag_sample_from_traces
    r = _max_lag_sample_from_traces([
        {"name": "Committed Lag", "y": ["5000", "NaN"], "x": ["1000000", "1010000"]},
    ])
    assert r == (5000.0, 1000.0)


@pytest.mark.parametrize("ys", [["NaN"], ["NaN", "nan"], ["inf"], ["-Infinity", "NaN"]])
def test_series_with_no_finite_value_is_no_data(ys):
    from psycopg.yb.yba_client import (
        _lag_report_from_traces, _max_lag_sample_from_traces,
    )
    traces = [{"name": "Committed Lag", "y": ys, "x": ["1000000"] * len(ys)}]
    assert _max_lag_sample_from_traces(traces) is None
    assert _max_lag_from_traces(traces) is None
    assert _lag_report_from_traces(traces) is None


def test_nan_node_does_not_hide_lag_in_report():
    """A node whose newest point is NaN keeps its last finite value and
    is flagged stale (its series ends early), so the lag wait does not
    count the sample."""
    from psycopg.yb.yba_client import _lag_report_from_traces
    xs = ["1000000", "1010000"]
    r = _lag_report_from_traces([
        _trace("n1", ["0.001", "0.001"], xs),
        _trace("n2", ["0.001", "NaN"], xs),
    ])
    assert r["values"] == {"n1": 0.001, "n2": 0.001}
    assert r["stale"] == ["n2"]


def test_extract_committed_lag_ms_nan_only_is_none():
    resp = {"tserver_async_replication_lag_micros": {"data": [
        {"name": "Committed Lag", "y": ["NaN"], "x": ["1000000"]},
    ]}}
    assert _extract_committed_lag_ms(resp) is None
