"""
A transient YBA failure at bootstrap must not turn Phase 2 off for good.

r12 218917 (startup_a_down, 4 of 4 cycles): A was down when the app
started, YBA timed out answering the xCluster config lookup
("GET /customers/.../xcluster_configs/... timeout: The read operation
timed out") and the bootstrap probe disabled the lag phase for the life
of the group. After A healed, the failback ran without a lag wait and
the stress suite failed driver_path ("Phase 2 enabled in the DSN but the
driver never started a lag wait").

Fix: ``YBAClientError.transient`` marks timeouts, connection errors and
5xx. A transient bootstrap failure keeps the params on the group, and the
drain retries the probe before its next lag wait, inside the lag-wait
timeout. Auth errors and unknown replication names stay permanent.
"""

# Copyright (C) 2026 Yugabyte

from __future__ import annotations

import threading
import time
import types

import pytest

from psycopg.pq import TransactionStatus
from psycopg.yb import drain as drain_mod
from psycopg.yb import yba_client as yba_mod
from psycopg.yb.drain import _retry_pending_yba, trigger_drain
from psycopg.yb.health import HealthResult
from psycopg.yb.registry import ClusterRegistry, FailoverGroup
from psycopg.yb.yba_client import YBAClient, YBAClientError


pytestmark = pytest.mark.yb_unit


# ------------------------------------------------------- error classification

class _FakeHTTP:
    def __init__(self, exc=None, status=200, body=b"{}"):
        self.exc, self.status, self.body = exc, status, body
        self.timeout = None
        self.sock = None

    def request(self, *a, **kw):
        if self.exc is not None:
            raise self.exc

    def getresponse(self):
        return types.SimpleNamespace(read=lambda: self.body, status=self.status)

    def close(self):
        pass


def _client_with(http) -> YBAClient:
    c = YBAClient("http://yba.invalid", "token")
    c._conn = http
    return c


@pytest.mark.parametrize("http, transient", [
    (_FakeHTTP(exc=TimeoutError("The read operation timed out")), True),
    (_FakeHTTP(status=503, body=b"unavailable"), True),
    (_FakeHTTP(status=500, body=b"Too many attempts"), True),
    (_FakeHTTP(status=401, body=b"bad token"), False),
    (_FakeHTTP(status=404, body=b"not found"), False),
])
def test_yba_error_transient_classification(http, transient):
    with pytest.raises(YBAClientError) as ei:
        _client_with(http)._request("GET", "/session_info")
    assert ei.value.transient is transient


def test_unknown_replication_name_is_permanent():
    c = YBAClient("http://yba.invalid", "token")
    c._customer_uuid = "cid"
    c._request = lambda *a, **kw: []   # no universes → no candidates
    with pytest.raises(YBAClientError) as ei:
        c.resolve_xcluster_config_uuid("missing")
    assert ei.value.transient is False


# ------------------------------------------------------- registry bootstrap + retry

class _ScriptedClient:
    """Stands in for ``YBAClient``: ``outcomes`` is shared across
    instances; each probe pops the next one (an exception or None)."""

    outcomes: list = []
    probes = 0

    def __init__(self, **kw):
        self.kw = kw

    def get_session_info(self, deadline=None):
        type(self).probes += 1
        outcome = type(self).outcomes.pop(0) if type(self).outcomes else None
        if outcome is not None:
            raise outcome
        return "cid-1"

    def resolve_xcluster_config_uuid(self, name, deadline=None):
        return {"a_to_b": "xcc-ab", "b_to_a": "xcc-ba"}[name]


def _params():
    return types.SimpleNamespace(
        lag_wait_enabled=True,
        yba_endpoint="http://yba.invalid",
        yba_api_token="token",
        yba_connect_timeout_s=2.0,
        yba_read_timeout_s=3.0,
        yba_verify_ssl=False,
        yba_ca_file="",
        replication_name="a_to_b",
        failback_replication_name="b_to_a",
        threshold_replication_lag_ms=0,
        lag_wait_timeout_s=30,
        lag_freshness_s=0,
        lag_confirmation_samples=1,
        failback_threshold_replication_lag_ms=0,
        failback_lag_wait_timeout_s=30,
        failback_lag_freshness_s=0,
        failback_lag_confirmation_samples=1,
        global_time_budget_secs=None,
        failback_global_time_budget_secs=None,
    )


def _group(fake_state, **kw) -> FailoverGroup:
    p = fake_state(("p1", "aws", "us-west", "us-west-1a", "primary"), uuid="P")
    s = fake_state(("s1", "aws", "us-east", "us-east-1a", "primary"), uuid="S")
    return FailoverGroup(
        primary=p, secondary=s, lock=threading.Lock(),
        primary_status=HealthResult.HEALTHY,
        secondary_status=HealthResult.HEALTHY,
        cooldown_s=0, **kw,
    )


@pytest.fixture
def scripted(monkeypatch):
    _ScriptedClient.outcomes = []
    _ScriptedClient.probes = 0
    monkeypatch.setattr(yba_mod, "YBAClient", _ScriptedClient)
    return _ScriptedClient


def test_transient_bootstrap_failure_keeps_params(fake_state, scripted):
    scripted.outcomes = [YBAClientError("timeout", transient=True)]
    group = _group(fake_state)
    params = _params()

    ClusterRegistry.instance()._wire_yba_lag_phase(group, params)

    assert group.yba_client is None
    assert group.yba_pending_params is params


@pytest.mark.parametrize("verify, cafile, expected_cafile", [
    (False, "", None),
    (True, "", None),
    (True, "/etc/yba/ca.pem", "/etc/yba/ca.pem"),
])
def test_wire_passes_tls_settings_to_client(
    fake_state, scripted, verify, cafile, expected_cafile,
):
    """Review #8: ybaVerifySsl / ybaCaFile reach the YBAClient."""
    group = _group(fake_state)
    params = _params()
    params.yba_verify_ssl = verify
    params.yba_ca_file = cafile

    ClusterRegistry.instance()._wire_yba_lag_phase(group, params)

    assert group.yba_client is not None
    assert group.yba_client.kw["verify_ssl"] is verify
    assert group.yba_client.kw["cafile"] == expected_cafile


def test_permanent_bootstrap_failure_disables_for_good(fake_state, scripted):
    scripted.outcomes = [YBAClientError("HTTP 401", transient=False)]
    group = _group(fake_state)

    ClusterRegistry.instance()._wire_yba_lag_phase(group, _params())

    assert group.yba_client is None
    assert group.yba_pending_params is None


def test_retry_enables_phase2_once_yba_answers(fake_state, scripted):
    scripted.outcomes = [YBAClientError("timeout", transient=True), None]
    group = _group(fake_state)
    reg = ClusterRegistry.instance()
    reg._wire_yba_lag_phase(group, _params())

    reg.retry_yba_lag_phase(group, deadline=time.monotonic() + 5)

    assert group.yba_client is not None
    assert group.yba_pending_params is None
    assert group.xcluster_config_uuid == "xcc-ab"
    assert group.failback_xcluster_config_uuid == "xcc-ba"


def test_retry_with_permanent_failure_stops_retrying(fake_state, scripted):
    scripted.outcomes = [
        YBAClientError("timeout", transient=True),
        YBAClientError("HTTP 403", transient=False),
    ]
    group = _group(fake_state)
    reg = ClusterRegistry.instance()
    reg._wire_yba_lag_phase(group, _params())

    reg.retry_yba_lag_phase(group, deadline=time.monotonic() + 5)

    assert group.yba_client is None
    assert group.yba_pending_params is None


# ------------------------------------------------------- drain integration

def test_retry_pending_shares_the_lag_wait_timeout(fake_state, scripted):
    scripted.outcomes = [YBAClientError("timeout", transient=True), None]
    group = _group(fake_state, lag_wait_timeout_s=7)
    ClusterRegistry.instance()._wire_yba_lag_phase(group, _params())
    group.lag_wait_timeout_s = 7

    before = time.monotonic()
    deadline = _retry_pending_yba(group, is_failback=False, budget_deadline=None)

    assert group.yba_client is not None
    assert deadline is not None and before + 7 - 0.5 <= deadline <= time.monotonic() + 7


def test_retry_pending_is_a_no_op_without_pending_params(fake_state, scripted):
    group = _group(fake_state)
    assert _retry_pending_yba(group, is_failback=True, budget_deadline=123.0) == 123.0
    assert scripted.probes == 0


def test_failback_after_transient_bootstrap_runs_the_lag_wait(fake_state, scripted, monkeypatch):
    """startup_a_down end to end at drain level: the bootstrap probe timed
    out, A healed, the failback drain retries the probe and runs Phase 2."""
    scripted.outcomes = [YBAClientError("timeout", transient=True), None]
    group = _group(fake_state)
    group.primary_status = HealthResult.UNHEALTHY
    ClusterRegistry.instance()._wire_yba_lag_phase(group, _params())
    waits = []
    monkeypatch.setattr(
        drain_mod, "_wait_for_replication_lag",
        lambda g, which, is_failback=False, budget_deadline=None, outgoing=None:
            waits.append((is_failback, budget_deadline)),
    )

    trigger_drain(group, "primary", HealthResult.HEALTHY, drain_timeout_s=0)

    assert waits and waits[0][0] is True, "failback ran without a lag wait"
    assert waits[0][1] is not None
    assert group.primary_status == HealthResult.HEALTHY


# ------------------------------------------------------- certificate errors

def _cert_error():
    return YBAClientError(
        "YBA GET /session_info TLS certificate verification failed: self-signed",
        cert_error=True,
    )


def test_cert_error_logs_at_error_and_marks_failback_refused(
    fake_state, scripted, caplog,
):
    """Review #8 follow-up: a certificate error turned the lag phase off
    with only a WARNING, and failbacks then skipped the lag check even
    under onPhase2Timeout=fail."""
    scripted.outcomes = [_cert_error()]
    group = _group(fake_state, failback_on_phase2_timeout="fail")

    with caplog.at_level("WARNING", logger="psycopg.yb.registry"):
        ClusterRegistry.instance()._wire_yba_lag_phase(group, _params())

    assert group.yba_client is None
    assert "certificate" in group.yba_cert_error
    errors = [r for r in caplog.records if r.levelname == "ERROR"]
    assert len(errors) == 1 and "failbacks are refused" in errors[0].getMessage()


@pytest.mark.parametrize("policy, failback_name", [
    ("proceed", "b_to_a"),
    ("fail", ""),
])
def test_cert_error_leaves_failback_alone_when_nothing_to_check(
    fake_state, scripted, policy, failback_name,
):
    scripted.outcomes = [_cert_error()]
    group = _group(fake_state, failback_on_phase2_timeout=policy)
    params = _params()
    params.failback_replication_name = failback_name

    ClusterRegistry.instance()._wire_yba_lag_phase(group, params)

    assert group.yba_cert_error == ""


def test_cert_error_on_retry_marks_failback_refused(fake_state, scripted):
    scripted.outcomes = [YBAClientError("timeout", transient=True), _cert_error()]
    group = _group(fake_state, failback_on_phase2_timeout="fail")
    reg = ClusterRegistry.instance()
    reg._wire_yba_lag_phase(group, _params())

    reg.retry_yba_lag_phase(group, deadline=time.monotonic() + 5)

    assert group.yba_pending_params is None
    assert "certificate" in group.yba_cert_error


def _cert_failed_group(fake_state):
    group = _group(fake_state, failback_on_phase2_timeout="fail")
    group.primary_status = HealthResult.UNHEALTHY
    ClusterRegistry.instance()._wire_yba_lag_phase(group, _params())
    return group


class _Conn:
    closed = False

    def close(self):
        self.closed = True


def test_failback_refused_after_cert_error(fake_state, scripted, caplog):
    """Review #26: refused before Phase 1, so the serving secondary is
    neither paused nor drained."""
    scripted.outcomes = [_cert_error(), _cert_error()]   # bootstrap, re-probe
    group = _cert_failed_group(fake_state)
    conn = _Conn()   # keep a ref: tracked_conns is weak
    group.secondary.tracked_conns.add(conn)

    with caplog.at_level("INFO", logger="psycopg.yb.drain"):
        trigger_drain(group, "primary", HealthResult.HEALTHY, drain_timeout_s=0)

    assert group.primary_status == HealthResult.UNHEALTHY
    assert group.last_phase2_timeout_ts > 0
    assert group.primary_last_transition_time > 0, "cooldown not re-armed"
    assert conn.closed is False
    assert group.last_failover_start_ts == 0.0
    assert not any("drain-wait start" in r.getMessage() for r in caplog.records)
    assert any("refused before the drain" in r.getMessage() for r in caplog.records)
    assert scripted.probes == 2


def test_fixed_cert_lets_the_failback_run_its_lag_wait(fake_state, scripted, monkeypatch):
    scripted.outcomes = [_cert_error(), None]   # bootstrap fails, re-probe works
    group = _cert_failed_group(fake_state)
    waits = []
    monkeypatch.setattr(
        drain_mod, "_wait_for_replication_lag",
        lambda g, which, is_failback=False, budget_deadline=None, outgoing=None:
            waits.append(is_failback),
    )

    trigger_drain(group, "primary", HealthResult.HEALTHY, drain_timeout_s=0)

    assert group.yba_cert_error == ""
    assert group.yba_client is not None
    assert waits == [True]
    assert group.primary_status == HealthResult.HEALTHY


def test_transient_reprobe_keeps_refusing(fake_state, scripted):
    scripted.outcomes = [_cert_error(), YBAClientError("timeout", transient=True)]
    group = _cert_failed_group(fake_state)

    trigger_drain(group, "primary", HealthResult.HEALTHY, drain_timeout_s=0)

    assert group.primary_status == HealthResult.UNHEALTHY
    assert group.yba_cert_error
    assert group.yba_cert_params is not None


def test_other_permanent_reprobe_error_disables_the_lag_phase(fake_state, scripted):
    """Certificate fixed but the token is now rejected: same as that
    error at bootstrap, the lag phase is off and the failback goes ahead."""
    scripted.outcomes = [_cert_error(), YBAClientError("HTTP 401", transient=False)]
    group = _cert_failed_group(fake_state)

    trigger_drain(group, "primary", HealthResult.HEALTHY, drain_timeout_s=0)

    assert group.yba_cert_error == ""
    assert group.yba_cert_params is None
    assert group.primary_status == HealthResult.HEALTHY


def test_proceed_policy_never_reprobes(fake_state, scripted):
    scripted.outcomes = [_cert_error()]
    group = _group(fake_state, failback_on_phase2_timeout="proceed")
    group.primary_status = HealthResult.UNHEALTHY
    ClusterRegistry.instance()._wire_yba_lag_phase(group, _params())

    trigger_drain(group, "primary", HealthResult.HEALTHY, drain_timeout_s=0)

    assert scripted.probes == 1
    assert group.primary_status == HealthResult.HEALTHY


def test_failover_still_flips_after_cert_error(fake_state, scripted):
    scripted.outcomes = [_cert_error()]
    group = _group(fake_state)
    ClusterRegistry.instance()._wire_yba_lag_phase(group, _params())

    trigger_drain(group, "primary", HealthResult.UNHEALTHY, drain_timeout_s=0)

    assert group.primary_status == HealthResult.UNHEALTHY
    assert group.last_phase2_timeout_ts == 0.0


def test_failback_out_of_no_viable_proceeds_after_cert_error(fake_state, scripted):
    """With the secondary down there is no healthy side to stay on, so the
    failback goes ahead, as it does on a lag-wait timeout."""
    scripted.outcomes = [_cert_error()]
    group = _group(fake_state)
    group.primary_status = HealthResult.UNHEALTHY
    group.secondary_status = HealthResult.UNHEALTHY
    group.active_before_no_viable = "secondary"
    ClusterRegistry.instance()._wire_yba_lag_phase(group, _params())

    trigger_drain(group, "primary", HealthResult.HEALTHY, drain_timeout_s=0)

    assert group.primary_status == HealthResult.HEALTHY
