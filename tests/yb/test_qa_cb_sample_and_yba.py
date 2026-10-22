"""
Integration-level repro tests for PR-H3 — DB-24852 and DB-24483.

DB-24852 — HealthAndLagCircuitBreaker never fails over when primary dies
            with stale lag above threshold. Reproduces Shubham's cycle 5:
            replication paused, lag frozen at 109222.7 ms, every A
            process killed, health probe tripped at t+13 s but the CB
            returned HEALTHY on 55 consecutive checks for 271 s because
            the lag reading from YBA never moved (the primary is dead,
            nobody writes to it, YBA keeps replaying the last pre-death
            sample for its ~5-minute Prometheus window). Routing stayed
            on A for the full 245 s harness timeout. 4,108 connects
            failed with "no eligible YugabyteDB node".

DB-24483 — Driver trusted a 30-second-old YBA lag reading on a planned
            flip. 724 batches that A had confirmed were missing on B at
            switch time; the real tserver lag was 17 s. YBA's default
            step over the 300 s lookback window returned a coarse
            interpolation; the newest sample we saw was from before
            replication was paused. Fix: send step=10 so YBA returns a
            point per scrape interval, and surface the sample timestamp
            in the Phase 2 log so operators can see freshness.

These tests mirror Shubham's observables at integration scope — the
CB running on a stream of controlled responses from a fake YBA, and
the yba_client's request body + response parsing exercised against a
realistic Prometheus-shaped payload.
"""

# Copyright (C) 2026 Yugabyte

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from psycopg.yb.health import HealthResult
from psycopg.yb.yba_client import (
    YBAClientError,
    _extract_committed_lag_ms,
    _extract_committed_lag_sample,
    _latest_xy,
    _max_lag_sample_from_traces,
)
from demo.samples.health_and_lag_cb import HealthAndLagCircuitBreaker


pytestmark = pytest.mark.yb_unit


# ===================================================================
# Shared test doubles
# ===================================================================


class _StubHealthProbe:
    def __init__(self, verdict: HealthResult) -> None:
        self.verdict = verdict
        self.calls = 0

    def check(self, group) -> HealthResult:
        self.calls += 1
        return self.verdict


class _StuckLagYBA:
    """YBA stand-in whose lag reading never moves — exactly the
    scenario when the primary is dead and no one is left to write to
    it. Shubham observed this with lag frozen at 109222.7 ms for 55
    consecutive checks."""

    def __init__(self, frozen_lag_ms: float) -> None:
        self.frozen_lag_ms = frozen_lag_ms
        self.calls = 0

    def get_committed_lag_ms(self, xcluster_uuid):
        self.calls += 1
        return self.frozen_lag_ms


def _group_stub(**attrs):
    base = dict(
        yba_client=None,
        xcluster_config_uuid="",
        failback_xcluster_config_uuid="",
    )
    base.update(attrs)
    return SimpleNamespace(**base)


# ===================================================================
# DB-24852 — bounded-hold backstop on sustained UNHEALTHY + stale lag
# ===================================================================


def test_db24852_held_forever_without_fix_proof_of_regression():
    """Sanity-check that the OLD behaviour (max_hold_ticks=0) still
    produces the regression — HEALTHY forever on sustained UNHEALTHY
    + stale lag. If this ever fails it means someone changed the
    sentinel semantics; the DB-24852 fix would then need re-review.
    """
    yba = _StuckLagYBA(frozen_lag_ms=109222.7)
    cb = HealthAndLagCircuitBreaker(
        which_cluster="primary",
        threshold_lag_ms=0,
        health_probe=_StubHealthProbe(HealthResult.UNHEALTHY),
        yba_client=yba,
        xcluster_config_uuid="xcc-1",
        max_hold_ticks=0,   # DISABLED — pre-fix behaviour
    )
    group = _group_stub()

    # 100 consecutive ticks over 500 s equivalent; without the backstop
    # the CB should return HEALTHY every time.
    for i in range(100):
        verdict = cb.check(group)
        assert verdict is HealthResult.HEALTHY, (
            f"max_hold_ticks=0 should preserve the hold-forever "
            f"behaviour; tick {i} returned {verdict}"
        )
    assert yba.calls == 100


def test_db24852_backstop_trips_after_max_hold_ticks():
    """Default behaviour: CB holds HEALTHY for `max_hold_ticks` - 1
    ticks, then on the N-th tick trips to UNHEALTHY, allowing the
    driver to fail over away from a dead primary."""
    yba = _StuckLagYBA(frozen_lag_ms=109222.7)
    cb = HealthAndLagCircuitBreaker(
        which_cluster="primary",
        threshold_lag_ms=0,
        health_probe=_StubHealthProbe(HealthResult.UNHEALTHY),
        yba_client=yba,
        xcluster_config_uuid="xcc-1",
        max_hold_ticks=5,   # trip on the 5th tick
    )
    group = _group_stub()

    verdicts = [cb.check(group) for _ in range(10)]

    # First 4 ticks held HEALTHY, 5th tripped to UNHEALTHY.
    assert verdicts[0:4] == [HealthResult.HEALTHY] * 4, (
        f"first 4 ticks should hold HEALTHY, got {verdicts[0:4]}"
    )
    assert verdicts[4] is HealthResult.UNHEALTHY, (
        f"tick 5 should trip (DB-24852 backstop), got {verdicts[4]}"
    )
    # Subsequent ticks continue to report UNHEALTHY (the trip is
    # recorded via _record_unhealthy).
    for i, v in enumerate(verdicts[5:], start=5):
        assert v is HealthResult.UNHEALTHY, (
            f"tick {i} after backstop should stay UNHEALTHY, got {v}"
        )


def test_db24852_hold_counter_resets_on_health_recovery():
    """If the health probe transitions back to HEALTHY mid-hold, the
    counter must reset so a later re-hold starts fresh."""
    probe = _StubHealthProbe(HealthResult.UNHEALTHY)
    yba = _StuckLagYBA(frozen_lag_ms=5000)
    cb = HealthAndLagCircuitBreaker(
        which_cluster="primary",
        threshold_lag_ms=0,
        health_probe=probe,
        yba_client=yba,
        xcluster_config_uuid="xcc-1",
        max_hold_ticks=5,
    )
    group = _group_stub()

    # 3 ticks of hold.
    for _ in range(3):
        assert cb.check(group) is HealthResult.HEALTHY
    assert cb._hold_ticks == 3

    # Health recovers for one tick.
    probe.verdict = HealthResult.HEALTHY
    assert cb.check(group) is HealthResult.HEALTHY
    assert cb._hold_ticks == 0, "hold counter should reset on recovery"

    # Then goes bad again — hold counter starts over.
    probe.verdict = HealthResult.UNHEALTHY
    for i in range(4):
        v = cb.check(group)
        assert v is HealthResult.HEALTHY, (
            f"tick {i} after recovery should hold HEALTHY, got {v}"
        )
    # 5th post-recovery tick trips.
    assert cb.check(group) is HealthResult.UNHEALTHY


def test_db24852_hold_counter_resets_on_lag_converge():
    """If lag drops below threshold on a tick (the normal "trip
    because standby caught up" path), the counter resets. A later
    re-hold starts fresh."""
    probe = _StubHealthProbe(HealthResult.UNHEALTHY)
    # Lag starts stuck, then drops to 0.
    yba = SimpleNamespace(calls=0)
    responses = iter([5000, 5000, 5000, 0, 5000, 5000, 5000, 5000, 0])

    def get_lag(uuid):
        yba.calls += 1
        return next(responses)
    yba.get_committed_lag_ms = get_lag

    cb = HealthAndLagCircuitBreaker(
        which_cluster="primary",
        threshold_lag_ms=0,
        health_probe=probe,
        yba_client=yba,
        xcluster_config_uuid="xcc-1",
        max_hold_ticks=5,
    )
    group = _group_stub()

    # Ticks 1-3: hold. Tick 4: lag drops to 0, CB trips to UNHEALTHY.
    assert cb.check(group) is HealthResult.HEALTHY   # hold 1
    assert cb.check(group) is HealthResult.HEALTHY   # hold 2
    assert cb.check(group) is HealthResult.HEALTHY   # hold 3
    assert cb.check(group) is HealthResult.UNHEALTHY  # lag=0 → trip
    assert cb._hold_ticks == 0


def test_db24852_backstop_configurable_per_cb():
    """Each CB instance has its own counter and threshold —
    primary and secondary CBs don't share state."""
    probe_p = _StubHealthProbe(HealthResult.UNHEALTHY)
    probe_s = _StubHealthProbe(HealthResult.UNHEALTHY)
    yba = _StuckLagYBA(frozen_lag_ms=5000)
    cb_p = HealthAndLagCircuitBreaker(
        which_cluster="primary", threshold_lag_ms=0,
        health_probe=probe_p, yba_client=yba,
        xcluster_config_uuid="xcc-1", max_hold_ticks=3,
    )
    cb_s = HealthAndLagCircuitBreaker(
        which_cluster="secondary", threshold_lag_ms=0,
        health_probe=probe_s, yba_client=yba,
        xcluster_config_uuid="xcc-1",
        max_hold_ticks=10,
    )
    group = _group_stub(
        xcluster_config_uuid="xcc-1",
        failback_xcluster_config_uuid="xcc-1",
    )

    # Tick 3 trips the primary (max=3), but secondary (max=10) still holds.
    for _ in range(3):
        cb_p.check(group)
        cb_s.check(group)
    assert cb_p._last_reported is HealthResult.UNHEALTHY
    assert cb_s._last_reported is HealthResult.HEALTHY


def test_db24852_init_validates_max_hold_ticks():
    with pytest.raises(ValueError, match="max_hold_ticks must be >= 0"):
        HealthAndLagCircuitBreaker(
            which_cluster="primary", max_hold_ticks=-1,
        )


# ===================================================================
# DB-24483 — YBA step=10 + sample timestamp
# ===================================================================


def test_db24483_step_10_is_sent_in_metrics_request_body():
    """Fix: driver must send step=10 so YBA's /metrics returns a
    point per scrape interval instead of a coarse interpolation."""
    captured = {}

    class _FakeClient:
        def get_session_info(self):
            return "cid-1"

        def _request(self, method, path, body=None):
            captured["method"] = method
            captured["path"] = path
            captured["body"] = body
            return {
                "tserver_async_replication_lag_micros": {
                    "data": [{"name": "Committed Lag", "y": [], "x": []}]
                }
            }

    # We're testing get_committed_lag_sample's request shape, so bind
    # the real method to our fake client via explicit method binding.
    from psycopg.yb.yba_client import YBAClient
    fake = _FakeClient()
    fake.get_committed_lag_sample = YBAClient.get_committed_lag_sample.__get__(fake)

    fake.get_committed_lag_sample("xcc-1")

    assert captured["method"] == "POST"
    assert captured["path"] == "/customers/cid-1/metrics"
    body = captured["body"]
    assert body["xClusterConfigUuid"] == "xcc-1"
    assert body["metrics"] == ["tserver_async_replication_lag_micros"]
    # DB-24483: step must be 10 (any value < default-coarse works, we
    # pin at 10 to match the scrape interval).
    assert body["step"] == "10", (
        f"DB-24483 fix: step=10 must appear in the metrics POST body, "
        f"got step={body.get('step')!r}"
    )


def test_db24483_sample_timestamp_returned_and_propagated():
    """Fix: _latest_xy returns the sample timestamp alongside the lag
    value so Phase 2 can log staleness. Verify the parse pipeline
    propagates x[-1] through _max_lag_sample_from_traces and
    _extract_committed_lag_sample."""
    # Prometheus-shaped response: timestamps in ms, values as strings.
    response = {
        "tserver_async_replication_lag_micros": {
            "data": [{
                "name": "Committed Lag",
                "x": ["1700000000000", "1700000010000", "1700000020000"],
                "y": ["50.0", "100.0", "150.0"],
            }],
        }
    }
    result = _extract_committed_lag_sample(response)
    assert result is not None
    lag_ms, sample_ts = result
    assert lag_ms == 150.0
    # Sample timestamp returned in SECONDS (converted from YBA's ms).
    assert sample_ts == 1700000020.0, (
        f"sample timestamp should be epoch seconds, got {sample_ts}"
    )


def test_db24483_sample_timestamp_none_when_x_axis_missing():
    """YBA could in principle return y without x (defensive).
    _latest_xy must handle this — y stays, x is None."""
    y_only_trace = {"y": ["42.0"]}
    y, x = _latest_xy(y_only_trace)
    assert y == 42.0
    assert x is None


def test_db24483_backward_compat_wrapper_still_returns_float():
    """`get_committed_lag_ms` was the pre-fix API; several callers
    still exist. Verify the backward-compat wrapper still returns a
    plain float (not a tuple) so no external caller breaks."""
    response = {
        "tserver_async_replication_lag_micros": {
            "data": [{"name": "Committed Lag", "x": ["1700000000000"], "y": ["42.0"]}]
        }
    }
    assert _extract_committed_lag_ms(response) == 42.0


def test_db24483_max_over_newest_x_when_all_caught_up():
    """When all traces report lag=0 (all caught up), the sample_ts
    returned is the newest observed x — gives operators 'when did we
    last observe caught-up?' instead of a random trace's timestamp."""
    traces = [
        {"name": "Committed Lag", "x": ["1700000000000"], "y": ["0.0"]},
        {"name": "Committed Lag", "x": ["1700000050000"], "y": ["0.0"]},
        {"name": "Committed Lag", "x": ["1700000020000"], "y": ["0.0"]},
    ]
    result = _max_lag_sample_from_traces(traces)
    assert result is not None
    lag_ms, sample_ts = result
    assert lag_ms == 0.0
    assert sample_ts == 1700000050.0   # newest x


def test_db24483_stale_sample_detectable_from_timestamp():
    """End-to-end: given a response with a sample from 30 s ago, the
    returned sample_ts lets a Phase 2 caller compute the age by
    comparing against time.time(). Mirrors the staleness check the
    Phase 2 log now surfaces."""
    import time as _time
    now = _time.time()
    stale_ts_ms = int((now - 30.0) * 1000)
    response = {
        "tserver_async_replication_lag_micros": {
            "data": [{
                "name": "Committed Lag",
                "x": [str(stale_ts_ms)],
                "y": ["0.0"],
            }]
        }
    }
    result = _extract_committed_lag_sample(response)
    assert result is not None
    lag_ms, sample_ts = result
    assert lag_ms == 0.0
    age_s = now - sample_ts
    assert 28.0 < age_s < 32.0, (
        f"sample should be detectable as ~30 s stale, got age={age_s:.1f}s"
    )
