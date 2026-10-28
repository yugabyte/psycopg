"""
DB-24483 — Phase 2 freshness + liveness gate.

Covers the full driver-only fix laid out in the design doc:

  * Freshness gate rejects any YBA sample whose ``sample_ts`` is older
    than ``barrier_ts + lag_freshness_s`` (default 25s). A sample that
    predates the barrier cannot describe post-barrier replication.
  * Liveness gate requires ``lag_confirmation_samples`` (default 2)
    distinct post-barrier samples at ``lag ≤ threshold`` before the
    drain flips routing. One pass is not enough: YBA replays its last
    sample for ~5min after the primary dies.
  * Opt-outs: ``lag_freshness_s=0`` disables the freshness gate;
    ``lag_confirmation_samples=1`` disables the liveness gate.
  * Bootstrap floor: when both gates are enabled,
    ``lag_wait_timeout_s`` must fit ``freshness + N × scrape_hint``;
    otherwise bootstrap raises ``ValueError``.

Tests parametrise the fake YBA client so each tick can return a
(lag_ms, sample_ts) pair the suite controls, which lets the gate's
clock-math be verified without wall-clock delays.
"""

# Copyright (C) 2026 Yugabyte

from __future__ import annotations

import threading
import time
from typing import Any

import pytest

from psycopg.pq import TransactionStatus
from psycopg.yb.drain import trigger_drain
from psycopg.yb.health import HealthResult
from psycopg.yb.params import (
    DEFAULT_LAG_CONFIRMATION_SAMPLES,
    DEFAULT_LAG_FRESHNESS_SEC,
    DEFAULT_LAG_WAIT_TIMEOUT_SEC,
    extract_yb_params,
)
from psycopg.yb.registry import FailoverGroup


pytestmark = pytest.mark.yb_unit


# -------------------------------------------------- controllable fake YBA

class _ScriptedYBAClient:
    """YBA test double. ``script`` is a list of (lag_ms, sample_ts)
    tuples; each ``get_committed_lag_sample`` call consumes the next
    entry. ``get_committed_lag_ms`` falls back to the lag component for
    the legacy-client code path. Repeats the last entry past the end."""

    def __init__(self, script: list[tuple[float | None, float | None]]) -> None:
        self.script = list(script)
        self.calls: list[str] = []

    def _next(self) -> tuple[float | None, float | None]:
        if not self.script:
            return (None, None)
        v = self.script[0]
        if len(self.script) > 1:
            self.script.pop(0)
        return v

    def get_committed_lag_sample(
        self, xcluster_uuid: str,
    ) -> tuple[float, float] | None:
        self.calls.append(xcluster_uuid)
        lag, ts = self._next()
        if lag is None:
            return None
        # Match the real client contract: never returns a None timestamp.
        return (lag, ts if ts is not None else time.time())

    def get_committed_lag_ms(self, xcluster_uuid: str) -> float | None:
        self.calls.append(xcluster_uuid)
        return self._next()[0]


# ---------------------------------------------------------- group fixture

def _make_group(
    fake_state,
    yba_client: _ScriptedYBAClient,
    threshold_ms: int = 0,
    lag_wait_timeout_s: int = 60,
    lag_freshness_s: int = DEFAULT_LAG_FRESHNESS_SEC,
    lag_confirmation_samples: int = DEFAULT_LAG_CONFIRMATION_SAMPLES,
) -> FailoverGroup:
    p = fake_state(("p1", "aws", "us-west", "us-west-1a", "primary"), uuid="P")
    s = fake_state(("s1", "aws", "us-east", "us-east-1a", "primary"), uuid="S")
    return FailoverGroup(
        primary=p, secondary=s, lock=threading.Lock(),
        primary_status=HealthResult.HEALTHY,
        secondary_status=HealthResult.HEALTHY,
        cooldown_s=0,
        yba_client=yba_client,
        xcluster_config_uuid="xcc-abcd",
        threshold_replication_lag_ms=threshold_ms,
        lag_wait_timeout_s=lag_wait_timeout_s,
        lag_freshness_s=lag_freshness_s,
        lag_confirmation_samples=lag_confirmation_samples,
    )


# -------------------------------------------------- freshness gate tests

def test_freshness_gate_rejects_sample_older_than_barrier_plus_floor(fake_state):
    """A sample whose ``sample_ts`` is BEFORE ``barrier_ts + freshness``
    is a known-stale read: it cannot have observed writes that landed
    before the barrier. The gate must NOT count it toward convergence."""
    # Pre-stamp the "fresh" moment at time.time() + 1000s so our scripted
    # pre-barrier samples (which use time.time() now) are definitionally
    # too old. The drain stamps barrier_ts at pause_dispatch; we use a
    # short wait so the test times out quickly instead of waiting for
    # the (nonexistent) fresh sample.
    stale_ts = time.time() - 5.0   # pretend sample was collected 5s ago
    client = _ScriptedYBAClient(script=[(0.0, stale_ts)])
    group = _make_group(
        fake_state, client,
        lag_wait_timeout_s=2,
        lag_freshness_s=25,
        lag_confirmation_samples=1,
    )
    group.on_phase2_timeout = "proceed"   # don't abort — we want the elapsed time
    t0 = time.monotonic()
    trigger_drain(group, "primary", HealthResult.UNHEALTHY, drain_timeout_s=0)
    elapsed = time.monotonic() - t0

    # The gate should have blocked every poll — Phase 2 ran to timeout
    # rather than converging on the stale sample.
    assert 1.5 <= elapsed <= 3.5, (
        "Phase 2 converged on a sample older than barrier_ts + "
        "lag_freshness_s, which it must not do"
    )
    # Multiple polls happened because convergence never fired.
    assert len(client.calls) >= 2


def test_freshness_gate_accepts_sample_past_floor(fake_state):
    """A sample stamped well AFTER the barrier window converges normally
    (with confirmation=1 for test speed)."""
    # Return samples timestamped WAY in the future so freshness always
    # passes regardless of when pause_dispatch stamped barrier_ts.
    future_ts = time.time() + 10_000.0
    client = _ScriptedYBAClient(script=[(0.0, future_ts)])
    group = _make_group(
        fake_state, client,
        lag_wait_timeout_s=10,
        lag_freshness_s=25,
        lag_confirmation_samples=1,
    )
    t0 = time.monotonic()
    trigger_drain(group, "primary", HealthResult.UNHEALTHY, drain_timeout_s=0)
    elapsed = time.monotonic() - t0

    assert elapsed < 1.5, "fresh + sufficient sample should converge immediately"
    assert len(client.calls) == 1


def test_freshness_zero_disables_gate(fake_state):
    """``lag_freshness_s=0`` is the explicit opt-out — pre-fix behaviour,
    accepts any sample. Operators who want short lag_wait_timeouts can
    disable the gate and own the DB-24483 data-loss risk."""
    stale_ts = time.time() - 60.0
    client = _ScriptedYBAClient(script=[(0.0, stale_ts)])
    group = _make_group(
        fake_state, client,
        lag_wait_timeout_s=10,
        lag_freshness_s=0,            # gate off
        lag_confirmation_samples=1,
    )
    t0 = time.monotonic()
    trigger_drain(group, "primary", HealthResult.UNHEALTHY, drain_timeout_s=0)
    elapsed = time.monotonic() - t0

    assert elapsed < 1.5, "gate off → stale sample converges"
    assert len(client.calls) == 1


# -------------------------------------------------- liveness gate tests

def test_liveness_gate_requires_two_distinct_samples(fake_state):
    """``lag_confirmation_samples=2`` (default) requires two distinct
    post-barrier passes. One is not enough."""
    future_ts_1 = time.time() + 10_000.0
    future_ts_2 = future_ts_1 + 11.0  # one scrape later
    # First tick: pass. Second tick: same timestamp (replay) — must NOT
    # count. Third tick: new timestamp — now we have 2 distinct passes.
    client = _ScriptedYBAClient(script=[
        (0.0, future_ts_1),
        (0.0, future_ts_1),
        (0.0, future_ts_2),
    ])
    group = _make_group(
        fake_state, client,
        lag_wait_timeout_s=10,
        lag_freshness_s=25,
        lag_confirmation_samples=2,
    )
    trigger_drain(group, "primary", HealthResult.UNHEALTHY, drain_timeout_s=0)

    # Three polls needed because the middle one was a replay.
    assert len(client.calls) == 3


def test_liveness_replay_pathology_never_converges(fake_state):
    """YBA replays the last sample for ~5min after the primary dies —
    the test models that with a single sample_ts repeated indefinitely.
    The gate must NOT converge, however many times the replay fires."""
    frozen_ts = time.time() + 10_000.0
    # ``_ScriptedYBAClient`` repeats the last entry past the end.
    client = _ScriptedYBAClient(script=[(0.0, frozen_ts)])
    group = _make_group(
        fake_state, client,
        lag_wait_timeout_s=3,
        lag_freshness_s=25,
        lag_confirmation_samples=2,
    )
    group.on_phase2_timeout = "proceed"
    t0 = time.monotonic()
    trigger_drain(group, "primary", HealthResult.UNHEALTHY, drain_timeout_s=0)
    elapsed = time.monotonic() - t0

    assert 2.5 <= elapsed <= 4.5, (
        "replayed sample must not progress the liveness count — "
        "Phase 2 should have hit its timeout"
    )
    # Many polls, all against the same replayed sample.
    assert len(client.calls) >= 3


def test_liveness_resets_when_lag_climbs_back(fake_state):
    """A tick where lag temporarily jumps above the threshold must reset
    the liveness counter — convergence has to earn N consecutive passes."""
    future_ts_1 = time.time() + 10_000.0
    future_ts_2 = future_ts_1 + 11.0
    future_ts_3 = future_ts_2 + 11.0
    future_ts_4 = future_ts_3 + 11.0
    client = _ScriptedYBAClient(script=[
        (0.0, future_ts_1),      # pass 1
        (500.0, future_ts_2),    # above threshold — RESET
        (0.0, future_ts_3),      # pass 1 again
        (0.0, future_ts_4),      # pass 2 — converge
    ])
    group = _make_group(
        fake_state, client,
        lag_wait_timeout_s=10,
        lag_freshness_s=25,
        lag_confirmation_samples=2,
    )
    trigger_drain(group, "primary", HealthResult.UNHEALTHY, drain_timeout_s=0)

    assert len(client.calls) == 4


def test_liveness_one_sample_disables_gate(fake_state):
    """``lag_confirmation_samples=1`` is the opt-out — one pass is
    enough, matching the pre-DB-24483 behaviour. Used by callers who've
    gated on freshness alone."""
    future_ts = time.time() + 10_000.0
    client = _ScriptedYBAClient(script=[(0.0, future_ts)])
    group = _make_group(
        fake_state, client,
        lag_wait_timeout_s=10,
        lag_freshness_s=25,
        lag_confirmation_samples=1,
    )
    trigger_drain(group, "primary", HealthResult.UNHEALTHY, drain_timeout_s=0)

    assert len(client.calls) == 1


# -------------------------------------------------- bootstrap floor tests

def _dsn(*extras: str) -> str:
    base = (
        "host=h1 load_balance_hosts=true "
        "yb.failover.secondaryClusterHosts=s1 "
        "yb.failover.ybaEndpoint=https://yba "
        "yb.failover.ybaApiToken=t "
        "yb.failover.replicationName=rep "
    )
    return base + " ".join(extras)


def test_bootstrap_warns_when_timeout_below_floor(caplog):
    """Both gates enabled and ``lag_wait_timeout_s`` below
    ``freshness + N × scrape_hint`` → bootstrap WARNS (does not raise).
    Phase 2 will time out on every planned failover under this config;
    the Phase 2 abort policy handles that safely. Raising a hard error
    at bootstrap would break stress-test harnesses that deliberately
    pick short timeouts to exercise the timeout path.
    """
    caplog.set_level("WARNING", logger="psycopg.yb.params")
    yb, _, _ = extract_yb_params(
        _dsn(
            "yb.failover.lagWaitTimeoutSecs=20",
            "yb.failover.lagFreshnessSecs=25",
            "yb.failover.lagConfirmationSamples=2",
        ),
        {},
    )
    # Bootstrap succeeded and the knobs are preserved.
    assert yb.lag_wait_timeout_s == 20
    assert yb.lag_freshness_s == 25
    assert yb.lag_confirmation_samples == 2
    # The warning fired with the specific "below the floor" phrase.
    assert any(
        "below the floor" in r.getMessage() and r.levelname == "WARNING"
        for r in caplog.records
    ), caplog.records


def test_bootstrap_accepts_timeout_at_floor():
    """``lag_wait_timeout_s`` exactly at the floor is allowed —
    operators who want the minimum should not need to pad."""
    yb, _, _ = extract_yb_params(
        _dsn(
            "yb.failover.lagWaitTimeoutSecs=45",     # 25 + 2*10
            "yb.failover.lagFreshnessSecs=25",
            "yb.failover.lagConfirmationSamples=2",
        ),
        {},
    )
    assert yb.lag_wait_timeout_s == 45
    assert yb.lag_freshness_s == 25
    assert yb.lag_confirmation_samples == 2


def test_bootstrap_freshness_zero_skips_floor():
    """``lag_freshness_s=0`` disables the gate; the floor doesn't apply
    and the operator may pick any lag_wait_timeout_s."""
    yb, _, _ = extract_yb_params(
        _dsn(
            "yb.failover.lagWaitTimeoutSecs=5",
            "yb.failover.lagFreshnessSecs=0",
        ),
        {},
    )
    assert yb.lag_wait_timeout_s == 5


def test_bootstrap_defaults_leave_gate_off():
    """The shipped defaults are gate-off (freshness=0, conf=1) so an
    upgrade cannot silently raise the Phase 2 wall-clock floor for a
    deployment that was previously running with lagWaitTimeoutSecs=30.
    Opt in by setting lagFreshnessSecs>0 and raising lagWaitTimeoutSecs
    above the derived floor. Regression guard against someone flipping
    the gate on by default without the harness coordination it needs."""
    yb, _, _ = extract_yb_params(_dsn(), {})
    assert yb.lag_wait_timeout_s == DEFAULT_LAG_WAIT_TIMEOUT_SEC
    assert yb.lag_freshness_s == DEFAULT_LAG_FRESHNESS_SEC
    assert yb.lag_confirmation_samples == DEFAULT_LAG_CONFIRMATION_SAMPLES
    assert yb.lag_freshness_s == 0
    assert yb.lag_confirmation_samples == 1


def test_bootstrap_failback_floor_applies_only_when_failback_phase2_enabled():
    """Failback floor applies when ``yb.failback.replicationName`` is
    wired (that's what turns on failback Phase 2) — not merely when
    xCluster is enabled. A DSN without a failback replication name may
    carry any failback_lag_wait_timeout_s because failback Phase 2 is
    skipped (fail-open) in that case."""
    yb, _, _ = extract_yb_params(
        _dsn(
            "yb.failback.lagWaitTimeoutSecs=5",       # below floor
            # NO yb.failback.replicationName → failback Phase 2 off.
        ),
        {},
    )
    assert yb.failback_lag_wait_timeout_s == 5

def test_bootstrap_failback_floor_warns_when_failback_phase2_enabled(caplog):
    """Wire failback replication AND opt in to the freshness gate → the
    floor applies and a timeout below it produces a WARNING (not an
    exception, per DB-24483 follow-up)."""
    caplog.set_level("WARNING", logger="psycopg.yb.params")
    yb, _, _ = extract_yb_params(
        _dsn(
            "yb.failback.replicationName=rep-b2a",
            # Opt in to the gate on the failback side; the shipped
            # default is 0 (gate off), so without this the floor check
            # is skipped and the warning never fires.
            "yb.failback.lagFreshnessSecs=25",
            "yb.failback.lagConfirmationSamples=2",
            "yb.failback.lagWaitTimeoutSecs=5",
        ),
        {},
    )
    assert yb.failback_lag_wait_timeout_s == 5
    assert any(
        "yb.failback.lagWaitTimeoutSecs" in r.getMessage()
        and "below the floor" in r.getMessage()
        and r.levelname == "WARNING"
        for r in caplog.records
    ), caplog.records


def test_bootstrap_floor_skipped_when_phase2_disabled():
    """A DSN that doesn't wire YBA at all → Phase 2 never runs → the
    floor doesn't apply, regardless of lag_wait_timeout_s. Preserves
    backward compatibility for pre-Phase-2 configs."""
    yb, _, _ = extract_yb_params(
        "host=h1 load_balance_hosts=true "
        "yb.failover.secondaryClusterHosts=s1 "
        "yb.failover.lagWaitTimeoutSecs=5",   # tiny; no YBA → fine
        {},
    )
    assert yb.lag_wait_enabled is False
    assert yb.lag_wait_timeout_s == 5


# -------------------------------------------------- barrier_ts stamping

def test_pause_dispatch_stamps_barrier_ts(fake_state):
    """``pause_dispatch`` must stamp a wall-clock ``barrier_ts`` so the
    freshness gate has something to compare against. ``resume_dispatch``
    does NOT clear it (it's a diagnostic for the next lag poll)."""
    p = fake_state(("p1", "aws", "us-west", "us-west-1a", "primary"), uuid="P")
    s = fake_state(("s1", "aws", "us-east", "us-east-1a", "primary"), uuid="S")
    group = FailoverGroup(
        primary=p, secondary=s, lock=threading.Lock(),
        primary_status=HealthResult.HEALTHY,
        secondary_status=HealthResult.HEALTHY,
    )
    assert group.barrier_ts == 0.0

    before = time.time()
    group.pause_dispatch()
    after = time.time()

    assert before <= group.barrier_ts <= after, (
        "barrier_ts must be stamped under pause_dispatch, "
        f"got {group.barrier_ts} outside [{before}, {after}]"
    )

    # resume_dispatch does not clear it — the stamp remains available
    # for the next cycle's logging and anyone diagnosing after a flip.
    stamped = group.barrier_ts
    group.resume_dispatch()
    assert group.barrier_ts == stamped
