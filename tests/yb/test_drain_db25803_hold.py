"""
DB-25803 — a failback that may be refused holds the secondary's conns
instead of closing them.

With ``yb.failback.onPhase2Timeout=fail`` (the default), Phase 1 of a
failback used to close every tracked conn on the secondary before the
lag wait. When the lag wait then failed, the failback was refused and
routing stayed on the secondary, but its conns were gone: a reconnect
wave on the serving side once per cooldown (review #18, drain.py:682).

Now those conns are held: their next round-trip blocks until the drain
decides. Refused, they carry on; failing back, they are closed as before.
A conn with a round-trip on the wire at the deadline is still closed.
"""

# Copyright (C) 2026 Yugabyte

from __future__ import annotations

import asyncio
import threading
import time

import pytest

from psycopg import errors as e
from psycopg.pq import TransactionStatus
from psycopg.yb import hold as yb_hold
from psycopg.yb.drain import trigger_drain
from psycopg.yb.health import HealthResult
from psycopg.yb.registry import FailoverGroup


pytestmark = pytest.mark.yb_unit


class _Info:
    def __init__(self, status: TransactionStatus) -> None:
        self.transaction_status = status


class _Conn:
    def __init__(self, status: TransactionStatus = TransactionStatus.IDLE) -> None:
        self.info = _Info(status)
        self.closed = False

    def close(self) -> None:
        self.closed = True
        self.info.transaction_status = TransactionStatus.UNKNOWN


class _LagYBA:
    def __init__(self, lag: float) -> None:
        self.lag = lag

    def get_committed_lag_sample(self, uuid):
        # Stamped well after the barrier, so the lag value alone decides.
        return (self.lag, time.time() + 10_000.0)


def _failback_group(fake_state, lag: float, policy: str = "fail") -> FailoverGroup:
    p = fake_state(("p1", "aws", "us-west", "us-west-1a", "primary"), uuid="P")
    s = fake_state(("s1", "aws", "us-east", "us-east-1a", "primary"), uuid="S")
    group = FailoverGroup(
        primary=p, secondary=s, lock=threading.Lock(),
        primary_status=HealthResult.UNHEALTHY,   # routing on the secondary
        secondary_status=HealthResult.HEALTHY,
        cooldown_s=0,
        yba_client=_LagYBA(lag),
        xcluster_config_uuid="xcc-ab",
        failback_xcluster_config_uuid="xcc-ba",
        failback_threshold_replication_lag_ms=0,
        failback_lag_wait_timeout_s=1,
        failback_lag_freshness_s=0,
        failback_lag_confirmation_samples=1,
    )
    group.failback_on_phase2_timeout = policy
    return group


def _fail_back(group: FailoverGroup) -> None:
    trigger_drain(group, "primary", HealthResult.HEALTHY, drain_timeout_s=0)


# ------------------------------------------------------------------ drain

def test_refused_failback_keeps_secondary_conns_open(fake_state):
    group = _failback_group(fake_state, lag=500.0)
    idle, in_txn = _Conn(), _Conn(TransactionStatus.INTRANS)
    group.secondary.tracked_conns.add(idle)
    group.secondary.tracked_conns.add(in_txn)

    _fail_back(group)

    assert group.primary_status == HealthResult.UNHEALTHY   # refused
    assert not idle.closed and not in_txn.closed
    assert getattr(idle, "_yb_hold", None) is None
    assert getattr(in_txn, "_yb_hold", None) is None


def test_converged_failback_closes_held_conns(fake_state):
    group = _failback_group(fake_state, lag=0.0)
    idle, in_txn = _Conn(), _Conn(TransactionStatus.INTRANS)
    group.secondary.tracked_conns.add(idle)
    group.secondary.tracked_conns.add(in_txn)

    _fail_back(group)

    assert group.primary_status == HealthResult.HEALTHY
    assert idle.closed and in_txn.closed


def test_round_trip_on_the_wire_at_the_deadline_is_closed(fake_state):
    """A hold only stops the next round-trip; one already sent (an
    autocommit statement) would commit on the secondary during Phase 2."""
    group = _failback_group(fake_state, lag=500.0)
    busy = _Conn(TransactionStatus.ACTIVE)
    busy._yb_busy = 1
    group.secondary.tracked_conns.add(busy)

    _fail_back(group)

    assert group.primary_status == HealthResult.UNHEALTHY
    assert busy.closed


@pytest.mark.parametrize("lag, outcome", [(500.0, "ran"), (0.0, "closed")])
def test_blocked_round_trip_follows_the_verdict(fake_state, lag, outcome):
    group = _failback_group(fake_state, lag=lag)
    conn = _Conn(TransactionStatus.INTRANS)
    group.secondary.tracked_conns.add(conn)
    seen: list[str] = []

    def app() -> None:
        deadline = time.monotonic() + 5
        while getattr(conn, "_yb_hold", None) is None:
            assert time.monotonic() < deadline
            time.sleep(0.005)
        try:
            yb_hold.block_if_held(conn)
        except e.OperationalError:
            seen.append("closed")
        else:
            yb_hold.done(conn)
            seen.append("ran")

    t = threading.Thread(target=app)
    t.start()
    _fail_back(group)
    t.join(5)

    assert seen == [outcome]


def test_failover_still_closes_conns(fake_state):
    group = _failback_group(fake_state, lag=0.0)
    group.primary_status = HealthResult.HEALTHY
    conn = _Conn()
    group.primary.tracked_conns.add(conn)

    trigger_drain(group, "primary", HealthResult.UNHEALTHY, drain_timeout_s=0)

    assert conn.closed
    assert getattr(conn, "_yb_hold", None) is None


def test_proceed_policy_closes_conns_without_holding(fake_state):
    group = _failback_group(fake_state, lag=500.0, policy="proceed")
    conn = _Conn()
    group.secondary.tracked_conns.add(conn)

    _fail_back(group)

    assert group.primary_status == HealthResult.HEALTHY
    assert conn.closed
    assert getattr(conn, "_yb_hold", None) is None


# ------------------------------------------------------------------- hold

def test_hold_conn_reports_a_round_trip_in_progress():
    conn = _Conn()
    hold = yb_hold.Hold()
    yb_hold.block_if_held(conn)           # not held: counts busy

    assert yb_hold.hold_conn(conn, hold) is True
    yb_hold.done(conn)
    assert yb_hold.hold_conn(conn, hold) is False


def test_pool_check_waits_for_the_verdict():
    conn = _Conn()
    hold = yb_hold.Hold()
    yb_hold.hold_conn(conn, hold)
    done = threading.Event()

    t = threading.Thread(target=lambda: (yb_hold.wait_settled(conn), done.set()))
    t.start()
    assert not done.wait(0.1)
    yb_hold.release(hold, [conn])
    assert done.wait(2)
    t.join()


def test_async_round_trip_waits_without_blocking_the_loop():
    conn = _Conn()
    hold = yb_hold.Hold()
    yb_hold.hold_conn(conn, hold)

    async def main() -> list[str]:
        ticks: list[str] = []

        async def ticker() -> None:
            for _ in range(3):
                await asyncio.sleep(0.01)
                ticks.append("tick")

        async def round_trip() -> None:
            await yb_hold.ablock_if_held(conn)
            ticks.append("ran")
            yb_hold.done(conn)

        task = asyncio.ensure_future(round_trip())
        await ticker()
        yb_hold.release(hold, [conn])
        await asyncio.wait_for(task, 2)
        return ticks

    assert asyncio.run(main()) == ["tick", "tick", "tick", "ran"]


def test_async_round_trip_raises_when_closed():
    conn = _Conn()
    hold = yb_hold.Hold()
    yb_hold.hold_conn(conn, hold)
    yb_hold.close(hold)

    with pytest.raises(e.OperationalError):
        asyncio.run(yb_hold.ablock_if_held(conn))


# ---------------------------------------------------- Connection.wait hook

def _instant_gen():
    return 42
    yield   # a generator that finishes without waiting on the socket


def _fake_conn(cluster: str | None = "secondary"):
    import types
    return types.SimpleNamespace(
        _yb_cluster=cluster, pgconn=types.SimpleNamespace(socket=-1),
    )


def test_connection_wait_blocks_while_held():
    import psycopg

    conn = _fake_conn()
    hold = yb_hold.Hold()
    yb_hold.hold_conn(conn, hold)
    result: list[int] = []

    t = threading.Thread(
        target=lambda: result.append(psycopg.Connection.wait(conn, _instant_gen()))
    )
    t.start()
    time.sleep(0.1)
    assert result == []
    yb_hold.release(hold, [conn])
    t.join(2)

    assert result == [42]
    assert conn._yb_busy == 0


def test_connection_wait_ignores_non_xcluster_conns():
    import psycopg

    conn = _fake_conn(cluster=None)
    conn._yb_hold = yb_hold.Hold()   # never consulted

    assert psycopg.Connection.wait(conn, _instant_gen()) == 42


def test_async_connection_wait_raises_when_closed():
    import psycopg

    conn = _fake_conn()
    hold = yb_hold.Hold()
    yb_hold.hold_conn(conn, hold)
    yb_hold.close(hold)

    with pytest.raises(e.OperationalError):
        asyncio.run(psycopg.AsyncConnection.wait(conn, _instant_gen()))
