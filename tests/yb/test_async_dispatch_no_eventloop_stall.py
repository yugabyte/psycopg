"""
Integration-level repro test for DB-24838 part 1 — PR-H2.

Shubham reported on #proj-smartdriver-xclusterfailover (Oct 20 2026):

  "the asyncio event loop stops for the whole switch (drain + lag
  wait, 130 s in one cycle)... From the first AsyncConnection.connect()
  after the drain barrier until the routing flip, no coroutine in the
  process runs. Cycle 21 failback: one connect() call took 129.93 s;
  the async writers committed nothing for 120 s while the sync writers
  kept committing 210-360 per 10 s."

Root cause: ``AsyncConnection.connect`` called sync
``wait_for_dispatch`` which uses ``threading.Condition.wait_for`` —
blocks the event-loop OS thread for the whole drain window.

Fix (this PR): added ``wait_for_dispatch_async`` that yields to the
loop via ``await asyncio.Event().wait()``. ``resume_dispatch`` on the
probe thread fans out to every registered waiter via
``loop.call_soon_threadsafe(event.set)`` so each loop wakes up on its
own thread.

These tests prove the fix by running the exact observable Shubham
measured: a heartbeat coroutine ticking every 100 ms, an
``AsyncConnection.connect()`` call entering the drain barrier, and a
drain that holds the barrier for N seconds. Assertion: the heartbeat
keeps ticking throughout the drain (max stall under one tick, not
drain-window length), proving the loop is NOT blocked.
"""

# Copyright (C) 2026 Yugabyte

from __future__ import annotations

import asyncio
import threading
import time
from typing import Any

import pytest

from psycopg.yb.health import HealthResult
from psycopg.yb.registry import FailoverGroup
from psycopg.yb.state import (
    wait_for_dispatch,
    wait_for_dispatch_async,
)


pytestmark = pytest.mark.yb_unit


# ===================================================================
# Harness — a FailoverGroup with no real conns, just the dispatch
# pause infra. We'll pause/resume it from a background "probe" thread
# to simulate the drain sequence.
# ===================================================================

def _make_group(fake_state) -> FailoverGroup:
    p = fake_state(("p1", "aws", "us-west", "us-west-1a", "primary"), uuid="P")
    s = fake_state(("s1", "aws", "us-east", "us-east-1a", "primary"), uuid="S")
    return FailoverGroup(
        primary=p, secondary=s, lock=threading.Lock(),
        primary_status=HealthResult.HEALTHY,
        secondary_status=HealthResult.HEALTHY,
        cooldown_s=0,
    )


# ===================================================================
# DB-24838 part 1 — event loop stays responsive during drain
# ===================================================================

@pytest.mark.anyio
async def test_db24838_heartbeat_ticks_through_drain_window(fake_state):
    """The exact repro of Shubham's observable: a 100 ms heartbeat
    coroutine ticks continuously; a separate coroutine calls
    ``wait_for_dispatch_async`` while the drain holds the barrier for
    1 second. Assert the heartbeat ticked ~10 times (NOT once at the
    end, which is what a blocked loop would produce).
    """
    group = _make_group(fake_state)
    group.pause_dispatch()   # arm barrier BEFORE the waiter enters

    heartbeat_ticks: list[float] = []
    stop_heartbeat = asyncio.Event()

    async def heartbeat():
        """Ticks every 100 ms while the loop runs. A blocked loop
        would produce zero ticks until it wakes up."""
        t_start = time.monotonic()
        while not stop_heartbeat.is_set():
            heartbeat_ticks.append(time.monotonic() - t_start)
            try:
                await asyncio.wait_for(stop_heartbeat.wait(), timeout=0.1)
            except asyncio.TimeoutError:
                pass

    async def connect_waiter():
        """Mirrors AsyncConnection.connect's call to
        wait_for_dispatch_async. If the sync version were used, this
        would block the whole loop."""
        return await wait_for_dispatch_async(group)

    def release_drain_after_1s():
        """Simulates the probe thread's drain completion — resumes
        dispatch from a different OS thread, exactly like
        ``trigger_drain`` does."""
        time.sleep(1.0)
        group.resume_dispatch()

    drain_thread = threading.Thread(
        target=release_drain_after_1s, name="fake-probe", daemon=True,
    )
    drain_thread.start()

    heartbeat_task = asyncio.create_task(heartbeat())
    try:
        t0 = time.monotonic()
        result = await asyncio.wait_for(connect_waiter(), timeout=3.0)
        elapsed = time.monotonic() - t0
    finally:
        stop_heartbeat.set()
        await heartbeat_task
        drain_thread.join(timeout=1.0)

    assert result is True, "wait_for_dispatch_async should return True on resume"
    # Barrier held ~1 s; we should have waited that long end-to-end.
    assert 0.9 < elapsed < 1.6, f"expected ~1s wait, got {elapsed:.2f}s"

    # The core assertion: heartbeat ticked many times during the drain
    # window. A blocked event loop (the DB-24838 bug) would produce
    # one tick at the start and one at the end — total 2 ticks over
    # the 1 s window. The fix produces ~10 ticks (one per 100 ms).
    assert len(heartbeat_ticks) >= 7, (
        f"heartbeat only ticked {len(heartbeat_ticks)} times during the "
        f"~1 s drain window — the event loop was blocked (DB-24838). "
        f"Expected ~10 ticks (one per 100 ms). Ticks: {heartbeat_ticks}"
    )

    # Also assert max inter-tick gap stayed near the 100 ms interval;
    # a blocked loop would show one gap ~= drain window length.
    if len(heartbeat_ticks) >= 2:
        gaps = [
            heartbeat_ticks[i + 1] - heartbeat_ticks[i]
            for i in range(len(heartbeat_ticks) - 1)
        ]
        max_gap = max(gaps)
        assert max_gap < 0.3, (
            f"max inter-tick gap was {max_gap:.3f}s — expected ~0.1s. "
            f"A gap approaching the drain-window length means the loop "
            f"stalled (DB-24838)."
        )


@pytest.mark.anyio
async def test_db24838_fast_path_when_not_paused(fake_state):
    """When dispatch is not paused, ``wait_for_dispatch_async`` must
    return immediately without registering a waiter or awaiting
    anything. Mirrors the sync fast-path contract."""
    group = _make_group(fake_state)
    # Note: dispatch_paused defaults to False.

    t0 = time.monotonic()
    result = await wait_for_dispatch_async(group)
    elapsed = time.monotonic() - t0

    assert result is True
    assert elapsed < 0.05, f"fast path should be instant, took {elapsed:.3f}s"
    assert group.async_dispatch_waiters == [], (
        "fast path should not register a waiter"
    )


@pytest.mark.anyio
async def test_db24838_timeout_returns_false_without_blocking_loop(fake_state):
    """Pause stays held past the caller's timeout. Assert False
    returned, assert the heartbeat kept ticking throughout, assert
    the waiter was cleaned out of the group's list."""
    group = _make_group(fake_state)
    group.pause_dispatch()
    # Will NOT be resumed in this test — the timeout should fire.

    heartbeat_ticks = []
    stop_heartbeat = asyncio.Event()

    async def heartbeat():
        t_start = time.monotonic()
        while not stop_heartbeat.is_set():
            heartbeat_ticks.append(time.monotonic() - t_start)
            try:
                await asyncio.wait_for(stop_heartbeat.wait(), timeout=0.05)
            except asyncio.TimeoutError:
                pass

    heartbeat_task = asyncio.create_task(heartbeat())
    try:
        t0 = time.monotonic()
        result = await wait_for_dispatch_async(group, timeout_s=0.4)
        elapsed = time.monotonic() - t0
    finally:
        stop_heartbeat.set()
        await heartbeat_task
        group.resume_dispatch()   # cleanup

    assert result is False
    assert 0.35 < elapsed < 0.8, f"expected ~0.4s wait, got {elapsed:.2f}s"
    # Heartbeat kept ticking even though we were waiting on a barrier
    # that never released.
    assert len(heartbeat_ticks) >= 5, (
        f"heartbeat only ticked {len(heartbeat_ticks)} times during a "
        f"0.4 s timeout wait — loop was stalled"
    )
    # Waiter was removed from the group on timeout.
    assert group.async_dispatch_waiters == [], (
        "timeout path should remove the waiter from the group"
    )


@pytest.mark.anyio
async def test_db24838_concurrent_async_waiters_all_wake_on_resume(fake_state):
    """Multiple concurrent AsyncConnection.connect() calls — all
    should resume together when the drain releases, and the heartbeat
    should stay alive throughout."""
    group = _make_group(fake_state)
    group.pause_dispatch()

    heartbeat_ticks = []
    stop_heartbeat = asyncio.Event()

    async def heartbeat():
        t_start = time.monotonic()
        while not stop_heartbeat.is_set():
            heartbeat_ticks.append(time.monotonic() - t_start)
            try:
                await asyncio.wait_for(stop_heartbeat.wait(), timeout=0.1)
            except asyncio.TimeoutError:
                pass

    def release_after(delay_s: float):
        time.sleep(delay_s)
        group.resume_dispatch()

    drain_thread = threading.Thread(
        target=release_after, args=(0.5,), daemon=True,
    )
    drain_thread.start()

    heartbeat_task = asyncio.create_task(heartbeat())
    try:
        # Fire 10 concurrent waiters — exactly the AsyncConnectionPool
        # pattern where many coroutines call connect() during a drain.
        t0 = time.monotonic()
        results = await asyncio.gather(
            *[wait_for_dispatch_async(group) for _ in range(10)]
        )
        elapsed = time.monotonic() - t0
    finally:
        stop_heartbeat.set()
        await heartbeat_task
        drain_thread.join(timeout=1.0)

    assert all(results), f"not all waiters returned True: {results}"
    assert 0.45 < elapsed < 1.0, (
        f"expected ~0.5s for all 10 to wake together, took {elapsed:.2f}s"
    )
    assert len(heartbeat_ticks) >= 4, (
        f"heartbeat only ticked {len(heartbeat_ticks)} times — the "
        f"10 concurrent waiters stalled the loop"
    )
    assert group.async_dispatch_waiters == [], (
        "all waiters should be removed after resume"
    )


# ===================================================================
# Sync path regression — the original wait_for_dispatch still works
# ===================================================================

def test_sync_wait_for_dispatch_still_works_after_async_addition(fake_state):
    """Adding the async variant must not have broken the sync one."""
    group = _make_group(fake_state)
    group.pause_dispatch()

    def release_soon():
        time.sleep(0.3)
        group.resume_dispatch()

    t = threading.Thread(target=release_soon, daemon=True)
    t.start()

    t0 = time.monotonic()
    result = wait_for_dispatch(group)
    elapsed = time.monotonic() - t0
    t.join(timeout=1.0)

    assert result is True
    assert 0.25 < elapsed < 0.8


# ===================================================================
# Cross-thread safety — async waiter in loop-thread A, resume from
# probe thread B (mirrors drain.py's real wiring)
# ===================================================================

@pytest.mark.anyio
async def test_db24838_resume_from_separate_thread_wakes_waiter(fake_state):
    """``trigger_drain`` runs on the probe daemon thread, not any
    event loop. ``resume_dispatch`` must correctly fan out to a waiter
    whose event loop runs on a different thread — exactly our real
    wiring, exercised via ``loop.call_soon_threadsafe``."""
    group = _make_group(fake_state)
    group.pause_dispatch()

    resumed_from = {"thread": None}

    def probe_thread_body():
        resumed_from["thread"] = threading.current_thread().name
        time.sleep(0.3)
        group.resume_dispatch()

    probe = threading.Thread(
        target=probe_thread_body, name="probe-primary", daemon=True,
    )
    probe.start()

    t0 = time.monotonic()
    result = await wait_for_dispatch_async(group)
    elapsed = time.monotonic() - t0
    probe.join(timeout=1.0)

    assert result is True
    assert 0.25 < elapsed < 0.8
    assert resumed_from["thread"] == "probe-primary", (
        "test setup bug — probe thread wasn't the one that resumed"
    )


# ===================================================================
# Spurious wake — resume → pause race must not release the waiter
# ===================================================================

@pytest.mark.anyio
async def test_waiter_rechecks_dispatch_paused_after_event(fake_state):
    """agent-K PR #14:126. The sync waiter calls
    ``Condition.wait_for(lambda: not dispatch_paused, ...)`` which
    re-reads the predicate on every wake. The async waiter must do the
    same, otherwise this race releases the waiter while a drain is
    still active:

      1. drain A completes → resume_dispatch() → event.set()
      2. drain B starts → pause_dispatch() → dispatch_paused=True again
      3. async waiter wakes from event.wait() → without a recheck,
         returns True → caller proceeds to connect into drain B

    The fix: after event.wait() returns, re-enter the lock and
    re-check dispatch_paused. If still set, loop (re-register, wait
    again). Only return True when dispatch_paused is observed False
    under the lock.
    """
    group = _make_group(fake_state)
    group.pause_dispatch()

    waiter_done = asyncio.Event()
    waiter_result = {}

    async def waiter():
        waiter_result["returned"] = await wait_for_dispatch_async(group)
        waiter_done.set()

    task = asyncio.create_task(waiter())

    # Give the waiter time to register.
    await asyncio.sleep(0.05)

    # Simulate the resume → pause race in rapid succession.
    group.resume_dispatch()                    # event.set() fires
    group.pause_dispatch()                     # pause re-armed

    # Waiter must NOT return — dispatch_paused is True again.
    try:
        await asyncio.wait_for(waiter_done.wait(), timeout=0.3)
        pytest.fail(
            "waiter returned despite dispatch_paused still being True — "
            "the recheck is missing"
        )
    except asyncio.TimeoutError:
        pass                                    # expected: waiter blocked

    # Final genuine resume — waiter must return True now.
    group.resume_dispatch()
    await asyncio.wait_for(waiter_done.wait(), timeout=0.5)
    task.result()                               # propagate any exception
    assert waiter_result["returned"] is True
