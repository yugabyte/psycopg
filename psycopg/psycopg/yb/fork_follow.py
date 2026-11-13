"""
DB-25060 — forked workers follow the parent's failover.

A pre-fork server (gunicorn, Celery prefork, multiprocessing) bootstraps
the FailoverGroup in the parent and forks workers. If every worker ran its
own probes, each would drain and flip on its own schedule: the parent's
barrier would not stop a worker's writes, and a worker could flip before
the parent. One application, several failovers.

Instead the parent stays the only process that probes and decides. Before
the first fork it allocates a small anonymous shared mapping (inherited by
every child) and publishes into it:

    paused / gen        the drain barrier, armed and released
    outgoing            which cluster is draining
    kill_gen            "force-close your outgoing conns now"
    primary/secondary   the routing statuses

Each child runs a follower thread that mirrors that state onto its own
FailoverGroup: it pauses dispatch with the parent, closes its own conns on
the outgoing cluster, reports its in-flight count back, and resumes on the
statuses the parent flipped to. The parent's Phase 1 counts the children's
in-flight transactions, so the barrier covers the whole application.

If the parent process goes away the child stops following and starts its
own probes.
"""

# Copyright (C) 2026 Yugabyte

from __future__ import annotations

import logging
import mmap
import os
import struct
import threading
import time
from typing import TYPE_CHECKING

from .health import HealthResult

if TYPE_CHECKING:
    from .registry import FailoverGroup

logger = logging.getLogger("psycopg.yb.fork_follow")

MAX_CHILDREN = 256

# Header: parent_pid u32, paused u8, primary u8, secondary u8, outgoing u8,
# gen u64, kill_gen u64.
_HDR = struct.Struct("=IBBBBQQ")
_HDR_SIZE = 64
# Slot: pid u32, in_flight u32, acked_gen u64, killed_gen u64, heartbeat f64.
_SLOT = struct.Struct("=IIQQd")
_SLOT_SIZE = 32

# A child whose heartbeat is older than this is treated as gone: its
# in-flight count no longer holds up the parent's Phase 1.
_HEARTBEAT_STALE_S = 1.0
# Follower poll interval.
_FOLLOW_POLL_S = 0.02

_OUTGOING_PRIMARY = 0
_OUTGOING_SECONDARY = 1


def _encode(status: HealthResult) -> int:
    return 1 if status == HealthResult.HEALTHY else 0


def _decode(value: int) -> HealthResult:
    return HealthResult.HEALTHY if value else HealthResult.UNHEALTHY


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class SharedFollowState:
    """The shared mapping. One writer per field: the parent writes the
    header, each child writes only its own slot."""

    def __init__(self, primary: HealthResult, secondary: HealthResult) -> None:
        self.buf = mmap.mmap(-1, _HDR_SIZE + MAX_CHILDREN * _SLOT_SIZE)
        self.parent_pid = os.getpid()
        self._lock = threading.Lock()      # parent-side header writes
        self._write_header(
            paused=0, primary=_encode(primary), secondary=_encode(secondary),
            outgoing=_OUTGOING_PRIMARY, gen=0, kill_gen=0,
        )

    # ------------------------------------------------------------ header

    def read(self) -> dict:
        pid, paused, primary, secondary, outgoing, gen, kill_gen = (
            _HDR.unpack_from(self.buf, 0)
        )
        return {
            "parent_pid": pid,
            "paused": bool(paused),
            "primary": _decode(primary),
            "secondary": _decode(secondary),
            "outgoing_primary": outgoing == _OUTGOING_PRIMARY,
            "gen": gen,
            "kill_gen": kill_gen,
        }

    def _write_header(self, **fields: int) -> None:
        values = dict(zip(
            ("pid", "paused", "primary", "secondary", "outgoing", "gen", "kill_gen"),
            _HDR.unpack_from(self.buf, 0),
        ))
        values.update(fields)
        values["pid"] = self.parent_pid
        _HDR.pack_into(
            self.buf, 0, values["pid"], values["paused"], values["primary"],
            values["secondary"], values["outgoing"], values["gen"],
            values["kill_gen"],
        )

    def publish_pause(self, gen: int, outgoing_primary: bool) -> None:
        with self._lock:
            # Outgoing and gen first, then paused, so a child that sees
            # paused=1 also sees which cluster to drain.
            self._write_header(
                outgoing=_OUTGOING_PRIMARY if outgoing_primary else _OUTGOING_SECONDARY,
                gen=gen,
            )
            self._write_header(paused=1)

    def publish_kill(self, gen: int) -> None:
        with self._lock:
            self._write_header(kill_gen=gen)

    def publish_statuses(self, primary: HealthResult, secondary: HealthResult) -> None:
        with self._lock:
            self._write_header(primary=_encode(primary), secondary=_encode(secondary))

    def publish_resume(self, primary: HealthResult, secondary: HealthResult) -> None:
        with self._lock:
            self._write_header(primary=_encode(primary), secondary=_encode(secondary))
            self._write_header(paused=0)

    # ------------------------------------------------------------- slots

    def _slot_off(self, slot: int) -> int:
        return _HDR_SIZE + slot * _SLOT_SIZE

    def _read_slot(self, slot: int) -> tuple:
        return _SLOT.unpack_from(self.buf, self._slot_off(slot))

    def _write_slot(self, slot: int, **fields) -> None:
        pid, in_flight, acked, killed, hb = self._read_slot(slot)
        values = {"pid": pid, "in_flight": in_flight, "acked_gen": acked,
                  "killed_gen": killed, "heartbeat": hb}
        values.update(fields)
        _SLOT.pack_into(
            self.buf, self._slot_off(slot), values["pid"], values["in_flight"],
            values["acked_gen"], values["killed_gen"], values["heartbeat"],
        )

    def reserve_slot(self) -> "int | None":
        """Parent side, just before fork: pick a free slot (never used, or
        whose child stopped heart-beating) and pre-stamp it so the child's
        first moments count as live."""
        now = time.monotonic()
        for slot in range(MAX_CHILDREN):
            pid, _, _, _, hb = self._read_slot(slot)
            if pid == 0 or now - hb > _HEARTBEAT_STALE_S * 5:
                _SLOT.pack_into(self.buf, self._slot_off(slot), 0, 0, 0, 0, now)
                return slot
        return None

    def claim_slot(self, slot: int, gen: int) -> None:
        self._write_slot(
            slot, pid=os.getpid(), in_flight=0, acked_gen=gen,
            killed_gen=gen, heartbeat=time.monotonic(),
        )

    def release_slot(self, slot: int) -> None:
        _SLOT.pack_into(self.buf, self._slot_off(slot), 0, 0, 0, 0, 0.0)

    def heartbeat(self, slot: int, in_flight: int) -> None:
        self._write_slot(slot, in_flight=in_flight, heartbeat=time.monotonic())

    def ack(self, slot: int, gen: int) -> None:
        self._write_slot(slot, acked_gen=gen)

    def ack_kill(self, slot: int, gen: int) -> None:
        self._write_slot(slot, killed_gen=gen, in_flight=0)

    def _live_slots(self) -> list[tuple]:
        now = time.monotonic()
        out = []
        for slot in range(MAX_CHILDREN):
            row = self._read_slot(slot)
            if row[0] == 0 or now - row[4] > _HEARTBEAT_STALE_S:
                continue
            out.append(row)
        return out

    def live_children(self) -> int:
        return len(self._live_slots())

    def children_in_flight(self, gen: int) -> int:
        """Parent side. In-flight transactions across live children for
        drain ``gen``. A child that has not yet seen the barrier counts as
        one, so the parent never declares a natural drain before every
        child has paused."""
        total = 0
        for _, in_flight, acked, _, _ in self._live_slots():
            total += in_flight if acked == gen else max(in_flight, 1)
        return total

    def wait_children_killed(self, gen: int, timeout_s: float) -> int:
        """Parent side. Wait until every live child has force-closed its
        outgoing conns for ``gen``. Returns how many had not by the
        timeout."""
        deadline = time.monotonic() + timeout_s
        while True:
            pending = sum(
                1 for _, _, _, killed, _ in self._live_slots() if killed != gen
            )
            if pending == 0 or time.monotonic() >= deadline:
                return pending
            time.sleep(_FOLLOW_POLL_S)


# --------------------------------------------------------------- parent side


def is_leader(group: "FailoverGroup") -> bool:
    follow = getattr(group, "_follow", None)
    return follow is not None and follow.parent_pid == os.getpid()


def ensure_shared_state(group: "FailoverGroup") -> None:
    """Allocate the mapping in the process that runs the probes. Called
    from ``_start_probe``, which always runs before an app forks."""
    if getattr(group, "_follow", None) is not None or not hasattr(os, "fork"):
        return
    group._follow = SharedFollowState(group.primary_status, group.secondary_status)


def before_fork(group: "FailoverGroup") -> None:
    follow = getattr(group, "_follow", None)
    group._follow_pending_slot = (
        follow.reserve_slot() if follow is not None else None
    )


# ---------------------------------------------------------------- child side


def start_follower(group: "FailoverGroup", on_parent_gone) -> bool:
    """Child side, from the fork hook. Returns False when there is nothing
    to follow (no mapping, no slot, or the parent already exited); the
    caller then starts the child's own probes."""
    follow = getattr(group, "_follow", None)
    slot = getattr(group, "_follow_pending_slot", None)
    group._follow_pending_slot = None
    if follow is None or slot is None or not _pid_alive(follow.parent_pid):
        return False
    snap = follow.read()
    follow.claim_slot(slot, snap["gen"])
    t = threading.Thread(
        target=_follow_loop, args=(group, follow, slot, snap, on_parent_gone),
        name="yb-xcluster-follow", daemon=True,
    )
    group._follower_thread = t
    t.start()
    logger.info(
        "xCluster fork: following parent pid=%d (slot %d) instead of "
        "running probes in this worker", follow.parent_pid, slot,
    )
    return True


def _active(group: "FailoverGroup", primary: HealthResult, secondary: HealthResult):
    if primary == HealthResult.HEALTHY:
        return group.primary
    if secondary == HealthResult.HEALTHY:
        return group.secondary
    return None


def _close_all(state) -> int:
    from .drain import _safe_close, _tracked_snapshot

    closed = 0
    for c in _tracked_snapshot(state):
        if c is None or getattr(c, "closed", False):
            continue
        if _safe_close(c) is None:
            closed += 1
    return closed


def _set_statuses(group: "FailoverGroup", primary: HealthResult, secondary: HealthResult) -> None:
    now = time.monotonic()
    with group.lock:
        if group.primary_status != primary:
            group.primary_status = primary
            group.primary_last_transition_time = now
        if group.secondary_status != secondary:
            group.secondary_status = secondary
            group.secondary_last_transition_time = now


def _follow_loop(group, follow: SharedFollowState, slot: int, snap: dict, on_parent_gone) -> None:
    from .drain import (
        _close_idle_tracked_during_drain, _in_flight_conns, _tracked_snapshot,
    )

    local_gen = snap["gen"]
    killed_gen = snap["gen"]
    paused = False
    outgoing = None
    # Start on the parent's current routing.
    _set_statuses(group, snap["primary"], snap["secondary"])
    while True:
        try:
            if not _pid_alive(follow.parent_pid):
                if paused:
                    group.resume_dispatch()
                follow.release_slot(slot)
                group._follow = None
                logger.warning(
                    "xCluster fork: parent pid=%d is gone — this worker "
                    "now runs its own probes", follow.parent_pid,
                )
                on_parent_gone(group)
                return
            snap = follow.read()
            if snap["paused"] and snap["gen"] != local_gen:
                local_gen = snap["gen"]
                outgoing = group.primary if snap["outgoing_primary"] else group.secondary
                group.pause_dispatch()
                paused = True
                follow.ack(slot, local_gen)
                logger.info(
                    "xCluster fork: parent armed barrier gen=%d — pausing "
                    "this worker's dispatch, draining %s",
                    local_gen, outgoing.uuid,
                )
            if paused:
                _close_idle_tracked_during_drain(_tracked_snapshot(outgoing), outgoing.uuid)
                in_flight = len(_in_flight_conns(_tracked_snapshot(outgoing)))
                if snap["kill_gen"] == local_gen and killed_gen != local_gen:
                    closed = _close_all(outgoing)
                    killed_gen = local_gen
                    follow.ack_kill(slot, local_gen)
                    in_flight = 0
                    logger.info(
                        "xCluster fork: closed %d conn(s) on %s at the "
                        "parent's flip boundary (gen=%d)",
                        closed, outgoing.uuid, local_gen,
                    )
                follow.heartbeat(slot, in_flight)
                if not snap["paused"]:
                    if killed_gen != local_gen:
                        _close_all(outgoing)
                        killed_gen = local_gen
                        follow.ack_kill(slot, local_gen)
                    _set_statuses(group, snap["primary"], snap["secondary"])
                    paused = False
                    group.resume_dispatch()
                    follow.heartbeat(slot, 0)
                    logger.info(
                        "xCluster fork: parent released barrier gen=%d — "
                        "routing primary=%s secondary=%s",
                        local_gen, snap["primary"].value, snap["secondary"].value,
                    )
            else:
                follow.heartbeat(slot, 0)
                if (snap["primary"], snap["secondary"]) != (
                    group.primary_status, group.secondary_status,
                ):
                    # Missed a whole barrier window between two polls, or
                    # a transition that did not move routing. Close our
                    # conns on a cluster we stop routing to, then adopt.
                    old = _active(group, group.primary_status, group.secondary_status)
                    new = _active(group, snap["primary"], snap["secondary"])
                    if old is not None and old is not new:
                        group.pause_dispatch()
                        _close_all(old)
                        _set_statuses(group, snap["primary"], snap["secondary"])
                        group.resume_dispatch()
                    else:
                        _set_statuses(group, snap["primary"], snap["secondary"])
        except Exception:
            logger.debug("xCluster fork: follower tick failed", exc_info=True)
        time.sleep(_FOLLOW_POLL_S)
