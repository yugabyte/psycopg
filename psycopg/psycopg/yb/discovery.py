"""
Run the YugabyteDB cluster-discovery query (`yb_servers()`) and parse the
result into a list of `NodeInfo` plus the cluster's `universe_uuid`.

The same logic underlies pgjdbc-yb's `LoadBalanceService.refresh()`.
"""

# Copyright (C) 2026 Yugabyte

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from .node import NodeInfo, Placement

if TYPE_CHECKING:
    from ..connection import Connection
    from ..connection_async import AsyncConnection

logger = logging.getLogger(__name__)


# pgjdbc-yb's GET_SERVERS_QUERY. Columns we consume below; future-proofed
# against the server adding more by selecting *.
GET_SERVERS_QUERY = "SELECT * FROM yb_servers()"

# Statement timeout for control-connection queries. Mirrors pgjdbc-yb's
# socketTimeout of 15s so a hung tserver can't wedge refresh forever.
CONTROL_QUERY_TIMEOUT_MS = 15_000


def _parse_rows(rows: list[tuple]) -> tuple[list[NodeInfo], str]:
    """Build NodeInfo list + universe_uuid from `yb_servers()` rows.

    Schema (positional, matching pgjdbc-yb's ResultSet usage):

        0: host            (text)
        1: port            (int)
        2: num_connections (int)  — not used
        3: node_type       (text) — "primary" | "read_replica"
        4: cloud           (text)
        5: region          (text)
        6: zone            (text)
        7: public_ip       (text, nullable)
        8: uuid            (text) — per-tserver uuid (not used yet)
        9: universe_uuid   (text) — same across every row for one cluster

    Returns (nodes, universe_uuid). When the server omits the universe_uuid
    column (older YB versions), returns "" — callers fall back to a
    stable-but-process-local identifier.
    """
    nodes: list[NodeInfo] = []
    universe_uuid = ""
    for row in rows:
        host = row[0]
        port = int(row[1])
        node_type = row[3]
        cloud = row[4]
        region = row[5]
        zone = row[6]
        public_ip = row[7] if len(row) > 7 else None
        if not universe_uuid and len(row) > 9 and row[9]:
            universe_uuid = str(row[9])
        nodes.append(NodeInfo(
            host=host,
            public_ip=public_ip if public_ip else None,
            port=port,
            placement=Placement(cloud=cloud, region=region, zone=zone),
            node_type=node_type,
        ))
    return nodes, universe_uuid


def _stable_fallback_uuid(rows: list[tuple]) -> str:
    """Derive a stable per-cluster identifier when yb_servers() doesn't return
    a universe_uuid column. Sorted host list is sufficient to keep registry
    state coherent within one process; we never persist this across processes."""
    hosts = sorted(r[0] for r in rows)
    return "fallback:" + ",".join(hosts)


def fetch_servers_sync(conn: "Connection") -> tuple[list[NodeInfo], str]:
    """Run yb_servers() and return (nodes, universe_uuid).

    universe_uuid is column 9 of every row; we read it from the first row.
    The connection itself is not closed — the caller stashes it as the
    cluster's control connection.
    """
    t0 = time.perf_counter()
    cur = conn.execute(GET_SERVERS_QUERY)
    rows = cur.fetchall()
    nodes, uuid = _parse_rows(rows)
    if not uuid:
        uuid = _stable_fallback_uuid(rows)
    logger.debug(
        "yb_servers() returned %d row(s) in %.1fms (uuid=%s)",
        len(rows), (time.perf_counter() - t0) * 1000, uuid,
    )
    return nodes, uuid


async def fetch_servers_async(conn: "AsyncConnection") -> tuple[list[NodeInfo], str]:
    t0 = time.perf_counter()
    cur = await conn.execute(GET_SERVERS_QUERY)
    rows = await cur.fetchall()
    nodes, uuid = _parse_rows(rows)
    if not uuid:
        uuid = _stable_fallback_uuid(rows)
    logger.debug(
        "yb_servers() returned %d row(s) in %.1fms (uuid=%s, async)",
        len(rows), (time.perf_counter() - t0) * 1000, uuid,
    )
    return nodes, uuid
