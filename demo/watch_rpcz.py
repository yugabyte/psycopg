#!/usr/bin/env python3
"""Live view of /rpcz 'client backend' counts across all six xCluster nodes.

Run this in a second terminal alongside ``xcluster_failover_demo.py``.

Each tserver exposes an HTTP debug endpoint at ``http://<host>:13000/rpcz``
that lists every active backend. We count occurrences of the literal
substring ``client backend`` per host — same technique pgjdbc-yb's
``FallbackOptionsLBTest`` uses — and re-render once a second so you can
see the count move from the primary cluster to the secondary as the
demo's failover unfolds.

Exit with ctrl+c.
"""

from __future__ import annotations

import sys
import time
import urllib.error
import urllib.request


HOSTS = [
    ("PRIMARY  ", "127.0.0.1"),
    ("PRIMARY  ", "127.0.0.2"),
    ("PRIMARY  ", "127.0.0.3"),
    ("SECONDARY", "127.0.0.4"),
    ("SECONDARY", "127.0.0.5"),
    ("SECONDARY", "127.0.0.6"),
]
RPCZ_PORT = 13000
POLL_S = 1.0
HTTP_TIMEOUT_S = 2.0


def rpcz_count(host: str) -> int | None:
    """Return the # of 'client backend' rows on ``host``'s /rpcz, or None
    if the tserver is unreachable (the operator hasn't started it yet,
    or has stopped it as part of the demo)."""
    url = f"http://{host}:{RPCZ_PORT}/rpcz"
    try:
        with urllib.request.urlopen(url, timeout=HTTP_TIMEOUT_S) as resp:
            body = resp.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, OSError):
        return None
    return body.count("client backend")


def render(counts: dict[str, int | None]) -> str:
    lines = []
    lines.append(f"{'role':<10} {'host':<14} {'client backends':<10}  bar")
    lines.append("-" * 60)
    # Max bar = 12 (room for ~12 conns) — adjust if the demo pool grows.
    bar_max = 12
    for role, host in HOSTS:
        cnt = counts[host]
        if cnt is None:
            shown = "  DOWN"
            bar = ""
        else:
            shown = f"{cnt:>3}"
            bar = "█" * min(cnt, bar_max)
            if cnt > bar_max:
                bar += f" (+{cnt - bar_max})"
        lines.append(f"{role:<10} {host:<14} {shown:>10}  {bar}")
    return "\n".join(lines)


def main() -> int:
    use_clear = sys.stdout.isatty()
    print(
        f"watching /rpcz on 6 nodes every {POLL_S:g}s (ctrl+c to exit)\n",
        flush=True,
    )
    try:
        while True:
            counts = {host: rpcz_count(host) for _, host in HOSTS}
            if use_clear:
                # ANSI: clear screen, cursor to home.
                sys.stdout.write("\033[2J\033[H")
            sys.stdout.write(
                f"xCluster /rpcz — {time.strftime('%H:%M:%S')}\n\n"
            )
            sys.stdout.write(render(counts))
            sys.stdout.write(f"\n\n(poll every {POLL_S:g}s, ctrl+c to exit)\n")
            sys.stdout.flush()
            time.sleep(POLL_S)
    except KeyboardInterrupt:
        print("\nwatch stopped")
        return 0


if __name__ == "__main__":
    sys.exit(main())
