"""
Connect+close benchmark, driver-agnostic.

Imports `psycopg` (whatever is currently installed under that import name)
and times N connect/close cycles. Used by `run_perf.sh` which orchestrates
two runs: one with upstream `psycopg==3.3.4` installed, one with our
`psycopg-yugabytedb` installed. Both runs go through the same code path
(this script), so anything different between the two numbers is
attributable to the driver itself.

Output is one line of JSON on stdout so the orchestrator can collect and
compare. Errors go to stderr.

Usage:

    python -m tests.yb.perf.bench \
        --dsn "host=127.0.0.1 port=5433 user=yugabyte dbname=yugabyte" \
        --label upstream \
        --load-balance absent \
        --n 100
"""

# Copyright (C) 2026 Yugabyte

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time

import psycopg


def _quantile(samples: list[float], q: float) -> float:
    """Return the q-th quantile (0 < q < 1) without depending on Python 3.13's
    statistics.quantiles edge-case behaviour."""
    if not samples:
        return float("nan")
    s = sorted(samples)
    k = max(0, min(len(s) - 1, int(round(q * (len(s) - 1)))))
    return s[k]


def measure(dsn: str, n: int, warmup: int = 10) -> dict[str, float]:
    """Time `n` connect/close cycles after `warmup` discarded cycles.

    Why warmup matters here: the smart-driver path has one-time costs on the
    FIRST connection — bootstrap (open contact-point conn, run yb_servers(),
    populate ClusterState, stash the control connection). If we measured that
    first connect alongside the others it would skew the percentiles for our
    fork while leaving upstream's numbers untouched, biasing the comparison.

    We also discard a few more samples past the bootstrap to absorb DNS
    caching, libpq internal state, and any Python-level first-call costs.
    `warmup=10` is generous; the bootstrap itself is just one connect.
    """
    # Warmup — bootstrap on the first iteration (for our fork), plus a margin
    # to absorb any other first-time costs.
    for _ in range(warmup):
        psycopg.connect(dsn).close()

    samples: list[float] = []
    for _ in range(n):
        t0 = time.perf_counter()
        conn = psycopg.connect(dsn)
        samples.append(time.perf_counter() - t0)
        conn.close()

    return {
        "n": float(n),
        "warmup": float(warmup),
        "p50_ms":  statistics.median(samples) * 1000,
        "p90_ms":  _quantile(samples, 0.90) * 1000,
        "p99_ms":  _quantile(samples, 0.99) * 1000,
        "mean_ms": statistics.mean(samples) * 1000,
        "min_ms":  min(samples) * 1000,
        "max_ms":  max(samples) * 1000,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsn", required=True,
                    help="Connection string (without load_balance_hosts)")
    ap.add_argument("--label", required=True,
                    help="Identifier for this run (printed in the result)")
    ap.add_argument(
        "--load-balance",
        choices=("true", "false", "absent"),
        default="absent",
        help=(
            "true: append load_balance_hosts=true (smart driver). "
            "false: append load_balance_hosts=false (our pass-through). "
            "absent: don't append anything (libpq-only)."
        ),
    )
    ap.add_argument("--n", type=int, default=100,
                    help="Number of connect/close cycles to measure")
    ap.add_argument("--warmup", type=int, default=10,
                    help="Number of warmup cycles to discard before measuring "
                         "(absorbs bootstrap cost on the smart-driver path)")
    args = ap.parse_args()

    dsn = args.dsn
    if args.load_balance == "true":
        dsn += " load_balance_hosts=true"
    elif args.load_balance == "false":
        dsn += " load_balance_hosts=false"

    try:
        stats = measure(dsn, args.n, warmup=args.warmup)
    except Exception as exc:
        print(json.dumps({"label": args.label, "error": repr(exc)}), file=sys.stdout)
        sys.exit(1)

    result = {
        "label":   args.label,
        "driver":  psycopg.__name__,
        "version": getattr(psycopg, "__version__", "?"),
        "lb":      args.load_balance,
        **stats,
    }
    print(json.dumps(result))


if __name__ == "__main__":
    main()
