#!/usr/bin/env bash
#
# Run the connect/close benchmark against upstream psycopg and against our
# psycopg-yugabytedb, sequentially. Because our distribution and upstream's
# both write to site-packages/psycopg/, they can't coexist in one venv, so
# this script does install → measure → uninstall → install → measure.
#
# At the end the script restores the editable install of our fork so the
# developer's environment is back to its pre-run state.
#
# Usage:
#     PSYCOPG_YB_TEST_DSN="host=127.0.0.1 port=5433 user=yugabyte dbname=yugabyte" \
#         bash tests/yb/perf/run_perf.sh
#
# Optional env:
#     N           — connect/close cycles per scenario (default 100)
#     PY          — Python interpreter (default: python3)
#     UPSTREAM_V  — upstream psycopg version to install (default: 3.3.4)

set -euo pipefail

DSN="${PSYCOPG_YB_TEST_DSN:-host=127.0.0.1 port=5433 user=yugabyte dbname=yugabyte}"
N="${N:-100}"
PY="${PY:-python3}"
UPSTREAM_V="${UPSTREAM_V:-3.3.4}"

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
DRIVER_DIR="$REPO_ROOT/psycopg"

echo "==> repo: $REPO_ROOT"
echo "==> python: $($PY -c 'import sys; print(sys.executable, sys.version.split()[0])')"
echo "==> dsn: $DSN"
echo "==> n: $N"
echo

# We use a quiet pip install path so the benchmark output isn't drowned out.
quiet_pip() {
    $PY -m pip "$@" --quiet 2>&1 | grep -vE '^(WARNING:|\[notice\])' || true
}

BENCH="$REPO_ROOT/tests/yb/perf/bench.py"

run_bench() {
    local label="$1"
    local lb="$2"
    # Run from /tmp to avoid the repo-root source-tree shadowing problem
    # (`import psycopg` would otherwise find the namespace dir before the
    # installed wheel). Invoke bench.py by path so /tmp doesn't need
    # `tests/` on sys.path.
    ( cd /tmp && $PY "$BENCH" \
        --dsn "$DSN" \
        --label "$label" \
        --load-balance "$lb" \
        --n "$N" )
}

cleanup_and_restore() {
    echo
    echo "==> restoring editable install of psycopg-yugabytedb"
    quiet_pip uninstall -y psycopg psycopg-yugabytedb || true
    quiet_pip install -e "$DRIVER_DIR"
}
trap cleanup_and_restore EXIT

# 1. Our fork ----------------------------------------------------------
echo "==> installing psycopg-yugabytedb (editable, from $DRIVER_DIR)"
quiet_pip uninstall -y psycopg psycopg-yugabytedb || true
quiet_pip install -e "$DRIVER_DIR"
echo
echo "--- psycopg-yugabytedb scenarios ---"
# Smart driver on. Hits the dispatcher, registry, policy, control conn.
RESULT_YB_ON=$(run_bench yb-on true)
echo "$RESULT_YB_ON"
# Pass-through. Same code as upstream, just with our dispatcher branching off
# early when load_balance_hosts is false.
RESULT_YB_OFF=$(run_bench yb-off false)
echo "$RESULT_YB_OFF"
echo

# 2. Swap to upstream --------------------------------------------------
echo "==> installing upstream psycopg==$UPSTREAM_V"
quiet_pip uninstall -y psycopg-yugabytedb
quiet_pip install "psycopg==$UPSTREAM_V"
echo
echo "--- upstream psycopg scenarios ---"
RESULT_UP=$(run_bench upstream absent)
echo "$RESULT_UP"
echo

# 3. Comparison --------------------------------------------------------
echo "==> comparison"
$PY - <<EOF
import json
results = [
    json.loads('''$RESULT_UP'''),
    json.loads('''$RESULT_YB_OFF'''),
    json.loads('''$RESULT_YB_ON'''),
]
print(f"{'scenario':<14} {'driver':<22} {'p50 ms':>8} {'p90 ms':>8} {'p99 ms':>8} {'mean ms':>9}")
print("-" * 75)
for r in results:
    print(f"{r['label']:<14} {r['driver']+' '+r['version']:<22} "
          f"{r['p50_ms']:>8.2f} {r['p90_ms']:>8.2f} {r['p99_ms']:>8.2f} {r['mean_ms']:>9.2f}")
up_p50 = results[0]['p50_ms']
yb_off_p50 = results[1]['p50_ms']
yb_on_p50  = results[2]['p50_ms']
print()
print(f"yb-off vs upstream: {(yb_off_p50 / up_p50 - 1) * 100:+.1f}% (p50)")
print(f"yb-on  vs upstream: {(yb_on_p50  / up_p50 - 1) * 100:+.1f}% (p50)")
EOF
