#!/usr/bin/env bash
# Tear down the xCluster + both yb-ctl clusters. Idempotent; safe to run
# multiple times even if pieces are already gone.
set -uo pipefail

: "${YB_CTL:?must set YB_CTL=/path/to/yb-ctl}"
YB_BIN="$(dirname "$YB_CTL")"
YB_ADMIN="${YB_BIN}/yb-admin"
PRIMARY_DIR="${PRIMARY_DIR:-$HOME/yb-primary}"
SECONDARY_DIR="${SECONDARY_DIR:-$HOME/yb-secondary}"
# Must match setup.sh — the consumer-side masters that know about the
# replication group id. Previously wrong (.4-.6 vs. a stale 5+5 layout);
# now consistent with setup.sh's 3+3 layout. agent-K PR #12:11.
SECONDARY_MASTERS=127.0.0.4:7100,127.0.0.5:7100,127.0.0.6:7100
REPL_GROUP_ID=failover_demo_repl

step() { echo; echo "==> $*"; }

step "delete xCluster replication"
# NOT || true-silenced — a stale replication group left behind here
# makes the next setup.sh fail with "replication group already exists".
# Report the error instead of hiding it. The `ignore-errors` arg to
# yb-admin still swallows "no such group" so idempotency is intact.
"$YB_ADMIN" -master_addresses "$SECONDARY_MASTERS" delete_universe_replication \
  "$REPL_GROUP_ID" ignore-errors || echo "warn: delete_universe_replication exited $?"

step "destroy primary cluster"
"$YB_CTL" --data_dir="$PRIMARY_DIR" destroy 2>&1 | tail -1 || true

step "destroy secondary cluster"
"$YB_CTL" --data_dir="$SECONDARY_DIR" destroy 2>&1 | tail -1 || true

step "hard-kill leftover yb processes from THIS demo's data dirs only"
# Scope to processes whose cmdline includes one of our data dirs — the
# previous version killed every yb-master / yb-tserver on the box,
# which would stomp on any other YB cluster the operator was running.
# agent-K PR #12:11.
for d in "$PRIMARY_DIR" "$SECONDARY_DIR"; do
  pkill -9 -f "yb-master.*$d" 2>/dev/null || true
  pkill -9 -f "yb-tserver.*$d" 2>/dev/null || true
done

step "remove data directories"
rm -rf "$PRIMARY_DIR" "$SECONDARY_DIR"

echo
echo "DONE."
