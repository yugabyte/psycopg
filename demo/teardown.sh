#!/usr/bin/env bash
# Tear down the xCluster + both yb-ctl clusters. Idempotent; safe to run
# multiple times even if pieces are already gone.
set -uo pipefail

: "${YB_CTL:?must set YB_CTL=/path/to/yb-ctl}"
YB_BIN="$(dirname "$YB_CTL")"
YB_ADMIN="${YB_BIN}/yb-admin"
PRIMARY_DIR="${PRIMARY_DIR:-$HOME/yb-primary}"
SECONDARY_DIR="${SECONDARY_DIR:-$HOME/yb-secondary}"
SECONDARY_MASTERS=127.0.0.4:7100,127.0.0.5:7100,127.0.0.6:7100
REPL_GROUP_ID=failover_demo_repl

step() { echo; echo "==> $*"; }

step "delete xCluster replication"
"$YB_ADMIN" -master_addresses "$SECONDARY_MASTERS" delete_universe_replication \
  "$REPL_GROUP_ID" ignore-errors 2>/dev/null || true

step "destroy primary cluster"
"$YB_CTL" --data_dir="$PRIMARY_DIR" destroy 2>&1 | tail -1 || true

step "destroy secondary cluster"
"$YB_CTL" --data_dir="$SECONDARY_DIR" destroy 2>&1 | tail -1 || true

step "hard-kill any leftover yb processes"
pkill -9 -f yb-master 2>/dev/null || true
pkill -9 -f yb-tserver 2>/dev/null || true

step "remove data directories"
rm -rf "$PRIMARY_DIR" "$SECONDARY_DIR"

echo
echo "DONE."
