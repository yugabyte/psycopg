#!/usr/bin/env bash
# Set up two yb-ctl clusters + UNIDIRECTIONAL xCluster replication
# (primary -> secondary) for the failover_demo table.
#
# Layout (Amogh's demo spec):
#   primary   : RF=3, 5 tservers, on 127.0.0.1 .. 127.0.0.5
#   secondary : RF=3, 5 tservers, on 127.0.0.6 .. 127.0.0.10
#
# Requires:
#   * $YB_CTL set to a yb-ctl binary (e.g. yugabyte-2024.1.0.0/bin/yb-ctl)
#   * Loopback aliases 127.0.0.2-.10 on macOS:
#       for i in 2 3 4 5 6 7 8 9 10; do
#         sudo ifconfig lo0 alias 127.0.0.$i/32 up
#       done
#   * `python` resolvable (yb-ctl shebangs itself with /usr/bin/env python).
#     On macOS where only python3 exists:
#       mkdir -p /tmp/py && ln -sf "$(command -v python3)" /tmp/py/python
#       export PATH=/tmp/py:$PATH
set -euo pipefail

: "${YB_CTL:?must set YB_CTL=/path/to/yb-ctl}"
YB_BIN="$(dirname "$YB_CTL")"
YB_ADMIN="${YB_BIN}/yb-admin"
YSQLSH="${YB_BIN}/ysqlsh"
PRIMARY_DIR="${PRIMARY_DIR:-$HOME/yb-primary}"
SECONDARY_DIR="${SECONDARY_DIR:-$HOME/yb-secondary}"
# Master quorum lives on the first 3 nodes of each side (RF=3).
PRIMARY_MASTERS=127.0.0.1:7100,127.0.0.2:7100,127.0.0.3:7100
SECONDARY_MASTERS=127.0.0.6:7100,127.0.0.7:7100,127.0.0.8:7100
REPL_GROUP_ID=failover_demo_repl

step() { echo; echo "==> $*"; }

step "[1/7] create PRIMARY cluster on 127.0.0.1..3 (rf=3)"
"$YB_CTL" --data_dir="$PRIMARY_DIR" create \
  --rf 3 --placement_info cloud1.datacenter1.rack1 --ip_start 1 >/dev/null

step "[2/7] extend PRIMARY to 5 tservers (127.0.0.4, .5)"
"$YB_CTL" --data_dir="$PRIMARY_DIR" add_node >/dev/null
"$YB_CTL" --data_dir="$PRIMARY_DIR" add_node >/dev/null

step "[3/7] create SECONDARY cluster on 127.0.0.6..8 (rf=3)"
"$YB_CTL" --data_dir="$SECONDARY_DIR" create \
  --rf 3 --placement_info cloud1.datacenter1.rack1 --ip_start 6 >/dev/null

step "[4/7] extend SECONDARY to 5 tservers (127.0.0.9, .10)"
"$YB_CTL" --data_dir="$SECONDARY_DIR" add_node >/dev/null
"$YB_CTL" --data_dir="$SECONDARY_DIR" add_node >/dev/null

step "[5/7] wait for ysql on both clusters"
for h in 127.0.0.1 127.0.0.6; do
  ready=0
  for _ in $(seq 30); do
    if "$YSQLSH" -h "$h" -p 5433 -U yugabyte -d yugabyte -c 'SELECT 1' >/dev/null 2>&1; then
      ready=1; break
    fi
    sleep 1
  done
  [ "$ready" = "1" ] || { echo "FAIL: $h:5433 never came up"; exit 1; }
done

step "[6/7] create the failover_demo table on BOTH clusters"
# Non-transactional xCluster requires the same schema on both ends BEFORE
# replication is set up. Idempotent — CREATE TABLE IF NOT EXISTS.
SQL="CREATE TABLE IF NOT EXISTS failover_demo (id INT PRIMARY KEY, ts TIMESTAMP, payload TEXT);"
"$YSQLSH" -h 127.0.0.1 -p 5433 -U yugabyte -d yugabyte -c "$SQL" >/dev/null
"$YSQLSH" -h 127.0.0.6 -p 5433 -U yugabyte -d yugabyte -c "$SQL" >/dev/null

step "look up the failover_demo table id on the producer (primary)"
TABLE_ID=$("$YB_ADMIN" -master_addresses "$PRIMARY_MASTERS" \
  list_tables include_table_id 2>/dev/null \
  | awk '/^yugabyte\.failover_demo / {print $2}')
if [ -z "$TABLE_ID" ]; then
  echo "FAIL: could not find failover_demo table_id on primary"
  exit 1
fi
echo "    table_id = $TABLE_ID"

step "[7/7] establish xCluster replication primary -> secondary"
# setup_universe_replication runs on the CONSUMER (secondary) and is
# pointed at the PRODUCER's masters. The replication group id is just a
# label we use later to alter/delete the stream.
"$YB_ADMIN" -master_addresses "$SECONDARY_MASTERS" setup_universe_replication \
  "$REPL_GROUP_ID" "$PRIMARY_MASTERS" "$TABLE_ID"

step "replication status (from the consumer's view)"
"$YB_ADMIN" -master_addresses "$SECONDARY_MASTERS" get_replication_status 2>&1 | head -10

cat <<EOF

================================================================
xCluster setup complete.

  producer (primary)   : 127.0.0.1..127.0.0.5   (rf=3, 5 tservers)
  consumer (secondary) : 127.0.0.6..127.0.0.10  (rf=3, 5 tservers)
  replicating table    : public.failover_demo (id, ts, payload)
  direction            : UNIDIRECTIONAL primary -> secondary
  repl_group_id        : ${REPL_GROUP_ID}

Next (Amogh's demo flow):
  # 1. app fails to start (no CB attached)
  python3 demo/xcluster_drain_timing_demo.py

  # 2. no-drain run: fail over + fail back
  ATTACH_CB=1 DRAIN_TIMEOUT_S=0 \\
      python3 demo/xcluster_drain_timing_demo.py

  # 3. drain run: fail over with drain window
  ATTACH_CB=1 DRAIN_TIMEOUT_S=15 \\
      python3 demo/xcluster_drain_timing_demo.py

Teardown: bash demo/teardown.sh
================================================================
EOF
