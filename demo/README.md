# xCluster failover live demo

End-to-end demo of the smart driver's failover working on top of **real
xCluster replication**: write rows on the primary cluster, kill the
primary, watch the pool's reads automatically reroute to the secondary
cluster and return the same rows.

## What it shows

```
┌──────────── what the SMART DRIVER does ─────────────┐  ┌────── what xCLUSTER does ───────┐
│ ConnectionPool routes new conns to primary while    │  │ Replicates failover_demo rows    │
│ healthy; on CB trip, evicts stale conns and opens   │  │ primary → secondary in real time │
│ replacements on secondary.                          │  │ (yb-admin setup_universe_replication) │
└─────────────────────────────────────────────────────┘  └──────────────────────────────────┘
                                  ↓ together ↓
                Application sees: zero-config read failover
                + data continuity through a primary outage
```

## Files

| File | Role |
|---|---|
| `setup.sh` | Brings up both clusters AND configures xCluster replication |
| `teardown.sh` | Tears down replication + destroys both clusters |
| `xcluster_failover_demo.py` | The demo: writes rows on primary, fails over, reads on secondary |
| `watch_rpcz.py` | Side-by-side `/rpcz` view of all 6 nodes (run in a second terminal) |

## Prereqs

* `psycopg-yugabytedb` + `psycopg_pool` installed (or `PYTHONPATH` set to this repo).
* A YugabyteDB install (2024.1+ recommended — older versions may need different `yb-admin` syntax).
* **macOS only**: loopback aliases for `.4`–`.6`:
  ```bash
  sudo ifconfig lo0 alias 127.0.0.4/32 up
  sudo ifconfig lo0 alias 127.0.0.5/32 up
  sudo ifconfig lo0 alias 127.0.0.6/32 up
  ```
* `yb-ctl` shebangs `#!/usr/bin/env python`. If only `python3` exists on PATH, add a shim:
  ```bash
  mkdir -p /tmp/py && ln -sf "$(command -v python3)" /tmp/py/python
  export PATH=/tmp/py:$PATH
  ```

## Running

```bash
# 1. Point at your yb-ctl + bring up both clusters + xCluster replication
export YB_CTL=/path/to/yugabyte-2024.1.0.0/bin/yb-ctl
bash demo/setup.sh

# 2. Make psycopg-yugabytedb resolvable
cd /path/to/psycopg-yb
export PYTHONPATH="$PWD/psycopg:$PWD/psycopg_pool:$PYTHONPATH"
```

**Terminal A** — live `/rpcz` view (updates every 1 s):

```bash
python3 demo/watch_rpcz.py
```

**Terminal B** — the demo. Walks through four phases pausing on `ENTER`:

```bash
python3 demo/xcluster_failover_demo.py
```

Sequence:

1. **Phase 1** — pool inserts 24 rows on primary, prints per-host
   distribution (2/2/2). The demo then opens a side-channel SELECT on
   the secondary and prints `secondary holds 24/24 rows`, proving
   xCluster is live.
2. **Phase 2** — prompts to stop the primary. In a third shell:
   ```bash
   yb-ctl --data_dir=~/yb-primary stop
   ```
   Then press `ENTER` in Terminal B.
3. Demo polls `group.status` with a per-second heartbeat. CB usually
   trips in under a second once primary's PG is fully gone.
4. **Phase 3** — re-borrows 6 conns. Each runs `SELECT count(*) FROM
   failover_demo`. All 6 land on secondary; all 6 see 24 rows. Watch
   terminal: `.1/.2/.3` flip to `DOWN`, `.4/.5/.6` light up.
5. **Phase 4** — holds the 6 secondary conns open. Verify in a side shell:
   ```bash
   ysqlsh -h 127.0.0.4 -U yugabyte -d yugabyte \
     -c "SELECT * FROM failover_demo ORDER BY id LIMIT 5;"
   ```
   Press `ENTER` to release and exit.

## Teardown

```bash
bash demo/teardown.sh
```

Deletes the replication stream, destroys both clusters, hard-kills any
stragglers, and removes the data dirs.

## Validated output (from a verified end-to-end run)

```
Phase 1:
  127.0.0.1  2 pool conn(s)  -> 8 rows inserted via this host
  127.0.0.2  2 pool conn(s)  -> 8 rows inserted via this host
  127.0.0.3  2 pool conn(s)  -> 8 rows inserted via this host
  ✓ secondary now holds 24/24 rows. xCluster is replicating live.

Phase 2: STOP primary → CB tripped in 0.0s (instant on primary refused).

Phase 3:
  127.0.0.4  2 pool conn(s)   sees 24 rows
  127.0.0.5  2 pool conn(s)   sees 24 rows
  127.0.0.6  2 pool conn(s)   sees 24 rows
  ✓ all 6 pool conns landed on the SECONDARY cluster
    and read back all 24 rows.

/rpcz:
  127.0.0.1  DOWN     127.0.0.4  3 backends
  127.0.0.2  DOWN     127.0.0.5  2 backends
  127.0.0.3  DOWN     127.0.0.6  2 backends
```

## Caveats / known limitations

* Replication is **unidirectional** (primary → secondary). Writes from
  the application during the outage land on the secondary but never
  replicate back to the primary on recovery — for a real failback story
  you'd configure bidirectional xCluster, which is a separate operator
  step.
* `TRUNCATE failover_demo` would fail (YB blocks table-rewriting DDL on
  xCluster source tables — see yugabyte-db #16625). The demo uses
  `DELETE FROM` for the per-run reset, which replicates cleanly.
* `setup.sh` uses table-level non-transactional xCluster
  (`setup_universe_replication`). For DB-scoped transactional xCluster
  (consistent reads across tables), use `create_xcluster_checkpoint` +
  `setup_xcluster_replication` instead — both work in 2024.1+.
