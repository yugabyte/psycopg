# `tests/yb/` — psycopg-yugabytedb tests

Three tiers, gated by pytest markers (auto-applied by `conftest.py` based on
filename — no decorators on individual tests):

| Marker     | Files                                            | Count | Needs                            | Typical runtime |
|------------|--------------------------------------------------|-------|----------------------------------|------------------|
| `yb_unit`  | `test_params`, `test_node`, `test_policy`, `test_registry` | 92 | nothing (pure Python)            | < 1 s total      |
| `yb`       | `test_smart_driver*.py` (sync + `_async` siblings) | 77 | real YB cluster + `yb-ctl`       | ~10 s/test (cluster cycle), ~25 min total |
| `perf`     | `perf/bench.py` (driven by `perf/run_perf.sh`)   | 3 scenarios | running cluster + pip access | ~30 s/scenario   |

The integration suite intentionally mirrors every sync scenario on the async
path. The sync surface lives in `test_smart_driver.py`,
`test_smart_driver_failover.py`, `test_smart_driver_topology_failover.py`,
`test_smart_driver_pool.py`, and `test_smart_driver_pool_failover.py`; the
async siblings live in the matching `*_async.py` files. Sync/async drift
is the failure mode this mirroring is set up to catch — keep both halves in
lockstep when adding scenarios.

The {direct-connect, pool} × {cluster-aware, topology-aware} × {steady,
failure-mode} cube is fully covered. Pool tests require `psycopg-pool`
installed (`pytest.importorskip`); install it with the fork's `[pool]`
extra: `pip install -e "./psycopg[pool]"`.

## Running

All commands assume your shell is in the fork root (the directory with
`psycopg/`, `tests/`, `tools/`) and that you have the fork installed
editably: `pip install -e ./psycopg`.

```bash
# Unit tests — pure Python, no cluster needed.
pytest --import-mode=importlib -m yb_unit tests/yb/

# Full integration suite. Needs yb-ctl. Each test destroys+creates a fresh
# cluster, so this takes ~15 min.
YB_CTL=/path/to/yugabyte-X.Y.Z.W/bin/yb-ctl \
    pytest --import-mode=importlib -m yb tests/yb/

# A single integration test, useful while iterating.
YB_CTL=/path/to/yugabyte-X.Y.Z.W/bin/yb-ctl \
    pytest --import-mode=importlib -m yb \
    tests/yb/test_smart_driver_failover.py::test_control_conn_reopens_on_host_loss_and_picks_up_added_node -v

# Perf benchmark — see the dedicated section below. You start the cluster
# manually; the script doesn't manage it.
/path/to/yb-ctl create --rf 3 --placement_info cloud1.datacenter1.rack1
PSYCOPG_YB_TEST_DSN="host=127.0.0.1,127.0.0.2,127.0.0.3 port=5433 user=yugabyte dbname=yugabyte" \
    N=200 bash tests/yb/perf/run_perf.sh
/path/to/yb-ctl destroy
```

### Why `--import-mode=importlib`

The fork's repo layout has a top-level `psycopg/` directory (the package
source) which, in pytest's default import mode, becomes an implicit
namespace package because `psycopg/` itself has no `__init__.py`. That
namespace package then shadows the installed driver, so `from psycopg import
…` imports an empty module and every smart-driver attribute is missing.
`importlib` mode bypasses `sys.path` manipulation and resolves `psycopg` to
the actual installed package.

## Prerequisites

### Get yb-ctl

`yb-ctl` ships with every YugabyteDB release. Either install YB locally or
extract the bin directory from a release tarball. Tested with 2025.2.0.0.
Set `YB_CTL` to the absolute path of the binary, or put its directory on
`PATH`.

### macOS only: loopback aliases for .4 onward

`yb-ctl` runs nodes on `127.0.0.1`, `.2`, `.3` (these are aliased by
`yb-ctl create` via sudo) and — for tests that call `add_node` — `.4`. macOS
only routes loopback addresses that are explicitly aliased, and `yb-ctl
add_node` does **not** alias them. Set them up once per boot:

```bash
sudo ifconfig lo0 alias 127.0.0.4/32 up
sudo ifconfig lo0 alias 127.0.0.5/32 up   # if any test ever adds a 5th node
```

These survive until reboot. Without them, every `add_node`-based test fails
with `Failed waiting for None tservers`.

Tests that hit this path:
`test_uniform_load_after_node_addition` (sync + async),
`test_control_conn_reopens_on_host_loss_and_picks_up_added_node` (sync + async),
`test_add_node_in_topology_zone_receives_traffic` (sync + async),
`test_add_node_outside_topology_zone_receives_no_traffic` (sync + async).

## Files

```
tests/yb/
├── conftest.py                                       pytest config + fixtures
├── test_params.py                                    [yb_unit] conninfo parsing
├── test_node.py                                      [yb_unit] NodeInfo, Placement
├── test_policy.py                                    [yb_unit] policy picker behaviour
├── test_registry.py                                  [yb_unit] ClusterRegistry, ClusterKey, control reopen
├── test_smart_driver.py                              [yb]      sync direct-connect integration
├── test_smart_driver_async.py                        [yb]      async direct-connect mirror
├── test_smart_driver_failover.py                     [yb]      sync cluster-aware failover suite
├── test_smart_driver_failover_async.py               [yb]      async cluster-aware failover mirror
├── test_smart_driver_topology_failover.py            [yb]      sync topology × node-state matrix
├── test_smart_driver_topology_failover_async.py      [yb]      async topology × node-state mirror
├── test_smart_driver_pool.py                         [yb]      sync ConnectionPool steady-state
├── test_smart_driver_pool_async.py                   [yb]      async AsyncConnectionPool steady-state
├── test_smart_driver_pool_failover.py                [yb]      sync pool × failover (cluster + topology)
├── test_smart_driver_pool_failover_async.py          [yb]      async pool × failover mirror
└── perf/
    ├── bench.py                                      driver-agnostic connect+close timer
    └── run_perf.sh                                   installs upstream vs fork sequentially, compares
```

## Cluster lifecycle and fixtures

Each integration test destroys and creates a fresh cluster — destroy ~4 s,
create ~6 s on a warm machine. We picked per-test create over a session
fixture because state leakage (residual connections, lingering counters,
half-failed `add_node` operations) was a real source of flakiness; ~10 s/test
overhead is the simpler trade. Destroys are aggressive: any stuck `yb-ctl`,
`yb-tserver`, or `yb-master` process gets killed and the data directory is
wiped, so a test that fails mid-mutation doesn't poison the next one.

After cluster create + readiness probe, the fixture polls `/rpcz` until the
"client backend" count stops changing across two consecutive reads (or 8 s
elapse). This settles the brief window where YB hasn't yet reaped the
readiness-probe connection's backend, so test assertions about exact /rpcz
counts have a stable starting point.

Two cluster shapes:

* **`yb_cluster`** — 3 nodes, all in `cloud1.datacenter1.rack1`. Default for
  the majority of integration tests.
* **`yb_multi_zone_cluster`** — 3 nodes: 2 in `cloud1.datacenter1.zoneA`,
  1 in `cloud1.datacenter1.zoneB`. Used by topology × node-state tests.

### The `assert_balanced` fixture

`assert_balanced(uuid, expected_per_host)` does two checks, both **exact**,
no tolerance knob:

1. **Driver side**: `ClusterRegistry.get_load(uuid, host) == expected` for
   each host. The least-loaded picker reserves+increments atomically under
   the per-cluster lock, so this can't drift even under concurrent connects.
2. **Server side** (`/rpcz` "client backend" count per host): asserts equal
   to `expected + 1` for the host(s) that host the smart driver's
   long-lived control connection(s), and equal to `expected` for the others.
   The fixture reads `state.control_sync.info.host` and
   `state.control_async.info.host` at assert time to determine which host(s)
   to add the +1 for. The dead-conn settle loop in `yb_cluster` ensures
   nothing else is on /rpcz.

Drift on either side is treated as a real failure, never as a flake.

## What's covered

### Unit tests (`yb_unit`, 92 tests, no DB)

**Conninfo parsing** (`test_params.py`):
`load_balance_hosts` parsing for `true` / `false` / `disable` / `random` /
absent, with the right ones stripped before libpq sees the conninfo.
`topology_keys` parsing (single entry, multiple entries, zone wildcard,
cloud and region wildcards rejected at parse time).
`yb_servers_refresh_interval` and `failed_host_reconnect_delay_secs`
(defaults, clamps). Dashed and underscore forms both accepted. YB-only keys
removed from kwargs before libpq parsing; unrelated kwargs preserved.

**NodeInfo / Placement** (`test_node.py`):
Placement.matches semantics including zone wildcard and directional
matching. Case sensitivity. NodeInfo defaults. Placement hashability.

**ClusterRegistry / ClusterKey** (`test_registry.py`):
Singleton identity. ClusterKey sort-invariant, hashable, password and SSL
mode excluded. Counter primitives (increment/decrement floors at 0,
unknown uuid/host is a no-op). Counter thread-safety stress (10 threads ×
1000 inc/dec, final count must be exactly 0). `mark_failed` sets is_down +
zeros counter + flags force_refresh. Cross-key dedup (two ClusterKeys
resolving to one uuid share state). `clear` / `aclear` drop all state and
close control connections.

**Control-connection re-open** (`test_registry.py`):
`_ensure_control_sync` returns the cached conn when present, opens a fresh
one against the first non-down node when not, skips `is_down` candidates,
falls over past hosts that refuse, returns None only when every candidate
refuses. Refusing hosts get marked failed so the dispatcher's next pick
doesn't waste a connect on them. `_claim_refresh_slot` honours the interval
window, force_refresh wins immediately, only one of N concurrent callers
wins per interval. End-to-end "control conn dies mid-fetch → next refresh
reopens against survivor" with all-internal mocks. `force_refresh` stays
flagged when no node accepts a fresh control conn (so the next caller
retries immediately, not after a full window).

**Policies** (`test_policy.py`):
Least-loaded pick of unique min, random tie-break statistical distribution
across 1000 picks, `attempted` set skip, down-node TTL semantics
(skipped within TTL, reconsidered after), zero TTL means always reconsider,
topology filter exact and wildcard, no-match returns None, multiple
topology keys union, single-pass invariant over `state.nodes.values()`.
`build_policy` selects topology-aware when keys are set, cluster-aware
otherwise.

### Integration tests (`yb`, 77 tests, real cluster)

Every bullet runs in **both sync and async** unless explicitly noted:

**Basic distribution & tagging** (`test_smart_driver*.py`):
* Pass-through paths (`load_balance_hosts=false`, absent) — registry never
  touched, connection not tagged with `_yb_uuid` / `_yb_host`
* Uniform load distribution, 12 conns over 3 nodes → exact 4/4/4 on the
  driver side, /rpcz cross-checked exactly (with control-conn accounting)
* `_yb_uuid` and `_yb_host` set on successful smart-driver connect
* Single-host bootstrap (one contact point) discovers the full cluster
* Topology exact match, wildcard zone (`cloud.region.*`), no-match raises
  `OperationalError`, invalid cloud wildcard rejected at conninfo-parse
  time as `ValueError`
* `close()` decrements counter; pass-through `close()` does not touch the
  registry
* `/rpcz` per-host == driver counter (after accounting for control conns)
* Concurrent connect (30 threads / async `gather`) distributes exactly
  10/10/10 — proves the atomic reserve+increment under the per-cluster
  lock holds under contention

**Failover** (`test_smart_driver_failover*.py`):
* Stopped node gets quarantined via `mark_failed` after first failed pick
* Contact host (bootstrap host) can die without breaking new connects —
  the cached cluster state and `mark_failed`-driven quarantine carry the
  rest
* `yb_servers_refresh_interval` honoured (smoke-tested by setting a 2 s
  interval and observing the driver picks up state changes inside seconds,
  not the 300 s default)
* Control-host failover — kill the bootstrap/control host, next connect
  re-opens control against a survivor via `_ensure_control_sync` /
  `_aensure_control_async`; the registry's ClusterState survives intact
* Uniform load after node addition — phase 1 distributes 3/3/3, add a 4th
  node, refresh fires, phase-2 picks distribute the catch-up share to the
  new node and then split the tie-breaks deterministically (sorted shape
  `[4, 4, 5, 5]`)
* **Control re-open + node addition** — combined scenario: 12 phase-1
  conns at 4/4/4 stay open, add a 4th node, stop the control host, wait
  for `yb_servers()` to see the new node, then open 12 phase-2 conns.
  The very first phase-2 connect drives single-cycle recovery: cached
  control conn dies mid-fetch → drop → re-open on a survivor → fetch new
  topology → merge. Resulting per-host distribution is asserted exactly:
  dead host = 0 (zeroed by mark_failed), the 3 healthy hosts have sorted
  shape `[6, 7, 7]` (derived in the test docstring by walking the
  least-loaded picker step-by-step)
* `ClusterRegistry.clear()` (sync) and `aclear()` (async) drop all state
  and close their respective control connections

**Async-only scenarios** (no sync analog because the failure modes don't
exist on the sync path):
* Cancellation mid-connect via `asyncio.Task.cancel()` does not leak the
  per-host counter reservation
* Mixed sync + async callers in the same process share one process-global
  `ClusterRegistry` and one set of per-host counters

**Topology × node-state matrix** (`test_smart_driver_topology_failover*.py`,
`yb_multi_zone_cluster`):
* Add node IN topology zone → it attracts new traffic, sorted shape
  derived deterministically
* Add node OUTSIDE topology zone → zero traffic to it (topology filter
  is strict, no cluster-wide fallback in v1)
* Stop node IN topology → traffic stays within topology even when only
  one in-topology node survives
* Stop node OUTSIDE topology → in-topology traffic unaffected, no detour
* Restart stopped node IN topology → re-enters rotation after the
  failed-host TTL, sorted shape derived deterministically

**ConnectionPool steady-state** (`test_smart_driver_pool*.py`, every bullet
in both sync (`ConnectionPool`) and async (`AsyncConnectionPool`)):
* `load_balance_hosts=false` pool — registry never touched, conns not
  tagged with `_yb_uuid`
* 12-conn pool with `load_balance_hosts=true` distributes exact 4/4/4 on
  the 3-node cluster (driver counter + `/rpcz` cross-check)
* Topology-aware pool (`topology_keys=zoneA`) on the multi-zone cluster
  distributes exact 6/6/0 — proves the topology filter runs underneath
  the pool path too
* Topology-no-match → `OperationalError` raised through the pool's
  startup
* Borrow + return doesn't mutate the per-host counter (the counter
  tracks open-on-server conns, not in-flight checkouts)
* Pool close drains every per-host count back to 0 (each managed conn's
  close path runs the smart driver's decrement)
* Pool growth from `min_size=3` to `max_size=12` under concurrent
  borrowers ends at exact 4/4/4 — the dispatcher's atomic reserve holds
  across the pool's growth path
* A pool-managed conn and a direct `psycopg.connect()` to the same
  cluster share the process-global registry

**ConnectionPool × failover matrix** (`test_smart_driver_pool_failover*.py`,
every bullet sync + async):
* Cluster-aware + stop_node → 10-conn pool over 3 nodes lands 5/0/5 on
  the survivors; the stopped node is `mark_failed`-quarantined on first
  failed pick and the rest go to the two live hosts
* Cluster-aware + add_node → pool₁(9 conns) + add node 4 + pool₂(9 conns)
  ends at sorted shape `[4, 4, 5, 5]` across all four hosts — same
  deterministic walk as the direct-connect node-addition test, through
  the pool path
* Topology + stop a zoneA node → all 6 pool conns land on the surviving
  zoneA node; stopped zoneA at 0 (quarantined), zoneB at 0 (filtered)
* Topology + stop the zoneB node → topology traffic on zoneA unaffected,
  6 pool conns split exactly 3/3 on zoneA; stopped zoneB stays at 0
* Topology + add a zoneA node → new zoneA node attracts its share, 9
  pool conns split exact 3/3/0/3
* Topology + add a zoneB node → new out-of-topology node receives zero
  traffic, 6 pool conns split exact 3/3/0/0

## Perf benchmark

Sequential install/uninstall comparison of upstream `psycopg` vs our fork,
in a single venv. **The two distributions cannot coexist** because both
write to `site-packages/psycopg/` (our distribution is renamed but the
import name is the same — we chose this so existing upstream users can
swap distributions without code changes). The benchmark script handles the
sequencing.

### How the script works

`tests/yb/perf/run_perf.sh`:

1. Editably install `psycopg-yugabytedb` from the fork's `psycopg/` source
   directory.
2. Run the bench script with `load_balance_hosts=true` (the "yb-on"
   scenario — exercises the full smart-driver path: bootstrap, registry,
   policy pick, per-host kwarg rewrite).
3. Run again with `load_balance_hosts=false` ("yb-off" — exercises our
   dispatcher's early-exit branch that hands off to `_aconnect_plain`,
   identical to upstream's path apart from one branch).
4. Uninstall `psycopg-yugabytedb`. Install `psycopg==3.3.4` from PyPI.
5. Run the bench script a third time, no `load_balance_hosts` parameter
   ("upstream" — pure upstream, baseline).
6. Print a comparison table with p50/p90/p99/mean and the percent delta of
   each fork scenario vs upstream.
7. `trap cleanup_and_restore EXIT` reinstalls the fork editably so the
   developer's environment is left as it was, even if the script fails
   mid-run.

The bench script itself (`perf/bench.py`) is **driver-agnostic** — it
imports `psycopg` (whichever is installed under that name), runs N warmup
connect/close cycles, then times N more. The smart-driver path has
one-time bootstrap costs on the very first connect (`yb_servers()` query +
ClusterState population + control conn open); warmup absorbs those so the
measured distribution is the steady-state cost per connect.

### Running

The script does not manage the cluster. Start one yourself, point the
script at it, tear it down when done.

```bash
# 1. Start a cluster (any shape; the bench is single-connection)
/path/to/yb-ctl create --rf 3 --placement_info cloud1.datacenter1.rack1

# 2. Run the benchmark. Multi-host DSN is what production code would use
#    with the smart driver, so use it here too.
PY=/path/to/python3 \
    PSYCOPG_YB_TEST_DSN="host=127.0.0.1,127.0.0.2,127.0.0.3 port=5433 user=yugabyte dbname=yugabyte" \
    N=200 \
    bash tests/yb/perf/run_perf.sh

# 3. Tear down
/path/to/yb-ctl destroy
```

Environment knobs:

| Var          | Default | What it controls                                |
|--------------|---------|-------------------------------------------------|
| `PSYCOPG_YB_TEST_DSN` | `host=127.0.0.1 port=5433 user=yugabyte dbname=yugabyte` | DSN passed to every scenario |
| `N`          | `100`   | Connect/close cycles per scenario (after a fixed 10-cycle warmup) |
| `PY`         | `python3` | Interpreter used for pip install/uninstall and the bench |
| `UPSTREAM_V` | `3.3.4` | Upstream psycopg version installed for the baseline scenario |

### Sample output

```
scenario       driver                   p50 ms   p90 ms   p99 ms   mean ms
---------------------------------------------------------------------------
upstream       psycopg 3.3.4             15.46    22.25    33.90     16.74
yb-off         psycopg 3.3.4.1rc1        14.98    19.70    25.07     15.70
yb-on          psycopg 3.3.4.1rc1        15.64    22.32    31.94     17.13

yb-off vs upstream: -3.1% (p50)
yb-on  vs upstream: +1.2% (p50)
```

### Budgets

The script reports raw deltas; the design plan's budgets (not enforced
automatically) are:

* **Pass-through (yb-off) within 5% of upstream.** Anything worse means the
  dispatcher's early-exit branch is doing real work it shouldn't.
* **Smart driver (yb-on) within 10% of upstream.** This covers registry
  lookup, the per-cluster lock, policy pick + tie-break, and the per-host
  kwarg rewrite — all amortised across the connect/close cycle.

If a perf regression pushes either number outside its budget, investigate
before merging.
