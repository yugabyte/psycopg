YugabyteDB Psycopg 3: Database adapter for Python
=================================================

Psycopg 3 is a modern implementation of a PostgreSQL adapter for Python.
This is the YugabyteDB smart-driver fork, distributed as ``psycopg-yugabytedb``.

This distribution contains the pure Python package ``psycopg``.

.. Note::

    Despite the lack of number in the package name, this package is the
    successor of psycopg2_.

    Please use the psycopg2 package if you are maintaining an existing program
    using psycopg2 as a dependency. If you are developing something new,
    Psycopg 3 is the most current implementation of the adapter.

    .. _psycopg2: https://pypi.org/project/psycopg2/


YugabyteDB smart driver
-----------------------

This fork (distribution name ``psycopg-yugabytedb``) extends psycopg 3 with
cluster-aware and topology-aware connection load balancing for `YugabyteDB`_.
The import name stays ``psycopg`` — existing code is unchanged. Adding
``load_balance_hosts=true`` to the conninfo string opts into the smart driver.

.. _YugabyteDB: https://www.yugabyte.com/

Opt in with one parameter:

.. code-block:: python

    import psycopg

    conn = psycopg.connect(
        "host=h1,h2,h3 port=5433 user=yugabyte dbname=yugabyte "
        "load_balance_hosts=true"
    )

The driver discovers every live tserver via ``yb_servers()`` on the first
connect, then distributes subsequent connects across them with a least-loaded
picker (random tie-break, atomic under a per-cluster lock). One contact point
in ``host=…`` is enough to bootstrap; the rest of the cluster is discovered.

Smart-driver conninfo parameters:

.. list-table::
   :header-rows: 1
   :widths: 30 14 12 44

   * - Parameter
     - Values
     - Default
     - Description
   * - ``load_balance_hosts``
     - ``true`` / ``false`` /
       ``disable`` / ``random``
     - absent
     - ``true`` enables the smart driver. ``false`` is equivalent to libpq's
       ``disable``. ``disable`` / ``random`` pass through to libpq unchanged.
   * - ``topology_keys``
     - ``cloud.region.zone``
       (comma-separated for
       multiple keys)
     - none
     - Restrict picks to tservers matching at least one placement. Zone may
       be ``*`` for any zone in that cloud/region. Cloud / region wildcards
       are rejected. Strict in v1 — no cluster-wide fallback if all
       matching nodes are down.
   * - ``yb_servers_refresh_interval``
     - seconds (integer)
     - ``300``
     - How often to re-query ``yb_servers()``. Clamped to ``[0, 600]``.
   * - ``failed_host_reconnect_delay_secs``
     - seconds (integer)
     - ``5``
     - How long to quarantine a node after a failed connect, before
       reconsidering it. Clamped to ``[0, 60]``.

Topology-aware example — bind traffic to one zone:

.. code-block:: python

    conn = psycopg.connect(
        "host=h1,h2,h3 port=5433 user=yugabyte dbname=yugabyte "
        "load_balance_hosts=true "
        "topology_keys=cloud1.datacenter1.zoneA"
    )

The upstream ``psycopg-pool`` connection pool works unchanged with the smart
driver — the dispatcher sits underneath the pool, so pool-managed connections
honour the configured policy too. Install it via the ``[pool]`` extra:

.. code-block:: bash

    pip install "psycopg-yugabytedb[pool]"

That pulls our driver plus the unmodified upstream ``psycopg-pool``; the
pool's ``import psycopg`` resolves to our driver and every conn the pool
opens goes through the dispatcher.

Install without the ``[pool]`` extra with ``pip install psycopg-yugabytedb``,
or pin to a specific version (``3.3.4.1`` is the first GA release). The fork
cannot coexist with upstream ``psycopg`` in the same environment — both
install into ``site-packages/psycopg/``.

xCluster failover
~~~~~~~~~~~~~~~~~

The driver can be configured with a secondary YugabyteDB cluster to fail
over to when the primary becomes unusable. Opt-in is gated on BOTH
``load_balance_hosts=true`` AND a non-empty
``yb.failover.secondaryClusterHosts`` — when those conditions hold, the
driver eagerly bootstraps both clusters, starts one background probe
thread per cluster, and routes new connections through whichever cluster
the dual-CB state machine picks. ``psycopg-pool``'s ``check=`` callback
evicts stale connections on failover via the bundled ``xcluster_check``
helper.

Architecture at a glance:

* Two ``CircuitBreaker`` instances per ``FailoverGroup`` — one per
  cluster. Each is polled independently by its own probe thread.
* State machine: primary HEALTHY → serve primary; primary UNHEALTHY +
  secondary HEALTHY → serve secondary; both UNHEALTHY → raise
  ``NoViableClusterError``.
* On a routing transition, the driver runs a **barrier-with-timeout
  drain**: new connects are paused, in-flight transactions on the
  outgoing cluster get up to ``drainTimeoutSecs`` to commit or abort,
  survivors are force-closed at the deadline, then dispatch resumes
  against the new active cluster.

Configuration (libpq conninfo keys; ``.``, ``_``, and ``-`` are all
accepted as separators between tokens):

.. list-table::
   :header-rows: 1
   :widths: 38 15 47

   * - Parameter
     - Default
     - Description
   * - ``yb.failover.secondaryClusterHosts``
     - (empty)
     - Comma-separated secondary-cluster host list. One host is enough —
       the rest of the cluster is discovered via ``yb_servers()``.
       Setting this is the opt-in trigger for xCluster failover.
   * - ``yb.failover.trackerTableTablets``
     - ``9``
     - Tablet count for the default tracker-table CB's DDL. Each tablet
       gets one seed row; the CB's UPDATE touches all rows, exercising
       every tablet leader on each tick.
   * - ``yb.failover.maxUpdateFailuresAllowed``
     - ``0``
     - Consecutive UPDATE failures tolerated on the tracker-table CB
       before it reports UNHEALTHY. Symmetric hysteresis — same
       threshold governs the UNHEALTHY→HEALTHY recovery direction.
   * - ``yb.failover.cooldownSecs``
     - ``1500``
     - Minimum interval between status transitions in either direction
       on a single CB. Prevents rapid ping-pong when a cluster flaps.
   * - ``yb.failover.checkTimeoutSecs``
     - ``max(1, refresh/2)``
     - Wall-clock cap on each ``CircuitBreaker.check()`` call. Ticks
       that exceed the cap are abandoned; the previous status is
       preserved. Prevents a blocking custom CB from stalling failover.
   * - ``yb.failover.drainTimeoutSecs``
     - ``10``
     - Barrier-with-timeout drain behaviour. Sentinel values:
       ``-1`` = wait indefinitely, never force-close;
       ``0`` = force-close in-flight transactions immediately;
       ``N > 0`` = wait up to N seconds then force-close survivors.

Example (application):

.. code-block:: python

    import psycopg

    conn = psycopg.connect(
        "host=primary1,primary2,primary3 port=5433 user=yugabyte dbname=yugabyte "
        "load_balance_hosts=true "
        "yb.failover.secondaryClusterHosts=secondary1,secondary2,secondary3 "
        "yb.failover.drainTimeoutSecs=10 "
        # Recommended: bound in-flight queries to the drain window (see below):
        "options='-c statement_timeout=10000'"
    )

With ``psycopg-pool``:

.. code-block:: python

    from psycopg.yb.pool import xcluster_check
    from psycopg_pool import ConnectionPool

    pool = ConnectionPool(
        "host=primary1,primary2,primary3 load_balance_hosts=true "
        "yb.failover.secondaryClusterHosts=secondary1,secondary2,secondary3 "
        "yb.failover.drainTimeoutSecs=10 options='-c statement_timeout=10000'",
        check=xcluster_check,
        min_size=4, max_size=20,
    )

Pluggable circuit breaker
^^^^^^^^^^^^^^^^^^^^^^^^^

Health signals enter the driver through a ``CircuitBreaker`` object —
any class with ``check(group) -> HealthResult`` satisfies the contract.
Attach a custom one at startup, per-cluster:

.. code-block:: python

    from psycopg.yb import bootstrap_failover_group
    from psycopg.yb.circuit_breaker import ExternalSignalCircuitBreaker

    group = bootstrap_failover_group(dsn)
    group.primary_circuit_breaker   = ExternalSignalCircuitBreaker(which_cluster="primary")
    group.secondary_circuit_breaker = ExternalSignalCircuitBreaker(which_cluster="secondary")

Provided implementations:

* ``TrackerTableCircuitBreaker`` — default. UPDATE-based probe over
  ``yb_cluster_health_tracker``. Tablet-splits the table so the UPDATE
  touches every leader, catching partial-cluster failures.
* ``AlwaysHealthyCircuitBreaker`` — inert; for tests.
* ``ExternalSignalCircuitBreaker`` — reads ``target_status`` from a
  well-known signal table. Operator writes a row to trigger a manual
  failover / failback:

  .. code-block:: sql

      INSERT INTO yb_failover_signals (group_id, target_status, reason)
      VALUES ('<primary-uuid>', 'UNHEALTHY', 'planned maintenance');

Manual failover from Python (test / migration):

.. code-block:: python

    from psycopg.yb.health import HealthResult
    from psycopg.yb.registry import ClusterRegistry

    group = ClusterRegistry.instance().get_failover_group(primary_uuid)
    group.force_primary_status(HealthResult.UNHEALTHY)   # → serve secondary
    # ...
    ClusterRegistry.instance().reset_failover_group(primary_uuid)  # failback

Server ``statement_timeout`` coordination
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

YugabyteDB does not cancel a running query when its client socket
closes (see `yugabyte-db#28983
<https://github.com/yugabyte/yugabyte-db/issues/28983>`_ and
`#29379 <https://github.com/yugabyte/yugabyte-db/issues/29379>`_). For
the drain's force-close to actually stop server-side execution, the
server-side ``statement_timeout`` must be no larger than the drain
window. Otherwise a session force-closed by the driver can still commit
its in-flight work AFTER the driver has flipped the dispatcher —
replicated to the other cluster by xCluster, that's a silent dual-write.

The driver **does not modify** ``statement_timeout``. It reads the GUC
once at bootstrap and emits a WARNING at ``psycopg.yb`` if it's
unbounded (``0``) or larger than ``drainTimeoutSecs``. Set it at the
DSN (``options='-c statement_timeout=<ms>'``) or role level.

Per-process semantics
^^^^^^^^^^^^^^^^^^^^^

Each Python process maintains its own ``ClusterRegistry`` and therefore
its own xCluster status flags. In a multi-worker deployment (gunicorn,
``ProcessPoolExecutor``, Celery), workers independently observe failures
and independently decide to fail over. Brief divergence during the
detection window is expected and accepted for v1.

Logging
~~~~~~~

Every smart-driver module logs to its own logger under ``psycopg.yb.*``.
Levels follow the standard library, plus a custom ``TRACE`` level (numeric
5, finer than ``DEBUG``) for counter-mutation-level firehose detail.

* ``WARNING`` — driver gave up (no eligible nodes, control conn re-open failed)
* ``INFO`` — cluster bootstrap, topology change observed, host quarantined,
  control conn re-opened on a survivor
* ``DEBUG`` — per-pick, per-refresh, per-control-conn open/close
* ``TRACE`` — every counter increment / decrement, every filtered candidate

Enable:

.. code-block:: python

    import logging
    from psycopg.yb import TRACE

    logging.basicConfig(level=logging.INFO)

    # Lifecycle events only (default INFO above is fine):
    logging.getLogger("psycopg.yb").setLevel(logging.INFO)

    # Per-operation debug:
    logging.getLogger("psycopg.yb").setLevel(logging.DEBUG)

    # Full firehose:
    logging.getLogger("psycopg.yb").setLevel(TRACE)

    # Or narrow to one subsystem:
    logging.getLogger("psycopg.yb.policy").setLevel(logging.DEBUG)


Installation
------------

In short, run the following::

    pip install --upgrade pip           # to upgrade pip
    pip install "psycopg[binary,pool]"  # to install package and dependencies

If something goes wrong, and for more information about installation, please
check out the `Installation documentation`__.

.. __: https://www.psycopg.org/psycopg3/docs/basic/install.html#


Hacking
-------

For development information check out `the project readme`__.

.. __: https://github.com/psycopg/psycopg#readme


Copyright (C) 2020 The Psycopg Team
