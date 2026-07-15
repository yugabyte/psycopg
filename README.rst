YugabyteDB Psycopg 3 -- Database adapter for Python
===================================================

Psycopg 3 is a modern implementation of a PostgreSQL adapter for Python.
This is the YugabyteDB smart-driver fork, distributed as ``psycopg-yugabytedb``.


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

Provided implementations: ``TrackerTableCircuitBreaker`` (default;
UPDATE-based probe), ``AlwaysHealthyCircuitBreaker`` (tests),
``ExternalSignalCircuitBreaker`` (operator-controlled — reads
``target_status`` from a well-known ``yb_failover_signals`` table).

Server ``statement_timeout`` coordination
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

YugabyteDB does not cancel a running query when its client socket
closes (see `yugabyte-db#28983
<https://github.com/yugabyte/yugabyte-db/issues/28983>`_ and
`#29379 <https://github.com/yugabyte/yugabyte-db/issues/29379>`_). For
the drain's force-close to actually stop server-side execution, the
server-side ``statement_timeout`` must be no larger than the drain
window. The driver does **not** modify ``statement_timeout``; it reads
the GUC once at bootstrap and emits a WARNING at ``psycopg.yb`` if it's
unbounded (``0``) or larger than ``drainTimeoutSecs``. Set it at the
DSN (``options='-c statement_timeout=<ms>'``) or role level.

Per-process semantics
^^^^^^^^^^^^^^^^^^^^^

Each Python process maintains its own ``ClusterRegistry`` and therefore
its own xCluster status flags. In a multi-worker deployment (gunicorn,
``ProcessPoolExecutor``, Celery), workers independently observe failures
and independently decide to fail over. Brief divergence during the
detection window is expected and accepted for v1 (see design doc §11).

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

Quick version::

    pip install "psycopg[binary,pool]"

For further information about installation please check `the documentation`__.

.. __: https://www.psycopg.org/psycopg3/docs/basic/install.html


.. _Hacking:

Hacking
-------

In order to work on the Psycopg source code, you must have the
``libpq`` PostgreSQL client library installed on the system. For instance, on
Debian systems, you can obtain it by running::

    sudo apt install libpq5

On macOS, run::

    brew install libpq

On Windows you can use EnterpriseDB's `installers`__ to obtain ``libpq``
which is included in the Command Line Tools.

.. __: https://www.enterprisedb.com/downloads/postgres-postgresql-downloads

You can then clone this repository to develop Psycopg::

    git clone https://github.com/psycopg/psycopg.git
    cd psycopg

Please note that the repository contains the source code of several Python
packages, which may have different requirements:

- The ``psycopg`` directory contains the pure python implementation of
  ``psycopg``. The package has only a runtime dependency on the ``libpq``, the
  PostgreSQL client library, which should be installed in your system.

- The ``psycopg_c`` directory contains an optimization module written in
  C/Cython. In order to build it you will need a few development tools: please
  look at `Local installation`__ in the docs for the details.

- The ``psycopg_pool`` directory contains the `connection pools`__
  implementations. This is kept as a separate package to allow a different
  release cycle.

.. __: https://www.psycopg.org/psycopg3/docs/basic/install.html#local-installation
.. __: https://www.psycopg.org/psycopg3/docs/advanced/pool.html

You can create a local virtualenv and install the packages `in
development mode`__, together with their development and testing
requirements::

    python -m venv .venv
    source .venv/bin/activate

    # Install the base Psycopg package in editable mode
    pip install --config-settings editable_mode=strict -e "./psycopg[dev,test]"

    # Install the connection pool package in editable mode
    pip install --config-settings editable_mode=strict -e ./psycopg_pool

    # Install the C speedup extension
    pip install ./psycopg_c

.. __: https://pip.pypa.io/en/stable/topics/local-project-installs/#editable-installs

The ``--config-settings editable_mode=strict`` will be probably required
to work around the problem of the `editable mode broken`__.

.. __: https://github.com/pypa/setuptools/issues/3557

Now hack away! You can run the tests using::

    psql -c 'create database psycopg_test'
    export PSYCOPG_TEST_DSN="dbname=psycopg_test"
    pytest

The project includes some `pre-commit`__ hooks to check that the code is valid
according to the project coding convention. Please make sure to install them
by running::

    pre-commit install

This will allow to check lint errors before submitting merge requests, which
will save you time and frustrations.

.. __: https://pre-commit.com/


Cross-compiling
---------------

To use cross-platform zipapps created with `shiv`__ that include Psycopg
as a dependency you must also have ``libpq`` installed. See
`the section above <Hacking_>`_ for install instructions.

.. __: https://github.com/linkedin/shiv
