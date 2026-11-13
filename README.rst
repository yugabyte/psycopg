YugabyteDB Psycopg 3 -- Database adapter for Python
===================================================

Psycopg 3 is a modern implementation of a PostgreSQL adapter for Python.
This is the YugabyteDB smart-driver fork, distributed as ``psycopg-yugabytedb``.
It adds two features on top of psycopg 3: cluster-aware **load balancing**
and **xCluster failover** between two `YugabyteDB`_ universes. The import
name stays ``psycopg``, so existing code is unchanged.

.. _YugabyteDB: https://www.yugabyte.com/

Installation
------------

.. code-block:: bash

    pip install "psycopg-yugabytedb[binary]"    # C speedups, prebuilt with libpq bundled
    pip install psycopg-yugabytedb              # pure Python, uses the system libpq
    pip install "psycopg-yugabytedb[pool]"      # adds the upstream psycopg-pool
    pip install "psycopg-yugabytedb[c]"         # C speedups, built from source
    pip install psycopg-yugabytedb==3.3.4.2     # pin a release

The ``[binary]`` extra installs a prebuilt C extension with ``libpq``
bundled, so nothing else is needed on the machine. Binary wheels are
published for Linux x86_64 and macOS on Apple Silicon (arm64), for
CPython 3.10 to 3.14. Elsewhere, install the plain package, which needs
the ``libpq`` client library on the system (``apt install libpq5``,
``brew install libpq``). The ``[c]`` extra compiles against it, so it also
needs a C compiler and the ``libpq`` headers; see `Local installation`__
in the psycopg docs.

.. __: https://www.psycopg.org/psycopg3/docs/basic/install.html#local-installation

The fork can't be installed next to upstream ``psycopg`` in the same
environment: both install into ``site-packages/psycopg/``. The ``[pool]``
extra pulls the unmodified upstream ``psycopg-pool``; its ``import psycopg``
resolves to this driver, so pooled connections get the same behaviour.


Load balancing
--------------

The driver can spread connections across the tservers of a YugabyteDB
cluster in two ways:

* **Cluster-aware**: across every tserver in the cluster.
* **Topology-aware**: only across tservers in the placements (cloud,
  region, zone) you list.

Both start the same way. On the first connect the driver asks the cluster
for its live tservers (``yb_servers()``). After that, each new connection
goes to the eligible tserver with the fewest connections from this process
(ties broken at random). One reachable host in ``host=`` is enough; the
rest are discovered, and the list is refreshed every
``yb_servers_refresh_interval`` seconds.

``psycopg-pool`` works unchanged with either: the pool's connections go
through the same load balancer.

Cluster-aware load balancing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Add ``load_balance_hosts=true`` to the connection string:

.. code-block:: python

    import psycopg

    conn = psycopg.connect(
        "host=h1,h2,h3 port=5433 user=yugabyte dbname=yugabyte "
        "load_balance_hosts=true"
    )

Connections go to every live tserver in the cluster.

Topology-aware load balancing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Add ``topology_keys`` as well, to keep connections in one or more zones or
regions, for example the ones closest to the application:

.. code-block:: python

    conn = psycopg.connect(
        "host=h1,h2,h3 port=5433 user=yugabyte dbname=yugabyte "
        "load_balance_hosts=true "
        "topology_keys=cloud1.datacenter1.zoneA"
    )

Each key is ``cloud.region.zone``. Use ``*`` for the zone to allow every
zone in a region (``cloud1.datacenter1.*``), and separate several keys
with commas (``cloud1.datacenter1.zoneA,cloud1.datacenter1.zoneB``).
Connections only go to tservers that match a key. If none of them are
reachable, connecting fails; the driver doesn't fall back to the rest of
the cluster.

Load-balancing parameters:

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
     - ``true`` enables the smart driver. ``false`` is the same as libpq's
       ``disable``. ``disable`` and ``random`` pass through to libpq
       unchanged.
   * - ``topology_keys``
     - ``cloud.region.zone``
       (comma-separated for
       multiple keys)
     - none
     - Only connect to tservers in one of these placements. The zone may
       be ``*`` for any zone in that cloud and region; wildcards for the
       cloud or region are rejected. There is no fallback to the rest of
       the cluster if every matching node is down.
   * - ``yb_servers_refresh_interval``
     - seconds (integer)
     - ``300``
     - How often to re-read ``yb_servers()``. Clamped to ``[0, 600]``.
       With xCluster failover this is also how often the health checks
       run (see below).
   * - ``failed_host_reconnect_delay_secs``
     - seconds (integer)
     - ``5``
     - How long to skip a node after a failed connect. Clamped to
       ``[0, 60]``.


xCluster failover
-----------------

xCluster failover lets an application keep running when a YugabyteDB
universe becomes unavailable. It is meant for two universes that already
replicate to each other with xCluster: a **primary** universe that the
application normally uses, and a **secondary** universe.

* **Failover.** When the primary becomes unavailable, the driver sends
  new connections to the secondary.
* **Failback.** When the primary is available again, the driver moves
  connections back to it.

The driver doesn't decide by itself whether a universe is available. The
application provides a **circuit breaker** for each universe: an object
that reports whether that universe is healthy, based on whatever signal
the application trusts (a probe query, a monitoring system, an operator).
The driver checks both circuit breakers periodically in the background.

The two universes, their circuit breakers and the current routing state
together make up a **failover group**. A process has one failover group
for each primary universe. Every connection to that universe belongs to
the group, even with a different database or user, so they all switch
universes together.

A switch from one universe to the other happens in two phases:

* **Drain.** The driver stops opening new connections and gives the
  transactions already running on the current universe time to finish.
  Connections that are still open when the time is up are closed, so no
  more writes reach the universe being left.
* **Replication-lag wait.** If YugabyteDB Anywhere (YBA) is configured,
  the driver checks the xCluster replication lag and waits until the
  other universe has received the writes made before the drain. Without
  YBA, this phase is skipped.

New connections then go to the other universe. If the replication-lag
wait runs out of time, a failover goes ahead anyway, since the primary is
unavailable. A failback is postponed and tried again after a cooldown, so
the application isn't moved back to a primary that is still missing
recent writes.

If both universes are unhealthy, new connections fail with
``NoViableClusterError``.

Processes that each create their own failover group switch independently.
In a pre-fork server (gunicorn ``--preload``, Celery prefork), the process
that created the group runs the health checks and makes the decision;
forked workers follow its switches, pausing, draining their own
connections and resuming with it. A worker starts its own health checks
only if that process exits.

Usage
~~~~~

These connection parameters enable xCluster failover and control its
main behaviour:

* ``host`` - the hosts of the **primary** universe, as for any
  connection. One reachable host is enough; the rest are discovered.
* ``load_balance_hosts=true`` - required; xCluster failover is part of
  the smart driver.
* ``yb.failover.secondaryClusterHosts`` - comma-separated hosts of the
  **secondary** universe. Setting it turns on xCluster failover.
* ``yb_servers_refresh_interval`` - how often, in seconds, the circuit
  breakers are checked (default ``300``). Set it to a few seconds for
  timely detection.
* ``yb.failover.drainTimeoutSecs`` - how long running transactions get to
  finish during the drain (default ``10``). ``-1`` waits for all of them;
  ``0`` closes connections immediately.
* ``yb.failover.lagWaitTimeoutSecs`` - how long the replication-lag wait
  may take (default ``30``).
* ``yb.failover.globalTimeBudgetSecs`` - an upper limit on the whole
  switch (unset by default). When set, it is split between the drain
  (40%) and the replication-lag wait (60%), unless those are set
  explicitly.

For example, an application whose primary universe has the nodes ``a1``,
``a2`` and ``a3`` and whose secondary has ``b1``, ``b2`` and ``b3``:

.. code-block:: python

    DSN = (
        "host=a1,a2,a3 port=5433 user=yugabyte dbname=yugabyte "
        "load_balance_hosts=true "
        "yb.failover.secondaryClusterHosts=b1,b2,b3 "
        "yb_servers_refresh_interval=5 "
        "yb.failover.drainTimeoutSecs=10 "
        "options='-c statement_timeout=10000'"
    )

With these settings the circuit breakers are checked every 5 seconds,
and running transactions get up to 10 seconds to finish before a switch.
The ``statement_timeout`` keeps queries within the drain window (see
`statement_timeout`_ below).

To enable the replication-lag wait, add the YBA endpoint and API token,
and the names of the two xCluster replication configs in YBA: the one
from the primary to the secondary (used on failover) and the one from the
secondary to the primary (used on failback):

.. code-block:: python

    DSN += (
        " yb.failover.ybaEndpoint=https://yba.example.com"
        " yb.failover.ybaApiToken=<YBA API token>"
        " yb.failover.replicationName=primary-to-secondary"
        " yb.failback.replicationName=secondary-to-primary"
        " yb.failover.lagWaitTimeoutSecs=30"
    )

Instead of setting the drain and lag-wait timeouts separately, you can set
one limit for the whole switch. ``yb.failover.globalTimeBudgetSecs=60``
gives a 24-second drain and a 36-second replication-lag wait.

Next, create the failover group with ``bootstrap_failover_group(DSN)``
(``abootstrap_failover_group`` in asyncio applications). Call it once
when the application starts: it connects to both universes, discovers
their nodes, starts the background health checks, and returns the group.
Then attach a circuit breaker for each universe to the group before
opening the first connection; connecting without them raises
``MissingCircuitBreakerError``. Later ``psycopg.connect(DSN)`` calls, and
pools created with the same DSN, use this group.

.. code-block:: python

    import psycopg
    from psycopg.yb import bootstrap_failover_group
    from psycopg.yb.health import HealthResult

    class MyCircuitBreaker:
        def __init__(self, which_cluster):
            self.which_cluster = which_cluster   # "primary" or "secondary"

        def check(self, group):
            # Ask your monitoring whether this universe is usable.
            if universe_is_down(self.which_cluster):
                return HealthResult.UNHEALTHY
            return HealthResult.HEALTHY

    group = bootstrap_failover_group(DSN)
    group.primary_circuit_breaker = MyCircuitBreaker("primary")
    group.secondary_circuit_breaker = MyCircuitBreaker("secondary")

    conn = psycopg.connect(DSN)

Every connection string that points at the same primary universe must
use the same failover settings; if one doesn't, ``bootstrap_failover_group`` raises
``ConflictingFailoverConfigError`` naming the settings that differ. Use a
separate circuit breaker instance for each universe. A ``check()``
call that takes longer than ``yb.failover.checkTimeoutSecs`` is ignored
and the previous status is kept.

Ready-made circuit breakers are in ``demo/samples/`` in this repository.
They aren't part of the installed package; copy the one you want.

* ``TrackerTableCircuitBreaker`` - reads a small table spread over every
  tablet server and reports ``UNHEALTHY`` when it can't.
* ``ExternalSignalCircuitBreaker`` - an operator inserts a row into a
  ``yb_failover_signals`` table to fail over or fail back on demand.
* ``HealthAndLagCircuitBreaker`` - a health check combined with a YBA
  lag check, for setups that don't configure the driver's own
  replication-lag wait.

With ``psycopg-pool``, pass ``xcluster_check`` so the pool drops
connections to the old universe after a switch:

.. code-block:: python

    from psycopg.yb.pool import xcluster_check   # xcluster_check_async for AsyncConnectionPool
    from psycopg_pool import ConnectionPool

    pool = ConnectionPool(DSN, check=xcluster_check, min_size=4, max_size=20)

With ``yb.failover.autoFailbackEnabled=false``, the driver stays on the
secondary after the primary recovers, until you fail back yourself. The
manual failback runs the same drain and replication-lag wait as an
automatic one, and blocks until they finish:

.. code-block:: python

    from psycopg.yb.registry import ClusterRegistry

    registry = ClusterRegistry.instance()
    registry.reset_failover_group(group.primary.uuid)

    # (primary status, secondary status, last change times)
    print(registry.get_failover_status(group.primary.uuid))

Call ``reset_failover_group`` in the process that created the group. In a
forked worker of a pre-fork server it has no effect. It returns ``False``
if a switch is already in progress; call it again once that has finished.
Like an automatic failback, it is postponed if the replication-lag wait
runs out of time.

All other parameters are listed below.

Other connection parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~

All options go in the connection string. ``.``, ``_`` and ``-`` are all
accepted between words (``yb.failover.drainTimeoutSecs``,
``yb_failover_drain_timeout_secs``). The ``yb.failback.*`` options apply to
the failback direction; each one you leave out uses the matching
``yb.failover.*`` value, except ``yb.failback.replicationName``.

General:

.. list-table::
   :header-rows: 1
   :widths: 38 15 47

   * - Parameter
     - Default
     - Description
   * - ``yb.failover.secondaryClusterHosts``
     - (empty)
     - Comma-separated hosts of the secondary universe. One is enough;
       the rest are discovered. Setting this (with
       ``load_balance_hosts=true``) turns on xCluster failover.
   * - ``yb_servers_refresh_interval``
     - ``300``
     - Seconds between circuit-breaker checks (and ``yb_servers()``
       refreshes). Lower it, e.g. to ``5``, for faster detection.
       The checks run at most once a second.
   * - ``yb.failover.checkTimeoutSecs``
     - ``max(1, refresh/2)``
     - Time limit for one ``check()`` call. A check that runs longer is
       ignored and the previous status is kept.
   * - ``yb.failover.cooldownSecs``
     - ``1500``
     - Minimum time between two status changes of the same circuit
       breaker, so a flapping universe doesn't switch back and forth.
       This also delays a failback after a failover.
   * - ``yb.failover.autoFailbackEnabled``
     - ``true``
     - ``false`` = stay on the secondary when the primary recovers, until
       an operator fails back. While the secondary is down, the driver
       fails back to a healthy primary anyway.

Drain:

.. list-table::
   :header-rows: 1
   :widths: 38 15 47

   * - Parameter
     - Default
     - Description
   * - ``yb.failover.drainTimeoutSecs``
     - ``10``
     - How long transactions on the old universe get to finish before
       their connections are closed. ``-1`` = wait as long as it takes,
       never close. ``0`` = close at once.
   * - ``yb.failover.closeFenceSecs``
     - ``1``
     - After the drain closed connections, wait until this many seconds
       have passed since the last close before switching, so an app that
       holds session state (such as an advisory lock) notices the close
       first. Time spent in the lag check counts towards it. ``0`` turns
       it off.
   * - ``yb.failover.globalTimeBudgetSecs``
     - (unset)
     - Upper limit on a switch, from detection to the new universe.
       Splits into drain (40%) and lag check (60%) unless those are set
       explicitly; setting them to more than the budget is an error.
       Two exceptions: a failback under ``onPhase2Timeout=fail`` can run
       up to ~11 s over while it waits for YBA's next sample (logged as
       a WARNING), and ``closeFenceSecs`` isn't counted.
   * - ``yb.failback.drainTimeoutSecs``
     - failover value
     - Drain timeout for failbacks.
   * - ``yb.failback.globalTimeBudgetSecs``
     - failover value
     - Time budget for failbacks.

Replication-lag check (YBA):

.. list-table::
   :header-rows: 1
   :widths: 38 15 47

   * - Parameter
     - Default
     - Description
   * - ``yb.failover.ybaEndpoint``
     - (empty)
     - YBA base URL, e.g. ``https://yba.example.com``. The lag check runs
       only when this, ``ybaApiToken`` and ``replicationName`` are all
       set.
   * - ``yb.failover.ybaApiToken``
     - (empty)
     - YBA API token.
   * - ``yb.failover.replicationName``
     - (empty)
     - Name of the A→B xCluster replication config in YBA.
   * - ``yb.failback.replicationName``
     - (empty)
     - Name of the B→A config. No fallback: if unset, failbacks skip the
       lag check.
   * - ``yb.failover.ybaVerifySsl``
     - ``true``
     - Verify YBA's TLS certificate. If verification fails, the lag check
       is turned off (logged at ERROR): failovers switch without it, and
       under ``onPhase2Timeout=fail`` failbacks are refused before any
       connection is drained. Each failback attempt checks the
       certificate again, so a fixed certificate works without a restart.
   * - ``yb.failover.ybaCaFile``
     - (empty)
     - CA bundle for a YBA with a self-signed certificate.
   * - ``yb.failover.thresholdReplicationLagMs``
     - ``0``
     - Switch once the lag is at or below this many milliseconds.
   * - ``yb.failover.lagWaitTimeoutSecs``
     - ``30``
     - How long to wait for the lag to reach the threshold. On failover
       the driver switches anyway when it runs out.
   * - ``yb.failover.lagFreshnessSecs``
     - ``0`` (off)
     - Ignore YBA samples taken less than this many seconds after the
       drain finished. ``25`` covers the tserver's 15 s metric update plus
       YBA's 10 s scrape. Raise ``lagWaitTimeoutSecs`` to at least
       ``lagFreshnessSecs + lagConfirmationSamples × 10``; the driver
       warns at startup if it's lower.
   * - ``yb.failover.lagConfirmationSamples``
     - ``1``
     - Number of YBA samples, at least 10 s apart, that must show the lag
       at or below the threshold before switching.
   * - ``yb.failback.onPhase2Timeout``
     - ``fail``
     - What a failback does when the lag check times out or YBA can't be
       reached. ``fail`` = cancel the failback, stay on the secondary,
       and retry after ``cooldownSecs``. ``proceed`` = fail back anyway.
       While the secondary is down, the failback goes ahead regardless.
       Failover always goes ahead.
   * - ``yb.failback.thresholdReplicationLagMs``,
       ``yb.failback.lagWaitTimeoutSecs``,
       ``yb.failback.lagFreshnessSecs``,
       ``yb.failback.lagConfirmationSamples``
     - failover value
     - The same settings for the failback direction.

``statement_timeout``
~~~~~~~~~~~~~~~~~~~~~

YugabyteDB doesn't cancel a running query when its client connection is
closed (`yugabyte-db#28983
<https://github.com/yugabyte/yugabyte-db/issues/28983>`_, `#29379
<https://github.com/yugabyte/yugabyte-db/issues/29379>`_). For the drain
to actually stop work on the old universe, set the server's
``statement_timeout`` no higher than ``drainTimeoutSecs``, in the
connection string (``options='-c statement_timeout=<ms>'``) or on the
role. The driver doesn't change it; it logs a WARNING at startup if it is
unset (``0``) or larger than the drain timeout.

Logging
-------

Every smart-driver module logs under ``psycopg.yb.*``, with the standard
levels plus ``TRACE`` (5, below ``DEBUG``):

* ``WARNING``: the driver gave up on something (no eligible nodes, a
  control connection couldn't reopen)
* ``INFO``: cluster bootstrap, topology changes, quarantined hosts,
  failover and failback steps
* ``DEBUG``: each pick, refresh and control-connection open or close
* ``TRACE``: every connection-counter change and filtered candidate

.. code-block:: python

    import logging
    from psycopg.yb import TRACE

    logging.basicConfig(level=logging.INFO)
    logging.getLogger("psycopg.yb").setLevel(logging.DEBUG)   # or TRACE
    logging.getLogger("psycopg.yb.policy").setLevel(logging.DEBUG)  # one module


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

    git clone https://github.com/yugabyte/psycopg.git
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
