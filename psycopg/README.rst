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
honour the configured policy too.

The fork is `pre-release`_. Install with ``pip install --pre psycopg-yugabytedb``
or pin to ``3.3.4.1rc1``. It cannot coexist with upstream ``psycopg`` in the
same environment (both install into ``site-packages/psycopg/``).

.. _pre-release: https://packaging.python.org/en/latest/specifications/version-specifiers/#pre-releases


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
